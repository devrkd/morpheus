"""HTTP surface: one Converse API in front of the harness."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import StreamingResponse

from ..core.registry import MODELS
from ..core.types import ConverseRequest, ConverseResponse, StreamEvent
from ..errors import HarnessError
from .deps import RunnerDep
from .security import PrincipalDep

router = APIRouter(prefix="/v1")


@router.get("/health", summary="Liveness plus per-provider credential status")
async def health(runner: RunnerDep) -> dict[str, object]:
    """Unauthenticated on purpose: it reveals no transcript and no secret.

    It reports which credential *source* each provider resolved — a source
    name, never a value — which is what makes it useful to a deployment check.
    """
    status = runner.provider_status()
    usable = [name for name, p in status.items() if p["available"] and p["credential_detected"]]
    return {
        "status": "ok" if usable else "degraded",
        "usable_providers": sorted(usable),
        "providers": status,
    }


@router.get("/whoami", summary="The calling principal's identity and limits")
async def whoami(principal: PrincipalDep) -> dict[str, object]:
    """Lets a client validate a key and render who is signed in.

    Returns no secret — not the key, not its hash.
    """
    return {
        "principal_id": principal.id,
        "allowed_models": (sorted(principal.allowed_models) if principal.allowed_models else None),
        "allowed_tools": (sorted(principal.allowed_tools) if principal.allowed_tools else None),
        "max_tokens_per_turn": principal.max_tokens_per_turn,
    }


@router.get("/models", summary="The routing and capability catalog")
async def list_models(runner: RunnerDep, principal: PrincipalDep) -> dict[str, object]:
    """Filtered to what the calling principal may actually use.

    Models outside the allowlist are omitted rather than listed as forbidden,
    so the catalog doubles as the answer to "what can I call?" and cannot be
    used to enumerate models this caller has no access to.
    """
    status = runner.provider_status()
    return {
        "models": [
            {
                "id": spec.id,
                "provider": spec.provider.value,
                "available": bool(status.get(spec.provider.value, {}).get("available")),
                "context_window": spec.context_window,
                "max_output_tokens": spec.max_output_tokens,
                "supports_effort": spec.supports_effort,
                "max_effort": spec.max_effort.value,
                "supports_images": spec.supports_images,
                "reasoning": spec.thinking.value,
                "aliases": list(spec.aliases),
            }
            for spec in sorted(MODELS.values(), key=lambda s: (s.provider.value, s.id))
            if principal.may_use(spec.id)
        ]
    }


@router.get("/tools", summary="Tools available to you, with their schemas")
async def list_tools(runner: RunnerDep, principal: PrincipalDep) -> dict[str, object]:
    """Every registered tool, annotated with whether this principal may use it.

    Unlike the model catalog, forbidden tools are listed rather than hidden — a
    caller seeing `permitted: false` next to `dangerous: true` learns what to
    ask an operator for, and tool names are not sensitive.
    """
    return {
        "tools": [
            {
                **spec,
                "permitted": principal.may_use_tool(spec["name"], dangerous=spec["dangerous"]),
            }
            for spec in runner.catalog.describe()
        ]
    }


@router.post(
    "/converse",
    response_model=ConverseResponse,
    summary="Send a turn and get the whole response",
)
async def converse(
    request: ConverseRequest,
    runner: RunnerDep,
    principal: PrincipalDep,
    response: Response,
) -> ConverseResponse:
    result = await runner.converse(request, principal)
    response.headers["X-Session-Id"] = result.session_id
    response.headers["X-Provider"] = result.provider
    return result


@router.post(
    "/converse-stream",
    summary="Send a turn and stream the response, tool calls included",
)
async def converse_stream(
    request: ConverseRequest, runner: RunnerDep, principal: PrincipalDep
) -> StreamingResponse:
    session_id, events = await runner.converse_stream(request, principal)

    async def body() -> AsyncIterator[bytes]:
        try:
            async for event in events:
                yield _sse(event)
        except HarnessError as exc:
            # The status line is already sent, so a mid-stream failure has to
            # be reported inside the stream rather than as an HTTP status.
            # Same sanitization rule as the exception handler: only what
            # to_payload() considers safe reaches the caller.
            payload = exc.to_payload()["error"]
            yield _sse(
                StreamEvent(
                    type="error",
                    session_id=session_id,
                    code=str(payload["code"]),
                    message=str(payload["message"]),
                    error_id=str(payload["error_id"]),
                    provider=exc.provider,
                )
            )
        yield b"data: [DONE]\n\n"

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={
            "X-Session-Id": session_id,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sessions", summary="Your most recently used sessions")
async def list_sessions(
    runner: RunnerDep,
    principal: PrincipalDep,
    limit: int = Query(default=100, ge=1, le=1000),
    detail: bool = Query(
        default=False, description="Return per-session metadata instead of bare ids"
    ),
) -> dict[str, object]:
    records = await runner.list_sessions(principal, limit)
    if detail:
        return {"sessions": [r.summary() for r in records]}
    return {"session_ids": [r.session_id for r in records]}


@router.get("/sessions/{session_id}", summary="A session's metadata and transcript")
async def get_session(
    session_id: str,
    runner: RunnerDep,
    principal: PrincipalDep,
    include_messages: bool = Query(
        default=False, description="Include the transcript, not just metadata"
    ),
) -> dict[str, object]:
    record = await runner.get_session(session_id, principal)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No session '{session_id}'")

    payload = record.summary()
    if include_messages:
        payload["messages"] = await runner.read_transcript(session_id)
    return payload


@router.delete("/sessions/{session_id}", status_code=204, summary="Forget a session")
async def delete_session(session_id: str, runner: RunnerDep, principal: PrincipalDep) -> Response:
    if not await runner.delete_session(session_id, principal):
        raise HTTPException(status_code=404, detail=f"No session '{session_id}'")
    return Response(status_code=204)


def _sse(event: StreamEvent) -> bytes:
    payload = json.dumps(event.model_dump(exclude_none=True), separators=(",", ":"))
    return f"event: {event.type}\ndata: {payload}\n\n".encode()
