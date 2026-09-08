"""HTTP surface: one Converse API in front of every provider."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import StreamingResponse

from ..core.registry import MODELS
from ..core.types import ConverseRequest, ConverseResponse, StreamEvent
from ..errors import HarnessError
from .deps import ServiceDep
from .security import PrincipalDep

router = APIRouter(prefix="/v1")


@router.get("/health", summary="Liveness plus per-provider credential status")
async def health(service: ServiceDep) -> dict[str, object]:
    """Unauthenticated on purpose: it reveals no transcript and no secret.

    It does report which credential *source* each provider resolved — a
    source name, never a value — which is what makes it useful to a probe or
    a deployment check.
    """
    status = service.provider_status()
    # "ok" requires a provider that both accepts requests and has a credential
    # this service could find. The Anthropic adapter reports itself available
    # even when detection found nothing (it resolves lazily, see
    # providers/credentials.py), so keying off `available` alone would report a
    # healthy service that cannot serve a single request.
    usable = [name for name, p in status.items() if p["available"] and p["credential_detected"]]
    return {
        "status": "ok" if usable else "degraded",
        "usable_providers": sorted(usable),
        "providers": status,
    }


@router.get("/whoami", summary="The calling principal's identity and limits")
async def whoami(principal: PrincipalDep) -> dict[str, object]:
    """Lets a client confirm a key is valid and render who is signed in.

    Returns no secret — not the key, not its hash.
    """
    return {
        "principal_id": principal.id,
        "allowed_models": (sorted(principal.allowed_models) if principal.allowed_models else None),
        "max_tokens_per_turn": principal.max_tokens_per_turn,
    }


@router.get("/models", summary="The routing and capability catalog")
async def list_models(service: ServiceDep, principal: PrincipalDep) -> dict[str, object]:
    """Filtered to what the calling principal may actually use.

    Models outside a principal's allowlist are omitted rather than listed as
    forbidden, so the catalog doubles as the answer to "what can I call?" and
    cannot be used to enumerate models this caller has no access to.
    """
    status = service.provider_status()
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
                "supports_sampling": spec.supports_sampling,
                "supports_images": spec.supports_images,
                "reasoning": spec.thinking.value,
                "aliases": list(spec.aliases),
            }
            for spec in sorted(MODELS.values(), key=lambda s: (s.provider.value, s.id))
            if principal.may_use(spec.id)
        ]
    }


@router.get("/tools", summary="Tools available to you, with their schemas")
async def list_tools(service: ServiceDep, principal: PrincipalDep) -> dict[str, object]:
    """Every registered tool, annotated with whether this principal may use it.

    Unlike the model catalog, forbidden tools are listed rather than hidden —
    a caller seeing `permitted: false` next to `dangerous: true` learns what
    to ask an operator for, and the tool names are not sensitive.
    """
    return {
        "tools": [
            {
                **tool.describe(),
                "permitted": principal.may_use_tool(tool.name, dangerous=tool.dangerous),
            }
            for tool in service.tools.all()
        ]
    }


@router.post(
    "/converse",
    response_model=ConverseResponse,
    summary="Send a turn and get the whole response",
)
async def converse(
    request: ConverseRequest,
    service: ServiceDep,
    principal: PrincipalDep,
    response: Response,
) -> ConverseResponse:
    result = await service.converse(request, principal)
    response.headers["X-Session-Id"] = result.session_id
    response.headers["X-Provider"] = result.provider
    return result


@router.post(
    "/converse-stream",
    summary="Send a turn and stream the response as canonical SSE events",
)
async def converse_stream(
    request: ConverseRequest, service: ServiceDep, principal: PrincipalDep
) -> StreamingResponse:
    session_id, events = await service.converse_stream(request, principal)

    async def body() -> AsyncIterator[bytes]:
        try:
            async for event in events:
                yield _sse(event)
        except HarnessError as exc:
            # The status line is already sent by this point, so a mid-stream
            # failure has to be reported inside the stream rather than as an
            # HTTP status. Clients must treat an `error` event as terminal.
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
    service: ServiceDep,
    principal: PrincipalDep,
    limit: int = Query(default=100, ge=1, le=1000),
    detail: bool = Query(
        default=False, description="Return per-session metadata instead of bare ids"
    ),
) -> dict[str, object]:
    if detail:
        return {"sessions": await service.list_session_summaries(principal, limit)}
    return {"session_ids": await service.list_sessions(principal, limit)}


@router.get("/sessions/{session_id}", summary="A session's metadata and transcript")
async def get_session(
    session_id: str,
    service: ServiceDep,
    principal: PrincipalDep,
    include_messages: bool = Query(
        default=False, description="Include the canonical transcript, not just metadata"
    ),
) -> dict[str, object]:
    session = await service.get_session(session_id, principal)
    if session is None:
        raise HTTPException(status_code=404, detail=f"No session '{session_id}'")

    payload = session.summary()
    if include_messages:
        payload["system"] = (
            [block.model_dump() for block in session.system] if session.system else None
        )
        payload["messages"] = [m.model_dump() for m in session.messages]
    return payload


@router.delete("/sessions/{session_id}", status_code=204, summary="Forget a session")
async def delete_session(session_id: str, service: ServiceDep, principal: PrincipalDep) -> Response:
    if not await service.delete_session(session_id, principal):
        raise HTTPException(status_code=404, detail=f"No session '{session_id}'")
    return Response(status_code=204)


def _sse(event: StreamEvent) -> bytes:
    payload = json.dumps(event.model_dump(exclude_none=True), separators=(",", ":"))
    return f"event: {event.type}\ndata: {payload}\n\n".encode()
