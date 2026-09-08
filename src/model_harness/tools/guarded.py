"""The tool implementations worth keeping after the Strands migration.

Strands supplies the tool loop, schema generation and an MCP client, so our
registry, our hand-written JSON Schemas and our execution wrapper are gone.
These two implementations are not, for different reasons:

* **HTTP.** Strands' vended ``http_request`` is 35 lines and accepts any URL
  with any method — no address validation at all. Handing a model that tool
  hands it cloud instance metadata at 169.254.169.254, private subnets, and
  this service on loopback. The guards in ``net.py`` are the reason ours stays.
* **The clock.** Trivial, but it is the single most common thing a model is
  asked and cannot know, and Strands does not ship one.

Code execution is deliberately absent. Strands ships real sandboxes
(``strands.sandbox``: docker, ssh, posix_shell) which are strictly better than
the bare subprocess we had, so re-implementing it here would be a regression.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from .net import AddressRejected, resolve_and_check

MAX_BODY_BYTES = 512_000
MAX_REDIRECTS = 3
HTTP_TIMEOUT = 15.0
SAFE_METHODS = frozenset({"GET", "HEAD"})
MAX_OUTPUT_CHARS = 20_000


class ToolError(Exception):
    """An expected failure, written for the model that will read it.

    Say what was wrong with the arguments and what would work instead. The
    caller turns this into a returned string rather than a raised exception,
    so the model can recover instead of losing the turn.
    """


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Cap a tool result, telling the model it was cut.

    Silent truncation is worse than none: the model reasons over what looks
    like a complete answer. Saying so lets it narrow the request instead.
    """
    if len(text) <= limit:
        return text
    return (
        text[:limit] + f"\n\n[truncated: output was {len(text)} characters, limit is "
        f"{limit}. Request a narrower slice if you need the rest.]"
    )


# --- clock ----------------------------------------------------------------


async def current_time(timezone: str = "UTC") -> str:
    try:
        zone = UTC if timezone.upper() == "UTC" else ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ToolError(
            f"Unknown timezone '{timezone}'. Use an IANA name such as 'Europe/Amsterdam' or 'UTC'."
        ) from exc

    now = datetime.now(zone)
    return json.dumps(
        {
            "iso8601": now.isoformat(),
            "human": now.strftime("%A %d %B %Y, %H:%M:%S"),
            "timezone": timezone,
            "utc_offset": now.strftime("%z"),
            "unix": int(now.timestamp()),
        },
        indent=2,
    )


# --- http -----------------------------------------------------------------


def allowed_hosts() -> frozenset[str] | None:
    raw = os.environ.get("HARNESS_TOOL_HTTP_ALLOWED_HOSTS", "").strip()
    if not raw:
        return None
    return frozenset(h.strip().lower() for h in raw.split(",") if h.strip())


def write_methods_enabled() -> bool:
    return os.environ.get("HARNESS_TOOL_HTTP_ALLOW_WRITES", "").lower() in {
        "1",
        "true",
        "yes",
    }


async def http_fetch(url: str, method: str = "GET", body: str | None = None) -> str:
    url = (url or "").strip()
    if not url:
        raise ToolError("The 'url' argument is required.")

    method = (method or "GET").upper()
    if method not in SAFE_METHODS and not write_methods_enabled():
        raise ToolError(
            f"Method {method} is not enabled on this deployment; only "
            f"{sorted(SAFE_METHODS)} are available. A request that changes remote "
            "state has to be turned on by an operator."
        )

    allowed = allowed_hosts()
    hops: list[str] = []
    current = url

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=False) as client:
        for _ in range(MAX_REDIRECTS + 1):
            try:
                # Revalidated every hop: a redirect target is chosen by the
                # remote server, not by us.
                await resolve_and_check(current, allowed_hosts=allowed)
            except AddressRejected as exc:
                raise ToolError(str(exc)) from exc

            try:
                response = await client.request(
                    method,
                    current,
                    headers={"user-agent": "model-harness/0.3 (+tool http_request)"},
                    content=body if method not in SAFE_METHODS else None,
                )
            except httpx.TimeoutException as exc:
                raise ToolError(
                    f"Request to {current} timed out after {HTTP_TIMEOUT:.0f}s."
                ) from exc
            except httpx.HTTPError as exc:
                raise ToolError(f"Request to {current} failed: {exc}") from exc

            if response.is_redirect and response.headers.get("location"):
                hops.append(current)
                current = str(response.next_request.url) if response.next_request else ""
                if not current:
                    break
                continue
            break
        else:
            raise ToolError(f"Too many redirects (more than {MAX_REDIRECTS}).")

    raw = response.content[:MAX_BODY_BYTES]
    try:
        text = raw.decode(response.encoding or "utf-8", errors="replace")
    except (LookupError, UnicodeDecodeError):
        text = raw.decode("utf-8", errors="replace")

    payload: dict[str, Any] = {
        "status": response.status_code,
        "url": str(response.url),
        "content_type": response.headers.get("content-type", ""),
        "redirect_chain": hops or None,
        "truncated": len(response.content) > len(raw),
        "body": text,
    }
    return json.dumps({k: v for k, v in payload.items() if v is not None}, indent=2)
