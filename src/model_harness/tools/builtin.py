"""The built-in tool set.

Three tools, in increasing order of how much trust they need:

* ``get_current_time`` — reads the clock. No side effects, no arguments worth
  abusing. Solves the single most common "the model is dumb" complaint.
* ``http_request`` — fetches a URL. Read-only by default and refuses internal
  addresses; see ``net.py`` for why that matters.
* ``run_python`` — executes code in a subprocess. Genuinely dangerous and
  **not a security sandbox**; read its docstring before enabling it.

Every handler returns a string. Anything a handler wants the model to know
about a failure goes in that string via :class:`ToolError`, never an
exception that would abort the turn.
"""

from __future__ import annotations

import asyncio
import json
import os
import resource
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from .base import Tool, ToolError, ToolRegistry, truncate
from .net import AddressRejected, resolve_and_check

# --- clock ----------------------------------------------------------------


async def _get_current_time(args: dict[str, Any]) -> str:
    zone_name = args.get("timezone") or "UTC"
    try:
        zone = UTC if zone_name.upper() == "UTC" else ZoneInfo(zone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ToolError(
            f"Unknown timezone '{zone_name}'. Use an IANA name such as 'Europe/Amsterdam' or 'UTC'."
        ) from exc

    now = datetime.now(zone)
    return json.dumps(
        {
            "iso8601": now.isoformat(),
            "human": now.strftime("%A %d %B %Y, %H:%M:%S"),
            "timezone": zone_name,
            "utc_offset": now.strftime("%z"),
            "unix": int(now.timestamp()),
        },
        indent=2,
    )


GET_CURRENT_TIME = Tool(
    name="get_current_time",
    description=(
        "Get the current date and time. Use this whenever the answer depends "
        "on what time it is now — the current date, the day of the week, how "
        "long until something, or any 'today'/'now' question. Optionally pass "
        "an IANA timezone name to get the time there."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "IANA timezone, e.g. 'Europe/Amsterdam'. Defaults to UTC.",
            }
        },
        "required": [],
        "additionalProperties": False,
    },
    handler=_get_current_time,
)


# --- http -----------------------------------------------------------------

_MAX_BODY_BYTES = 512_000
_MAX_REDIRECTS = 3
_HTTP_TIMEOUT = 15.0
_SAFE_METHODS = frozenset({"GET", "HEAD"})


def _allowed_hosts() -> frozenset[str] | None:
    raw = os.environ.get("HARNESS_TOOL_HTTP_ALLOWED_HOSTS", "").strip()
    if not raw:
        return None
    return frozenset(h.strip().lower() for h in raw.split(",") if h.strip())


def _write_methods_enabled() -> bool:
    return os.environ.get("HARNESS_TOOL_HTTP_ALLOW_WRITES", "").lower() in {
        "1",
        "true",
        "yes",
    }


async def _http_request(args: dict[str, Any]) -> str:
    url = str(args.get("url") or "").strip()
    if not url:
        raise ToolError("The 'url' argument is required.")

    method = str(args.get("method") or "GET").upper()
    if method not in _SAFE_METHODS and not _write_methods_enabled():
        raise ToolError(
            f"Method {method} is not enabled on this deployment; only "
            f"{sorted(_SAFE_METHODS)} are available. A request that changes "
            "remote state has to be turned on by an operator."
        )

    allowed = _allowed_hosts()
    hops: list[str] = []
    current = url

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=False) as client:
        for _ in range(_MAX_REDIRECTS + 1):
            try:
                # Revalidated every hop: a redirect is a fresh destination
                # chosen by the remote server, not by us.
                await resolve_and_check(current, allowed_hosts=allowed)
            except AddressRejected as exc:
                raise ToolError(str(exc)) from exc

            try:
                response = await client.request(
                    method,
                    current,
                    headers={"user-agent": "model-harness/0.2 (+tool http_request)"},
                    content=args.get("body") if method not in _SAFE_METHODS else None,
                )
            except httpx.TimeoutException as exc:
                raise ToolError(
                    f"Request to {current} timed out after {_HTTP_TIMEOUT:.0f}s."
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
            raise ToolError(f"Too many redirects (more than {_MAX_REDIRECTS}).")

    body = response.content[:_MAX_BODY_BYTES]
    try:
        text = body.decode(response.encoding or "utf-8", errors="replace")
    except (LookupError, UnicodeDecodeError):
        text = body.decode("utf-8", errors="replace")

    payload = {
        "status": response.status_code,
        "url": str(response.url),
        "content_type": response.headers.get("content-type", ""),
        "redirect_chain": hops or None,
        "truncated": len(response.content) > len(body),
        "body": text,
    }
    return json.dumps({k: v for k, v in payload.items() if v is not None}, indent=2)


HTTP_REQUEST = Tool(
    name="http_request",
    description=(
        "Fetch a public URL over HTTP or HTTPS and return the status and body. "
        "Use this to read documentation, call a public API, or check a web "
        "page. Only public internet addresses work: requests to localhost, "
        "private networks, and cloud metadata endpoints are refused. Large "
        "bodies are truncated, so prefer specific URLs over whole sites."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Absolute http(s) URL to fetch."},
            "method": {
                "type": "string",
                "enum": ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"],
                "description": "HTTP method. Only GET and HEAD unless writes are enabled.",
            },
            "body": {
                "type": "string",
                "description": "Request body, for write methods when enabled.",
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    },
    handler=_http_request,
    dangerous=True,
)


# --- code execution -------------------------------------------------------

_PY_TIMEOUT = 20.0
_PY_MEMORY_BYTES = 512 * 1024 * 1024


def _apply_limits() -> None:
    """Pre-exec limits for the child process.

    CPU and address space are capped so a runaway loop or allocation cannot
    take the host down, and core dumps are disabled. These bound *accidents*.
    They are not a security boundary — see the tool's docstring.
    """
    resource.setrlimit(resource.RLIMIT_CPU, (int(_PY_TIMEOUT), int(_PY_TIMEOUT) + 2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (_PY_MEMORY_BYTES, _PY_MEMORY_BYTES))
    except (ValueError, OSError):
        # Some platforms refuse RLIMIT_AS; the timeout still applies.
        pass


async def _run_python(args: dict[str, Any]) -> str:
    code = args.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ToolError("The 'code' argument is required and must be Python source.")

    workdir = tempfile.mkdtemp(prefix="harness-py-")
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",  # isolated: ignore PYTHON* env vars and the user's site-packages
            "-c",
            code,
            cwd=workdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # A deliberately bare environment: nothing inherited means no API
            # keys, no tokens, no provider credentials visible to the code.
            env={"PATH": "/usr/bin:/bin", "HOME": workdir, "TMPDIR": workdir},
            preexec_fn=_apply_limits,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=_PY_TIMEOUT)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise ToolError(
                f"Execution exceeded {_PY_TIMEOUT:.0f}s and was killed. Make the "
                "code faster or split the work."
            ) from None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    return json.dumps(
        {
            "exit_code": process.returncode,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
        },
        indent=2,
    )


RUN_PYTHON = Tool(
    name="run_python",
    description=(
        "Execute a short Python 3 program and return its stdout, stderr and "
        "exit code. Use it for calculation, data manipulation, parsing, and "
        "anything easier to compute than to reason about. The program runs "
        "with no network access to internal systems and no access to this "
        "service's credentials, in a temporary directory that is deleted "
        "afterwards. Print what you want to see — nothing is returned "
        "implicitly. Only the standard library is available."
    ),
    input_schema={
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python 3 source to execute."}},
        "required": ["code"],
        "additionalProperties": False,
    },
    handler=_run_python,
    dangerous=True,
)


def default_registry() -> ToolRegistry:
    """Every built-in tool.

    Registration is not permission: a tool still has to be requested per turn
    and granted to the calling principal. ``dangerous`` tools additionally
    require an explicit grant.
    """
    return ToolRegistry([GET_CURRENT_TIME, HTTP_REQUEST, RUN_PYTHON])


async def execute(tool: Tool, arguments: dict[str, Any]) -> tuple[str, bool, int]:
    """Run one tool call. Returns ``(output, is_error, duration_ms)``.

    Never raises. A tool that raises would abort the whole turn and discard
    the work already done in the loop; the model can instead read the error
    and try something else, which is usually what a person would do.
    """
    started = time.perf_counter()
    try:
        output = await tool.handler(arguments)
        is_error = False
    except ToolError as exc:
        output, is_error = f"Error: {exc}", True
    except Exception as exc:  # noqa: BLE001 - a tool bug must not kill the turn
        output, is_error = f"Error: {type(exc).__name__}: {exc}", True

    duration_ms = int((time.perf_counter() - started) * 1000)
    return truncate(output, tool.max_output_chars), is_error, duration_ms
