"""The tool contract and registry.

A tool is a name, a description, a JSON Schema for its arguments, and an async
handler. Nothing here is provider-specific: the adapters translate these into
Anthropic's ``tools`` and OpenAI's ``tools`` shapes, and translate the model's
requests back into :class:`~model_harness.core.types.ToolUseBlock`.

Two rules the whole design leans on:

* **The description is the interface.** It is the only documentation the model
  gets. A vague description produces wrong calls far more often than a bad
  schema does.
* **A tool must not raise at the model.** A handler that fails returns an error
  *result*, which the model can read and react to. Raising would abort the
  turn and throw away the work already done in the loop.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]


class ToolError(Exception):
    """Raised by a handler for an expected failure.

    The message is returned to the model as an error result, so write it for
    that reader: say what was wrong with the arguments and what would work
    instead.
    """


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler = field(compare=False)

    dangerous: bool = False
    """Whether this tool can affect anything outside the process.

    Surfaced in ``GET /v1/tools`` and required to be explicitly granted to a
    principal, so no caller gets side-effecting capability by default.
    """

    max_output_chars: int = 20_000
    """Ceiling on a single result.

    Tool output flows straight into the next request's context, so an
    unbounded result is both a cost problem and a way to blow the context
    window in one call.
    """

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "dangerous": self.dangerous,
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def all(self) -> list[Tool]:
        return [self._tools[name] for name in self.names()]

    def __len__(self) -> int:
        return len(self._tools)


def truncate(text: str, limit: int) -> str:
    """Cap a tool result, telling the model it was cut.

    Silent truncation is worse than none: the model reasons over what looks
    like a complete answer. Saying so lets it narrow the request instead.
    """
    if len(text) <= limit:
        return text
    return (
        text[:limit] + f"\n\n[truncated: output was {len(text)} characters, limit is {limit}. "
        "Request a narrower slice if you need the rest.]"
    )
