"""Tool registry for the agent.

Each tool is a name, a description, a JSON schema and a handler. The registry
turns them into the `tools` parameter of the Messages API and executes calls
safely: unknown tools and handler exceptions become error results the model
can read and recover from, instead of crashing the agent loop.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ops_agent.knowledge_base import KnowledgeBase

ToolHandler = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler

    def to_api_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "strict": True,
            "input_schema": self.input_schema,
        }


@dataclass(frozen=True)
class ToolResult:
    content: str
    is_error: bool = False


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool]):
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"Duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.to_api_schema() for tool in self._tools.values()]

    def execute(self, name: str, tool_input: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(f"Unknown tool: {name}", is_error=True)
        try:
            return ToolResult(tool.handler(tool_input))
        except Exception as exc:  # noqa: BLE001 - the model should see any tool failure
            return ToolResult(f"Tool '{name}' failed: {exc}", is_error=True)


def build_knowledge_base_tool(kb: KnowledgeBase, top_k: int = 3) -> Tool:
    def handler(tool_input: dict[str, Any]) -> str:
        results = kb.search(tool_input["query"], top_k=top_k)
        if not results:
            return "No matching knowledge base articles found."
        return "\n\n".join(
            f"[{r.entry.id}] {r.entry.title}\n{r.entry.content}" for r in results
        )

    return Tool(
        name="search_knowledge_base",
        description=(
            "Search the internal IT support knowledge base (VPN, accounts and "
            "passwords, MFA, email, printers, Wi-Fi, software requests, internal "
            "services). Returns the most relevant articles, each prefixed with "
            "its ID. Use short keyword queries, e.g. 'vpn dns internal sites'. "
            "Call it again with different keywords if the first results miss."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords describing the user's problem.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def default_registry(kb: KnowledgeBase | None = None) -> ToolRegistry:
    kb = kb or KnowledgeBase.from_file()
    return ToolRegistry([build_knowledge_base_tool(kb)])
