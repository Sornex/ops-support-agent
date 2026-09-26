"""Core agent loop.

Sends the conversation to the Claude Messages API, executes any tool calls the
model requests, feeds the results back, and repeats until the model produces a
final answer. The loop is written by hand (no framework) so each step of the
tool-use protocol stays visible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import anthropic
from dotenv import load_dotenv

from ops_agent.tools import ToolRegistry, default_registry

DEFAULT_MODEL = "claude-opus-5"
MAX_TOKENS = 16000
MAX_ITERATIONS = 10

SYSTEM_PROMPT = """\
You are an IT operations support agent for internal company staff.

- For any how-to or troubleshooting question, search the knowledge base first \
and answer from what it returns. Cite the article IDs you used, e.g. [vpn-002].
- If the knowledge base does not cover the question, say so plainly. Do not \
invent procedures, hostnames or policies.
- Only claim to have done something if a tool result confirms it. You can only \
use the tools you have been given.
- Keep answers short and practical: numbered steps for procedures, no filler.
"""


@dataclass(frozen=True)
class ToolCall:
    name: str
    input: dict[str, Any]
    result: str
    is_error: bool


@dataclass
class AgentReply:
    text: str
    stop_reason: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    iterations: int = 0


class OpsAgent:
    def __init__(
        self,
        client: anthropic.Anthropic,
        tools: ToolRegistry,
        model: str = DEFAULT_MODEL,
        max_iterations: int = MAX_ITERATIONS,
    ):
        self.client = client
        self.tools = tools
        self.model = model
        self.max_iterations = max_iterations
        self.messages: list[dict[str, Any]] = []

    @classmethod
    def from_env(cls) -> OpsAgent:
        load_dotenv()
        return cls(
            client=anthropic.Anthropic(),
            tools=default_registry(),
            model=os.environ.get("OPS_AGENT_MODEL", DEFAULT_MODEL),
        )

    def run(self, user_message: str) -> AgentReply:
        history_len = len(self.messages)
        self.messages.append({"role": "user", "content": user_message})
        try:
            return self._loop()
        except Exception:
            del self.messages[history_len:]
            raise

    def _loop(self) -> AgentReply:
        tool_calls: list[ToolCall] = []

        for iteration in range(1, self.max_iterations + 1):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=self.tools.schemas(),
                messages=self.messages,
            )
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "tool_use":
                self.messages.append(
                    {"role": "user", "content": self._run_tools(response, tool_calls)}
                )
                continue

            return AgentReply(
                text=self._final_text(response),
                stop_reason=response.stop_reason,
                tool_calls=tool_calls,
                iterations=iteration,
            )

        return AgentReply(
            text=(
                "I couldn't finish this request within the allowed number of "
                "steps. Please rephrase it or open a ticket."
            ),
            stop_reason="max_iterations",
            tool_calls=tool_calls,
            iterations=self.max_iterations,
        )

    def _run_tools(self, response: Any, tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
        # All results for one assistant turn go back in a single user message.
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = self.tools.execute(block.name, block.input)
            tool_calls.append(
                ToolCall(block.name, dict(block.input), result.content, result.is_error)
            )
            item: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result.content,
            }
            if result.is_error:
                item["is_error"] = True
            results.append(item)
        return results

    @staticmethod
    def _final_text(response: Any) -> str:
        if response.stop_reason == "refusal":
            return "I can't help with that request."
        text = "\n".join(b.text for b in response.content if b.type == "text").strip()
        if response.stop_reason == "max_tokens":
            text += "\n\n[Response was cut off: token limit reached.]"
        return text
