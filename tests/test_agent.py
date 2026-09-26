from types import SimpleNamespace

import pytest

from ops_agent.agent import SYSTEM_PROMPT, OpsAgent
from ops_agent.tools import Tool, ToolRegistry, default_registry


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use_block(block_id, name, tool_input):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=tool_input)


def response(stop_reason, *blocks):
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks))


class FakeClient:
    """Replays scripted responses and records every request."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        # Snapshot: the agent mutates its history list after each call.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def echo_registry():
    echo = Tool(
        name="echo",
        description="Echo the text back.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        handler=lambda tool_input: f"echo:{tool_input['text']}",
    )
    return ToolRegistry([echo])


def test_direct_answer_without_tools():
    client = FakeClient(response("end_turn", text_block("Hello!")))
    agent = OpsAgent(client, echo_registry(), model="test-model")

    reply = agent.run("hi")

    assert reply.text == "Hello!"
    assert reply.stop_reason == "end_turn"
    assert reply.tool_calls == []
    assert reply.iterations == 1
    request = client.requests[0]
    assert request["model"] == "test-model"
    assert request["system"] == SYSTEM_PROMPT
    assert request["tools"][0]["name"] == "echo"
    assert request["messages"] == [{"role": "user", "content": "hi"}]


def test_tool_call_round_trip():
    tool_turn = response(
        "tool_use",
        text_block("Let me check."),
        tool_use_block("toolu_1", "echo", {"text": "ping"}),
    )
    client = FakeClient(tool_turn, response("end_turn", text_block("Done: echo:ping")))
    agent = OpsAgent(client, echo_registry())

    reply = agent.run("say ping")

    assert reply.text == "Done: echo:ping"
    assert reply.iterations == 2
    assert [(c.name, c.input, c.result, c.is_error) for c in reply.tool_calls] == [
        ("echo", {"text": "ping"}, "echo:ping", False)
    ]

    second_request = client.requests[1]["messages"]
    assert [m["role"] for m in second_request] == ["user", "assistant", "user"]
    assert second_request[1]["content"] is tool_turn.content
    assert second_request[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "echo:ping"}
    ]


def test_parallel_tool_calls_return_results_in_one_message():
    client = FakeClient(
        response(
            "tool_use",
            tool_use_block("toolu_a", "echo", {"text": "a"}),
            tool_use_block("toolu_b", "echo", {"text": "b"}),
        ),
        response("end_turn", text_block("ok")),
    )
    agent = OpsAgent(client, echo_registry())

    reply = agent.run("do both")

    results = client.requests[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["toolu_a", "toolu_b"]
    assert len(reply.tool_calls) == 2


def test_unknown_tool_becomes_error_result_and_loop_continues():
    client = FakeClient(
        response("tool_use", tool_use_block("toolu_1", "nope", {})),
        response("end_turn", text_block("Sorry, can't do that.")),
    )
    agent = OpsAgent(client, echo_registry())

    reply = agent.run("use nope")

    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    assert "Unknown tool" in result["content"]
    assert reply.text == "Sorry, can't do that."
    assert reply.tool_calls[0].is_error


def test_tool_exception_becomes_error_result():
    def boom(_tool_input):
        raise RuntimeError("disk on fire")

    registry = ToolRegistry([Tool("boom", "Always fails.", {"type": "object"}, boom)])
    client = FakeClient(
        response("tool_use", tool_use_block("toolu_1", "boom", {})),
        response("end_turn", text_block("It failed.")),
    )
    agent = OpsAgent(client, registry)

    agent.run("go")

    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    assert "disk on fire" in result["content"]


def test_max_iterations_stops_a_runaway_loop():
    looping = [
        response("tool_use", tool_use_block(f"toolu_{i}", "echo", {"text": "x"}))
        for i in range(3)
    ]
    client = FakeClient(*looping)
    agent = OpsAgent(client, echo_registry(), max_iterations=3)

    reply = agent.run("loop forever")

    assert reply.stop_reason == "max_iterations"
    assert reply.iterations == 3
    assert len(client.requests) == 3


def test_history_persists_across_turns():
    client = FakeClient(
        response("end_turn", text_block("first")),
        response("end_turn", text_block("second")),
    )
    agent = OpsAgent(client, echo_registry())

    agent.run("one")
    agent.run("two")

    roles = [m["role"] for m in client.requests[1]["messages"]]
    assert roles == ["user", "assistant", "user"]


def test_api_error_rolls_back_history():
    client = FakeClient(RuntimeError("network down"), response("end_turn", text_block("ok")))
    agent = OpsAgent(client, echo_registry())

    with pytest.raises(RuntimeError):
        agent.run("first try")
    assert agent.messages == []

    agent.run("second try")
    assert client.requests[1]["messages"] == [{"role": "user", "content": "second try"}]


def test_refusal_returns_safe_message():
    client = FakeClient(response("refusal"))
    reply = OpsAgent(client, echo_registry()).run("something bad")
    assert reply.stop_reason == "refusal"
    assert reply.text == "I can't help with that request."


def test_max_tokens_is_flagged_in_text():
    client = FakeClient(response("max_tokens", text_block("partial answer")))
    reply = OpsAgent(client, echo_registry()).run("long question")
    assert reply.text.startswith("partial answer")
    assert "cut off" in reply.text


def test_registry_rejects_duplicate_names():
    tool = Tool("dup", "d", {"type": "object"}, lambda _i: "")
    with pytest.raises(ValueError):
        ToolRegistry([tool, tool])


def test_default_registry_exposes_strict_knowledge_base_tool():
    schemas = default_registry().schemas()
    assert [s["name"] for s in schemas] == ["search_knowledge_base"]
    assert schemas[0]["strict"] is True
    assert schemas[0]["input_schema"]["additionalProperties"] is False


def test_knowledge_base_tool_returns_articles_with_ids():
    result = default_registry().execute("search_knowledge_base", {"query": "vpn dns internal"})
    assert not result.is_error
    assert "[vpn-002]" in result.content


def test_knowledge_base_tool_reports_no_match():
    result = default_registry().execute("search_knowledge_base", {"query": "zzzqqq"})
    assert result.content == "No matching knowledge base articles found."
