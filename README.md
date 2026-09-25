# Ops Support Agent

An AI support agent that answers IT/ops questions from a knowledge base and can
safely take action — creating tickets, checking service status, restarting
services — through explicit, logged tool calls with human confirmation on
anything destructive.

Built to demonstrate practical **agentic tool use**: an LLM that doesn't just
chat, but plans which tool to call, calls it, and reports back — with
guardrails so it never does anything risky without confirmation.

## Why this exists

Most "AI agent" demos are RAG chatbots with no side effects. This one actually
*does* things — but every action-tool is logged, and anything irreversible
(like restarting a service) requires an explicit `--confirm` before it runs.
That guardrail pattern is the interesting part of this project.

## Features (planned / in progress)

- [ ] Knowledge-base lookup (FAQ-style retrieval, no vector DB needed for v1)
- [ ] Claude API integration with tool use (function calling)
- [ ] `create_ticket` tool — logs a structured support ticket to disk
- [ ] `check_service_status` tool — read-only, safe to call anytime
- [ ] `restart_service` tool — destructive, requires explicit confirmation
- [ ] Full audit log of every tool call the agent makes
- [ ] CLI chat interface

## Tech stack

- Python 3.11+
- [Anthropic Claude API](https://docs.anthropic.com/) (tool use / function calling)
- No framework — the agent loop is written by hand to keep the mechanics visible

## Project structure

```
ops-support-agent/
├── src/ops_agent/
│   ├── agent.py            # Core agent loop (Claude API + tool dispatch)
│   ├── tools.py             # Tool definitions and implementations
│   ├── knowledge_base.py    # FAQ lookup
│   └── cli.py                # Command-line chat interface
├── data/
│   └── knowledge_base.json  # Sample support FAQ
└── tests/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env           # add your ANTHROPIC_API_KEY
```

## Status

Early development — building this incrementally, one commit per working
step. See commit history for progress.
