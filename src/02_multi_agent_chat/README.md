# Multi-Agent Chat: Family Trip Discussion

Three LLMs play a family planning a weekend trip on a $300 budget. Each reply is addressed to one other family member, who answers next, and a judge LLM checks every reply before it is accepted.

## Agents

| Member | Role | Model |
|---|---|---|
| Alex, 24 | The planner: lists, backup plans, keeps everyone happy | `ollama_chat/qwen3:1.7b` |
| Dad, 52 | Driver and budget keeper | `openai/gpt-4.1` |
| Mia, 15 | Little sister | `ollama_chat/llama3.2` |
| Judge | Checks each reply stays in character and in format | `ollama_chat/qwen3:8b` |

All calls go through [LiteLLM](https://docs.litellm.ai) (`acompletion`), so cloud and local models are used the same way.

## How it works

1. The system opens by asking Alex for a plan.
2. Every reply must be one JSON object: `{"to": "Alex" | "Dad" | "Mia", "content": "..."}`. A member can't message themselves.
3. The judge receives the speaker's role, the message being answered and the reply. It returns `{"valid": bool, "reason": "..."}`.
4. Invalid JSON or a rejected reply is retried up to `maxAttempts` (3) times. If all attempts fail, the run stops.
5. The member named in `to` answers next. The run ends after `maxRounds` (100) turns.

All members share one chat history. It starts with the trip scenario, and every message is added as "From X to Y: ...". Each call sends that history with the member's own role prompt. The conversation is printed with [Rich](https://rich.readthedocs.io), with one color per speaker.

## Setup

- `OPENAI_API_KEY` in `.env`, for Dad.
- Ollama running on `localhost:11434`, with the local models pulled:
  ```bash
  ollama pull qwen3:1.7b
  ollama pull qwen3:8b
  ollama pull llama3.2
  ```

## Run

From this folder:

```bash
uv run python family_trip_discussion.py
```
