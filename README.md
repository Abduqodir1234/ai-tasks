# AI Tasks

Small Python experiments with LLMs: calling OpenAI, Gemini, Claude and local Ollama models, scraping websites with Playwright, a multi-agent conversation, Gradio chat apps with tool calling, a Python → C++ converter, and a RAG assistant over the PostHog docs.

## Project structure

Each project folder has its own README with setup and run details.

```
.
├── src/
│   ├── 01_llm_basics/          # One chat call per provider, a website summarizer, a best-price finder
│   ├── 02_multi_agent_chat/    # 3 LLM "family members" plan a trip, a judge LLM validates replies
│   ├── 03_chat_using_gradio/   # Gradio apps: plain chat, brochure writer, shop and airline assistants
│   ├── code_converter/         # Convert Python to C++ with an LLM, run both and compare
│   ├── assistant_in_posthog/   # RAG assistant over the PostHog docs and handbook, with an eval harness
│   └── vector_db/              # Chroma db written by assistant_in_posthog/ingest.py (not committed)
├── .env.example
├── pyproject.toml / uv.lock    # uv project files
└── requirements.txt            # same dependencies, for plain pip
```

| Project | README |
|---|---|
| LLM basics | [src/01_llm_basics](src/01_llm_basics/README.md) |
| Multi-agent chat | [src/02_multi_agent_chat](src/02_multi_agent_chat/README.md) |
| Gradio chat apps | [src/03_chat_using_gradio](src/03_chat_using_gradio/README.md) |
| Code converter | [src/code_converter](src/code_converter/README.md) |
| PostHog assistant | [src/assistant_in_posthog](src/assistant_in_posthog/README.md) |

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [Ollama](https://ollama.com) for the local-model projects. Pull the models the project you run needs:
  ```bash
  ollama pull llama3.2
  ollama pull qwen3:1.7b
  ollama pull qwen3:8b
  ollama pull gemma3:270m   # brochure creator
  ollama pull gpt-oss       # PostHog assistant
  ```
- A C++ compiler available as `c++`, for the code converter

## Setup

### With uv (recommended)

```bash
uv sync
uv run playwright install chromium
cp .env.example .env   # then add your API keys
```

### With pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
```

## Environment variables

| Variable         | Used by |
|------------------|---------|
| `OPENAI_API_KEY` | `openai_fun_fact.py`, `best_price_finder.py`, `family_trip_discussion.py`, `airline-assistant.py`, `code_converter` |
| `GEMINI_API_KEY` | `gemini_fun_fact.py` |
| `CLAUDE_API_KEY` | `code_converter` (Claude models) |
| `HF_TOKEN`       | `assistant_in_posthog` (Hugging Face embedding and reranker models) |

Ollama runs locally on `http://localhost:11434` and needs no key.

## Running

```bash
# 01 - LLM basics
uv run src/01_llm_basics/openai_fun_fact.py
uv run src/01_llm_basics/gemini_fun_fact.py
uv run src/01_llm_basics/ollama_fun_fact.py
uv run src/01_llm_basics/website_summarizer.py
uv run src/01_llm_basics/best_price_finder.py https://example-shop.com "iphone 15"

# 02 - Multi-agent chat
uv run src/02_multi_agent_chat/family_trip_discussion.py

# 03 - Gradio chat apps
uv run src/03_chat_using_gradio/real-chat-with-gradio.py
uv run src/03_chat_using_gradio/brochure_creator_using_gradio.py
uv run src/03_chat_using_gradio/assistant-in-shopping.py
uv run src/03_chat_using_gradio/airline-assistant.py

# Code converter
uv run src/code_converter/main.py

# PostHog assistant (see its README for ingest and eval)
uv run src/assistant_in_posthog/app.py
```

Without uv, activate the venv and use `python` instead of `uv run`.

## Updating dependencies

```bash
uv add <package>
uv export --no-hashes --no-header --no-emit-project -o requirements.txt
```
