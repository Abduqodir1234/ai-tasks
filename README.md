# AI Tasks

Small Python experiments with LLMs: calling OpenAI, Gemini and local Ollama models, scraping websites with Playwright, and a multi-agent conversation.

## Project structure

```
.
├── src/
│   ├── 01_llm_basics/
│   │   ├── openai_fun_fact.py       # Simple OpenAI chat completion
│   │   ├── gemini_fun_fact.py       # Gemini through its OpenAI-compatible API
│   │   ├── ollama_fun_fact.py       # Local Llama 3.2 through Ollama
│   │   ├── website_summarizer.py    # Scrape a page with Playwright, summarize with Ollama
│   │   └── best_price_finder.py     # Search a shop site, let GPT pick the cheapest offers
│   └── 02_multi_agent_chat/
│       └── family_trip_discussion.py  # 3 LLM "family members" plan a trip, a judge LLM validates replies
├── .env.example
├── pyproject.toml / uv.lock     # uv project files
└── requirements.txt             # same dependencies, for plain pip
```

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [Ollama](https://ollama.com) for the local-model scripts, with these models pulled:
  ```bash
  ollama pull llama3.2
  ollama pull qwen3:1.7b
  ollama pull qwen3:8b
  ```

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

| Variable         | Used by                                                                 |
|------------------|-------------------------------------------------------------------------|
| `OPENAI_API_KEY` | `openai_fun_fact.py`, `best_price_finder.py`, `family_trip_discussion.py` |
| `GEMINI_API_KEY` | `gemini_fun_fact.py`                                                    |

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
```

Without uv, activate the venv and use `python` instead of `uv run`.

## Updating dependencies

```bash
uv add <package>
uv export --no-hashes --no-header --no-emit-project -o requirements.txt
```
