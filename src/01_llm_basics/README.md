# LLM Basics

First steps with LLM APIs: one chat call each to OpenAI, Gemini and a local Ollama model, then two small tools that feed a scraped web page to an LLM.

## Scripts

| Script | Model | What it does |
|---|---|---|
| `openai_fun_fact.py` | `gpt-4.1-nano` (OpenAI) | Asks for a fun fact |
| `gemini_fun_fact.py` | `gemini-3.1-pro-preview` (Gemini, via its OpenAI-compatible API) | Lists the available models, then asks for a fun fact |
| `ollama_fun_fact.py` | `llama3.2` (Ollama, via its OpenAI-compatible API) | Asks for a fun fact |
| `website_summarizer.py` | `llama3.2` (Ollama) | Opens a page with Playwright, keeps only the main content, and returns a JSON summary of the site |
| `best_price_finder.py` | `gpt-4.1-mini` (OpenAI) | Opens a shop site, finds its search box, searches for a product and ranks the cheapest matching offers |

All of them use the `openai` client; Gemini and Ollama are reached by changing `base_url`.

## Setup

- `OPENAI_API_KEY` in `.env` for `openai_fun_fact.py` and `best_price_finder.py`.
- `GEMINI_API_KEY` in `.env` for `gemini_fun_fact.py`.
- Ollama running on `localhost:11434` with `llama3.2` pulled, for the Ollama scripts.
- Chromium for Playwright: `uv run playwright install chromium`.

## Run

From this folder:

```bash
uv run python openai_fun_fact.py
uv run python gemini_fun_fact.py
uv run python ollama_fun_fact.py
uv run python website_summarizer.py
uv run python best_price_finder.py <website_url> "<product>"
```

- `website_summarizer.py` summarizes `https://asaxiy.uz/`. Change the URL at the bottom of the file to summarize another site.
- `best_price_finder.py` asks for the URL and product when no arguments are given.

## How the two tools work

**`website_summarizer.py`**
- Picks the page's main block: `main`, `article`, or the element holding the most paragraph text.
- Strips navigation, footers, banners and ads from it.
- Removes duplicate lines and sends the text to the model.
- The model returns JSON with `summary`, `site_type`, `audience`, `offerings`, `primary_action`, `language` and `confidence`.

**`best_price_finder.py`**
- Finds the search box in three ways, in order:
  1. A search field already on screen.
  2. A search icon that opens one.
  3. The first text field on the page.
- Recognizes search elements in English, Russian and Uzbek.
- After searching, scrolls to load lazy product cards.
- Sends up to 20,000 chars of page text and 150 links to the model.
- The model returns a Markdown table of the top 5 offers ranked by price, using only data from the page.
