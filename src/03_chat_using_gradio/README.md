# Chat Apps with Gradio

Four Gradio apps built on local Ollama models through [LiteLLM](https://docs.litellm.ai): a plain chat, a brochure writer, and two tool-calling assistants backed by SQLite.

## Apps

| Script | Model | What it does |
|---|---|---|
| `real-chat-with-gradio.py` | `qwen3:8b` | Streaming chat with a custom assistant persona ("Nova") |
| `brochure_creator_using_gradio.py` | `qwen3:8b`, `llama3.2` or `gemma3:270m` | Scrapes a company homepage and writes a Markdown brochure |
| `assistant-in-shopping.py` | `qwen3:8b` | Clothing store assistant: browse items and place orders |
| `airline-assistant.py` | `qwen3:8b`, plus OpenAI `tts-1` and `gpt-image-2` | Flight search assistant with a spoken reply and a destination image |

## Setup

- Ollama running on `localhost:11434`, with the models pulled:
  ```bash
  ollama pull qwen3:8b
  ollama pull llama3.2
  ollama pull gemma3:270m
  ```
- Chromium for Playwright, for the brochure creator: `uv run playwright install chromium`.
- `OPENAI_API_KEY` in `.env`, for the airline assistant's voice and images.

## Run

From this folder:

```bash
uv run python real-chat-with-gradio.py
uv run python brochure_creator_using_gradio.py
uv run python assistant-in-shopping.py
uv run python airline-assistant.py
```

Each app opens in your browser.

## Details

### `real-chat-with-gradio.py`
- A `gr.ChatInterface` that converts Gradio's chat history to LiteLLM messages and streams the reply.
- The system prompt gives the model the persona "Nova". It tells the model not to reveal the underlying model and to reply in the user's language (Uzbek, Russian or English).

### `brochure_creator_using_gradio.py`
- Input: a model, a company name and a URL. The page's visible text and links are scraped with Playwright.
- Each model has its own prompt and brochure style:
  - `qwen3:8b` writes a business brochure for clients and investors.
  - `llama3.2` writes a friendly one for customers and future employees.
  - `gemma3:270m` writes a short bullet-point summary.
- All prompts use only facts found on the page.

### `assistant-in-shopping.py`
- "Nova", the assistant for a clothing store. On start it creates `shop.db` (SQLite) if it's missing. If the tables are empty, it fills them with categories, 12 sample items and generated test items, all priced in UZS.
- Tools:
  - `get_category_list` lists the categories.
  - `get_paged_list_of_items_in_shop` filters by category, name search and stock, a page at a time.
  - `create_order` checks the items, sizes, colors and stock and returns a summary. It does not place the order.
  - `confirm_order` places the order prepared by `create_order`.
- Two-step ordering: `create_order` stores a pending order per chat session. `confirm_order` refuses to run in the same turn, so the customer must see the summary and confirm it in a later message.

### `airline-assistant.py`
- An assistant for the fictional airline "Airsoft Airway". The system prompt holds the fare types, baggage, child, refund and check-in rules.
- Tool: `ticketListByCity` searches the `flights` table in `airline.db` by origin, destination and an exact date or date range. It returns 10 flights per page.
- `airline.db` is not created by the script and is not in git. It must exist next to the script, with a `flights` table.
- After each reply:
  - A second `qwen3:8b` call works out the destination the user is currently asking about, and only counts it if flights to it were found. When there is one, `gpt-image-2` draws a travel poster of that city.
  - `tts-1` reads the reply aloud.
- The UI is password-protected, with username `test` and password `test`.
