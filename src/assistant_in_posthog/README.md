# PostHog Assistant

A local chat assistant that answers questions about PostHog from its [docs](https://posthog.com/docs) and [company handbook](https://posthog.com/handbook). Pages are scraped into a Chroma vector database, and a tool-calling LLM searches it before every answer and cites the pages it used. Everything runs locally on Ollama and Hugging Face models.

## How it works

```
question
  └─ rewrite (gpt-oss)             make the latest message stand alone, using the chat history
      └─ agent (gpt-oss + tools)   must call search_posthog before replying
          └─ search_posthog        up to 8 query variants → Chroma (Qwen3-Embedding-0.6B)
              └─ rerank            BAAI/bge-reranker-v2-m3 cross-encoder → top_k chunks
          └─ answer                grounded in the results only, with source links
```

The agent loop in `llm_as_controller.py` guards against gpt-oss quirks:

- A reply with no search, or an empty reply, is nudged up to `MAX_NUDGES` times. If the model still hasn't searched, the search is run for it with the rewritten prompt.
- After `MAX_TOOL_ROUNDS` tool rounds, the model is asked to answer from the results it already has.
- Tool-call JSON that Ollama can't parse is retried up to `MAX_INVOKE_ATTEMPTS` times.
- An empty final answer is replaced with `EMPTY_ANSWER_FALLBACK`.

## Models

| Model | Where | Used for |
|---|---|---|
| `gpt-oss` (Ollama) | `llm_as_controller.py` | Agent: tool calls and the final answer; prompt rewriting |
| `BAAI/bge-reranker-v2-m3` (Hugging Face) | `llm_as_controller.py`, `llm.py` | Reranking retrieved chunks |
| `Qwen/Qwen3-Embedding-0.6B` (Hugging Face) | ingest and search | Embeddings; searches use its `query` prompt |
| `qwen3:8b` (Ollama) | `ingest.py`, `evaluation/evaluator.py` | Rewriting pages before chunking; LLM judge |

`llm.py` is the older pipeline without tool calling (`qwen3:8b` answers, `qwen3:1.7b` rewrites). The Gradio app uses `llm_as_controller.py`.

## Project structure

```
assistant_in_posthog/
├── app.py                    # Gradio chat UI
├── llm_as_controller.py      # Tool-calling agent (used by app.py and the evaluator)
├── llm.py                    # Older fixed pipeline: retrieve → rerank → answer
├── ingest.py                 # Scrape docs + handbook into the vector db
├── vector_db_ui.py           # Gradio UI to inspect the db: stats and a t-SNE plot of the vectors
├── constants/
│   └── assistant_constant.py # Model names and all prompts
├── tools/
│   └── retrieve_from_rag.py  # Similarity search on the db
├── utils/
│   ├── llm_utils.py          # Rerank, prompt rewrite, page optimizing
│   ├── split_text.py         # Chunking and page titles on chunks
│   ├── html_cleaner.py       # Fetch a page (Markdown at <url>.md, else cleaned HTML)
│   ├── link_collector.py     # Read page links from the sitemap and llms.txt
│   └── ...
└── evaluation/
    ├── evaluator.py          # Retrieval + answer quality eval with an HTML report
    ├── tests.json            # Tests the evaluator runs by default
    ├── tests.full.json       # Full test set
    └── results/              # One folder per run (not committed)
```

The vector db is stored in `src/vector_db`, next to this folder.

## Setup

1. Install the dependencies from the repo root, as described in the [main README](../../README.md).
2. Install [Ollama](https://ollama.com), then pull the models:
   ```bash
   ollama pull gpt-oss
   ollama pull qwen3:8b
   ```
3. Set `HF_TOKEN` in the repo's `.env`. The Hugging Face models download on first run; the reranker is about 2.3 GB.

Run every command below from `src/assistant_in_posthog`.

## 1. Ingest

```bash
uv run python ingest.py
```

This reads the handbook pages from the posthog.com sitemap and the docs pages from `llms.txt`. For each page it:

1. Fetches it as Markdown (`<url>.md`), falling back to cleaned HTML.
2. Rewrites it with `qwen3:8b` so every section has a descriptive heading and makes sense alone. A page over 40,000 chars is rewritten in parts split at headings, each sent with the page outline. A rewrite that fails or comes back too short is dropped, and the original page is kept.
3. Splits it into 1000-char chunks with 200 chars of overlap and starts each chunk with the page title.
4. Stores the chunks with `source`, `title` and `section` (`docs` or `handbook`) metadata.

OpenAPI spec pages are skipped (`EXCLUDED_PATHS`). Pages already in the db are skipped too, so to rebuild the db after changing chunking or the rewrite, move `src/vector_db` aside and run the ingest again.

If fetching fails with `URLError: [Errno 8] nodename nor servname provided`, your DNS can't resolve posthog.com. Check for a DNS blocker, or switch to a public resolver such as 1.1.1.1.

## 2. Chat

```bash
uv run python app.py
```

Opens the Gradio chat in your browser.

To look inside the db:

```bash
uv run python vector_db_ui.py
```

## 3. Evaluate

```bash
uv run python -m evaluation.evaluator                                  # tests.json
uv run python -m evaluation.evaluator --tests evaluation/tests.full.json
uv run python -m evaluation.evaluator --limit 5                        # first 5 tests
uv run python -m evaluation.evaluator --index 3                        # one test, with full answer and judge feedback
```

Each test is a JSON object; only `question` and `reference_answer` are required:

```json
{"question": "...", "reference_answer": "...", "keywords": ["..."], "source": "https://posthog.com/handbook/...", "category": "how_to"}
```

Each test is scored on:

- **Retrieval:** MRR and nDCG of the `source` page, keyword coverage, and whether the source was retrieved and cited.
- **Answer:** an LLM judge (`qwen3:8b`, change it with `--judge-model`) scores accuracy, completeness, relevance and clarity from 1 to 5 against the reference answer.

Results go to `evaluation/results/<timestamp>/` as `results.json` and `report.html`.

The evaluator imports `llm_as_controller` once per run, so code and prompt changes take effect on the next run.
