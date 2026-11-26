# Python to C++ Converter

A Gradio app that converts Python code to fast C++ with an LLM, then runs both versions with the same arguments so you can compare output and speed.

## Models

Pick one in the UI:

| Label | Model | Provider |
|---|---|---|
| `gpt-5` | `gpt-5` | OpenAI |
| `gpt-4.1` | `gpt-4.1` | OpenAI |
| `claude-opus` | `claude-opus-5-5` | Anthropic, via its OpenAI-compatible endpoint |
| `claude-sonnet` | `claude-sonnet-5-5` | Anthropic, via its OpenAI-compatible endpoint |
| `qwen3-8b` | `qwen3:8b` | Ollama (`localhost:11434`) |
| `llama3.2` | `llama3.2` | Ollama (`localhost:11434`) |

- All models are called through the `openai` client. The client is picked by the model name's prefix (`clientForModel` in `converter/llm.py`).
- Temperature is `0.0` for every model except `gpt-5`, which only accepts its default.

## How it works

1. **Convert.** The system prompt asks for correct C++20 first and speed second. It covers algorithm choice, fewer allocations, cache-friendly data, fast I/O and no undefined behavior. The reply streams into the UI.
2. **Extract.** The reply also holds a build command, an optimizations list and behavior notes. `extractCppCode` picks out the `cpp` code block, or the longest block if none is tagged. The full reply is shown in a collapsible panel.
3. **Run.**
   - "Run Python code" runs the source with the current interpreter.
   - "Run C++ code" compiles with `c++ -std=c++20 -O3 -march=native -DNDEBUG`, then runs the binary.
   - Both runs happen in a temp folder, get the same arguments, and time out after 120 s.
   - Compile time and run time are shown separately.

## Project structure

```
code_converter/
├── main.py              # Gradio UI
└── converter/
    ├── llm.py           # Prompts, model clients, streaming, C++ extraction
    └── execute.py       # Run Python, compile and run C++
```

## Setup

- A C++ compiler available as `c++`, e.g. Xcode Command Line Tools on macOS.
- `OPENAI_API_KEY` in `.env`, for the OpenAI models.
- `CLAUDE_API_KEY` in `.env`, for the Claude models.
- Ollama with `qwen3:8b` and `llama3.2` pulled, for the local models.

## Run

From this folder:

```bash
uv run python main.py
```

The generated C++ and your Python code run on your machine without a sandbox, so only run code you trust.
