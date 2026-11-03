import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
openaiClient = OpenAI()

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
CLAUDE_BASE_URL =  "https://api.anthropic.com/v1"
claudeClient = OpenAI(base_url=CLAUDE_BASE_URL, api_key=CLAUDE_API_KEY)

QWEN_BASE_URL = "http://localhost:11434/v1"
qwenClient = OpenAI(base_url=QWEN_BASE_URL, api_key="ollama")

SYSTEM_PROMPT = """
    You are an expert C++ performance engineer. You translate Python code into modern C++ (C++20) that is correct first and as fast as possible second.

    Rules:
    1. Correctness: The C++ output must produce identical results to the original for all valid inputs, including edge cases (empty input, overflow, unicode, NaN, ordering). If the original's behavior is ambiguous or relies on language quirks (e.g., Python big ints, floor division, dict ordering), state how you handled it.
    2. Performance priorities, in order:
    - Pick the best algorithm and data structure first. Fix asymptotic inefficiencies in the original if behavior is preserved, and say so.
    - Minimize heap allocations: reserve() vectors, reuse buffers, prefer stack/arena storage, avoid std::shared_ptr unless ownership truly requires it.
    - Cache-friendly layout: contiguous arrays (std::vector, std::array), struct-of-arrays where it helps, avoid pointer chasing and node-based containers (std::map, std::list) unless needed. Prefer flat hash maps (e.g., ankerl::unordered_dense or absl::flat_hash_map) over std::unordered_map when hashing is hot.
    - Avoid copies: pass by const&, use std::string_view / std::span, move semantics, return by value (RVO).
    - Fast I/O: no std::endl, disable iostream sync or use fread/fwrite or memory-mapped files for large input.
    - Make hot loops vectorization-friendly: no virtual calls, no exceptions, no branches that can be avoided, use constexpr and inline where useful.
    - Use multithreading (std::thread, std::execution::par, or OpenMP) only when work is large and parallelizable; note thread-safety assumptions.
    3. No undefined behavior. No signed overflow, out-of-bounds access, uninitialized reads, or strict-aliasing violations.
    4. Use only the standard library unless a third-party library gives a major speedup; if you use one, name it and give a std-only fallback.
    5. Code style: single self-contained file unless asked otherwise, clear names, brief comments only where an optimization is non-obvious.

    Output format:
    1. The complete C++ code in one code block.
    2. Build command with recommended flags, e.g.:
    g++ -std=c++20 -O3 -march=native -flto -DNDEBUG main.cpp -o main
    3. A short "Optimizations" list: what you changed versus a literal translation and why it's faster.
    4. "Behavior differences" — anything that differs from the original, or "None".
    5. Optional: a minimal benchmark or test harness if the code is a library/function.
"""

def generateUserPrompt(sourceCode: str) -> str:
    return f"""
        Convert the following Python code to C++ and make it as fast as possible.

    Code:
    ```python
    {sourceCode}
    ```
    """


def getMessage(sourceCode: str) -> Any:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": generateUserPrompt(sourceCode)},
    ]

def clientForModel(model: str) -> OpenAI:
    if model.startswith("claude"):
        return claudeClient
    elif model.startswith("qwen"):
        return qwenClient
    elif model.startswith("llama"):
        return qwenClient
    else:
        return openaiClient


def temperatureFor(model: str) -> dict:
    # gpt-5 only accepts the default temperature
    return {} if model.startswith("gpt-5") else {"temperature": 0.0}


def convertCode(sourceCode: str, model: str = "gpt-4o") -> str | None:
    messages = getMessage(sourceCode)
    response = clientForModel(model).chat.completions.create(
        model=model,
        messages=messages,
        **temperatureFor(model),
    )

    return response.choices[0].message.content


def convertCodeStream(sourceCode: str, model: str):
    """Yield the response text accumulated so far as chunks arrive."""
    messages = getMessage(sourceCode)
    stream = clientForModel(model).chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        **temperatureFor(model),
    )

    text = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
            yield text


models = {
    "gpt-5": "gpt-5",
    "gpt-4.1": "gpt-4.1",
    "claude-opus": "claude-opus-5-5",
    "claude-sonnet": "claude-sonnet-5-5",
    "qwen3-8b": "qwen3:8b",
    "llama3.2": "llama3.2",
}


def extractCppCode(response: str) -> str:
    """Pick the C++ block out of the model's reply (it also contains build commands and notes)."""
    blocks = re.findall(r"```([\w+#-]*)[^\n]*\n(.*?)```", response, re.DOTALL)
    if not blocks:
        return response
    for lang, code in blocks:
        if lang.lower() in ("cpp", "c++", "cxx", "cc"):
            return code
    return max((code for _, code in blocks), key=len)
