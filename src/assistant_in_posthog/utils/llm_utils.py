


from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from constants.assistant_constant import (
    OPTIMIZE_DOC_SYSTEM_PROMPT,
    OPTIMIZE_DOC_USER_PROMPT,
    RERANK_SYSTEM_PROMPT,
    REWRITE_PROMPT_SYSTEM_PROMPT,
    REWRITE_PROMPT_USER_PROMPT,
)
from dtos.llm_dto import GradioChatHistory


class RerankResult(BaseModel):
    ids: list[int] = Field(description="Ids of all chunks, most relevant first")


class RewrittenPrompt(BaseModel):
    prompt: str = Field(description="The latest message rewritten to stand alone, with any relevant context from history")


class OptimizedPage(BaseModel):
    content: str = Field(description="The whole page rewritten in Markdown, starting with a '# <title>' heading")


def combined_question(question: str, history: list[GradioChatHistory] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def rerank_context(question: str, docs: list[Document], rerankLlm: BaseChatModel) -> list[Document]:
    """
    Ask rerankLlm to order docs by relevance to question and return them in that order.
    rerankLlm must support with_structured_output; a small fast model is enough.

    Chunks are numbered 1..n for the LLM (short numbers are easier for a small
    model than Chroma ids). Its answer is cleaned up: unknown and repeated ids
    are dropped, and chunks it left out are added after the ranked ones in
    retriever order, so a short answer can't push the right chunk out of top_k.
    If the call fails, no valid id is left, or the order is just counting up or
    down (see isCountingOrder), the original order is returned.
    """
    if len(docs) < 2:
        return docs

    chunks = "\n\n".join(f"[{id}]\n{doc.page_content}" for id, doc in enumerate(docs, start=1))
    messages = [
        SystemMessage(content=RERANK_SYSTEM_PROMPT),
        HumanMessage(content=f"Question:\n{question}\n\nChunks:\n{chunks}"),
    ]

    try:
        result = rerankLlm.with_structured_output(RerankResult).invoke(messages)
        rankedIds = RerankResult.model_validate(result).ids
    except Exception as e:
        print(f"Rerank failed, keeping retriever order: {type(e).__name__}: {e}")
        return docs

    order = []
    for id in rankedIds:
        if 1 <= id <= len(docs) and id not in order:
            order.append(id)
    if not order:
        print("Rerank returned no valid ids, keeping retriever order")
        return docs
    if isCountingOrder(order):
        print(f"Rerank order {order} only counts up or down, keeping retriever order")
        return docs
    skipped = [id for id in range(1, len(docs) + 1) if id not in order]
    print(f"Rerank order: {order}" + (f", skipped (kept last): {skipped}" if skipped else ""))

    return [docs[id - 1] for id in order + skipped]


# qwen3:8b sometimes writes the ids out in sequence, e.g. [12, 11, 10, ..., 1],
# instead of ranking them; that order says nothing about relevance
MIN_COUNTING_LENGTH = 5
COUNTING_SHARE = 0.8


def isCountingOrder(order: list[int]) -> bool:
    """True when at least COUNTING_SHARE of neighbouring ids in order differ by exactly 1."""
    if len(order) < MIN_COUNTING_LENGTH:
        return False
    steps = sum(abs(b - a) == 1 for a, b in zip(order, order[1:]))
    return steps / (len(order) - 1) >= COUNTING_SHARE


def improve_current_prompt(prompt: str, history: list[GradioChatHistory], llm: BaseChatModel) -> str:
    """
    Ask llm to improve the prompt by adding context from history.
    llm must support with_structured_output. If the call fails or the answer is
    empty, the prompt is returned as given.
    """

    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    messages = [
        SystemMessage(content=REWRITE_PROMPT_SYSTEM_PROMPT),
        HumanMessage(content=REWRITE_PROMPT_USER_PROMPT.format(prior_messages=prior, current_prompt=prompt)),
    ]
    try:
        result = llm.with_structured_output(RewrittenPrompt).invoke(messages)
        rewritten = RewrittenPrompt.model_validate(result).prompt.strip()
    except Exception as e:
        print(f"Rewrite failed, keeping original prompt: {type(e).__name__}: {e}")
        return prompt
    print("Rewritten prompt:", rewritten)
    return rewritten or prompt


def optimize_document(content: str, title: str | None, llm: BaseChatModel, maxChars: int = 40000, minRatio: float = 0.5) -> str:
    """
    Ask llm to rewrite a whole page so its chunks are easier to retrieve:
    descriptive headings, sections that make sense alone, schemas turned into
    sentences. See OPTIMIZE_DOC_SYSTEM_PROMPT. llm must support with_structured_output.

    The original content is returned when the page is longer than maxChars
    (it would not fit the model's context), when the call fails, or when the
    answer is shorter than minRatio of the original, which means the model
    summarized or cut the page instead of rewriting it.
    """
    if not content.strip() or len(content) > maxChars:
        return content

    messages = [
        SystemMessage(content=OPTIMIZE_DOC_SYSTEM_PROMPT),
        HumanMessage(content=OPTIMIZE_DOC_USER_PROMPT.format(title=title or "", content=content)),
    ]
    try:
        result = llm.with_structured_output(OptimizedPage).invoke(messages)
        optimized = OptimizedPage.model_validate(result).content
    except Exception as e:
        print(f"Optimize failed for {title!r}, keeping original: {type(e).__name__}: {e}")
        return content

    if len(optimized.strip()) < len(content) * minRatio:
        print(f"Optimize answer for {title!r} too short, keeping original")
        return content
    return optimized.strip()
