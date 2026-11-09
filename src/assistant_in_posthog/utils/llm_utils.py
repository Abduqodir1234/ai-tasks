

from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from constants.assistant_constant import RERANK_SYSTEM_PROMPT, REWRITE_PROMPT_SYSTEM_PROMPT, REWRITE_PROMPT_USER_PROMPT
from dtos.llm_dto import GradioChatHistory


class RerankResult(BaseModel):
    ids: list[int] = Field(description="Ids of all chunks, most relevant first")


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
    are dropped, and chunks it left out are appended in their original order,
    so no chunk is lost. If the call fails the original order is returned.
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
    missing = [id for id in range(1, len(docs) + 1) if id not in order]
    print(f"Rerank order: {order}" + (f", missing appended: {missing}" if missing else ""))

    return [docs[id - 1] for id in order + missing]


def improve_current_prompt(prompt: str, history: list[GradioChatHistory], llm: BaseChatModel) -> str | list[str | dict[Any, Any]]:
    """
    Ask llm to improve the prompt by adding context from history.
    """
    if not history:
        return prompt

    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    messages = [
        SystemMessage(content=REWRITE_PROMPT_SYSTEM_PROMPT),
        HumanMessage(content=REWRITE_PROMPT_USER_PROMPT.format(prior_messages=prior, current_prompt=prompt)),
    ]
    llmRes = llm.invoke(messages)

    return llmRes.content
