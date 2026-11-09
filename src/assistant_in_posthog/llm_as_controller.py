import os
from pathlib import Path
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from dotenv.main import rewrite
from langchain_community.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from tools.retrieve_from_rag import fetch_related_context
from dtos.llm_dto import GradioChatHistory
from constants.assistant_constant import EMBEDDING_MODEL_NAME, RELEVANT_CONTEXT_COUNT, TOOL_SYSTEM_PROMPT, VECTOR_DB_NAME, RELEVANT_DOCS_COUNT
from utils.get_vector_db import getDb
from utils.llm_utils import rerank_context, improve_current_prompt
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages

load_dotenv(override=True)


llm = ChatOllama(
    base_url="http://localhost:11434", 
    model="qwen3:8b", 
    verbose=True, 
    temperature=0.2,
)
# temperature 0 so the same question and chunks give the same order;
# reasoning off so <think> text can't break the JSON
rerankLlm = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen3:1.7b",
    temperature=0,
    reasoning=False,
)


HF_TOKEN = os.environ.get("HF_TOKEN")
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"token": HF_TOKEN})

# current directory and can open a different, empty db
DB_NAME = str(Path(__file__).parent.parent / VECTOR_DB_NAME)
db = getDb(DB_NAME, embedding)
retriever = db.as_retriever()


MAX_QUERY_VARIANTS = 3


def searchSection(queries: list[str], question: str, top_k: int, section: Literal["docs", "handbook"]) -> str:
    """
    Search section with every query variant, drop chunks found by more than one
    variant, rerank the rest against question and return the top_k as text,
    each with its page title and url so the answer can cite it.
    """
    print(f"Searching {section} for queries: {queries} with top_k={top_k}")
    results = []
    seen = set()
    for query in queries[:MAX_QUERY_VARIANTS]:
        for doc in fetch_related_context(db, query, top_k=top_k, section=section):
            key = (doc.metadata.get("source"), doc.page_content)
            if key not in seen:
                seen.add(key)
                results.append(doc)

    ranked_results = rerank_context(question, results, rerankLlm)[:top_k]
    return "\n\n".join(
        f"Title: {doc.metadata.get('title', '')}\nSource: {doc.metadata.get('source', '')}\n{doc.page_content}"
        for doc in ranked_results
    )


@tool("search_posthog_docs", description="Search the PostHog docs (posthog.com/docs). Use for questions about using the product: setup, features, SDKs, API, integrations, troubleshooting, billing. Pass 3 differently worded search queries for the same need (keyword-style, a natural question, a one-sentence hypothetical answer), and the user's question rewritten to stand on its own.")
def search_posthog_docs(queries: list[str], question: str, top_k: int = 5) -> str:
    return searchSection(queries, question, top_k, "docs")


@tool("search_posthog_handbooks", description="Search the PostHog company handbook (posthog.com/handbook). Use for questions about how PostHog the company works: values, strategy, hiring, internal engineering, sales, support, marketing, incident post-mortems. Pass 3 differently worded search queries for the same need (keyword-style, a natural question, a one-sentence hypothetical answer), and the user's question rewritten to stand on its own.")
def search_posthog_handbooks(queries: list[str], question: str, top_k: int = 5) -> str:
    return searchSection(queries, question, top_k, "handbook")


tools = [search_posthog_docs, search_posthog_handbooks]
tools_by_name = {t.name: t for t in tools}
llm = llm.bind_tools(tools)

rewritePromptLlm = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen3:1.7b",
    temperature=0,
    reasoning=False,
)


def main(prompt: str, history: list[GradioChatHistory] = []) -> str | list[str | dict[Any, Any]]:
    """
    Generate a response to a prompt using the LLM and the vector database.
    """
    messages = [SystemMessage(content = TOOL_SYSTEM_PROMPT)]
    # messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=improve_current_prompt(prompt, history, rewritePromptLlm)))
    llmRes = llm.invoke(messages)
    messages.append(llmRes)
    while llmRes.tool_calls:
        for call in llmRes.tool_calls:
            tool_msg = tools_by_name[call["name"]].invoke(call)  # returns a ToolMessage
            messages.append(tool_msg)
        llmRes = llm.invoke(messages)

    return llmRes.content

