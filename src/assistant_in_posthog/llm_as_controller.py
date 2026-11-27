import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from dotenv.main import rewrite
from langchain_community.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder

from tools.retrieve_from_rag import fetch_related_context
from dtos.llm_dto import GradioChatHistory
from constants.assistant_constant import EMBEDDING_MODEL_NAME, RERANK_MODEL_NAME, RELEVANT_CONTEXT_COUNT, TOOL_SYSTEM_PROMPT, VECTOR_DB_NAME, RELEVANT_DOCS_COUNT
from utils.get_vector_db import getDb
from utils.llm_utils import rerank_context, improve_current_prompt
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage, convert_to_messages
from langchain_core.runnables import Runnable
from ollama import ResponseError

load_dotenv(override=True)


llm = ChatOllama(
    base_url="http://localhost:11434", 
        model="gpt-oss", 
    # model="qwen3:8b", 
    verbose=True, 
    temperature=0.3, 
    num_ctx=32768,
    reasoning="low",
    top_p=0.9,
    top_k=40,
)


HF_TOKEN = os.environ.get("HF_TOKEN")
# Cross-encoder scores each chunk against the question on its own, so the order
# is deterministic and the chunk count isn't limited by an LLM's context
reranker = CrossEncoder(RERANK_MODEL_NAME, token=HF_TOKEN, max_length=1024)
# Qwen3-Embedding expects searches to carry its "query" instruction prefix and stored
# chunks none, so only searches use it and the ingested vectors stay valid
embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"token": HF_TOKEN},
    query_encode_kwargs={"prompt_name": "query"},
)

# current directory and can open a different, empty db
DB_NAME = str(Path(__file__).parent.parent / VECTOR_DB_NAME)
db = getDb(DB_NAME, embedding)
retriever = db.as_retriever()


MAX_QUERY_VARIANTS = 8
# gpt-oss can keep searching and never write an answer, so after this many
# tool rounds it is asked to answer from the results it already has
MAX_TOOL_ROUNDS = 5
FORCE_ANSWER_MESSAGE = "Stop searching. Answer now using only the search results above; if they don't answer the question, say so."
# gpt-oss sometimes replies without searching, or with neither text nor a tool call,
# so it is nudged this many times; if it still hasn't searched, the search is run for it
MAX_NUDGES = 2
SEARCH_FIRST_MESSAGE = "You must search with `search_posthog` before replying, even if the question looks out of scope or unclear."
EMPTY_AFTER_SEARCH_MESSAGE = "Your reply was empty. Answer now using the search results above; if they don't answer the question, say so."
EMPTY_ANSWER_FALLBACK = "Sorry, I couldn't come up with an answer. Please try rephrasing your question, or check https://posthog.com/docs."
# Ollama returns a 500 when gpt-oss writes tool-call JSON it can't parse
MAX_INVOKE_ATTEMPTS = 3


@tool("search_posthog", response_format="content_and_artifact", description="Search the PostHog docs (posthog.com/docs) and company handbook (posthog.com/handbook) together: using the product (setup, features, SDKs, API, integrations, troubleshooting, billing) and how PostHog the company works (values, strategy, people, internal engineering, sales, support, marketing, incident post-mortems). Pass 3 differently worded search queries for the same need, worded like documentation text, not questions (keyword-style, a page or section title, a short passage as the page would state it), and the user's question rewritten to stand on its own.")
def search_posthog(queries: list[str], question: str, top_k: int = 5) -> tuple[str, list[Document]]:
    """
    Search docs and handbook with every query variant, drop chunks found by more than one
    variant, rerank the rest against question and return the top_k as text,
    each with its page title and url so the answer can cite it.
    The top_k chunks are returned too, as the tool's artifact, so the evaluator can score retrieval.
    """
    print(f"Searching for queries: {queries} with top_k={top_k}")
    results = []
    seen = set()
    queries = queries[:MAX_QUERY_VARIANTS]
    # The variants are searched at once; map keeps query order, so the dedup below gives the same results
    with ThreadPoolExecutor(max_workers=max(len(queries), 1)) as executor:
        # No section filter: picking docs or handbook up front sent handbook questions to the docs
        perQuery = executor.map(lambda query: fetch_related_context(db, query, top_k=top_k), queries)
        for docs in perQuery:
            for doc in docs:
                key = (doc.metadata.get("source"), doc.page_content)
                if key not in seen:
                    seen.add(key)
                    results.append(doc)

    ranked_results = rerank_context(question, results, reranker)[:top_k]
    text = "\n\n".join(
        f"Title: {doc.metadata.get('title', '')}\nSource: {doc.metadata.get('source', '')}\n{doc.page_content}"
        for doc in ranked_results
    )
    return text, ranked_results


tools = [search_posthog]
tools_by_name = {t.name: t for t in tools}
# Kept without tools so the answer after MAX_TOOL_ROUNDS can only be text
answerLlm = llm
llm = llm.bind_tools(tools)

rewritePromptLlm = ChatOllama(
    base_url="http://localhost:11434",
    model="gpt-oss",
    temperature=0,
    reasoning=False,
)


def main(prompt: str, history: list[GradioChatHistory] = []) -> str | list[str | dict[Any, Any]]:
    """
    Generate a response to a prompt using the LLM and the vector database.
    """
    return answer_question(prompt, history)[0]


def answer_question(prompt: str, history: list[GradioChatHistory] = []) -> tuple[str | list[str | dict[Any, Any]], list[Document]]:
    """
    Same as main, but also returns every chunk the searches handed to the LLM,
    in the order they were given, so the evaluator can score retrieval.
    """
    retrievedDocs: list[Document] = []
    messages: list[BaseMessage] = [SystemMessage(content = TOOL_SYSTEM_PROMPT)]
    messages.extend(convert_to_messages(history))
    rewrittenPrompt = improve_current_prompt(prompt, history, rewritePromptLlm)
    messages.append(HumanMessage(content=rewrittenPrompt))
    llmRes = invoke_with_retry(llm, messages)
    toolRounds = 0
    nudges = 0
    while True:
        if llmRes.tool_calls and toolRounds < MAX_TOOL_ROUNDS:
            # The reply holding the tool calls must come before their ToolMessages
            messages.append(llmRes)
            toolRounds += 1
            for call in llmRes.tool_calls:
                tool_msg = tools_by_name[call["name"]].invoke(call)  # returns a ToolMessage
                messages.append(tool_msg)
                retrievedDocs.extend(tool_msg.artifact or [])
        elif not llmRes.tool_calls and (toolRounds == 0 or is_empty(llmRes.content)) and nudges < MAX_NUDGES:
            nudges += 1
            print(f"Reply without a search or text after {toolRounds} tool rounds, nudging ({nudges}/{MAX_NUDGES})")
            # The reply itself is left out so the model doesn't copy it
            messages.append(HumanMessage(content=SEARCH_FIRST_MESSAGE if toolRounds == 0 else EMPTY_AFTER_SEARCH_MESSAGE))
        else:
            break
        llmRes = invoke_with_retry(llm, messages)

    if toolRounds == 0:
        print(f"Still no search after {MAX_NUDGES} nudges, searching with the rewritten prompt")
        args = {"queries": [rewrittenPrompt], "question": rewrittenPrompt, "top_k": 8}
        # Invoked as a tool call so it returns a ToolMessage carrying the chunks as its artifact
        tool_msg = search_posthog.invoke({"name": search_posthog.name, "args": args, "id": "forced-search", "type": "tool_call"})
        retrievedDocs.extend(tool_msg.artifact or [])
        # Passed as a user message: a ToolMessage needs an assistant tool call before it
        messages.append(HumanMessage(content=f"Search results for the question:\n\n{tool_msg.content}\n\n{FORCE_ANSWER_MESSAGE}"))
        llmRes = invoke_with_retry(answerLlm, messages)

    if llmRes.tool_calls:
        print(f"Still calling tools after {MAX_TOOL_ROUNDS} rounds, asking for an answer without tools")
        messages.append(HumanMessage(content=FORCE_ANSWER_MESSAGE))
        llmRes = invoke_with_retry(answerLlm, messages)

    if is_empty(llmRes.content):
        return EMPTY_ANSWER_FALLBACK, retrievedDocs
    return llmRes.content, retrievedDocs


def invoke_with_retry(model: Runnable, messages: list[BaseMessage]) -> AIMessage:
    """
    Invoke model, retrying when Ollama can't parse the tool call gpt-oss wrote.
    Any other error, or the last failed attempt, is raised.
    """
    for attempt in range(1, MAX_INVOKE_ATTEMPTS + 1):
        try:
            return model.invoke(messages)
        except ResponseError as e:
            if "error parsing tool call" not in str(e) or attempt == MAX_INVOKE_ATTEMPTS:
                raise
            print(f"Malformed tool call, retrying ({attempt}/{MAX_INVOKE_ATTEMPTS}): {e}")
    raise AssertionError("unreachable")


def is_empty(content: str | list[str | dict[Any, Any]]) -> bool:
    return not content or (isinstance(content, str) and not content.strip())

