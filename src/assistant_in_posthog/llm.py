import os
from pathlib import Path
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from dtos.llm_dto import GradioChatHistory
from constants.assistant_constant import EMBEDDING_MODEL_NAME, RELEVANT_CONTEXT_COUNT, SYSTEM_PROMPT, VECTOR_DB_NAME, RELEVANT_DOCS_COUNT
from utils.get_vector_db import getDb
from utils.llm_utils import combined_question, rerank_context, improve_current_prompt
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

rewritePromptLlm = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen3:1.7b",
    temperature=0,
    reasoning=False,
)
HF_TOKEN = os.environ.get("HF_TOKEN")
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"token": HF_TOKEN})

# Same absolute path as ingest.py; a bare "vector_db" resolves against the
# current directory and can open a different, empty db
DB_NAME = str(Path(__file__).parent.parent / VECTOR_DB_NAME)
db = getDb(DB_NAME, embedding)
retriever = db.as_retriever()


# def inspectVectorDb(sampleSize=3):
#     """Print how many vectors the collection holds and a few sample rows."""
#     print(f"Vector DB: {DB_NAME}")
#     print(f"Collection: {db._collection.name}, vectors: {db._collection.count()}")

#     sample = db.get(limit=sampleSize, include=["documents", "metadatas", "embeddings"])
#     for id, document, metadata, vector in zip(
#         sample["ids"], sample["documents"], sample["metadatas"], sample["embeddings"]
#     ):
#         print(f"- id: {id}")
#         print(f"  metadata: {metadata}")
#         print(f"  text: {document[:200]!r}")
#         print(f"  vector: {len(vector)} dims, first 5: {[round(float(x), 4) for x in vector[:5]]}")


def main(prompt: str, history: list[GradioChatHistory] = []) -> str | list[str | dict[Any, Any]]:
    """
    Generate a response to a prompt using the LLM and the vector database.
    """
    # inspectVectorDb()
    improvedPrompt = improve_current_prompt(prompt, history, rewritePromptLlm)
    original_combined_prompt = combined_question(prompt, history)
    improved_combined_prompt = combined_question(improvedPrompt, history)
    print(f"Combined prompt: {original_combined_prompt}")
    original_prompt_docs = retriever.invoke(original_combined_prompt,k = RELEVANT_DOCS_COUNT)
    improved_prompt_docs = retriever.invoke(improved_combined_prompt,k = RELEVANT_DOCS_COUNT)

    # Rank by the current question only; older questions would pull the order off-topic
    docs = rerank_context(improvedPrompt, original_prompt_docs+improved_prompt_docs, rerankLlm)[:RELEVANT_CONTEXT_COUNT]
    print(f"Kept top {len(docs)} documents after rerank")
    relevant_context = "\n\n".join([doc.page_content for doc in docs])
    messages = [SystemMessage(content = SYSTEM_PROMPT.format(context=relevant_context))]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=prompt))
    llmRes = llm.invoke(messages)
    return llmRes.content

