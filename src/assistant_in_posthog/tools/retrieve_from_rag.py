from typing import Literal

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


def fetch_related_context(
    db: VectorStore,
    query: str,
    top_k: int = 5,
    section: Literal["docs", "handbook"] | None = None,
) -> list[Document]:
    """
    Retrieve relevant data from RAG (Retrieval-Augmented Generation) based on the query.

    Args:
        db (VectorStore): The vector db to search, e.g. from `getDb()`.
        query (str): The input query for which relevant data needs to be retrieved.
        top_k (int): The number of top relevant results to retrieve. Default is 5.
        section: Search only "docs" or only "handbook" chunks. None searches both.

    Returns:
        list[Document]: The closest chunks. Each one's metadata has the "source" url and "title" of its page.
    """
    # similarity_search, not retriever.invoke: invoke ignores k and has no filter
    return db.similarity_search(query, k=top_k, filter={"section": section} if section else None)
