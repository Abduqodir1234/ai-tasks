from langchain_chroma import Chroma

def insertChunksIntoVectorDB(chunks: list[str],vectorDb: Chroma, metadata: dict | None = None):
    """
    Add chunks to the vector db. metadata (e.g. {"source": url, "title": ...}) is
    stored with every chunk. Chroma rejects None values, so those keys are dropped.
    """
    if not chunks:
        print("No chunks to insert into vector DB.")
        return

    metadata = {key: value for key, value in (metadata or {}).items() if value is not None}
    vectorDb.add_texts(chunks, metadatas=[metadata] * len(chunks) if metadata else None)
