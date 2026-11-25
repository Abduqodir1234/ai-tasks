from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text_into_chunks(text: str, chunk_size: int=1000, chunk_overlap: int=200):
    """
    Split text into chunks of a specified size with a specified overlap.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(text)


def add_title_to_chunks(chunks: list[str], title: str | None) -> list[str]:
    """
    Start every chunk with its page title, so a chunk split off from the page's
    intro (e.g. a bare list of book titles) still matches searches for the page's topic.
    Chunks whose first line already holds the title (usually the first one) are kept as they are.
    """
    if not title:
        return chunks
    return [
        chunk if title.lower() in chunk.split("\n", 1)[0].lower() else f"{title}\n\n{chunk}"
        for chunk in chunks
    ]
