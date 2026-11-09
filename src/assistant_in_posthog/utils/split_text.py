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
