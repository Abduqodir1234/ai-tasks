import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from constants.assistant_constant import EMBEDDING_MODEL_NAME, VECTOR_DB_NAME
from utils import clean_html_from_given_link, get_links_from_llms_txt, get_links_from_sitemap, split_text, getDb, insert_chunks_into_vector_db
from utils.llm_utils import optimize_document
from utils.text_formatter import normalize_text


load_dotenv(override=True)
HF_TOKEN = os.environ.get("HF_TOKEN")

SITEMAP_URL = "https://posthog.com/sitemap/sitemap-index.xml"
# Index of every docs page's .md version. It has no handbook pages, so those still come from the sitemap
LLMS_TXT_URL = "https://posthog.com/llms.txt"
# Each section is ingested from its own index and path and tagged so searches can filter by it
SECTION_SOURCES = {"handbook": (SITEMAP_URL, "/handbook/"), "docs": (LLMS_TXT_URL, "/docs/")}
# Pages whose url contains one of these are not ingested. The OpenAPI spec pages are
# schema JSON that made up 76% of all chunks and crowded prose pages out of search results
EXCLUDED_PATHS = ("/open-api-spec/",)
MAX_WORKERS = 8
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"token": HF_TOKEN})
DB_NAME = str(Path(__file__).parent.parent / VECTOR_DB_NAME)
# Rewrites each page before chunking (see optimize_document). temperature 0 and
# reasoning off so the answer is the page only; num_ctx fits a 40k-char page and its rewrite
optimizeLlm = ChatOllama(
    base_url="http://localhost:11434",
    # model="gpt-oss",
    model="qwen3:8b",
    temperature=0.7, 
    top_p=0.8, 
    top_k=20,
    reasoning=False,
    num_ctx=32768,
)


def fetchPage(link, chunkSize, chunkOverlap, optimizeLlm=None):
    """Download, clean, optionally rewrite with optimizeLlm, and split one page. Runs in a worker thread."""
    pageStart = time.perf_counter()
    # posthog.com serves each page as Markdown at <url>.md; pages without one fall back to the HTML
    htmlData = clean_html_from_given_link(link, timeout=30, prefer_markdown=True)
    # Rewrite the whole page before splitting so every chunk gets a descriptive heading and stands alone.
    # Before normalize_text, which would strip any inline Markdown the LLM adds
    if optimizeLlm is not None:
        # Kept so main can show the first page before and after the rewrite
        htmlData["original_content"] = htmlData["content"]
        htmlData["content"] = optimize_document(htmlData["content"], htmlData["title"], optimizeLlm)
    # Strip inline Markdown (`code`, **bold**, \[ escapes, curly quotes) so stored text matches plain-text searches
    htmlData["content"] = normalize_text(htmlData["content"])
    chunks = split_text.split_text_into_chunks(htmlData["content"], chunk_size=chunkSize, chunk_overlap=chunkOverlap)
    chunks = split_text.add_title_to_chunks(chunks, htmlData["title"])
    return htmlData, chunks, time.perf_counter() - pageStart


def printOptimizedPage(htmlData):
    """Print a page before and after optimize_document, to check what the LLM rewrite does."""
    original, optimized = htmlData["original_content"], htmlData["content"]
    line = "=" * 80
    print(f"{line}\nOPTIMIZE PREVIEW: {htmlData['url']} ({htmlData['title']!r})")
    if optimized == normalize_text(original):
        print("Rewrite was not used (failed or too short), the original page is stored")
    print(f"{line}\nBEFORE ({len(original)} chars)\n{line}\n{original}")
    print(f"{line}\nAFTER ({len(optimized)} chars, after normalize_text)\n{line}\n{optimized}\n{line}")


def main(indexUrl, pathPrefix, section, dbDirectory, embedding, maxWorkers=8, chunkSize=1000, chunkOverlap=200, skipExisting=True, optimizeLlm=None):
    """
    Ingest every page listed under pathPrefix in indexUrl (a sitemap, or an
    llms.txt when it ends in .txt) into the vector db.
    Every chunk is tagged with section ("docs" or "handbook") in its metadata.

    Pages are downloaded, cleaned and split in maxWorkers threads at once, since
    that part is mostly waiting on the network. Inserting (embedding + writing to
    Chroma) happens on this thread, one page at a time, as each page finishes:
    the embedding model gains nothing from parallel calls and the local Chroma
    db is safest with a single writer.

    Every page is tried even if some fail; the failures are listed at the end
    and returned as [(link, error message)].

    With skipExisting=True, pages whose url is already stored as a chunk's source
    are skipped, so a run that stopped half way picks up where it left off
    without adding duplicates.

    With optimizeLlm, each page is rewritten by that LLM before it is split
    (see optimize_document); pages it can't rewrite are stored as they are.
    """
    # Without the db or the link list there is nothing to loop over, so these stop the run
    try:
        db = getDb(dbDirectory, embedding)
        getLinks = get_links_from_llms_txt if indexUrl.endswith(".txt") else get_links_from_sitemap
        links = getLinks(indexUrl, path_prefix=pathPrefix, timeout=30)
        links = [link for link in links if not any(path in link for path in EXCLUDED_PATHS)]
        # reset_collection, not delete_collection: delete leaves db without a
        # collection and every add_texts after it fails
        # db.delete_collection()  # Clear the collection before inserting new data
    except Exception as e:
        print(f"Error generating data from {indexUrl}: {type(e).__name__}: {e}")
        return [(indexUrl, f"{type(e).__name__}: {e}")]
    print(f"Found {len(links)} links on {indexUrl}, fetching with {maxWorkers} threads")

    if skipExisting:
        # A page's chunks are added in one add_texts call, so a stored source means the whole page is in
        # Read in pages: one get over every chunk hits SQLite's "too many SQL variables"
        storedSources = set()
        offset = 0
        while True:
            stored = db.get(where={"section": section}, include=["metadatas"], limit=5000, offset=offset)["metadatas"]
            storedSources.update((metadata or {}).get("source", "").rstrip("/") for metadata in stored)
            if len(stored) < 5000:
                break
            offset += 5000
        linkCount = len(links)
        links = [link for link in links if link.rstrip("/") not in storedSources]
        print(f"Skipped {linkCount - len(links)} pages already in the db, {len(links)} left to fetch")

    totalChunks = 0
    failures = []
    runStart = time.perf_counter()
    with ThreadPoolExecutor(max_workers=maxWorkers) as executor:
        futures = {executor.submit(fetchPage, link, chunkSize, chunkOverlap, optimizeLlm): link for link in links}

        # Pages finish in any order, so index counts finished pages, not sitemap position
        for index, future in enumerate(as_completed(futures), start=1):
            link = futures[future]

            # One bad page must not stop the run: record the error and move on
            try:
                htmlData, chunks, fetchSeconds = future.result()
                if index == 1 and "original_content" in htmlData:
                    printOptimizedPage(htmlData)
                insertStart = time.perf_counter()
                # Every chunk of a page carries the page's url and title so answers can cite it
                metadata = {"source": htmlData["url"] or link, "title": htmlData["title"], "section": section}
                insert_chunks_into_vector_db.insertChunksIntoVectorDB(chunks, db, metadata)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                failures.append((link, error))
                print(f"[{index}/{len(links)}] FAILED {link}\n    {error}")
                continue

            totalChunks += len(chunks)
            print(
                f"[{index}/{len(links)}] {link}\n"
                f"    {len(htmlData['content'])} chars, title: {htmlData['title']!r}, "
                f"fetched in {fetchSeconds:.1f}s, stored {len(chunks)} chunks "
                f"in {time.perf_counter() - insertStart:.1f}s (total {totalChunks})"
            )

    print(
        f"Done: {len(links) - len(failures)}/{len(links)} pages, {totalChunks} chunks "
        f"in {time.perf_counter() - runStart:.0f}s"
    )

    if failures:
        print(f"\n{len(failures)} page(s) failed:")
        for failedLink, error in failures:
            print(f"  - {failedLink}\n      {error}")

    return failures


if __name__ == "__main__":
    for section, (indexUrl, pathPrefix) in SECTION_SOURCES.items():
        main(indexUrl, pathPrefix, section, DB_NAME, embedding, maxWorkers=1, optimizeLlm=optimizeLlm)

# print(clean_html_from_given_link("https://posthog.com/handbook", timeout=30))
