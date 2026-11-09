import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from constants.assistant_constant import EMBEDDING_MODEL_NAME, VECTOR_DB_NAME
from utils import clean_html_from_given_link, get_links_from_sitemap, split_text, getDb, insert_chunks_into_vector_db
from utils.text_formatter import normalize_text


load_dotenv(override=True)
HF_TOKEN = os.environ.get("HF_TOKEN")

SITEMAP_URL = "https://posthog.com/sitemap/sitemap-index.xml"
# Each section is ingested from its own path and tagged so searches can filter by it
SECTION_PATH_PREFIXES = {"handbook": "/handbook/","docs": "/docs/", }
MAX_WORKERS = 8
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"token": HF_TOKEN})
DB_NAME = str(Path(__file__).parent.parent / VECTOR_DB_NAME)


def fetchPage(link, chunkSize, chunkOverlap):
    """Download, clean and split one page. Runs in a worker thread."""
    pageStart = time.perf_counter()
    htmlData = clean_html_from_given_link(link, timeout=30)
    # Strip inline Markdown (`code`, **bold**, \[ escapes, curly quotes) so stored text matches plain-text searches
    htmlData["content"] = normalize_text(htmlData["content"])
    chunks = split_text.split_text_into_chunks(htmlData["content"], chunk_size=chunkSize, chunk_overlap=chunkOverlap)
    return htmlData, chunks, time.perf_counter() - pageStart


def main(sitemapUrl, pathPrefix, section, dbDirectory, embedding, maxWorkers=8, chunkSize=1000, chunkOverlap=200):
    """
    Ingest every page listed in a sitemap under pathPrefix into the vector db.
    Every chunk is tagged with section ("docs" or "handbook") in its metadata.

    Pages are downloaded, cleaned and split in maxWorkers threads at once, since
    that part is mostly waiting on the network. Inserting (embedding + writing to
    Chroma) happens on this thread, one page at a time, as each page finishes:
    the embedding model gains nothing from parallel calls and the local Chroma
    db is safest with a single writer.

    Every page is tried even if some fail; the failures are listed at the end
    and returned as [(link, error message)].
    """
    # Without the db or the link list there is nothing to loop over, so these stop the run
    try:
        db = getDb(dbDirectory, embedding)
        links = get_links_from_sitemap(sitemapUrl, path_prefix=pathPrefix, timeout=30)
        # reset_collection, not delete_collection: delete leaves db without a
        # collection and every add_texts after it fails
        # db.delete_collection()  # Clear the collection before inserting new data
    except Exception as e:
        print(f"Error generating data from {sitemapUrl}: {type(e).__name__}: {e}")
        return [(sitemapUrl, f"{type(e).__name__}: {e}")]
    print(f"Found {len(links)} links on {sitemapUrl}, fetching with {maxWorkers} threads")

    totalChunks = 0
    failures = []
    runStart = time.perf_counter()
    with ThreadPoolExecutor(max_workers=maxWorkers) as executor:
        futures = {executor.submit(fetchPage, link, chunkSize, chunkOverlap): link for link in links}

        # Pages finish in any order, so index counts finished pages, not sitemap position
        for index, future in enumerate(as_completed(futures), start=1):
            link = futures[future]

            # One bad page must not stop the run: record the error and move on
            try:
                htmlData, chunks, fetchSeconds = future.result()
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
    for section, pathPrefix in SECTION_PATH_PREFIXES.items():
        main(SITEMAP_URL, pathPrefix, section, DB_NAME, embedding, maxWorkers=24)

# print(clean_html_from_given_link("https://posthog.com/handbook", timeout=30))
