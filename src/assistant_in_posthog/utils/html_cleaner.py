import re
from urllib.error import HTTPError

import trafilatura

from .link_collector import fetch_html

# PostHog prefixes every .md page with a note for AI agents and a canonical url line
AGENT_NOTE = re.compile(r"\A> AI agents:[^\n]*\n+")
CANONICAL_URL = re.compile(r"^Canonical URL: \S+[ \t]*\n+", re.MULTILINE)
TITLE = re.compile(r"^# (.+)$", re.MULTILINE)
# "On this page" is a list of #anchor links to the headings below it
TABLE_OF_CONTENTS = re.compile(r"^## On this page\n+(?:[ \t]*- \[[^\]]*\]\(#[^)]*\)\n+)+", re.MULTILINE)
FENCED_CODE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.DOTALL | re.MULTILINE)
IMAGE = re.compile(r"!\[[^\]]*\]\((?:[^()\s]|\([^()\s]*\))*\)")
# Keeps only the link text, to match clean_html(), which drops links
LINK = re.compile(r"\[([^\]]+)\]\((?:[^()\s]|\([^()\s]*\))*\)")


def clean_html(html, url=None):
    """
    Strip an HTML page down to the text worth putting in a vector db.
    trafilatura finds the main content and drops menus, footers, sidebars and
    other boilerplate. The content is returned as Markdown so chunkers can
    split on headings.

    Returns {"url", "title", "description", "content"}.
    """
    content = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        include_links=False,
        include_images=False,
        include_comments=False,
    )
    metadata = trafilatura.extract_metadata(html, default_url=url)

    return {
        "url": metadata.url if metadata else url,
        "title": metadata.title if metadata else None,
        "description": metadata.description if metadata else None,
        "content": content or "",
    }


def _strip_links_and_images(markdown):
    """Drop images and turn links into their text, leaving fenced code blocks as they are."""
    parts = []
    last = 0
    for block in FENCED_CODE.finditer(markdown):
        parts.append(LINK.sub(r"\1", IMAGE.sub("", markdown[last:block.start()])))
        parts.append(block.group())
        last = block.end()
    parts.append(LINK.sub(r"\1", IMAGE.sub("", markdown[last:])))
    return "".join(parts)


def clean_markdown(markdown, url=None):
    """
    Clean a page's raw Markdown version (posthog.com serves one at <page url>.md)
    into the same shape clean_html() returns: the agent note, canonical url line,
    "On this page" list, images and link targets are removed.

    Returns {"url", "title", "description", "content"}.
    """
    markdown = AGENT_NOTE.sub("", markdown.replace("\r\n", "\n"))

    # url is kept as given: the canonical url can point at another page (python -> python/reference)
    markdown = CANONICAL_URL.sub("", markdown, count=1)
    title = TITLE.search(markdown)
    markdown = TABLE_OF_CONTENTS.sub("", markdown, count=1)

    return {
        "url": url,
        "title": title.group(1).strip() if title else None,
        "description": None,
        "content": _strip_links_and_images(markdown).strip(),
    }


def fetch_markdown_from_given_link(url, timeout=30):
    """
    Download the Markdown version of the page at url (url + ".md") and return
    clean_markdown() of it, or None when the site has no Markdown version of it.
    """
    try:
        markdown, _ = fetch_html(url.rstrip("/") + ".md", timeout)
    except HTTPError as e:
        if e.code == 404:
            return None
        raise
    # A site without .md pages may answer with its HTML page instead of a 404
    if markdown.lstrip().startswith("<"):
        return None
    return clean_markdown(markdown, url)


def clean_html_from_given_link(url, timeout=30, prefer_markdown=False):
    """
    Download the page at url and return clean_html() of it.
    With prefer_markdown=True the page's Markdown version (url + ".md") is used
    instead when the site has one, and the HTML page is only the fallback.
    """
    if prefer_markdown:
        page = fetch_markdown_from_given_link(url, timeout)
        if page is not None:
            return page
    html, final_url = fetch_html(url, timeout)
    return clean_html(html, final_url)
