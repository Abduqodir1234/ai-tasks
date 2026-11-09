import trafilatura

from .link_collector import fetch_html


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


def clean_html_from_given_link(url, timeout=30):
    """Download the page at url and return clean_html() of it."""
    html, final_url = fetch_html(url, timeout)
    return clean_html(html, final_url)
