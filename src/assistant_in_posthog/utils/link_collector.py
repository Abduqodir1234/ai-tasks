import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from courlan import extract_links

USER_AGENT = "Mozilla/5.0 (compatible; posthog-assistant-ingest/1.0)"
SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"


def fetch_html(url, timeout=30):
    """Download url and return (html, final_url_after_redirects)."""
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace"), response.geturl()


def parse_links_from_html(html, base_url, same_domain_only=True):
    """
    Return the links in an HTML string as sorted, de-duplicated absolute urls.
    Relative links are resolved against base_url. With same_domain_only=True
    only links on base_url's site are returned (courlan also counts its
    subdomains, e.g. app.posthog.com, as the same site).
    """
    links = extract_links(html, url=base_url, external_bool=False)
    if not same_domain_only:
        links |= extract_links(html, url=base_url, external_bool=True)
    return sorted(links)


def parse_links_from_given_link(url, same_domain_only=True, timeout=30):
    """Download the page at url and return parse_links_from_html() of it."""
    html, base_url = fetch_html(url, timeout)
    return parse_links_from_html(html, base_url, same_domain_only)


def get_links_from_sitemap(sitemap_url, path_prefix=None, timeout=30):
    """
    Return every page url listed in a sitemap, following sitemap indexes.
    With path_prefix (e.g. "/handbook/") only urls whose path starts with it
    are returned.
    """
    xml, _ = fetch_html(sitemap_url, timeout)
    root = ET.fromstring(xml)

    locs = [loc.text.strip() for loc in root.iter(f"{SITEMAP_NS}loc") if loc.text]

    if root.tag == f"{SITEMAP_NS}sitemapindex":
        urls = []
        for child_sitemap in locs:
            urls.extend(get_links_from_sitemap(child_sitemap, path_prefix, timeout))
        return urls

    return [
        url for url in locs
        if path_prefix is None or urlparse(url).path.startswith(path_prefix)
    ]
