"""
Find the best price for a product on a shopping website.

Opens the site in a real browser (Playwright), searches for the product,
scrapes the results page and asks an LLM to pick the cheapest matching offers.

Usage:
    python best_price_finder.py <website_url> "<product>"
"""
import sys
from urllib.parse import urljoin

from dotenv import load_dotenv
from openai import OpenAI
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

load_dotenv(override=True)
openai_client = OpenAI()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

MODEL = "gpt-4.1-mini"

# How much of the page we send to the LLM
MAX_TEXT_CHARS = 20_000
MAX_LINKS = 150

# Browser behavior (milliseconds)
PAGE_LOAD_TIMEOUT = 60_000
NETWORK_IDLE_TIMEOUT = 15_000
WAIT_AFTER_OPEN = 3_000
WAIT_AFTER_CLICK = 1_000
SCROLL_PAUSE = 1_000
SCROLL_TIMES = 5
SCROLL_STEP_PX = 3_000
TYPING_DELAY = 50

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
)
VIEWPORT = {"width": 1366, "height": 900}


# ---------------------------------------------------------------------------
# Search box detection
# ---------------------------------------------------------------------------

# Words that mark an element as search-related (English, Russian, Uzbek)
SEARCH_WORDS = ["search", "поиск", "найти", "qidir", "izla"]

# Anything a user can type into: real inputs, textareas, contenteditable divs, ARIA widgets
TYPEABLE = ", ".join([
    "input:not([type=hidden]):not([type=checkbox]):not([type=radio]):not([type=submit])",
    "textarea",
    "[contenteditable='']",
    "[contenteditable='true']",
    "[role=searchbox]",
    "[role=combobox]",
    "[role=textbox]",
])

# Things you click to open a hidden search box: icons, buttons, divs acting as buttons
CLICKABLE = "button, a, [role=button], div, span, svg"

# Attributes that usually reveal what an element is for
HINT_ATTRS = ["type", "name", "id", "class", "placeholder", "aria-label", "title", "data-testid", "role"]

# Don't inspect more than this many candidates per selector (keeps it fast)
MAX_CANDIDATES = 300

# JS: join the element's hint attributes and the start of its text into one string
JS_ELEMENT_HINTS = """
(el, attrs) => {
    const attrText = attrs.map(a => el.getAttribute(a) || '').join(' ');
    const innerText = (el.innerText || '').slice(0, 50);
    return attrText + ' ' + innerText;
}
"""


def looks_like_search(element):
    """True if the element's attributes or text mention search."""
    hints = element.evaluate(JS_ELEMENT_HINTS, HINT_ATTRS).lower()
    return any(word in hints for word in SEARCH_WORDS)


def find_visible(page, selector, must_look_like_search=True):
    """Return the first visible element matching the selector, or None."""
    candidates = page.locator(selector)
    for i in range(min(candidates.count(), MAX_CANDIDATES)):
        element = candidates.nth(i)
        try:
            if not element.is_visible():
                continue
            if must_look_like_search and not looks_like_search(element):
                continue
            return element
        except Exception:
            # Element disappeared or changed while we were checking it
            continue
    return None


def find_visible_search_box(page):
    """Strategy 1: a search box that is already on screen."""
    searchbox = page.get_by_role("searchbox").first
    if searchbox.count() and searchbox.is_visible():
        return searchbox
    return find_visible(page, TYPEABLE)


def open_hidden_search_box(page):
    """Strategy 2: click a search icon/button that reveals the search box."""
    trigger = find_visible(page, CLICKABLE)
    if trigger is None:
        return None

    trigger.click()
    page.wait_for_timeout(WAIT_AFTER_CLICK)
    return find_visible(page, TYPEABLE) or find_visible(page, TYPEABLE, must_look_like_search=False)


def first_typeable_element(page):
    """Strategy 3 (last resort): the first visible text field, usually the header search bar."""
    return find_visible(page, TYPEABLE, must_look_like_search=False)


def type_and_submit(page, box, text):
    """Type into the box like a real user. Works for inputs and contenteditable divs."""
    box.click()
    page.keyboard.press("ControlOrMeta+A")
    page.keyboard.press("Backspace")
    page.keyboard.type(text, delay=TYPING_DELAY)
    page.keyboard.press("Enter")
    page.wait_for_load_state("domcontentloaded")


def search_on_website(page, product):
    """Find the site's search box, search for the product. Returns True on success."""
    strategies = [find_visible_search_box, open_hidden_search_box, first_typeable_element]

    for strategy in strategies:
        box = strategy(page)
        if box is not None:
            type_and_submit(page, box, product)
            return True

    return False


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

# JS: collect [link text, href] for every link on the page
JS_COLLECT_LINKS = """
links => links.map(a => [
    a.innerText.trim().replace(/\\s+/g, ' '),
    a.getAttribute('href'),
])
"""


def wait_until_loaded(page):
    """Wait for network requests to settle; give up quietly on busy sites."""
    try:
        page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_TIMEOUT)
    except PlaywrightTimeout:
        pass


def scroll_to_load(page):
    """Scroll down a few times so lazy-loaded product cards render."""
    for _ in range(SCROLL_TIMES):
        page.mouse.wheel(0, SCROLL_STEP_PX)
        page.wait_for_timeout(SCROLL_PAUSE)


def collect_links(page):
    """Return [(text, absolute_url), ...] for links that have both text and href."""
    raw_links = page.eval_on_selector_all("a[href]", JS_COLLECT_LINKS)
    return [
        (text, urljoin(page.url, href))
        for text, href in raw_links
        if text and href
    ]


def fetch_search_results(website, product, headless=True):
    """
    Open the website, search for the product and
    return (title, text, links) of the results page.
    """
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        page = browser.new_page(user_agent=USER_AGENT, viewport=VIEWPORT)

        page.goto(website, wait_until="domcontentloaded", timeout=PAGE_LOAD_TIMEOUT)
        page.wait_for_timeout(WAIT_AFTER_OPEN)

        if not search_on_website(page, product):
            print("Search box not found, analyzing the given page as is.")

        wait_until_loaded(page)
        scroll_to_load(page)

        title = page.title()
        text = page.inner_text("body")
        links = collect_links(page)

        browser.close()

    return title, text[:MAX_TEXT_CHARS], links[:MAX_LINKS]


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a product research assistant. Your job is to analyze the content of websites (product listings, catalog pages, search results, product detail pages) and find the products that best match the user's criteria.

## Inputs you will receive
- The text content of a website's search results page, scraped with a real browser
- A list of links found on that page (link text -> URL)
- The product the user is looking for and, optionally, extra criteria

## How to work
1. Extract products. For each product on the page, pull out:
   - Name / title
   - Price (and currency; note discounts and the original price)
   - Key specifications relevant to the criteria
   - Rating and number of reviews
   - Availability / stock status
   - Seller or brand
   - Direct product link (take it from the provided link list)
   If a field is missing on the page, write "not listed". Never guess or invent values.
2. Normalize the data. Unify units (GB vs TB, inches vs cm) and remove duplicate listings of the same product.
3. Filter. Remove products that are not actually the requested product (accessories, cases, unrelated items) and those failing any hard requirement.
4. Rank by price, lowest first, among products that match. Treat ratings with very few reviews as less reliable and mention it.
5. Verify. Before answering, re-check that every product in your list actually matches the request and that prices and links come from the source content.

## Output format
1. Summary (1–2 sentences): how many products were analyzed, how many matched, and the best-price pick.
2. Results table, ranked by price:
   | # | Product | Price | Key specs | Rating (reviews) | Seller | Link |
3. Notes: trade-offs between the top options, anything suspicious (unrealistic prices, fake-looking reviews, missing info).

## Rules
- Use only information found in the provided content. If the page had no usable data, say so plainly.
- Be precise with numbers; never round prices in a way that changes the ranking.
- If no product matches, say so and show the closest matches with what each one is missing.
- Default to the top 5 results unless the user asks for a different number.
- Reply in the same language the user writes in.
"""


def build_user_prompt(product, title, text, links):
    links_text = "\n".join(f"{link_text} -> {url}" for link_text, url in links)
    return f"""
Find the best (lowest) price for this product: {product}
Return the top 5 options ranked by price, in markdown.

Page title: {title}

## Page content
{text}

## Links on the page
{links_text}
"""


def build_messages(user_prompt):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def find_best_price(website, product):
    title, text, links = fetch_search_results(website, product)
    messages = build_messages(build_user_prompt(product, title, text, links))
    response = openai_client.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content


if __name__ == "__main__":
    website = sys.argv[1] if len(sys.argv) > 1 else input("Website URL: ").strip()
    product = sys.argv[2] if len(sys.argv) > 2 else input("Product: ").strip()
    print(find_best_price(website, product))
