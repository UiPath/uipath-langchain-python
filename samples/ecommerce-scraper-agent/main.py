"""General E-commerce Scraper Agent

A UiPath LangGraph agent that uses LLM analysis to dynamically determine
how to extract product data from any e-commerce site.

Architecture:
  - Coordinator node: manages the URL queue, dispatches batches to sub-agents.
  - Scraper node (sub-agent): fetches pages, uses LLM to classify page type
    and determine extraction strategy (CSS selectors), extracts product data
    and navigation/product links.
  - Finalize node: deduplicates products, resolves currency symbols.
  - Sub-agents run in parallel per round via Send.

Flow:
  START -> coordinator -> [scrapers in parallel via Send] -> coordinator
        -> ... (until no new URLs) ...
        -> finalize -> END
"""

import asyncio
import json
import os
import re
from operator import add
from typing import Annotated, TypedDict
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Browser
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send
from uipath.platform import UiPath


# ── Configuration ──────────────────────────────────────────────────────────────

NUM_SUB_AGENTS = 5
CONCURRENT_PER_AGENT = 10
HTTP_TIMEOUT = 30.0
JS_RENDER_WAIT_MS = 3000
MIN_CONTENT_LENGTH = 200
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


# ── Page Fetching (httpx with playwright fallback) ─────────────────────────────

_browser: Browser | None = None
_browser_lock = asyncio.Lock()


def _page_looks_empty(html: str) -> bool:
    """Return True if the HTML has too little visible text content."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return len(text) < MIN_CONTENT_LENGTH


async def _get_browser() -> Browser:
    """Lazily launch a shared headless browser instance."""
    global _browser
    async with _browser_lock:
        if _browser is None or not _browser.is_connected():
            pw = await async_playwright().start()
            _browser = await pw.chromium.launch(headless=True)
            print("[browser] Launched headless Chromium")
    return _browser


async def _fetch_with_browser(url: str) -> str:
    """Fetch a page using a headless browser for JS-rendered content."""
    browser = await _get_browser()
    page = await browser.new_page(user_agent=USER_AGENT)
    try:
        await page.goto(url, wait_until="networkidle", timeout=HTTP_TIMEOUT * 1000)
        await page.wait_for_timeout(JS_RENDER_WAIT_MS)
        return await page.content()
    finally:
        await page.close()


async def _fetch_page(client: httpx.AsyncClient, url: str) -> str:
    """Fetch a page: try httpx first, fall back to playwright if content looks JS-rendered."""
    resp = await client.get(url)
    resp.raise_for_status()
    html = resp.text

    if _page_looks_empty(html):
        print(f"[fetch] JS-rendered page detected, using browser: {url}")
        html = await _fetch_with_browser(url)

    return html


# ── I/O Schemas ────────────────────────────────────────────────────────────────

class GraphInput(BaseModel):
    start_url: str = Field(
        default="https://sandbox.oxylabs.io/products",
        description="The starting URL to begin scraping",
    )


class GraphOutput(BaseModel):
    products: list[dict] = Field(
        description="All scraped products with extracted fields and url"
    )
    total_products: int = Field(default=0, description="Number of unique products")
    urls_scraped: int = Field(default=0, description="Number of URLs scraped")


# ── Graph State ────────────────────────────────────────────────────────────────

class GraphState(TypedDict):
    start_url: str
    url_chunks: list[list[str]]
    visited_hashes: set[int]
    scraped_urls: list[str]
    raw_products: Annotated[list[dict], add]
    discovered_urls: Annotated[list[str], add]
    products: list[dict]
    total_products: int
    urls_scraped: int


# ── LLM Page Analysis ─────────────────────────────────────────────────────────

class PageStrategy(BaseModel):
    """LLM-determined strategy for extracting data from a page type."""
    page_type: str = Field(
        description=(
            "'listing' for pages showing multiple products, "
            "'product' for single product detail pages, "
            "'other' for non-product pages"
        )
    )
    product_links_css: str = Field(
        default="",
        description="CSS selector for <a> elements linking to individual product pages",
    )
    navigation_links_css: str = Field(
        default="",
        description="CSS selector for pagination or category navigation <a> elements",
    )
    product_fields_css: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "For product detail pages: mapping of field names "
            "(e.g. 'name', 'price', 'description') to CSS selectors"
        ),
    )
    total_pages: int = Field(
        default=0,
        description=(
            "For listing pages: total number of pages if discoverable "
            "from the HTML or embedded JSON (e.g. __NEXT_DATA__). 0 if unknown."
        ),
    )
    pagination_url_template: str = Field(
        default="",
        description=(
            "For listing pages: URL template for generating all page URLs, "
            "with {page} as placeholder for the page number. "
            "E.g. 'https://example.com/products?page={page}'. Empty if unknown."
        ),
    )


_strategy_cache: dict[str, PageStrategy] = {}


def _url_pattern(url: str) -> str:
    """Normalize a URL to a cacheable pattern for strategy reuse.

    Replaces ID-like path segments with {id} and keeps only query param keys.
    """
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    segments = path.split("/")
    normalized = []
    for i, seg in enumerate(segments):
        if seg and i > 0 and (re.match(r"^\d+$", seg) or len(seg) > 30):
            normalized.append("{id}")
        else:
            normalized.append(seg)
    pattern = "/".join(normalized)
    if parsed.query:
        params = sorted(re.findall(r"([^&=]+)=", parsed.query))
        if params:
            pattern += "?" + "&".join(f"{p}=" for p in params)
    return f"{parsed.scheme}://{parsed.netloc}{pattern}"


def _clean_html_for_llm(html: str, max_chars: int = 20000) -> str:
    """Clean and truncate HTML for LLM analysis."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["style", "link", "meta", "noscript", "svg", "img"]):
        tag.decompose()
    for script in soup.find_all("script"):
        if script.get("id") != "__NEXT_DATA__":
            script.decompose()
    text = str(soup)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n<!-- truncated -->"
    return text


async def _analyze_page(html: str, url: str) -> PageStrategy:
    """Analyze a page with LLM to determine extraction strategy.

    Cached per URL pattern so the LLM is only called once per page type.
    """
    pattern = _url_pattern(url)
    if pattern in _strategy_cache:
        print(f"[analyze] Cache hit for pattern: {pattern}")
        return _strategy_cache[pattern]

    cleaned = _clean_html_for_llm(html)
    sdk = UiPath()

    response = await sdk.llm.chat_completions(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a web scraping expert. Analyze the HTML and return a JSON object with:\n"
                    "- page_type: 'listing' if it shows multiple products, 'product' if single product detail, 'other' otherwise\n"
                    "- product_links_css: CSS selector for <a> tags linking to individual product pages (empty string if none)\n"
                    "- navigation_links_css: CSS selector for pagination/category <a> tags (empty string if none)\n"
                    "- product_fields_css: for product detail pages only, map field names (name, price, description, etc.) to CSS selectors that extract each field's text content (empty object for listing/other pages)\n"
                    "- total_pages: for listing pages, the total number of pages if you can find it in the HTML or embedded JSON like __NEXT_DATA__ (look for pageCount, totalPages, last page number in pagination, etc.). 0 if unknown.\n"
                    "- pagination_url_template: for listing pages, the URL pattern for paginated pages with {page} as placeholder for the page number (e.g. 'https://example.com/products?page={page}'). Derive this from pagination links in the HTML. Empty string if unknown.\n\n"
                    "IMPORTANT: Look carefully at embedded JSON data (like <script id=\"__NEXT_DATA__\">) for total page counts.\n"
                    "Use robust CSS selectors based on class names or semantic structure.\n"
                    "Return ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": f"URL: {url}\n\nHTML:\n{cleaned}",
            },
        ],
        model="gpt-4o-mini-2024-07-18",
        temperature=0,
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)
    strategy = PageStrategy(**data)
    _strategy_cache[pattern] = strategy
    print(f"[analyze] Strategy for {pattern}: type={strategy.page_type}")
    return strategy


def _extract_links(soup: BeautifulSoup, css_selector: str, base_url: str) -> list[str]:
    """Extract absolute URLs from elements matching a CSS selector."""
    if not css_selector:
        return []
    try:
        elements = soup.select(css_selector)
    except Exception:
        return []
    urls = []
    for el in elements:
        href = el.get("href", "")
        if href and not href.startswith(("#", "javascript:", "mailto:")):
            urls.append(urljoin(base_url, href))
    return urls


def _strip_label_prefix(text: str, field: str) -> str:
    """Remove label prefixes like 'Developer:', 'Platform:' from extracted text."""
    # Try stripping a prefix that looks like the field name (case-insensitive)
    pattern = re.compile(rf"^\s*{re.escape(field)}\s*:\s*", re.IGNORECASE)
    stripped = pattern.sub("", text)
    # Also strip any generic "Label:" prefix at the start
    if not stripped and ":" in text:
        stripped = text.split(":", 1)[1].strip()
    return stripped


def _extract_product(
    soup: BeautifulSoup, fields_css: dict[str, str], url: str
) -> dict:
    """Extract product data from a page using CSS selectors."""
    product = {"url": url}
    for field, selector in fields_css.items():
        try:
            el = soup.select_one(selector)
            if el is None:
                product[field] = ""
                continue
            # Remove label child tags before extracting text
            for label_tag in el.find_all(["strong", "label", "b"], recursive=False):
                label_tag.extract()
            text = el.get_text(strip=True)
            # Also strip label-like prefixes from the text itself
            product[field] = _strip_label_prefix(text, field)
        except Exception:
            product[field] = ""
    return product


# ── Price / Currency helpers ──────────────────────────────────────────────────

def _split_price(price_text: str) -> tuple[str, str]:
    """Split '91,99 €' into ('91,99', '€'). Returns (value, symbol)."""
    price_text = price_text.strip()
    if not price_text:
        return "", ""
    m = re.match(r"^([^\d]*?)\s*([\d.,]+)\s*([^\d]*)$", price_text)
    if m:
        prefix, value, suffix = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        symbol = prefix or suffix
        return value, symbol
    return price_text, ""


# ── In-memory currency cache ──

_currency_cache: dict[str, str] = {}


async def _resolve_currency_iso(symbol: str) -> str:
    """Resolve a currency symbol (e.g. '€') to ISO 4217 code (e.g. 'EUR').

    1. Check in-memory cache.
    2. Ask LLM.
    """
    if not symbol:
        return ""
    if symbol in _currency_cache:
        print(f"[currency] Cache hit: '{symbol}' -> '{_currency_cache[symbol]}'")
        return _currency_cache[symbol]

    sdk = UiPath()

    try:
        response = await sdk.llm.chat_completions(
            messages=[
                {
                    "role": "system",
                    "content": "Reply with ONLY the ISO 4217 three-letter currency code. No explanation.",
                },
                {
                    "role": "user",
                    "content": f"What is the ISO 4217 currency code for this symbol: {symbol}",
                },
            ],
            model="gpt-4o-mini-2024-07-18",
            max_tokens=10,
            temperature=0,
        )
        text = response.choices[0].message.content.strip().upper()
        iso = re.search(r"\b([A-Z]{3})\b", text)
        if iso:
            _currency_cache[symbol] = iso.group(1)
            print(f"[currency] Resolved '{symbol}' -> '{iso.group(1)}' via LLM")
            return iso.group(1)
    except Exception as exc:
        print(f"[currency] LLM failed for '{symbol}': {exc}")

    _currency_cache[symbol] = symbol
    return symbol


# ── Product Enrichment via Context Grounding ──────────────────────────────────

async def _enrich_products(products: list[dict]) -> None:
    """Enrich scraped products with data from a Context Grounding catalog index.

    Searches the index by product name and merges any matched catalog fields
    (e.g. category, sku, brand) into the product dict. Skipped if the
    PRODUCT_CATALOG_CG_INDEX env var is not set.
    """
    cg_index = os.environ.get("PRODUCT_CATALOG_CG_INDEX")
    if not cg_index:
        return

    sdk = UiPath()
    enriched_count = 0

    for p in products:
        name = p.get("name", "")
        if not name:
            continue
        try:
            results = await sdk.context_grounding.search_async(
                name=cg_index,
                query=name,
                number_of_results=1,
            )
            if results:
                try:
                    catalog_data = json.loads(results[0].content)
                except (json.JSONDecodeError, TypeError):
                    catalog_data = {"catalog_info": results[0].content.strip()}
                for key, value in catalog_data.items():
                    if key not in p:
                        p[key] = value
                enriched_count += 1
        except Exception as exc:
            print(f"[enrich] Failed for '{name}': {exc}")

    if enriched_count:
        print(f"[enrich] Enriched {enriched_count}/{len(products)} products from catalog")


# ── Graph Nodes ────────────────────────────────────────────────────────────────

async def coordinator(state: GraphState) -> dict:
    """Manage the URL queue and dispatch work to sub-agents.

    Accepts all discovered URLs (product pages, listing pages) as long as
    they are on the same domain and haven't been visited yet.
    """
    visited: set[int] = state.get("visited_hashes") or set()
    scraped_urls: list[str] = state.get("scraped_urls") or []
    discovered = state.get("discovered_urls") or []
    raw_products = state.get("raw_products") or []

    start_url = state.get("start_url") or GraphInput().start_url

    # First run: seed with start URL
    if not visited:
        print(f"[coordinator] Starting scrape from {start_url}")
        return {
            "start_url": start_url,
            "url_chunks": [[start_url]],
            "visited_hashes": {hash(start_url)},
            "scraped_urls": [start_url],
            "total_products": 0,
            "urls_scraped": 0,
        }

    # Filter to same-domain, unvisited URLs
    start_domain = urlparse(start_url).netloc
    new_urls: list[str] = []
    for u in set(discovered):
        if hash(u) not in visited and urlparse(u).netloc == start_domain:
            new_urls.append(u)

    if not new_urls:
        print(
            f"[coordinator] Scraping complete! "
            f"{len(raw_products)} products from {len(scraped_urls)} URLs"
        )
        return {
            "url_chunks": [],
            "visited_hashes": visited,
            "scraped_urls": scraped_urls,
            "total_products": len(raw_products),
            "urls_scraped": len(scraped_urls),
        }

    print(f"[coordinator] Dispatching {len(new_urls)} URLs to scrape")

    new_visited = visited | {hash(u) for u in new_urls}
    num_chunks = min(NUM_SUB_AGENTS, len(new_urls))
    chunks: list[list[str]] = [[] for _ in range(num_chunks)]
    for i, url in enumerate(new_urls):
        chunks[i % num_chunks].append(url)

    return {
        "url_chunks": chunks,
        "visited_hashes": new_visited,
        "scraped_urls": scraped_urls + new_urls,
        "total_products": len(raw_products),
        "urls_scraped": len(scraped_urls),
    }


def dispatch_scrapers(state: GraphState) -> list[Send] | str:
    """Route: send URL chunks to parallel sub-agents, or go to finalize."""
    chunks = state.get("url_chunks") or []
    if not chunks:
        return "finalize"
    return [Send("scraper", {"urls": chunk}) for chunk in chunks]


async def scraper(state: dict) -> dict:
    """Sub-agent: fetches pages, analyzes with LLM, extracts data and links.

    For listing pages: extracts product links and navigation links.
    For product pages: extracts product data fields and any additional links.
    """
    urls: list[str] = state["urls"]
    all_products: list[dict] = []
    all_discovered: list[str] = []
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(CONCURRENT_PER_AGENT)

    async def fetch_one(client: httpx.AsyncClient, url: str) -> None:
        async with sem:
            try:
                html = await _fetch_page(client, url)

                strategy = await _analyze_page(html, url)
                soup = BeautifulSoup(html, "html.parser")

                products: list[dict] = []
                discovered: list[str] = []

                if strategy.page_type == "listing":
                    discovered.extend(
                        _extract_links(soup, strategy.product_links_css, url)
                    )
                    discovered.extend(
                        _extract_links(soup, strategy.navigation_links_css, url)
                    )
                    # Generate all pagination URLs if total_pages is known
                    if strategy.total_pages > 1 and strategy.pagination_url_template:
                        for page_num in range(1, strategy.total_pages + 1):
                            discovered.append(
                                strategy.pagination_url_template.replace(
                                    "{page}", str(page_num)
                                )
                            )
                elif strategy.page_type == "product":
                    if strategy.product_fields_css:
                        product = _extract_product(
                            soup, strategy.product_fields_css, url
                        )
                        if any(v for k, v in product.items() if k != "url"):
                            products.append(product)
                    discovered.extend(
                        _extract_links(soup, strategy.product_links_css, url)
                    )
                    discovered.extend(
                        _extract_links(soup, strategy.navigation_links_css, url)
                    )

                async with lock:
                    all_products.extend(products)
                    all_discovered.extend(discovered)
            except Exception as exc:
                print(f"  [sub-agent] ERROR {url}: {exc}")

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=HTTP_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        await asyncio.gather(*[fetch_one(client, url) for url in urls])

    print(
        f"  [sub-agent] Done: {len(urls)} pages "
        f"-> {len(all_products)} products, {len(all_discovered)} links"
    )

    return {"raw_products": all_products, "discovered_urls": all_discovered}


async def finalize(state: GraphState) -> dict:
    """Deduplicate products by URL, resolve currency symbols to ISO codes."""
    raw = state.get("raw_products") or []
    scraped = state.get("scraped_urls") or []

    seen: set[str] = set()
    unique: list[dict] = []
    for p in raw:
        key = p.get("url", "")
        if key and key not in seen:
            seen.add(key)
            unique.append(p)

    print(
        f"[finalize] Deduplicated {len(raw)} -> {len(unique)} unique products"
    )

    # ── Enrich from product catalog (if configured) ──
    await _enrich_products(unique)

    # ── Resolve currency from price fields ──
    for p in unique:
        price_raw = p.get("price", "")
        if price_raw:
            value, symbol = _split_price(price_raw)
            if symbol:
                p["price"] = value
                p["currency"] = await _resolve_currency_iso(symbol)

    return {
        "products": unique,
        "total_products": len(unique),
        "urls_scraped": len(scraped),
    }


# ── Build Graph ────────────────────────────────────────────────────────────────

builder = StateGraph(GraphState, input=GraphInput, output=GraphOutput)

builder.add_node("coordinator", coordinator)
builder.add_node("scraper", scraper)
builder.add_node("finalize", finalize)

builder.add_edge(START, "coordinator")
builder.add_conditional_edges(
    "coordinator", dispatch_scrapers, ["scraper", "finalize"]
)
builder.add_edge("scraper", "coordinator")
builder.add_edge("finalize", END)

graph = builder.compile()
