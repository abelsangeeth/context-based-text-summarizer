"""
Vision-Augmented LLM Pipeline for Long-Form Document Summarization — URL / Web Scraper Handler
══════════════════════════════════════════
Three-tier extraction:
  Tier 1 → trafilatura  (precision-first)
  Tier 2 → newspaper3k  (good headline / body separation)
  Tier 3 → BeautifulSoup raw fallback
"""

from urllib.parse import urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup

from input_handlers.cleaner import clean_text, chunk_text


# ── News domains for model routing ─────────────────────────────────────────
_NEWS_DOMAINS = frozenset([
    "bbc.com", "bbc.co.uk", "cnn.com", "reuters.com", "nytimes.com",
    "theguardian.com", "apnews.com", "ndtv.com", "thehindu.com",
    "washingtonpost.com", "aljazeera.com", "bloomberg.com",
    "economictimes.com", "hindustantimes.com", "timesofindia.com",
])


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════
def scrape_url(url: str) -> dict:
    """
    Scrape article text from *url* using a three-tier fallback strategy.

    Returns
    -------
    dict
        type, source, domain, title, text, chunks, word_count, scrape_method
    """
    if not validate_url(url):
        raise ValueError("Invalid URL format.")

    text = ""
    title = None
    method = ""

    # ── Tier 1: trafilatura ─────────────────────────────────
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                include_images=False,
                favor_precision=True,
            ) or ""
            if text and len(text.split()) > 50:
                method = "trafilatura"
    except Exception:
        pass

    # ── Tier 2: newspaper3k ─────────────────────────────────
    if not method:
        try:
            from newspaper import Article

            article = Article(url)
            article.download()
            article.parse()
            text = article.text or ""
            title = article.title
            if text and len(text.split()) > 50:
                method = "newspaper"
        except Exception:
            pass

    # ── Tier 3: BeautifulSoup ───────────────────────────────
    if not method:
        try:
            resp = requests.get(url, timeout=10, headers={
                "User-Agent": "Mozilla/5.0 (ContextFreeX/1.0)"
            })
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove noisy tags
            for tag in soup(["script", "style", "nav", "footer",
                             "header", "aside", "form"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else None
            if text:
                method = "bs4"
        except requests.exceptions.Timeout:
            raise ValueError("Could not reach URL. Check connection.")
        except requests.exceptions.RequestException:
            raise ValueError("Could not reach URL. Check connection.")

    if not text or len(text.split()) < 10:
        raise ValueError("Could not extract article text from this URL.")

    # ── Post-process ────────────────────────────────────────
    domain = urlparse(url).netloc.replace("www.", "")
    cleaned = clean_text(text)
    chunks = chunk_text(cleaned, max_words=800, overlap=100)
    word_count = len(cleaned.split())

    return {
        "type": "url",
        "source": url,
        "domain": domain,
        "title": title,
        "text": cleaned,
        "chunks": chunks,
        "word_count": word_count,
        "scrape_method": method,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════
def validate_url(url: str) -> bool:
    """Return True if *url* looks like a valid HTTP(S) URL."""
    if not url:
        return False
    if not url.startswith(("http://", "https://")):
        return False
    parsed = urlparse(url)
    return bool(parsed.netloc)


def is_news_article(url: str, text: str = "") -> bool:
    """
    Heuristic: is this URL a news article?
    Used to recommend BART-CNN model automatically.
    """
    domain = urlparse(url).netloc.replace("www.", "")
    if any(nd in domain for nd in _NEWS_DOMAINS):
        return True

    path = urlparse(url).path.lower()
    news_path_markers = ["/article/", "/news/", "/story/",
                         "/politics/", "/world/", "/opinion/"]
    if any(marker in path for marker in news_path_markers):
        return True

    return False
