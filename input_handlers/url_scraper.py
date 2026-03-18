"""URL content extraction using trafilatura."""

import trafilatura


def scrape_url(url: str) -> str:
    """
    Extract main article body from any URL — news, blogs, research pages.
    Returns cleaned article text, or empty string on failure.
    """
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return ""
    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
    )
    return text or ""
