from __future__ import annotations

import re
from urllib.parse import urlparse, urljoin, parse_qs, unquote
import requests
from bs4 import BeautifulSoup

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def is_google_news_url(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return host.endswith("news.google.com")


def _extract_external_url_from_google_html(html: str, base_url: str) -> str | None:
    """
    Google News landing pages contain outbound links to the original publisher.
    We heuristically pick the best external URL.
    """
    soup = BeautifulSoup(html, "html.parser")

    candidates: list[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()

        # Make relative links absolute
        href_abs = urljoin(base_url, href)

        # Some links embed the target as a query param, decode if present
        parsed = urlparse(href_abs)
        qs = parse_qs(parsed.query)
        for key in ("url", "u"):
            if key in qs and qs[key]:
                candidates.append(unquote(qs[key][0]))

        # Direct external link?
        if href_abs.startswith("http"):
            host = urlparse(href_abs).netloc.lower()
            if (
                host
                and not host.endswith("news.google.com")
                and not host.endswith("google.com")
                and "accounts.google.com" not in host
            ):
                candidates.append(href_abs)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for c in candidates:
        c = c.strip()
        if c and c not in seen:
            seen.add(c)
            deduped.append(c)

    if not deduped:
        return None

    # Heuristic: prefer "article-like" URLs; otherwise choose the longest
    def score(u: str) -> tuple[int, int]:
        articleish = int(bool(re.search(r"/(article|news|story|stories)/|/20\d{2}/", u)))
        return (articleish, len(u))

    return sorted(deduped, key=score, reverse=True)[0]


def resolve_google_news_to_publisher(url: str, timeout: float = 15.0, debug: bool = False) -> str | None:
    """
    Resolves a Google News RSS/article URL to the publisher's original URL.
    """
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout, allow_redirects=True)
    except Exception as e:
        if debug:
            print(f"[resolve_google_news] request failed: {e!r}")
        return None

    final_url = r.url
    if debug:
        print(f"[resolve_google_news] final_url={final_url}")

    # Sometimes after redirects you already land on the publisher domain
    if not is_google_news_url(final_url) and final_url.startswith("http"):
        return final_url

    print("r.text: ", r.text)

    # Otherwise, parse the Google News HTML to find the outbound publisher link
    publisher_url = _extract_external_url_from_google_html(r.text, base_url=final_url)
    if debug:
        print(f"[resolve_google_news] publisher_url={publisher_url}")

    return publisher_url
