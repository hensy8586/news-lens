import feedparser
from datetime import datetime
from dateutil import parser as dateparser
from typing import List, Dict, Any, Optional
import trafilatura
import hashlib
import uuid
from bs4 import BeautifulSoup
import requests
import json
from urllib.parse import urlparse, urljoin
from googlenewsdecoder import gnewsdecoder


UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)

DEFAULT_HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


def parse_entry_published(entry: Dict[str, Any]) -> Optional[datetime]:
    """ Best-effort parse of published date from a feed entry """

    for key in ("published", "updated", "pubDate"):
        val = entry.get(key)
        if val:
            try:
                return dateparser.parse(val)
            except Exception:
                pass

    # Some feeds put it in entry.get("published_parsed") as struct_time
    if entry.get("published_parsed"):
        try:
            return datetime(*entry.published_parsed[:6])
        except Exception:
            pass

    return None


def extract_entry_content(entry: Dict[str, Any]) -> Optional[str]:
    """ Extract content of feed entry if present """

    # 1) Full-content field (if the publisher includes it)
    if "content" in entry and entry.content:
        # content is usually a list of dicts with 'value' and 'type'
        for c in entry.content:
            if "value" in c:
                return c["value"]

    # 2) Fallbacks that often contain HTML snippets or sometimes full text
    if "summary" in entry:
        return entry.summary
    if "description" in entry:
        return entry.description

    return None


def make_article_id(published_dt: str, link: str) -> str:
    """ Deterministic UUID based on (source_outlet, link) """

    key = f"{published_dt}::{link}".encode("utf-8")
    digest = hashlib.sha256(key).digest()  # 32 bytes
    uid = uuid.UUID(bytes=digest[:16])     # take first 16 bytes as UUID
    return str(uid)


def _is_google_news_url(url: str) -> bool:
    try:
        return urlparse(url).netloc.endswith("news.google.com")
    except Exception:
        return False


def _normalize_url(url: str, debug: bool = False) -> str:

    # Normalize to https
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]

    # Decode Google News URL -> publisher URL
    if _is_google_news_url(url):
        try:
            out = gnewsdecoder(url)
            # out = {"status": True/False, "decoded_url": "..."} or {"status": False, "message": "..."}
            if not out or not out.get("status"):
                if debug:
                    print(f"[gnewsdecoder] failed for {url}: {out}")
                return None
            url = out.get("decoded_url")
            if debug:
                print(f"[gnewsdecoder] decoded -> {url}")
        except Exception as e:
            if debug:
                print(f"[gnewsdecoder] exception: {e!r}")
            return None

        if not url:
            return None

    return url


def _looks_paywalled(html: str) -> bool:
    h = (html or "").lower()
    signals = [
        "subscribe to read",
        "subscription",
        "sign up",
        "already a subscriber",
        "meteredContent",
        "paywall",
        "Be civil. Be kind."  # from boston.com
    ]
    return any(s in h for s in signals)


def fetch_rss_articles(
    feed_url: str, *,
    source_outlet: str, region: str, topic: str, language: str = "en",
) -> List[Dict[str, Any]]:
    """ Fetch and normalize articles from an RSS/Atom feed into a common schema """

    parsed = feedparser.parse(feed_url)

    articles: List[Dict[str, Any]] = []
    for entry in parsed.entries:
        published_at = parse_entry_published(entry)
        content_html = extract_entry_content(entry)
        link = _normalize_url(entry.get("link"))
        if not link or _looks_paywalled(link):  # skip any article behind paywall
            continue

        title = entry.get("title")
        if _is_google_news_url(entry.get("link")):
            source_outlet = title.split(" - ")[-1]
            title = title[:-len(" - "+source_outlet)]

        article = {
            "id": make_article_id(published_at.isoformat(), link),
            "source_type": "rss",
            "source_outlet": source_outlet,
            "region": region,       # e.g. "us", "europe", "world"
            "topic": topic,         # e.g. "general", "world", "politics"
            "language": language,   # "en"
            "title": title,
            "summary": entry.get("summary") or entry.get("description"),
            "content_html": content_html,
            "link": link,
            "published_at": published_at.isoformat(),
            "raw_entry": entry,     # keep for debugging/enrichment
        }
        articles.append(article)

    return articles


def fetch_article_fulltext(url: str, debug: bool = False) -> str | None:
    """Fetch full text given the link to an article (handles Google News wrapper URLs)."""

    # Fetch + extract via trafilatura
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        if debug:
            print(f"[fetch_article_fulltext] fetch_url() returned nothing for {url}")
        return None

    try:
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=False,  # IMPORTANT for mixed publishers
        )
    except Exception as e:
        if debug:
            print(f"[fetch_article_fulltext] Error extracting {url}: {e!r}")
        return None

    if not text and debug:
        print(f"[fetch_article_fulltext] No text extracted for {url}")

    return text or None


def _jsonld_iter_objects(obj):
    """Yield dict objects from JSON-LD blobs that may be dict/list/@graph."""
    if obj is None:
        return
    if isinstance(obj, dict):
        yield obj
        g = obj.get("@graph")
        if isinstance(g, list):
            for x in g:
                if isinstance(x, dict):
                    yield x
    elif isinstance(obj, list):
        for x in obj:
            yield from _jsonld_iter_objects(x)


def _extract_image_from_jsonld_obj(d: dict) -> list[str]:
    out = []

    def add_image(v):
        if not v:
            return
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, list):
            for item in v:
                add_image(item)
        elif isinstance(v, dict):
            # common shapes: {"url": "..."} or {"@type":"ImageObject","url":"..."}
            if isinstance(v.get("url"), str):
                out.append(v["url"])
            elif isinstance(v.get("@id"), str):
                out.append(v["@id"])

    # Many articles: type NewsArticle/Article/ReportageBlogPosting etc.
    add_image(d.get("image"))
    add_image(d.get("thumbnailUrl"))

    return out


def _looks_like_junk_image(url: str) -> bool:
    u = url.lower()
    junk_substrings = [
        "logo", "sprite", "icon", "favicon", "share=", "pixel", "spacer",
        "blank", "placeholder"
    ]
    return any(s in u for s in junk_substrings)


def extract_headline_image(url: str, timeout: float = 12.0, debug: bool = False) -> str | None:
    """Extract a likely headline image URL from an article page."""

    with requests.Session() as s:
        r = s.get(url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True)
        final_url = r.url
        html = r.text

    if debug:
        print(f"[image] status={r.status_code} final={final_url}")

    # If we got blocked or redirected to a generic consent/bot page, bail early
    # (CNBC sometimes does this depending on region/cookies)
    if r.status_code >= 400:
        return None
    if "consent" in final_url.lower() or "privacy" in final_url.lower():
        if debug:
            print("[image] likely consent wall; no reliable og:image available")
        return None

    soup = BeautifulSoup(html, "html.parser")
    candidates: list[tuple[str, int]] = []  # (url, score)

    # --- Meta tags (collect ALL) ---
    # property=og:image is most useful; CNBC often uses it when you get real HTML
    meta_props = [
        ("property", "og:image"),
        ("property", "og:image:url"),
        ("name", "twitter:image"),
        ("name", "twitter:image:src"),
    ]
    for attr, key in meta_props:
        for tag in soup.find_all("meta", attrs={attr: key}):
            content = tag.get("content")
            if content:
                candidates.append((content, 50))

    # Some sites use relative URLs in meta
    normalized = []
    for u, sc in candidates:
        if u.startswith("//"):
            u = "https:" + u
        elif u.startswith("/"):
            u = urljoin(final_url, u)
        normalized.append((u, sc))
    candidates = normalized

    # Try to boost if og:image:width exists and looks large
    # (If multiple candidates, larger images are usually the headline.)
    og_width = None
    wtag = soup.find("meta", attrs={"property": "og:image:width"})
    if wtag and wtag.get("content") and wtag["content"].isdigit():
        og_width = int(wtag["content"])

    boosted = []
    for u, sc in candidates:
        if og_width and og_width >= 800:
            boosted.append((u, sc + 10))
        else:
            boosted.append((u, sc))
    candidates = boosted

    # --- JSON-LD (support dict/list/@graph and ImageObject) ---
    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or script.get_text(strip=False) or ""
        raw = raw.strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue

        for obj in _jsonld_iter_objects(data):
            if not isinstance(obj, dict):
                continue
            for u in _extract_image_from_jsonld_obj(obj):
                # Normalize like above
                if u.startswith("//"):
                    u = "https:" + u
                elif u.startswith("/"):
                    u = urljoin(final_url, u)
                candidates.append((u, 40))

    # --- Filter and pick best ---
    # Remove junk and duplicates
    seen = set()
    filtered: list[tuple[str, int]] = []
    for u, sc in candidates:
        u = u.strip()
        if not u or u in seen:
            continue
        seen.add(u)
        if _looks_like_junk_image(u):
            continue
        # prefer https images
        if u.startswith("https://"):
            sc += 3
        filtered.append((u, sc))

    if debug:
        print("[image] candidates:")
        for u, sc in sorted(filtered, key=lambda x: x[1], reverse=True)[:8]:
            print(" ", sc, u)

    if not filtered:
        return None

    # pick highest score; tie-break by longer URL (often indicates CDN params for big image)
    filtered.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
    return filtered[0][0]
