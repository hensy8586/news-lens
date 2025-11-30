import feedparser
from datetime import datetime
from dateutil import parser as dateparser
from typing import List, Dict, Any, Optional
import trafilatura
import hashlib
import uuid


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


def make_article_id(source_outlet: str, link: str) -> str:
    """Deterministic UUID based on (source_outlet, link)."""
    key = f"{source_outlet}::{link}".encode("utf-8")
    digest = hashlib.sha256(key).digest()  # 32 bytes
    uid = uuid.UUID(bytes=digest[:16])     # take first 16 bytes as UUID
    return str(uid)


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
        link = entry.get("link")

        article = {
            "id": make_article_id(source_outlet, link),
            "source_type": "rss",
            "source_outlet": source_outlet,
            "region": region,       # e.g. "us", "europe", "world"
            "topic": topic,         # e.g. "general", "world", "politics"
            "language": language,   # "en"
            "title": entry.get("title"),
            "summary": entry.get("summary") or entry.get("description"),
            "content_html": content_html,
            "link": link,
            "published_at": published_at,
            "raw_entry": entry,     # keep for debugging/enrichment
        }
        articles.append(article)

    return articles


def fetch_article_fulltext(url: str, debug: bool = False) -> str | None:
    # Normalize to https for safety
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]

    # Let trafilatura handle the HTTP requests and quirks
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        if debug:
            print(f"[fetch_article_fulltext] trafilatura.fetch_url() returned nothing for {url}")
        return None

    try:
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=True,
        )
    except Exception as e:
        if debug:
            print(f"[fetch_article_fulltext] Error extracting {url}: {repr(e)}")
        return None

    if not text and debug:
        print(f"[fetch_article_fulltext] No text extracted for {url}")
        text = None

    return text
