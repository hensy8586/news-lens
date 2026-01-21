import os
from typing import List, Dict, Any, Iterable, Tuple, Optional

from supabase import create_client

from .rss_sources import RSS_FEEDS
from .rss_utils import fetch_rss_articles, fetch_article_fulltext, extract_headline_image


def get_supabase_client():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)


def fetch_rss_items() -> List[Dict[str, Any]]:
    """
    Ingest RSS items (no AI). Enrich each item with fulltext + headline image.
    """

    all_items: List[Dict[str, Any]] = []
    for feed_dict in RSS_FEEDS.values():
        filtered_articles = []
        feed_articles = fetch_rss_articles(
            feed_url=feed_dict["feed_url"],
            source_outlet=feed_dict["sub_source"],
            region=feed_dict["source_region"],
            topic=feed_dict["topic_region"],
            language=feed_dict["language"],
        )

        # Enrich each article with full text + image URL
        for article in feed_articles:
            link = article.get("link")
            if not link:
                continue

            content_text = fetch_article_fulltext(link)
            if isinstance(content_text, str) and len(content_text) > 1000:
                article["content_text"] = content_text
                article["image_url"] = extract_headline_image(link)
                filtered_articles.append(article)

        all_items.extend(filtered_articles)

    return all_items


def _chunks(lst: List[Dict[str, Any]], n: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _make_dedupe_key(a: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """
    Dedupe key MUST match your on_conflict key.
    Here: (link, published_at)

    Note: published_at should be normalized consistently by fetch_rss_articles
    (ideally ISO string).
    """

    link = a.get("link")
    published_at = a.get("published_at")
    if not link or not published_at:
        return None
    return (str(link), str(published_at))


def _dedupe_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for item in items:
        key = _make_dedupe_key(item)
        if key is None:
            # Skip malformed entries (no key for upsert)
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _prepare_rows(news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prepare rows for Supabase upsert.

    IMPORTANT: We explicitly set ai_eligible/archived defaults for NEW ingested items.
    This prevents surprises if defaults ever change and makes the pipeline intent clear.
    """

    rows: List[Dict[str, Any]] = []

    for a in news_items:
        source_outlet = a.get("source_outlet")
        link = a.get("link")
        published_at = a.get("published_at")

        if not source_outlet or not link or not published_at:
            # Skip malformed entries
            continue

        rows.append(
            {
                "source_outlet": source_outlet,
                "region": a.get("region"),
                "topic": a.get("topic"),
                "language": a.get("language"),
                "title": a.get("title"),
                "summary": a.get("summary"),
                "link": link,
                "image_url": a.get("image_url"),
                "published_at": published_at,
                "content_html": a.get("content_html"),
                "content_text": a.get("content_text"),
                "raw_entry": a.get("raw_entry"),
                "ai_eligible": True,
                "archived": False,
                "archived_at": None,
                "archive_reason": None,
            }
        )

    # Dedupe based on the same key you use in upsert conflict handling
    return _dedupe_items(rows)


def upsert_news_items(client, items: List[Dict[str, Any]], chunk_size: int = 100):
    if not items:
        return

    rows = _prepare_rows(items)

    for batch in _chunks(rows, chunk_size):
        # NOTE: on_conflict must match a UNIQUE constraint/index in Postgres.
        # Make sure you have UNIQUE(link, published_at) in your DB, or this will error.
        (
            client.table("articles")
            .upsert(
                batch,
                on_conflict="link,published_at",
                ignore_duplicates=False,
            )
            .execute()
        )


def main():
    client = get_supabase_client()
    items = fetch_rss_items()
    upsert_news_items(client, items)


if __name__ == "__main__":
    main()
