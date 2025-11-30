import os
from typing import List, Dict, Any

from supabase import create_client
from openai import OpenAI   # if you want LLM summaries

from .rss_sources import RSS_FEEDS
from .rss_utils import fetch_rss_articles, fetch_article_fulltext


def get_supabase_client():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)


def fetch_rss_items():
    all_items = []
    for feed_dict in RSS_FEEDS.values():
        feed_articles = fetch_rss_articles(
            feed_url=feed_dict["feed_url"],
            source_outlet=feed_dict["sub_source"],
            region=feed_dict["source_region"],
            topic=feed_dict["topic_region"],
            language=feed_dict["language"]
        )
        for article in feed_articles:
            article["content_text"] = fetch_article_fulltext(article["link"])

        all_items.extend(feed_articles)
    return all_items


def _prepare_rows(news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for a in news_items:
        source_outlet = a.get("source_outlet")
        link = a.get("link")
        if not source_outlet or not link:
            # Skip malformed entries
            continue

        rows.append({
            "id": a.get("id"),
            "source_outlet": source_outlet,
            "region": a.get("region"),
            "topic": a.get("topic"),
            "language": a.get("language"),
            "title": a.get("title"),
            "summary": a.get("summary"),
            "link": link,
            "published_at": a.get("published_at"),
            "content_html": a.get("content_html"),
            "content_text": a.get("content_text"),
            "raw_entry": a.get("raw_entry"),
        })
    return rows


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def upsert_news_items(client, items: List[Dict[str, Any]], chunk_size: int = 100):
    if not items:
        return

    rows = _prepare_rows(items)
    for batch in _chunks(rows, chunk_size):
        res = (
            client.table("articles")
            .upsert(
                batch,
                on_conflict=["id"],  # upsert by deterministic UUID
                ignore_duplicates=False
            )
            .execute()
        )


def main():
    client = get_supabase_client()
    items = fetch_rss_items()
    upsert_news_items(client, items)


if __name__ == "__main__":
    main()
