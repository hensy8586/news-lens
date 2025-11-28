import os
from datetime import datetime, timezone

import feedparser
from supabase import create_client
from openai import OpenAI   # if you want LLM summaries

from .rss_sources import RSS_FEEDS


def get_supabase_client():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)


def fetch_rss_items():
    all_items = []
    for feed_url in RSS_FEEDS:
        d = feedparser.parse(feed_url)
        for entry in d.entries:
            all_items.append(
                {
                    "source": feed_url,
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "summary_raw": entry.get("summary", ""),
                    "published_at": _parse_published(entry),
                }
            )
    return all_items


def _parse_published(entry):
    # Very rough; you can improve later
    if "published_parsed" in entry and entry.published_parsed:
        dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        return dt.isoformat()
    return datetime.now(timezone.utc).isoformat()


def upsert_news_items(client, items):
    if not items:
        return

    # Simplest approach: just insert; de-dupe later
    client.table("news_items").insert(items).execute()


def main():
    client = get_supabase_client()
    items = fetch_rss_items()
    upsert_news_items(client, items)


if __name__ == "__main__":
    main()
