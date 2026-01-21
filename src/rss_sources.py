# NOTE(HY): Currently using Google News to fetch media articles from multiple sources
# Google controls what to provide via its url; it's on a "relevance" base rather than chronological

RSS_FEEDS = {
    "BBC World": {
        "feed_url": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "source": "BBC",
        "sub_source": "BBC World",
        "source_region": "eu",
        "source_country": "uk",
        "topic_region": "world",
        "language": "en"
    },
    "BBC Main Headlines": {
        "feed_url": "http://feeds.bbci.co.uk/news/rss.xml",
        "source": "BBC",
        "sub_source": "BBC Main Headlines",
        "source_region": "eu",
        "source_country": "uk",
        "topic_region": "world",
        "language": "en"
    },
    "The Guardian US News": {
        "feed_url": "https://www.theguardian.com/us-news/rss",
        "source": "The Guardian",
        "sub_source": "The Guardian US News",
        "source_region": "eu",
        "source_country": "uk",
        "topic_region": "us",
        "language": "en"
    },
    "The Guardian World News": {
        "feed_url": "https://www.theguardian.com/world/rss",
        "source": "The Guardian",
        "sub_source": "The Guardian World News",
        "source_region": "eu",
        "source_country": "uk",
        "topic_region": "world",
        "language": "en"
    },
    "The Guardian Europe News": {
        "feed_url": "https://www.theguardian.com/world/europe-news/rss",
        "source": "The Guardian",
        "sub_source": "The Guardian Europe News",
        "source_region": "eu",
        "source_country": "uk",
        "topic_region": "eu",
        "language": "en"
    },
    "Google News": {
        "feed_url": "https://news.google.com/rss",
        "source": "Google News",
        "sub_source": "Google News",
        "source_region": "google",
        "source_country": "google",
        "topic_region": "google",
        "language": "en"
    }
}