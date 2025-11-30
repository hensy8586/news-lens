# NOTE(HY): Based on information from ChatGPT below; holding off on getting news from Reuters and AP

# ON **REUTERS**
# Reuters used to publish an official RSS directory (e.g. reuters/topNews, reuters/worldNews).
# An archived version shows that pattern. However, since then they’ve tightened up crawling and
# RSS access, and people often use bridge services or custom scraping instead.

# ON **AP NEWS**
# AP used to provide RSS directly; now they no longer do so officially.
# Options:
# 1) Skip AP in your first version and rely on AP content via aggregators (GNews/NewsAPI/etc.).
# 2) Use Google News RSS filtered for site:apnews.com if that fits your risk appetite and ToS.
# 3) Use a third-party generator that turns AP pages into RSS (again, check ToS carefully).

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
    "NPR": {
        "feed_url": "https://feeds.npr.org/1001/rss.xml",
        "source": "NPR",
        "sub_source": "NPR",
        "source_region": "na",
        "source_country": "us",
        "topic_region": "world",
        "language": "en"
    }
}