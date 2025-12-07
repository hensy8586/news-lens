import os
import requests
import streamlit as st
from streamlit.runtime.secrets import StreamlitSecretNotFoundError
import html
import urllib.parse

from app_css import card_css
from app_utils import clean_summary

# ---------- Config: API base URL ----------

# Priority:
# 1. st.secrets["API_BASE_URL"]  (Streamlit Cloud / local secrets.toml)
# 2. env var API_BASE_URL       (if you prefer envs locally)
# 3. fallback to localhost:8000 (dev default)
DEFAULT_API_BASE = "http://127.0.0.1:8080"
try:
    # Try Streamlit secrets first
    API_BASE = st.secrets["API_BASE_URL"]
except StreamlitSecretNotFoundError:
    # Fall back to env var, then to default
    API_BASE = os.getenv("API_BASE_URL", DEFAULT_API_BASE)

# ---------- Streamlit page setup ----------

st.set_page_config(
    page_title="News Lens",
    layout="wide",
)
card_css()

st.title("News Lens (your personal news feed)")
st.caption(f"Using API base: `{API_BASE}`")

# Controls
with st.sidebar:
    st.header("Options")
    limit = st.slider("Number of articles", min_value=5, max_value=50, value=20, step=5)
    show_raw = st.checkbox("Show raw summary text", value=True)


# ---------- Helper: call FastAPI ----------

@st.cache_data(show_spinner=False)
def fetch_latest_news(limit: int):
    url = f"{API_BASE}/news/latest"
    resp = requests.get(url, params={"limit": limit}, timeout=15)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # Attach backend’s response body to the error so Streamlit can display it
        backend_body = resp.text
        raise RuntimeError(
            f"HTTP {resp.status_code} from backend\nURL: {resp.url}\n\nBody:\n{backend_body}"
        ) from e
    return resp.json()


# ---------- Main content ----------

try:
    with st.spinner("Loading latest news from API..."):
        news_items = fetch_latest_news(limit)
except Exception as e:
    st.error("Could not load data from the FastAPI backend.")
    st.code(str(e))
    st.stop()

if not news_items:
    st.info("No news items returned from the API yet.")
    st.stop()

for item in news_items:
    title = item.get("title") or "(no title)"
    source = item.get("source_outlet") or "(unknown source)"
    link = item.get("link")
    published_at = item.get("published_at")
    summary_raw = item.get("summary")
    content_text = item.get("content_text") or ""
    image_url = item.get("image_url")

    # Build a clean summary (your existing logic)
    if summary_raw:
        summary = clean_summary(summary_raw)
    elif content_text:
        summary = (content_text[:400] + "...") if len(content_text) > 400 else content_text
    else:
        summary = ""

    # Escape for HTML
    title_html = html.escape(title)
    source_html = html.escape(source)
    published_html = html.escape(published_at) if published_at else ""
    summary_html = html.escape(summary)
    link_html = html.escape(link, quote=True) if link else ""

    meta_parts = [f"Source: {source_html}"]
    if published_html:
        meta_parts.append(f"Published: {published_html}")
    meta_html = " · ".join(meta_parts)

    # ---------- Decide what URL the card should open ----------
    # Prefer original article link; if missing, fall back to data: URL with full content_text
    target_url = None

    if link_html:
        target_url = link_html
    elif content_text:
        content_html = f"""
        <html>
          <head>
            <meta charset='utf-8'>
            <title>{title_html}</title>
          </head>
          <body style="font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; line-height: 1.5; padding: 1.5rem; max-width: 800px; margin: 0 auto;">
            <h1>{title_html}</h1>
            <p style="color: #555;">{meta_html}</p>
            <pre style="white-space: pre-wrap; font-family: inherit; font-size: 1rem;">
{html.escape(content_text)}
            </pre>
          </body>
        </html>
        """
        data_url = "data:text/html;charset=utf-8," + urllib.parse.quote(content_html)
        target_url = data_url

    # ---------- Card HTML ----------
    # Build the inner card structure once
    inner_card_html = f"""
<div class="news-card">
    <div class="news-card-inner">
    <div class="news-card-image">
        {'<img src="' + image_url + '" alt="thumbnail">' if image_url else ''}
    </div>
    <div class="news-card-content">
        <div class="news-card-title">{title_html}</div>
        <div class="news-card-meta">{meta_html}</div>
        <div class="news-card-summary">{summary_html}</div>
    </div>
    </div>
</div>
"""

    # If we have a target URL, wrap the card in an <a> so the whole card is clickable
    if target_url:
        card_html = f"""
<a href="{target_url}" target="_blank" class="news-card-link-wrapper">
    {inner_card_html}
</a>
"""
    else:
        # No URL available – just render a non-clickable card
        card_html = inner_card_html

    st.markdown(card_html, unsafe_allow_html=True)
