import os
import requests
import streamlit as st

# ---------- Config: API base URL ----------

# Priority:
# 1. st.secrets["API_BASE_URL"]  (Streamlit Cloud / local secrets.toml)
# 2. env var API_BASE_URL       (if you prefer envs locally)
# 3. fallback to localhost:8000 (dev default)
DEFAULT_API_BASE = "http://localhost:8000"
API_BASE = st.secrets.get(
    "API_BASE_URL",
    os.getenv("API_BASE_URL", DEFAULT_API_BASE),
)

# ---------- Streamlit page setup ----------

st.set_page_config(
    page_title="News Lens",
    layout="wide",
)

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
    resp.raise_for_status()
    return resp.json()


# ---------- Main content ----------

try:
    with st.spinner("Loading latest news from API..."):
        news_items = fetch_latest_news(limit)
except requests.exceptions.RequestException as e:
    st.error("Could not reach the FastAPI backend.")
    st.code(repr(e))
    st.stop()

if not news_items:
    st.info("No news items returned from the API yet.")
    st.stop()

for item in news_items:
    title = item.get("title") or "(no title)"
    source = item.get("source") or "(unknown source)"
    link = item.get("link")
    published_at = item.get("published_at")
    summary_raw = item.get("summary_raw")

    st.subheader(title)

    meta_parts = []
    meta_parts.append(f"Source: `{source}`")
    if published_at:
        meta_parts.append(f"Published: `{published_at}`")

    st.write(" · ".join(meta_parts))

    if show_raw and summary_raw:
        st.write(summary_raw)

    if link:
        st.markdown(f"[Original article]({link})")

    st.markdown("---")

