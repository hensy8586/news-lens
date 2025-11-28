import requests
import streamlit as st

API_BASE = "http://localhost:8000"  # or your deployed FastAPI URL

st.title("US ↔ China News Summaries")

limit = st.slider("How many articles?", min_value=5, max_value=50, value=20)

resp = requests.get(f"{API_BASE}/news/latest", params={"limit": limit})
data = resp.json()

for item in data:
    st.subheader(item["title"])
    st.write(f"Source: {item['source']}")
    st.write(item.get("summary_raw", ""))
    st.write(f"[Original link]({item['link']})")
    st.write("---")
