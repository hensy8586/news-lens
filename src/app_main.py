from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .db import get_supabase_client

app = FastAPI(
    title="News API",
    version="0.1.0",
)

# Allow your Streamlit origin; for testing you can be loose and tighten later
origins = [
    "http://localhost:8501",      # local Streamlit
    "https://your-streamlit-url.streamlit.app",  # deployed Streamlit
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/news/latest")
def get_latest_news(limit: int = 20):
    client = get_supabase_client()
    resp = (
        client.table("news_items")
        .select("*")
        .order("published_at", desc=True)
        .limit(limit)
        .execute()
    )
    return resp.data
