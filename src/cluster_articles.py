import os
import time
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Iterable

from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
from supabase import create_client, Client as SupabaseClient

# ----------------------------
# Config
# ----------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

CLUSTER_BATCH_LIMIT = int(os.getenv("CLUSTER_BATCH_LIMIT", "100"))
SLEEP_BETWEEN_ARTICLES_SEC = float(os.getenv("SLEEP_BETWEEN_ARTICLES_SEC", "0.1"))
MAX_CHARS_FOR_EMBED = int(os.getenv("MAX_CHARS_FOR_EMBED", "4000"))

CLUSTER_LOOKBACK_HOURS = int(os.getenv("CLUSTER_LOOKBACK_HOURS", os.getenv("AI_LOOKBACK_HOURS", "72")))

STORYLINE_SIM_THRESHOLD = float(os.getenv("STORYLINE_SIM_THRESHOLD", "0.55"))
STORYLINE_MATCH_COUNT = int(os.getenv("STORYLINE_MATCH_COUNT", "15"))

EVENT_SIM_THRESHOLD = float(os.getenv("EVENT_SIM_THRESHOLD", "0.80"))
EVENT_MATCH_COUNT = int(os.getenv("EVENT_MATCH_COUNT", "15"))

# Prompt version for storyline/event creation (only used when no match)
STORYLINE_VER = os.getenv("STORYLINE_VER", "v1")
EVENT_VER = os.getenv("EVENT_VER", "v1")

# ----------------------------
# Clients
# ----------------------------
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def get_supabase_client() -> SupabaseClient:
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

# ----------------------------
# Helpers
# ----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def clip_text(s: str, max_chars: int) -> str:
    return (s or "").strip()[:max_chars]

def article_embed_text(title: str, content_text: str) -> str:
    body = clip_text(content_text, MAX_CHARS_FOR_EMBED)
    return f"{(title or '').strip()}\n\n{body}".strip()

def safe_iso_from_article(article: Dict[str, Any]) -> str:
    p = (article.get("published_at") or "").strip()
    return p or utc_now_iso()

# simple normalization helpers (optional)
STOPWORDS = {"and","or","the","a","an","of","to","in","for","with","on","at","by","from","during","after","before","over","under","between","into","amid"}

def normalize_storyline_title(title: str, max_words: int = 3) -> str:
    if not title:
        return ""
    words = [re.sub(r"^[\W_]+|[\W_]+$", "", w) for w in title.split()]
    words = [w for w in words if w and w.lower() not in STOPWORDS]
    return " ".join(words[:max_words]).strip()

def normalize_event_title(title: str, max_words: int = 5) -> str:
    if not title:
        return ""
    words = [w.strip(".,:;!?()[]{}\"'") for w in title.split()]
    words = [w for w in words if w and w.lower() not in STOPWORDS]
    return " ".join(words[:max_words]).strip()

# ----------------------------
# Embedding call
# ----------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=20))
def embed_text(oa: OpenAI, text: str) -> List[float]:
    resp = oa.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

# ----------------------------
# Supabase ops
# ----------------------------
def fetch_articles_to_cluster(sb: SupabaseClient, limit: int) -> List[Dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=CLUSTER_LOOKBACK_HOURS)
    cutoff_iso = cutoff.isoformat()

    res = (
        sb.table("articles")
        .select("id,title,link,content_text,published_at,ai_eligible,archived,clustered_at,ai_processing_started_at,primary_storyline_id,primary_event_id")
        .eq("ai_eligible", True)
        .eq("archived", False)
        .is_("clustered_at", "null")
        .gte("published_at", cutoff_iso)
        .order("published_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []

def claim_article(sb: SupabaseClient, article_id: str) -> bool:
    """
    Claim a row so two runs can't both process it.
    Only claim if ai_processing_started_at is null and clustered_at is null.
    """
    res = (
        sb.table("articles")
        .update({"ai_processing_started_at": utc_now_iso(), "ai_processing_error": None})
        .eq("id", article_id)
        .is_("ai_processing_started_at", "null")
        .is_("clustered_at", "null")
        .execute()
    )
    return bool(res.data)

def mark_clustered(
    sb: SupabaseClient,
    *,
    article_id: str,
    primary_storyline_id: Optional[str],
    primary_event_id: Optional[str],
) -> None:
    patch: Dict[str, Any] = {
        "clustered_at": utc_now_iso(),
        "primary_storyline_id": primary_storyline_id,
        "primary_event_id": primary_event_id,
    }
    sb.table("articles").update(patch).eq("id", article_id).execute()

def set_processing_error(sb: SupabaseClient, article_id: str, err: str) -> None:
    sb.table("articles").update({"ai_processing_error": clip_text(err, 500)}).eq("id", article_id).execute()

def upsert_article_embedding(sb: SupabaseClient, article_id: str, embedding: List[float]) -> None:
    sb.table("article_embeddings").upsert(
        {"article_id": article_id, "embedding": embedding, "model_name": EMBED_MODEL}
    ).execute()

def rpc_match_storylines(sb: SupabaseClient, query_embedding: List[float], threshold: float, match_count: int) -> List[Dict[str, Any]]:
    res = sb.rpc(
        "match_storylines",
        {"query_embedding": query_embedding, "match_threshold": threshold, "match_count": match_count},
    ).execute()
    return res.data or []

def rpc_match_events_in_storyline(
    sb: SupabaseClient,
    storyline_id: str,
    query_embedding: List[float],
    threshold: float,
    match_count: int,
) -> List[Dict[str, Any]]:
    res = sb.rpc(
        "match_events_in_storyline",
        {"storyline_id": storyline_id, "query_embedding": query_embedding, "match_threshold": threshold, "match_count": match_count},
    ).execute()
    return res.data or []

def create_storyline(
    sb: SupabaseClient,
    *,
    title: str,
    description: str,
    started_at: str,
    last_updated_at: str,
    model_name: str,
    prompt_version: str,
    prompt_template_id: Optional[str] = None,
    main_category_id: Optional[str] = None,
) -> str:
    payload = {
        "title": title,
        "description": description,
        "main_category_id": main_category_id,
        "started_at": started_at,
        "last_updated_at": last_updated_at,
        "model_name": model_name,
        "prompt_version": prompt_version,
        "prompt_template_id": prompt_template_id,
    }
    res = sb.table("storylines").insert(payload).execute()
    return res.data[0]["id"]

def update_storyline_last_updated(sb: SupabaseClient, storyline_id: str, ts: str) -> None:
    sb.table("storylines").update({"last_updated_at": ts}).eq("id", storyline_id).execute()

def create_event(
    sb: SupabaseClient,
    *,
    title: str,
    description: str,
    started_at: str,
    last_updated_at: str,
    model_name: str,
    prompt_version: str,
    prompt_template_id: Optional[str] = None,
    main_category_id: Optional[str] = None,
    storyline_id: Optional[str] = None,
) -> str:
    payload = {
        "title": title,
        "description": description,
        "main_category_id": main_category_id,
        "started_at": started_at,
        "last_updated_at": last_updated_at,
        "model_name": model_name,
        "prompt_version": prompt_version,
        "prompt_template_id": prompt_template_id,
        "storyline_id": storyline_id,
    }
    res = sb.table("events").insert(payload).execute()
    return res.data[0]["id"]

def link_article_event(sb: SupabaseClient, *, article_id: str, event_id: str, similarity: float) -> None:
    sb.table("article_events").upsert(
        {"article_id": article_id, "event_id": event_id, "similarity": float(similarity), "is_primary": True}
    ).execute()

def _avg_update(old: List[float], old_n: int, new: List[float]) -> Tuple[List[float], int]:
    n2 = old_n + 1
    updated = [(old[i] * old_n + new[i]) / n2 for i in range(len(new))]
    return updated, n2

def upsert_storyline_centroid(sb: SupabaseClient, storyline_id: str, embedding: List[float]) -> None:
    row = sb.table("storyline_embeddings").select("embedding,embedding_count").eq("storyline_id", storyline_id).execute().data
    if row:
        old = row[0]["embedding"]
        old_n = int(row[0]["embedding_count"])
        new_emb, new_n = _avg_update(old, old_n, embedding)
        sb.table("storyline_embeddings").update(
            {"embedding": new_emb, "embedding_count": new_n, "model_name": EMBED_MODEL, "updated_at": utc_now_iso()}
        ).eq("storyline_id", storyline_id).execute()
    else:
        sb.table("storyline_embeddings").insert(
            {"storyline_id": storyline_id, "embedding": embedding, "embedding_count": 1, "model_name": EMBED_MODEL, "updated_at": utc_now_iso()}
        ).execute()

def upsert_event_centroid(sb: SupabaseClient, event_id: str, embedding: List[float]) -> None:
    row = sb.table("event_embeddings").select("embedding,embedding_count").eq("event_id", event_id).execute().data
    if row:
        old = row[0]["embedding"]
        old_n = int(row[0]["embedding_count"])
        new_emb, new_n = _avg_update(old, old_n, embedding)
        sb.table("event_embeddings").update(
            {"embedding": new_emb, "embedding_count": new_n, "model_name": EMBED_MODEL, "updated_at": utc_now_iso()}
        ).eq("event_id", event_id).execute()
    else:
        sb.table("event_embeddings").insert(
            {"event_id": event_id, "embedding": embedding, "embedding_count": 1, "model_name": EMBED_MODEL, "updated_at": utc_now_iso()}
        ).execute()

# ----------------------------
# Minimal storyline/event creation without LLM (Phase A)
# ----------------------------
def fallback_storyline_from_article(title: str) -> Tuple[str, str]:
    # Keep it broad; you can replace with an LLM later if you want.
    t = normalize_storyline_title(title, max_words=3) or clip_text(title, 60)
    desc = f"Auto-created storyline from article title: {clip_text(title, 140)}"
    return t, desc

def fallback_event_from_article(title: str) -> Tuple[str, str]:
    t = normalize_event_title(title, max_words=5) or clip_text(title, 80)
    desc = f"Auto-created event from article title: {clip_text(title, 160)}"
    return t, desc

# ----------------------------
# Core Phase A logic
# ----------------------------
def cluster_one_article(sb: SupabaseClient, oa: OpenAI, article: Dict[str, Any]) -> None:
    article_id = article["id"]
    title = article.get("title") or ""
    content_text = article.get("content_text") or ""

    if not (content_text or "").strip():
        # Still mark clustered so it won't be retried forever
        mark_clustered(sb, article_id=article_id, primary_storyline_id=None, primary_event_id=None)
        set_processing_error(sb, article_id, "empty_content_text")
        return

    emb_input = article_embed_text(title, content_text)
    embedding = embed_text(oa, emb_input)
    upsert_article_embedding(sb, article_id, embedding)

    # --- Storyline match or create ---
    matches_story = rpc_match_storylines(sb, embedding, threshold=STORYLINE_SIM_THRESHOLD, match_count=STORYLINE_MATCH_COUNT)
    chosen_storyline_id: Optional[str] = None
    if matches_story:
        chosen_storyline_id = matches_story[0].get("storyline_id") or matches_story[0].get("id")

    if not chosen_storyline_id:
        sl_title, sl_desc = fallback_storyline_from_article(title)
        chosen_storyline_id = create_storyline(
            sb,
            title=sl_title,
            description=sl_desc,
            started_at=safe_iso_from_article(article),
            last_updated_at=utc_now_iso(),
            model_name=EMBED_MODEL,
            prompt_version=STORYLINE_VER,
            prompt_template_id=None,
            main_category_id=None,
        )
        upsert_storyline_centroid(sb, chosen_storyline_id, embedding)
    else:
        upsert_storyline_centroid(sb, chosen_storyline_id, embedding)
        update_storyline_last_updated(sb, chosen_storyline_id, utc_now_iso())

    # --- Event match or create (within storyline) ---
    matches_evt = rpc_match_events_in_storyline(
        sb,
        chosen_storyline_id,
        embedding,
        threshold=EVENT_SIM_THRESHOLD,
        match_count=EVENT_MATCH_COUNT,
    )

    chosen_event_id: Optional[str] = None
    chosen_similarity: float = 1.0

    if matches_evt:
        chosen_event_id = matches_evt[0].get("event_id") or matches_evt[0].get("id")
        chosen_similarity = float(matches_evt[0].get("similarity", 1.0))
        link_article_event(sb, article_id=article_id, event_id=chosen_event_id, similarity=chosen_similarity)
        upsert_event_centroid(sb, chosen_event_id, embedding)
    else:
        evt_title, evt_desc = fallback_event_from_article(title)
        chosen_event_id = create_event(
            sb,
            title=evt_title,
            description=evt_desc,
            started_at=safe_iso_from_article(article),
            last_updated_at=utc_now_iso(),
            model_name=EMBED_MODEL,
            prompt_version=EVENT_VER,
            prompt_template_id=None,
            main_category_id=None,
            storyline_id=chosen_storyline_id,
        )
        upsert_event_centroid(sb, chosen_event_id, embedding)
        link_article_event(sb, article_id=article_id, event_id=chosen_event_id, similarity=1.0)

    # Mark clustered LAST
    mark_clustered(sb, article_id=article_id, primary_storyline_id=chosen_storyline_id, primary_event_id=chosen_event_id)

def main():
    sb = get_supabase_client()
    oa = get_openai_client()

    articles = fetch_articles_to_cluster(sb, limit=CLUSTER_BATCH_LIMIT)
    if not articles:
        print("No articles to cluster.")
        return

    print(f"Clustering {len(articles)} article(s)...")
    for i, a in enumerate(articles, start=1):
        article_id = a.get("id")
        try:
            if not article_id:
                continue

            if not claim_article(sb, article_id):
                print(f"SKIP (already claimed/clustered): {article_id}")
                continue

            print(f"[{i}/{len(articles)}] {str(a.get('title','(no title)'))[:80]}")
            cluster_one_article(sb, oa, a)
            time.sleep(SLEEP_BETWEEN_ARTICLES_SEC)

        except Exception as e:
            set_processing_error(sb, article_id, str(e))
            print(f"ERROR: article_id={article_id} err={e}")

    print("Done.")

if __name__ == "__main__":
    main()
