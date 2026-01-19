import os
import json
import time
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
from supabase import create_client, Client as SupabaseClient

from .prompt_store import PromptTemplateRow, get_or_seed_prompt_template


# ----------------------------
# Config
# ----------------------------
EMBED_MODEL = "text-embedding-3-small"

BATCH_LIMIT = int(os.getenv("BATCH_LIMIT", "25"))
SLEEP_BETWEEN_ARTICLES_SEC = float(os.getenv("SLEEP_BETWEEN_ARTICLES_SEC", "0.2"))

MAX_CHARS_FOR_LLM = int(os.getenv("MAX_CHARS_FOR_LLM", "12000"))

STORYLINE_SIM_THRESHOLD = float(os.getenv("STORYLINE_SIM_THRESHOLD", "0.55"))
STORYLINE_MATCH_COUNT = int(os.getenv("STORYLINE_MATCH_COUNT", "15"))

EVENT_SIM_THRESHOLD = float(os.getenv("EVENT_SIM_THRESHOLD", "0.80"))  # within-storyline matching
EVENT_MATCH_COUNT = int(os.getenv("EVENT_MATCH_COUNT", "15"))

# Prompt versions (env-driven)
SUMMARY_VERSION = os.getenv("SUMMARY_VER", "v1")
CATS_VERSION = os.getenv("CATS_VER", "v1")
EVENT_VERSION = os.getenv("EVENT_VER", "v1")
TITLE_VERSION = os.getenv("TITLE_VER", "v1")
TRANSLATE_VERSION = os.getenv("TRANSLATE_VER", "v1")
STORYLINE_VERSION = os.getenv("STORYLINE_VER", "v1")

# Prompt tasks
TASK_SUMMARY = "summary"
TASK_CATEGORIES = "category"
TASK_EVENT = "event"
TASK_TITLE = "title"
TASK_TRANSLATE = "translate"
TASK_STORYLINE = "storyline"

# Title storage mode
TITLE_STORAGE_MODE = os.getenv("TITLE_STORAGE_MODE", "article_titles")

# DeepSeek model name
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


# ----------------------------
# Clients
# ----------------------------
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def get_deepseek_client() -> OpenAI:
    """
    DeepSeek is OpenAI-API compatible. We use the OpenAI SDK with base_url.
    """
    return OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )


def get_supabase_client() -> SupabaseClient:
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


# ----------------------------
# Helpers
# ----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clip_text(s: str, max_chars: int) -> str:
    return (s or "").strip()[:max_chars]


def normalize_bullets(md: str) -> str:
    """
    Normalize to markdown bullet list lines starting with "- ".
    Keeps non-bullet text if model returns a paragraph (better than dropping it).
    """
    if not md:
        return md
    lines = [ln.rstrip() for ln in md.strip().splitlines() if ln.strip()]
    out: List[str] = []
    for ln in lines:
        s = ln.strip()
        if s.startswith(("-", "•")):
            out.append("- " + s.lstrip("-•").strip())
        elif len(s) >= 3 and s[0].isdigit() and s[1] in (".", "。"):
            out.append("- " + s[2:].strip())
        else:
            out.append(s)
    return "\n".join(out).strip()


def looks_truncated_zh(text: str) -> bool:
    """
    Heuristic for detecting cut-off output in Chinese bullets.
    """
    t = (text or "").strip()
    if not t:
        return True
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return True
    last = lines[-1]
    if last.startswith("-"):
        if last[-1] not in "。！？…）】」”":
            if len(last) > 12:
                return True
    if t.endswith(("建议", "指出", "认为", "包括", "例如", "以及", "等", "专家建议")):
        return True
    return False


class SafeDict(dict):
    def __missing__(self, key):
        return ""


def render_template(template: str, variables: Dict[str, Any]) -> str:
    """
    Uses Python .format(). Any literal JSON braces in templates MUST be escaped:
      { -> {{   and   } -> }}
    """
    return (template or "").format_map(SafeDict(**variables))


def build_messages(pt: PromptTemplateRow, variables: Dict[str, Any]) -> List[Dict[str, str]]:
    user_part = render_template(pt.user_prompt_template, variables)
    msgs: List[Dict[str, str]] = []
    if pt.system_prompt and pt.system_prompt.strip():
        sys_part = render_template(pt.system_prompt, variables)
        msgs.append({"role": "system", "content": sys_part})
    msgs.append({"role": "user", "content": user_part})
    return msgs


def build_single_prompt(pt: PromptTemplateRow, variables: Dict[str, Any]) -> str:
    user_part = render_template(pt.user_prompt_template, variables)
    if pt.system_prompt and pt.system_prompt.strip():
        sys_part = render_template(pt.system_prompt, variables)
        return f"{sys_part}\n\n{user_part}".strip()
    return user_part.strip()


def language_instruction(lang: str) -> str:
    return "Write in English." if lang == "en" else "用简体中文撰写。"


def article_embed_text(title: str, content_text: str) -> str:
    body = clip_text(content_text, 4000)
    return f"{(title or '').strip()}\n\n{body}".strip()


def safe_iso_from_article(article: Dict[str, Any], fallback: Optional[str] = None) -> str:
    """
    Prefer article.published_at if present; else fallback; else utc_now_iso.
    Assumes published_at is already ISO-ish from DB.
    """
    p = (article.get("published_at") or "").strip()
    if p:
        return p
    if fallback:
        return fallback
    return utc_now_iso()


STOPWORDS = {
    "and","or","the","a","an","of","to","in","for","with","on","at","by","from",
    "during","after","before","over","under","between","into","amid"
}


def normalize_event_title(title: str, max_words: int = 4) -> str:
    """
    Normalize event titles to be BROAD, reusable buckets.

    Rules:
    - Remove stopwords
    - Prefer location-based or situation-based phrasing
    - Cap to max_words
    """
    if not title:
        return title

    words = [
        w.strip(".,:;!?()[]{}\"'")
        for w in title.split()
    ]
    words = [w for w in words if w and w.lower() not in STOPWORDS]

    if not words:
        return ""

    # If ICE appears, keep it
    if "ICE" in words:
        idx = words.index("ICE")
        words = words[idx:]  # drop leading fluff

    return " ".join(words[:max_words])


def normalize_storyline_title(title: str, max_words: int = 3) -> str:
    """
    Turn an LLM storyline label into a short reusable tag.

    - Remove stopwords (so you never end with 'and')
    - Remove punctuation
    - Cap to max_words
    """
    if not title:
        return ""

    # split on whitespace, strip punctuation
    words = [re.sub(r"^[\W_]+|[\W_]+$", "", w) for w in title.split()]
    words = [w for w in words if w]  # drop empties

    # remove stopwords
    words = [w for w in words if w.lower() not in STOPWORDS]

    if not words:
        return ""

    return " ".join(words[:max_words])


def claim_article_for_processing(sb: SupabaseClient, article_id: str) -> bool:
    """
    Best-effort lock: only claim if not already started and not already processed.
    Returns True if claimed by this run.
    """
    res = (
        sb.table("articles")
        .update({"ai_processing_started_at": utc_now_iso(), "ai_processing_error": None})
        .eq("id", article_id)
        .is_("ai_processing_started_at", "null")
        .is_("ai_processed_at", "null")
        .execute()
    )
    return bool(res.data)  # if update affected a row, you claimed it


# ----------------------------
# OpenAI + DeepSeek calls
# ----------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=20))
def run_openai_text(oa: OpenAI, pt: PromptTemplateRow, variables: Dict[str, Any]) -> str:
    prompt = build_single_prompt(pt, variables)
    resp = oa.responses.create(
        model=pt.model_name,
        input=prompt,
        temperature=float(pt.temperature or 0.0),
        max_output_tokens=pt.max_output_tokens,
    )
    return (resp.output_text or "").strip()


def _extract_json(txt: str) -> str:
    t = (txt or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"\s*```$", "", t).strip()
    if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
        return t
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        return m.group(0).strip()
    m = re.search(r"\[.*\]", t, flags=re.DOTALL)
    if m:
        return m.group(0).strip()
    return ""

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=20))
def run_openai_json(oa: OpenAI, pt: PromptTemplateRow, variables: Dict[str, Any]) -> Dict[str, Any]:
    txt = run_openai_text(oa, pt, variables)
    if not (txt or "").strip():
        raise ValueError("Empty output (expected JSON).")
    candidate = _extract_json(txt)
    if not candidate:
        raise ValueError(f"No JSON found in output: {txt[:300]}")
    return json.loads(candidate)



@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=20))
def run_deepseek_text(ds: OpenAI, pt: PromptTemplateRow, variables: Dict[str, Any]) -> str:
    messages = build_messages(pt, variables)
    resp = ds.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=float(pt.temperature or 0.0),
        max_tokens=int(pt.max_output_tokens or 400),
    )
    return (resp.choices[0].message.content or "").strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=20))
def embed_text(oa: OpenAI, text: str) -> List[float]:
    resp = oa.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


# ----------------------------
# Supabase ops
# ----------------------------
def fetch_categories(sb: SupabaseClient) -> List[Dict[str, Any]]:
    res = sb.table("categories").select("id,slug,display_name").execute()
    return res.data or []


def categories_as_lines(categories: List[Dict[str, Any]]) -> str:
    return "\n".join([f"- {c['slug']}: {c.get('display_name','')}" for c in categories])


def fetch_articles_to_process(sb: SupabaseClient, limit: int) -> List[Dict[str, Any]]:
    # Optional: keep processing bounded to recent items
    # e.g. only last 72 hours
    hours = int(os.getenv("AI_LOOKBACK_HOURS", "72"))
    # Supabase Python client can't do SQL intervals directly; pass an ISO timestamp
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    cutoff_iso = cutoff.isoformat()

    res = (
        sb.table("articles")
        .select(
            "id,title,link,content_text,published_at,primary_event_id,primary_category_id,ai_processed_at,primary_storyline_id,ai_eligible,archived"
        )
        .is_("ai_processed_at", "null")
        .eq("ai_eligible", True)
        .eq("archived", False)
        .gte("published_at", cutoff_iso)   # optional but strongly recommended
        .order("published_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []



def upsert_article_embedding(sb: SupabaseClient, article_id: str, embedding: List[float]) -> None:
    sb.table("article_embeddings").upsert(
        {"article_id": article_id, "embedding": embedding, "model_name": EMBED_MODEL}
    ).execute()


def upsert_article_summary(
    sb: SupabaseClient,
    *,
    article_id: str,
    language: str,
    summary_short_md: str,
    model_name: str,
    prompt_version: str,
    prompt_template_id: str,
) -> None:
    sb.table("article_summaries").upsert(
        {
            "article_id": article_id,
            "language": language,
            "summary_short_md": summary_short_md,
            "summary_long_md": None,
            "model_name": model_name,
            "prompt_version": prompt_version,
            "prompt_template_id": prompt_template_id,
        }
    ).execute()


def upsert_article_category(
    sb: SupabaseClient,
    *,
    article_id: str,
    category_id: str,
    confidence: float,
    is_primary: bool,
    prompt_template_id: Optional[str] = None,
) -> None:
    payload = {
        "article_id": article_id,
        "category_id": category_id,
        "confidence": float(confidence),
        "is_primary": bool(is_primary),
    }
    if prompt_template_id is not None:
        payload["prompt_template_id"] = prompt_template_id
    sb.table("article_categories").upsert(payload).execute()


def upsert_article_title_zh(
    sb: SupabaseClient,
    *,
    article_id: str,
    title_zh: str,
    pt_title: PromptTemplateRow,
) -> None:
    t = (title_zh or "").strip()
    if not t:
        return

    if TITLE_STORAGE_MODE == "article_titles":
        sb.table("article_titles").upsert(
            {
                "article_id": article_id,
                "language": "zh",
                "title": t,
                "model_name": pt_title.model_name,
                "prompt_version": pt_title.version,
                "prompt_template_id": pt_title.id,
            }
        ).execute()
    elif TITLE_STORAGE_MODE == "articles_column":
        sb.table("articles").update({"title_zh": t}).eq("id", article_id).execute()
    else:
        raise ValueError(f"Unknown TITLE_STORAGE_MODE={TITLE_STORAGE_MODE!r}")


def update_article_primary_fields(
    sb: SupabaseClient,
    *,
    article_id: str,
    primary_category_id: Optional[str] = None,
    primary_event_id: Optional[str] = None,
    primary_storyline_id: Optional[str] = None,
    ai_processed_at: Optional[str] = None,
) -> None:
    patch: Dict[str, Any] = {}
    if primary_category_id is not None:
        patch["primary_category_id"] = primary_category_id
    if primary_event_id is not None:
        patch["primary_event_id"] = primary_event_id
    if primary_storyline_id is not None:
        patch["primary_storyline_id"] = primary_storyline_id
    if ai_processed_at is not None:
        patch["ai_processed_at"] = ai_processed_at
    if patch:
        sb.table("articles").update(patch).eq("id", article_id).execute()


def create_event(
    sb: SupabaseClient,
    *,
    title: str,
    description: str,
    main_category_id: Optional[str],
    started_at: Optional[str],
    last_updated_at: Optional[str],
    model_name: str,
    prompt_version: str,
    prompt_template_id: str,
) -> str:
    res = sb.table("events").insert(
        {
            "title": title,
            "description": description,
            "main_category_id": main_category_id,
            "started_at": started_at,
            "last_updated_at": last_updated_at,
            "model_name": model_name,
            "prompt_version": prompt_version,
            "prompt_template_id": prompt_template_id,
        }
    ).execute()
    return res.data[0]["id"]


def link_article_event(sb: SupabaseClient, *, article_id: str, event_id: str, similarity: float) -> None:
    sb.table("article_events").upsert(
        {"article_id": article_id, "event_id": event_id, "similarity": float(similarity), "is_primary": True}
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
        {
            "storyline_id": storyline_id,
            "query_embedding": query_embedding,
            "match_threshold": threshold,
            "match_count": match_count,
        },
    ).execute()
    return res.data or []


def create_storyline(
    sb: SupabaseClient,
    *,
    title: str,
    description: str,
    main_category_id: Optional[str],
    started_at: Optional[str],
    last_updated_at: Optional[str],
    model_name: str,
    prompt_version: str,
    prompt_template_id: str,
) -> str:
    res = sb.table("storylines").insert(
        {
            "title": title,
            "description": description,
            "main_category_id": main_category_id,
            "started_at": started_at,
            "last_updated_at": last_updated_at,
            "model_name": model_name,
            "prompt_version": prompt_version,
            "prompt_template_id": prompt_template_id,
        }
    ).execute()
    return res.data[0]["id"]


def update_storyline_last_updated(sb: SupabaseClient, storyline_id: str, ts: str) -> None:
    sb.table("storylines").update({"last_updated_at": ts}).eq("id", storyline_id).execute()


def update_event_storyline(sb: SupabaseClient, event_id: str, storyline_id: str) -> None:
    sb.table("events").update({"storyline_id": storyline_id}).eq("id", event_id).execute()


def _avg_update(old: List[float], old_n: int, new: List[float]) -> Tuple[List[float], int]:
    n2 = old_n + 1
    updated = [(old[i] * old_n + new[i]) / n2 for i in range(len(new))]
    return updated, n2


def upsert_storyline_centroid(sb: SupabaseClient, storyline_id: str, embedding: List[float], model_name: str) -> None:
    row = (
        sb.table("storyline_embeddings")
        .select("embedding,embedding_count")
        .eq("storyline_id", storyline_id)
        .execute()
        .data
    )
    if row:
        old = row[0]["embedding"]
        old_n = int(row[0]["embedding_count"])
        new_emb, new_n = _avg_update(old, old_n, embedding)
        sb.table("storyline_embeddings").update(
            {"embedding": new_emb, "embedding_count": new_n, "model_name": model_name, "updated_at": utc_now_iso()}
        ).eq("storyline_id", storyline_id).execute()
    else:
        sb.table("storyline_embeddings").insert(
            {"storyline_id": storyline_id, "embedding": embedding, "embedding_count": 1, "model_name": model_name, "updated_at": utc_now_iso()}
        ).execute()


def upsert_event_centroid(sb: SupabaseClient, event_id: str, embedding: List[float], model_name: str) -> None:
    row = sb.table("event_embeddings").select("embedding,embedding_count").eq("event_id", event_id).execute().data
    if row:
        old = row[0]["embedding"]
        old_n = int(row[0]["embedding_count"])
        new_emb, new_n = _avg_update(old, old_n, embedding)
        sb.table("event_embeddings").update(
            {"embedding": new_emb, "embedding_count": new_n, "model_name": model_name, "updated_at": utc_now_iso()}
        ).eq("event_id", event_id).execute()
    else:
        sb.table("event_embeddings").insert(
            {"event_id": event_id, "embedding": embedding, "embedding_count": 1, "model_name": model_name, "updated_at": utc_now_iso()}
        ).execute()


# ----------------------------
# Processing logic
# ----------------------------
def generate_summaries_openai_then_deepseek(
    oa: OpenAI,
    ds: OpenAI,
    *,
    pt_summary_en: PromptTemplateRow,
    pt_translate_zh: PromptTemplateRow,
    title: str,
    content_for_llm: str,
) -> Tuple[str, str]:
    summary_en = run_openai_text(
        oa,
        pt_summary_en,
        {"title": title, "content": content_for_llm, "language_instruction": language_instruction("en")},
    )
    summary_en = normalize_bullets(summary_en)

    summary_zh = run_deepseek_text(
        ds,
        pt_translate_zh,
        {"summary_en": summary_en, "title_en": title, "extra_rules": ""},
    )
    summary_zh = normalize_bullets(summary_zh)

    if looks_truncated_zh(summary_zh):
        summary_zh = run_deepseek_text(
            ds,
            pt_translate_zh,
            {
                "summary_en": summary_en,
                "title_en": title,
                "extra_rules": "上次输出可能被截断。请更精炼，但不得漏信息；保持项目符号数量一致。",
            },
        )
        summary_zh = normalize_bullets(summary_zh)

    return summary_en, summary_zh


def process_one_article(
    sb: SupabaseClient,
    oa: OpenAI,
    ds: OpenAI,
    *,
    article: Dict[str, Any],
    categories: List[Dict[str, Any]],
    pt_summary_en: PromptTemplateRow,
    pt_translate_zh: PromptTemplateRow,
    pt_cats: PromptTemplateRow,
    pt_event: PromptTemplateRow,
    pt_title: PromptTemplateRow,
    pt_storyline: PromptTemplateRow,
) -> None:
    article_id = article["id"]
    title = article.get("title") or ""
    content_text = article.get("content_text") or ""

    if not content_text.strip():
        sb.table("articles").update(
            {"ai_processed_at": utc_now_iso(), "ai_processing_error": "empty_content_text"}
        ).eq("id", article_id).execute()
        return


    content_for_llm = clip_text(content_text, MAX_CHARS_FOR_LLM)

    # 1) EN summary + ZH summary
    summary_en, summary_zh = generate_summaries_openai_then_deepseek(
        oa,
        ds,
        pt_summary_en=pt_summary_en,
        pt_translate_zh=pt_translate_zh,
        title=title,
        content_for_llm=content_for_llm,
    )

    # 2) ZH title
    title_zh = run_openai_text(oa, pt_title, {"title_en": title, "summary_en": summary_en}).strip()

    # 3) Categories
    cat_lines = categories_as_lines(categories)
    cls = run_openai_json(
        oa,
        pt_cats,
        {"title": title, "content": clip_text(content_text, 8000), "categories": cat_lines},
    )

    cat_map = {c["slug"]: c["id"] for c in categories}
    primary_slug = cls.get("primary_slug")
    secondary_slugs = cls.get("secondary_slugs") or []
    scores = cls.get("scores") or {}

    primary_cat_id: Optional[str] = None
    if primary_slug in cat_map:
        primary_cat_id = cat_map[primary_slug]
        upsert_article_category(
            sb,
            article_id=article_id,
            category_id=primary_cat_id,
            confidence=float(scores.get(primary_slug, 0.8)),
            is_primary=True,
            prompt_template_id=pt_cats.id,
        )

    for slug in secondary_slugs:
        if slug in cat_map and slug != primary_slug:
            upsert_article_category(
                sb,
                article_id=article_id,
                category_id=cat_map[slug],
                confidence=float(scores.get(slug, 0.6)),
                is_primary=False,
                prompt_template_id=pt_cats.id,
            )

    # 4) Embedding for matching
    emb_input = article_embed_text(title, content_text)
    embedding = embed_text(oa, emb_input)
    upsert_article_embedding(sb, article_id, embedding)

    # ----------------------------
    # STORYLINE: match or create
    # ----------------------------
    matches_story = rpc_match_storylines(
        sb,
        embedding,
        threshold=STORYLINE_SIM_THRESHOLD,
        match_count=STORYLINE_MATCH_COUNT,
    )

    chosen_storyline_id: Optional[str] = None
    if matches_story:
        # expected columns from RPC: storyline_id, similarity (or distance), etc.
        chosen_storyline_id = matches_story[0].get("storyline_id") or matches_story[0].get("id")

    if not chosen_storyline_id:
        # Ask LLM to define storyline title/description based on this article
        sl = run_openai_json(
            oa,
            pt_storyline,
            {"title": title, "summary_en": summary_en},
        )

        # Support both schema variants (your prompt likely returns storyline_title/storyline_description)
        sl_title_raw = (
            (sl.get("storyline_title") or sl.get("title") or "").strip()
            or clip_text(title, 140)
        )
        sl_desc = (
            (sl.get("storyline_description") or sl.get("description") or "").strip()
            or clip_text(summary_en, 800)
        )

        # ✅ Normalize so you never end up with "... and"
        sl_title = normalize_storyline_title(sl_title_raw, max_words=2)  # or 3 if you prefer

        # Fallback if normalization removes everything
        if not sl_title:
            sl_title = clip_text(sl_title_raw, 80)

        started_at = safe_iso_from_article(article)
        last_updated_at = utc_now_iso()

        chosen_storyline_id = create_storyline(
            sb,
            title=sl_title,
            description=sl_desc,
            main_category_id=primary_cat_id,
            started_at=started_at,
            last_updated_at=last_updated_at,
            model_name=pt_storyline.model_name,
            prompt_version=pt_storyline.version,
            prompt_template_id=pt_storyline.id,
        )
        upsert_storyline_centroid(sb, chosen_storyline_id, embedding, model_name=EMBED_MODEL)
    else:
        # Update centroid & last_updated
        upsert_storyline_centroid(sb, chosen_storyline_id, embedding, model_name=EMBED_MODEL)
        update_storyline_last_updated(sb, chosen_storyline_id, utc_now_iso())

    # Write storyline onto article
    update_article_primary_fields(sb, article_id=article_id, primary_storyline_id=chosen_storyline_id)

    # ----------------------------
    # EVENT: match or create (within storyline)
    # ----------------------------
    matches_evt = rpc_match_events_in_storyline(
        sb,
        chosen_storyline_id,
        embedding,
        threshold=EVENT_SIM_THRESHOLD,
        match_count=EVENT_MATCH_COUNT,
    )

    chosen_event_id: Optional[str] = None
    chosen_event_similarity: float = 1.0

    if matches_evt:
        chosen_event_id = matches_evt[0].get("event_id") or matches_evt[0].get("id")
        chosen_event_similarity = float(matches_evt[0].get("similarity", 1.0))
        link_article_event(sb, article_id=article_id, event_id=chosen_event_id, similarity=chosen_event_similarity)
        upsert_event_centroid(sb, chosen_event_id, embedding, model_name=EMBED_MODEL)
    else:
        evt = run_openai_json(
            oa,
            pt_event,
            {
                "title": title,
                "summary_en": summary_en,
                "category_slug": primary_slug or "",
                "category_name": cls.get("primary_name", ""),
            },
        )

        evt_title_raw = (evt.get("title") or "").strip() or clip_text(title, 140)
        evt_desc = (evt.get("description") or "").strip() or clip_text(summary_en, 900)
        evt_title = normalize_event_title(evt_title_raw, max_words=5)

        started_at = safe_iso_from_article(article)
        last_updated_at = utc_now_iso()

        chosen_event_id = create_event(
            sb,
            title=evt_title,
            description=evt_desc,
            main_category_id=primary_cat_id,
            started_at=started_at,
            last_updated_at=last_updated_at,
            model_name=pt_event.model_name,
            prompt_version=pt_event.version,
            prompt_template_id=pt_event.id,
        )

        update_event_storyline(sb, chosen_event_id, chosen_storyline_id)
        upsert_event_centroid(sb, chosen_event_id, embedding, model_name=EMBED_MODEL)
        link_article_event(sb, article_id=article_id, event_id=chosen_event_id, similarity=1.0)

    # ---- Persist derived content ----
    upsert_article_summary(
        sb,
        article_id=article_id,
        language="en",
        summary_short_md=summary_en,
        model_name=pt_summary_en.model_name,
        prompt_version=pt_summary_en.version,
        prompt_template_id=pt_summary_en.id,
    )
    upsert_article_summary(
        sb,
        article_id=article_id,
        language="zh",
        summary_short_md=summary_zh,
        model_name=DEEPSEEK_MODEL,
        prompt_version=pt_translate_zh.version,
        prompt_template_id=pt_translate_zh.id,
    )

    upsert_article_title_zh(sb, article_id=article_id, title_zh=title_zh, pt_title=pt_title)

    # Mark processed LAST
    update_article_primary_fields(
        sb,
        article_id=article_id,
        primary_category_id=primary_cat_id,
        primary_event_id=chosen_event_id,
        primary_storyline_id=chosen_storyline_id,
        ai_processed_at=utc_now_iso(),
    )


def main():
    sb = get_supabase_client()
    oa = get_openai_client()
    ds = get_deepseek_client()

    pt_summary_en = get_or_seed_prompt_template(sb, TASK_SUMMARY, SUMMARY_VERSION)
    pt_cats = get_or_seed_prompt_template(sb, TASK_CATEGORIES, CATS_VERSION)
    pt_event = get_or_seed_prompt_template(sb, TASK_EVENT, EVENT_VERSION)
    pt_title = get_or_seed_prompt_template(sb, TASK_TITLE, TITLE_VERSION)
    pt_translate_zh = get_or_seed_prompt_template(sb, TASK_TRANSLATE, TRANSLATE_VERSION)
    pt_storyline = get_or_seed_prompt_template(sb, TASK_STORYLINE, STORYLINE_VERSION)

    categories = fetch_categories(sb)
    if not categories:
        raise RuntimeError("No categories found. Seed categories table first.")

    articles = fetch_articles_to_process(sb, limit=BATCH_LIMIT)
    if not articles:
        print("No articles to process.")
        return

    print(f"Processing {len(articles)} article(s)...")
    for i, a in enumerate(articles, start=1):
        article_id = a.get("id")
        if not article_id:
            continue

        # Claim it BEFORE spending tokens
        if not claim_article_for_processing(sb, article_id):
            print(f"SKIP (already claimed/processed): {article_id}")
            continue

        try:
            print(f"[{i}/{len(articles)}] {str(a.get('title','(no title)'))[:80]}")
            process_one_article(
                sb,
                oa,
                ds,
                article=a,
                categories=categories,
                pt_summary_en=pt_summary_en,
                pt_translate_zh=pt_translate_zh,
                pt_cats=pt_cats,
                pt_event=pt_event,
                pt_title=pt_title,
                pt_storyline=pt_storyline,
            )
            time.sleep(SLEEP_BETWEEN_ARTICLES_SEC)
        except Exception as e:
            sb.table("articles").update({"ai_processing_error": str(e)}).eq("id", article_id).execute()
            print(f"ERROR: article_id={article_id} err={e}")

    print("Done.")


if __name__ == "__main__":
    main()
