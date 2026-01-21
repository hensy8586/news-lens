import os
import json
import time
import re
from datetime import datetime, timezone, timedelta, date
from typing import Any, Dict, List, Optional, Tuple, Set

from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
from supabase import create_client, Client as SupabaseClient

from .prompt_store import PromptTemplateRow, get_or_seed_prompt_template


# ----------------------------
# Config
# ----------------------------
BATCH_SLEEP = float(os.getenv("SLEEP_BETWEEN_ARTICLES_SEC", "0.2"))
MAX_CHARS_FOR_LLM = int(os.getenv("MAX_CHARS_FOR_LLM", "12000"))

LOOKBACK_HOURS = int(os.getenv("PUBLISH_LOOKBACK_HOURS", "72"))
TOP_N_STORYLINES = int(os.getenv("PUBLISH_TOP_N_STORYLINES", "6"))
K_PER_STORYLINE = int(os.getenv("PUBLISH_K_ARTICLES_PER_STORYLINE", "3"))

MIN_ARTICLES_PER_STORYLINE = int(os.getenv("PUBLISH_MIN_ARTICLES_PER_STORYLINE", "2"))
MIN_SOURCES_PER_STORYLINE = int(os.getenv("PUBLISH_MIN_SOURCES_PER_STORYLINE", "2"))

SUMMARY_VERSION = os.getenv("SUMMARY_VER", "v1")
DIGEST_EN_VERSION = os.getenv("DIGEST_EN_VER", "v1")
DIGEST_ZH_VERSION = os.getenv("DIGEST_ZH_VER", "v1")
POLISH_ZH_TITLE_VERSION = os.getenv("POLISH_ZH_TITLE_VER", "v1")

ENABLE_ZH_TITLE_POLISH = os.getenv("ENABLE_ZH_TITLE_POLISH", "true").lower() in ("1", "true", "yes", "y")

# Tasks
TASK_SUMMARY = "summary"
TASK_DIGEST_EN = "storyline_digest_en"
TASK_DIGEST_ZH = "storyline_digest_translate_zh"
TASK_TITLE_ZH_POLISH = "storyline_title_zh_polish"

# DeepSeek model name
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


# ----------------------------
# Clients
# ----------------------------
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def get_deepseek_client() -> OpenAI:
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


def utc_today_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()  # YYYY-MM-DD


def clip_text(s: str, max_chars: int) -> str:
    return (s or "").strip()[:max_chars]


def normalize_bullets(md: str) -> str:
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


class SafeDict(dict):
    def __missing__(self, key):
        return ""


def render_template(template: str, variables: Dict[str, Any]) -> str:
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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=20))
def run_deepseek_text(ds: OpenAI, pt: PromptTemplateRow, variables: Dict[str, Any]) -> str:
    messages = build_messages(pt, variables)
    resp = ds.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=float(pt.temperature or 0.0),
        max_tokens=int(pt.max_output_tokens or 700),
    )
    return (resp.choices[0].message.content or "").strip()


def _extract_json(txt: str) -> str:
    t = (txt or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"\s*```$", "", t).strip()
    if t.startswith("{") and t.endswith("}"):
        return t
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    return (m.group(0).strip() if m else "")


def parse_json_strict(txt: str) -> Dict[str, Any]:
    candidate = _extract_json(txt)
    if not candidate:
        raise ValueError(f"Expected JSON, got: {txt[:300]}")
    return json.loads(candidate)


def make_publish_batch_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")


# ----------------------------
# Supabase ops
# ----------------------------
def fetch_recent_clustered_articles(sb: SupabaseClient) -> List[Dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)

    res = (
        sb.table("articles")
        .select("id,title,link,source_outlet,published_at,content_text,primary_storyline_id,clustered_at,ai_processed_at,ai_eligible,archived")
        .eq("ai_eligible", True)
        .eq("archived", False)
        .not_.is_("clustered_at", "null")
        .not_.is_("primary_storyline_id", "null")
        .gte("published_at", cutoff.isoformat())
        .order("published_at", desc=True)
        .limit(5000)
        .execute()
    )
    return res.data or []


def fetch_articles_for_storyline(sb: SupabaseClient, storyline_id: str) -> List[Dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)

    res = (
        sb.table("articles")
        .select("id,title,link,source_outlet,published_at,content_text,ai_processed_at")
        .eq("primary_storyline_id", storyline_id)
        .eq("ai_eligible", True)
        .eq("archived", False)
        .gte("published_at", cutoff.isoformat())
        .order("published_at", desc=True)
        .limit(200)
        .execute()
    )
    return res.data or []


def mark_articles_selected(sb: SupabaseClient, article_ids: List[str], batch_id: str) -> None:
    if not article_ids:
        return
    sb.table("articles").update(
        {"selected_for_publish_at": utc_now_iso(), "publish_batch_id": batch_id}
    ).in_("id", article_ids).execute()


def upsert_article_summary_en(
    sb: SupabaseClient,
    *,
    article_id: str,
    summary_en: str,
    pt_summary: PromptTemplateRow,
) -> None:
    sb.table("article_summaries").upsert(
        {
            "article_id": article_id,
            "language": "en",
            "summary_short_md": summary_en,
            "summary_long_md": None,
            "model_name": pt_summary.model_name,
            "prompt_version": pt_summary.version,
            "prompt_template_id": pt_summary.id,
        }
    ).execute()


def fetch_existing_en_summaries(sb: SupabaseClient, article_ids: List[str]) -> Dict[str, str]:
    """
    Returns {article_id: summary_short_md} for language='en'.
    """
    if not article_ids:
        return {}
    res = (
        sb.table("article_summaries")
        .select("article_id,summary_short_md")
        .eq("language", "en")
        .in_("article_id", article_ids)
        .execute()
    )
    out: Dict[str, str] = {}
    for r in (res.data or []):
        aid = r.get("article_id")
        s = (r.get("summary_short_md") or "").strip()
        if aid and s:
            out[aid] = s
    return out


def mark_article_processed(sb: SupabaseClient, article_id: str) -> None:
    sb.table("articles").update({"ai_processed_at": utc_now_iso()}).eq("id", article_id).execute()


def update_storyline_display(
    sb: SupabaseClient,
    storyline_id: str,
    *,
    title_en: str,
    summary_en: str,
    title_zh: str,
    summary_zh: str,
    publish_batch_id: Optional[str] = None,
) -> None:
    patch = {
        "display_title_en": (title_en or "").strip(),
        "display_summary_en": (summary_en or "").strip(),
        "display_title_zh": (title_zh or "").strip(),
        "display_summary_zh": (summary_zh or "").strip(),
        "last_curated_at": utc_now_iso(),
    }
    if publish_batch_id is not None:
        patch["publish_batch_id"] = publish_batch_id

    sb.table("storylines").update(patch).eq("id", storyline_id).execute()


def upsert_storyline_digest_history(
    sb: SupabaseClient,
    *,
    storyline_id: str,
    digest_date: str,          # YYYY-MM-DD (UTC)
    publish_batch_id: str,
    title_en: str,
    summary_en: str,
    title_zh: str,
    summary_zh: str,
    representative_article_ids: List[str],
    article_count: int,
    source_count: int,
    lookback_hours: int,
    pt_digest_en: PromptTemplateRow,
    pt_digest_zh: PromptTemplateRow,
    pt_polish: Optional[PromptTemplateRow] = None,
) -> None:
    payload: Dict[str, Any] = {
        "storyline_id": storyline_id,
        "digest_date": digest_date,
        "publish_batch_id": publish_batch_id,
        "title_en": (title_en or "").strip(),
        "summary_en": (summary_en or "").strip(),
        "title_zh": (title_zh or "").strip(),
        "summary_zh": (summary_zh or "").strip(),
        "representative_article_ids": representative_article_ids,
        "article_count": int(article_count),
        "source_count": int(source_count),
        "lookback_hours": int(lookback_hours),

        "digest_en_model_name": pt_digest_en.model_name,
        "digest_en_prompt_version": pt_digest_en.version,
        "digest_en_prompt_template_id": pt_digest_en.id,

        "digest_zh_model_name": DEEPSEEK_MODEL,
        "digest_zh_prompt_version": pt_digest_zh.version,
        "digest_zh_prompt_template_id": pt_digest_zh.id,
    }

    if pt_polish is not None:
        payload.update({
            "polish_model_name": pt_polish.model_name,
            "polish_prompt_version": pt_polish.version,
            "polish_prompt_template_id": pt_polish.id,
        })

    sb.table("storyline_digests").upsert(
        payload,
        on_conflict="storyline_id,digest_date"
    ).execute()


# ----------------------------
# Selection logic
# ----------------------------
def rank_storylines(articles: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    agg: Dict[str, Dict[str, Any]] = {}
    for a in articles:
        sid = a.get("primary_storyline_id")
        if not sid:
            continue
        row = agg.setdefault(sid, {"count": 0, "sources": set(), "latest": None})
        row["count"] += 1
        src = a.get("source_outlet") or ""
        if src:
            row["sources"].add(src)
        pub = a.get("published_at")
        if pub and (row["latest"] is None or pub > row["latest"]):
            row["latest"] = pub

    ranked: List[Tuple[str, Dict[str, Any]]] = []
    for sid, st in agg.items():
        st["source_count"] = len(st["sources"])
        if st["count"] >= MIN_ARTICLES_PER_STORYLINE and st["source_count"] >= MIN_SOURCES_PER_STORYLINE:
            ranked.append((sid, st))

    ranked.sort(key=lambda x: (x[1]["count"], x[1]["source_count"], x[1]["latest"] or ""), reverse=True)
    return ranked


def pick_representative_articles(candidates: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    candidates = sorted(candidates, key=lambda a: a.get("published_at") or "", reverse=True)

    chosen: List[Dict[str, Any]] = []
    used_sources: Set[str] = set()

    # pass 1: prefer unprocessed + distinct sources
    for a in candidates:
        if len(chosen) >= k:
            break
        if a.get("ai_processed_at") is not None:
            continue
        src = a.get("source_outlet") or ""
        if src and src in used_sources:
            continue
        chosen.append(a)
        if src:
            used_sources.add(src)

    # pass 2: fill remaining with unprocessed (allow repeats)
    if len(chosen) < k:
        for a in candidates:
            if len(chosen) >= k:
                break
            if a.get("ai_processed_at") is not None:
                continue
            if a["id"] in {x["id"] for x in chosen}:
                continue
            chosen.append(a)

    # pass 3: allow already-processed if still short
    if len(chosen) < k:
        for a in candidates:
            if len(chosen) >= k:
                break
            if a["id"] in {x["id"] for x in chosen}:
                continue
            chosen.append(a)

    return chosen


# ----------------------------
# Digest input assembly (EN-only)
# ----------------------------
def build_digest_input_en(selected_articles: List[Dict[str, Any]], per_article_summary_en: Dict[str, str]) -> str:
    blocks: List[str] = []
    for a in selected_articles:
        aid = a["id"]
        src = a.get("source_outlet") or ""
        title = a.get("title") or ""
        summ = (per_article_summary_en.get(aid) or "").strip()
        if not summ:
            continue
        blocks.append(f"Outlet: {src}\nTitle: {title}\nSummary:\n{summ}".strip())
    return "\n\n---\n\n".join(blocks).strip()


# ----------------------------
# Main
# ----------------------------
def main():
    sb = get_supabase_client()
    oa = get_openai_client()
    ds = get_deepseek_client()

    pt_summary = get_or_seed_prompt_template(sb, TASK_SUMMARY, SUMMARY_VERSION)
    pt_digest_en = get_or_seed_prompt_template(sb, TASK_DIGEST_EN, DIGEST_EN_VERSION)
    pt_digest_zh = get_or_seed_prompt_template(sb, TASK_DIGEST_ZH, DIGEST_ZH_VERSION)

    pt_polish = None
    if ENABLE_ZH_TITLE_POLISH:
        pt_polish = get_or_seed_prompt_template(sb, TASK_TITLE_ZH_POLISH, POLISH_ZH_TITLE_VERSION)

    recent = fetch_recent_clustered_articles(sb)
    if not recent:
        print("No recent clustered articles found.")
        return

    ranked = rank_storylines(recent)
    if not ranked:
        print("No storylines meet MIN_ARTICLES/MIN_SOURCES thresholds.")
        return

    top = ranked[:TOP_N_STORYLINES]
    batch_id = make_publish_batch_id()
    digest_date = utc_today_date()

    print(f"Phase B batch_id={batch_id} digest_date={digest_date} (top {len(top)} storylines)")

    for idx, (sid, stats) in enumerate(top, start=1):
        try:
            print(
                f"\n[{idx}/{len(top)}] storyline_id={sid} "
                f"count={stats['count']} sources={stats['source_count']} latest={stats['latest']}"
            )

            candidates = fetch_articles_for_storyline(sb, sid)
            selected = pick_representative_articles(candidates, k=K_PER_STORYLINE)
            if not selected:
                print("  -> No articles selected.")
                continue

            selected_ids = [a["id"] for a in selected]
            mark_articles_selected(sb, selected_ids, batch_id=batch_id)
            print(f"  -> Selected {len(selected_ids)} article(s)")

            # 1) Load existing EN summaries for selected
            per_article_en: Dict[str, str] = fetch_existing_en_summaries(sb, selected_ids)

            # 2) For any selected article missing EN summary, create it (and mark processed)
            for a in selected:
                aid = a["id"]
                if (per_article_en.get(aid) or "").strip():
                    continue  # already have EN summary in DB

                title_en = a.get("title") or ""
                content_text = a.get("content_text") or ""
                if not content_text.strip():
                    print(f"  -> Skip empty content_text article_id={aid}")
                    mark_article_processed(sb, aid)
                    continue

                summary_en = run_openai_text(
                    oa,
                    pt_summary,
                    {
                        "title": title_en,
                        "content": clip_text(content_text, MAX_CHARS_FOR_LLM),
                        "language_instruction": "Write in English.",
                    },
                )
                summary_en = normalize_bullets(summary_en)

                upsert_article_summary_en(sb, article_id=aid, summary_en=summary_en, pt_summary=pt_summary)
                mark_article_processed(sb, aid)
                per_article_en[aid] = summary_en

                time.sleep(BATCH_SLEEP)

            digest_input = build_digest_input_en(selected, per_article_en)
            if not digest_input:
                print("  -> No digest input (no EN summaries available). Skipping digest for this storyline.")
                continue

            # 3) English storyline digest (JSON)
            out_en = run_openai_text(
                oa,
                pt_digest_en,
                {
                    "digest_input": digest_input,
                    "article_count": stats["count"],
                    "source_count": stats["source_count"],
                },
            )
            obj_en = parse_json_strict(out_en)
            title_en = (obj_en.get("title_en") or "").strip()
            summary_en = normalize_bullets((obj_en.get("summary_en") or "").strip())

            # 4) Translate digest to ZH (JSON)
            out_zh = run_deepseek_text(
                ds,
                pt_digest_zh,
                {"title_en": title_en, "summary_en": summary_en},
            )
            obj_zh = parse_json_strict(out_zh)
            title_zh = (obj_zh.get("title_zh") or "").strip()
            summary_zh = normalize_bullets((obj_zh.get("summary_zh") or "").strip())

            # 5) Optional polish title
            if pt_polish and title_zh:
                polished = run_openai_text(oa, pt_polish, {"title_zh": title_zh}).strip()
                if polished:
                    title_zh = polished

            # 6) Update rolling snapshot on storyline
            update_storyline_display(
                sb,
                sid,
                title_en=title_en,
                summary_en=summary_en,
                title_zh=title_zh,
                summary_zh=summary_zh,
                publish_batch_id=batch_id,
            )
            print("  -> Updated storylines.display_* (rolling snapshot)")

            # 7) Write daily history row (multi-day continuity)
            upsert_storyline_digest_history(
                sb,
                storyline_id=sid,
                digest_date=digest_date,
                publish_batch_id=batch_id,
                title_en=title_en,
                summary_en=summary_en,
                title_zh=title_zh,
                summary_zh=summary_zh,
                representative_article_ids=selected_ids,
                article_count=stats["count"],
                source_count=stats["source_count"],
                lookback_hours=LOOKBACK_HOURS,
                pt_digest_en=pt_digest_en,
                pt_digest_zh=pt_digest_zh,
                pt_polish=pt_polish,
            )
            print("  -> Upserted storyline_digests history row")

            # Optional: print for logs
            print("\n--- PUBLISHABLE (ZH) ---")
            print(f"【{title_zh}】")
            print(summary_zh)
            print("--- END ---\n")

        except Exception as e:
            print(f"ERROR storyline_id={sid} err={e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
