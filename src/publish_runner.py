# src/phase_b_runner.py
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict

from openai import OpenAI
from supabase import create_client, Client as SupabaseClient

from .prompt_store import PromptTemplateRow, get_or_seed_prompt_template

DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ---- clients ----
def get_supabase_client() -> SupabaseClient:
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

def get_openai_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def get_deepseek_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def clip_text(s: str, max_chars: int) -> str:
    return (s or "").strip()[:max_chars]

def normalize_bullets(md: str) -> str:
    if not md:
        return md
    lines = [ln.rstrip() for ln in md.strip().splitlines() if ln.strip()]
    out = []
    for ln in lines:
        s = ln.strip()
        if s.startswith(("-", "•")):
            out.append("- " + s.lstrip("-•").strip())
        else:
            out.append(s)
    return "\n".join(out).strip()

def looks_truncated_zh(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return True
    last = lines[-1]
    if last.startswith("-") and len(last) > 12 and last[-1] not in "。！？…）】」”":
        return True
    return False

# ---- prompt render helpers (same conventions as your ai_process.py) ----
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

def run_openai_text(oa: OpenAI, pt: PromptTemplateRow, variables: Dict[str, Any]) -> str:
    prompt = build_single_prompt(pt, variables)
    resp = oa.responses.create(
        model=pt.model_name,
        input=prompt,
        temperature=float(pt.temperature or 0.0),
        max_output_tokens=pt.max_output_tokens,
    )
    return (resp.output_text or "").strip()

def run_deepseek_text(ds: OpenAI, pt: PromptTemplateRow, variables: Dict[str, Any]) -> str:
    messages = build_messages(pt, variables)
    resp = ds.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=float(pt.temperature or 0.0),
        max_tokens=int(pt.max_output_tokens or 700),
    )
    return (resp.choices[0].message.content or "").strip()

# ---- data fetch ----
def fetch_recent_clustered_articles(sb: SupabaseClient, lookback_hours: int, limit: int = 5000) -> List[Dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    res = (
        sb.table("articles")
        .select("id,title,link,source_outlet,published_at,content_text,primary_storyline_id,clustered_at,ai_processed_at,ai_eligible,archived")
        .eq("ai_eligible", True)
        .eq("archived", False)
        .is_("clustered_at", "not.null")
        .is_("primary_storyline_id", "not.null")
        .gte("published_at", cutoff.isoformat())
        .order("published_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []

def fetch_articles_for_storyline(sb: SupabaseClient, storyline_id: str, lookback_hours: int, limit: int = 200) -> List[Dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    res = (
        sb.table("articles")
        .select("id,title,link,source_outlet,published_at,content_text,ai_processed_at")
        .eq("primary_storyline_id", storyline_id)
        .eq("ai_eligible", True)
        .eq("archived", False)
        .gte("published_at", cutoff.isoformat())
        .order("published_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []

# ---- ranking & selection ----
def rank_storylines(recent_articles: List[Dict[str, Any]], min_articles: int, min_sources: int) -> List[Tuple[str, Dict[str, Any]]]:
    agg: Dict[str, Dict[str, Any]] = {}
    for a in recent_articles:
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

    ranked = []
    for sid, st in agg.items():
        st["source_count"] = len(st["sources"])
        if st["count"] >= min_articles and st["source_count"] >= min_sources:
            ranked.append((sid, st))

    ranked.sort(key=lambda x: (x[1]["count"], x[1]["source_count"], x[1]["latest"] or ""), reverse=True)
    return ranked

def pick_representative_articles(candidates: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    candidates = sorted(candidates, key=lambda a: a.get("published_at") or "", reverse=True)
    chosen: List[Dict[str, Any]] = []
    used_sources: Set[str] = set()

    # pass 1: unprocessed + distinct sources
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

    # pass 2: fill with unprocessed (allow repeats)
    if len(chosen) < k:
        for a in candidates:
            if len(chosen) >= k:
                break
            if a.get("ai_processed_at") is not None:
                continue
            if a["id"] in {x["id"] for x in chosen}:
                continue
            chosen.append(a)

    # pass 3: allow already-processed
    if len(chosen) < k:
        for a in candidates:
            if len(chosen) >= k:
                break
            if a["id"] in {x["id"] for x in chosen}:
                continue
            chosen.append(a)

    return chosen

# ---- AI for selected articles ----
def summarize_then_translate(
    oa: OpenAI,
    ds: OpenAI,
    *,
    pt_summary: PromptTemplateRow,
    pt_translate: PromptTemplateRow,
    title_en: str,
    content_text: str,
    max_chars_for_llm: int = 12000,
) -> Tuple[str, str]:
    content_for_llm = clip_text(content_text, max_chars_for_llm)

    summary_en = run_openai_text(oa, pt_summary, {"title": title_en, "content": content_for_llm})
    summary_en = normalize_bullets(summary_en)

    summary_zh = run_deepseek_text(ds, pt_translate, {"summary_en": summary_en, "title_en": title_en, "extra_rules": ""})
    summary_zh = normalize_bullets(summary_zh)

    if looks_truncated_zh(summary_zh):
        summary_zh = run_deepseek_text(
            ds,
            pt_translate,
            {"summary_en": summary_en, "title_en": title_en, "extra_rules": "上次输出可能被截断。请更精炼但不得漏信息；保持项目符号。"},
        )
        summary_zh = normalize_bullets(summary_zh)

    return summary_en, summary_zh

def build_publishable_post_zh(
    *,
    storyline_title_zh: str,
    storyline_digest_zh: str,
    reps: List[Dict[str, Any]],
    rep_title_zh_map: Dict[str, str],
    rep_summary_zh_map: Dict[str, str],
) -> str:
    lines: List[str] = []
    lines.append(f"【{storyline_title_zh}】")
    lines.append(storyline_digest_zh.strip())
    lines.append("")
    lines.append("—— 代表报道 ——")
    for a in reps:
        aid = a["id"]
        src = a.get("source_outlet") or ""
        link = a.get("link") or ""
        tzh = rep_title_zh_map.get(aid, a.get("title") or "")
        sz = rep_summary_zh_map.get(aid, "")
        lines.append(f"\n来源：{src}\n标题：{tzh}\n{sz}\n链接：{link}")
    return "\n".join([ln.rstrip() for ln in lines if ln is not None]).strip()
