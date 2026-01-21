from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass(frozen=True)
class PromptSpec:
    task: str                   # e.g. 'summary', 'category', 'event'
    version: str                # e.g. 'v1'
    name: str
    system_prompt: Optional[str]
    user_prompt_template: str   # uses {title}, {content}, etc.
    model_name: str
    temperature: float = 0.0
    max_output_tokens: Optional[int] = None
    is_active: bool = True      # you can set false for deprecated seeds


# Registry keyed by (task, version)
PROMPT_SPECS: Dict[Tuple[str, str], PromptSpec] = {}

def register(spec: PromptSpec) -> None:
    key = (spec.task, spec.version)
    if key in PROMPT_SPECS:
        raise ValueError(f"Duplicate PromptSpec registered for {key}")
    PROMPT_SPECS[key] = spec


# ----------------------------
# TITLE v1
# ----------------------------
register(PromptSpec(
    task="title",
    version="v1",
    name="Chinese headline from EN title + EN summary",
    system_prompt="You write concise news headlines.",
    user_prompt_template=(
        "Write a Simplified Chinese news headline.\n"
        "Rules:\n"
        "- <= 28 Chinese characters if possible\n"
        "- Use common translation for names if exists, otherwise translate phonetically\n"
        "- No ending punctuation\n"
        "- Keep it factual and specific\n"
        "- No book title quotes 《》, no quotation marks.\n"
        "- Output ONLY the headline text\n\n"
        "English title:\n{title_en}\n\n"
        "English bullet summary:\n{summary_en}\n"
    ),
    model_name="gpt-4.1-mini",
    temperature=0.2,
    max_output_tokens=60,
))

# ----------------------------
# SUMMARY v1
# ----------------------------
register(PromptSpec(
    task="summary",
    version="v1",
    name="News bullet summary (language via variable)",
    system_prompt="You are a neutral, factual news summarizer. Follow instructions precisely.",
    user_prompt_template=(
        "Language:\n{language_instruction}\n\n"
        "Task:\n"
        "- Summarize the article into 4-6 bullet points.\n"
        "- Each bullet is ONE sentence.\n"
        "- Each bullet should be **under ~20 words**.\n"
        "- Avoid semicolons; keep one clause.\n"
        "- Include key facts: who/what/when/where/why and important numbers.\n"
        "- Avoid speculation and loaded language.\n"
        "- Output ONLY markdown bullet points.\n\n"
        "Article title:\n{title}\n\n"
        "Article text:\n\"\"\"{content}\"\"\""
    ),
    model_name="gpt-4.1-mini",
    temperature=0.2,
    max_output_tokens=500,
))

# ----------------------------
# CATEGORY v1
# ----------------------------
register(PromptSpec(
    task="category",
    version="v1",
    name="Fixed taxonomy classification",
    system_prompt="You classify news articles into a fixed taxonomy.",
    user_prompt_template=(
        "Allowed categories (slug: name):\n{categories}\n\n"
        "Rules:\n"
        "- Choose exactly 1 primary category.\n"
        "- Optionally choose up to 2 secondary categories.\n"
        "- Provide confidence scores 0.0-1.0 for chosen categories only.\n"
        "- Output STRICT JSON only with these rules: "
        "(1) secondary_slugs must be an empty array if none.\n"
        "(2) scores must include primary and any secondary only.\n\n"
        "Return schema:\n"
        "{{\n"
        '  "primary_slug": "...",\n'
        '  "secondary_slugs": ["...", "..."],\n'
        '  "scores": {{"slug": 0.0}}\n'
        "}}\n\n"
        "Article title:\n{title}\n\n"
        "Article text:\n\"\"\"{content}\"\"\""
    ),
    model_name="gpt-4.1-mini",
    temperature=0.0,
    max_output_tokens=250,
))

# ----------------------------
# EVENT v1
# ----------------------------
register(PromptSpec(
    task="event",
    version="v1",
    name="Event title + description from one EN summary",
    system_prompt="""
You assign news EVENTS.

An EVENT is a broad, ongoing situation that can include many related articles.
It should be broader than a single incident, court ruling, or statement.

Rules:
- Event title should be 2-5 words
- Prefer LOCATION + ISSUE (e.g., "ICE in Minneapolis")
- Do NOT include dates
- Do NOT include names of individuals
- Do NOT describe a single court ruling or quote
- The same event should be reusable for future articles

Good examples:
- ICE in Minneapolis
- Immigration Protests in Minnesota
- Federal Immigration Raids
- ICE Activity in Los Angeles

Bad examples:
- Judge restricts ICE use of pepper spray
- Court limits ICE tactics during protests
- Sheriff criticizes ICE after shooting

Output STRICT JSON only.

Return schema:
{{
  "event_title": "...",
  "event_description": "1 sentence describing the ongoing situation"
}}

""",
    user_prompt_template="""
Create an EVENT for the article below.

Article title:
{title}

English summary:
{summary_en}

""",
    model_name="gpt-4.1-mini",
    temperature=0.2,
    max_output_tokens=200,
))

# ----------------------------
# TRANSLATION v1
# ----------------------------
register(PromptSpec(
    task="translate",
    version="v1",
    name="Translate EN bullet summary to Simplified Chinese (no new facts)",
    system_prompt="You are a precise translator. Do not add or remove facts.",
    user_prompt_template=(
        "Translate the following English bullet points into Simplified Chinese.\n"
        "Rules:\n"
        "- Do NOT add or remove any facts.\n"
        "- Keep the SAME number of bullets.\n"
        "- Keep names, numbers, and locations unchanged.\n"
        "- Use common translation for names if exists, otherwise translate phonetically.\n"
        "- Output ONLY markdown bullet points.\n"
        "{extra_rules}\n\n"
        "English title:\n{title_en}\n\n"
        "English bullet points:\n{summary_en}\n"
    ),
    model_name="deepseek-chat",      # This field is not used by DeepSeek call here, but ok to store
    temperature=0.2,
    max_output_tokens=450,
))

# ----------------------------
# STORYLINE v1
# ----------------------------
register(PromptSpec(
    task="storyline",
    version="v1",
    name="Storyline title + description from EN summary (broad grouping)",
    system_prompt="""
You name BROAD developing news storylines.

A storyline is an umbrella that spans MULTIPLE events across places and dates.
It must NOT be tied to one person, one city, or one single incident.
Prefer 2-3 words.

Bad: "Philadelphia Sheriff's Criticism and Response to ICE Following Fatal Shooting"
Good: "Escalation of ICE Enforcement and Backlash"
Good: "ICE Raids and Community Resistance"

Output STRICT JSON only.

""",
    user_prompt_template="""
Create a broad storyline for the article below.

Rules:
- storyline_title must be reusable across many related articles.
- storyline_title, with the exception of being located in the United States as national events, MUST include a geographic or named anchor (e.g., "Uganda", "Kampala", "EU", "Gaza") unless the story is truly global.
- Avoid generic category labels like "Election disputes", "Political repression", "Human rights abuses" with no geographic anchors.
- With geographic/named anchor, prefer: "<Place> <Situation>" (e.g., "Uganda Post-Election Crackdown")
- Avoid names of individuals and avoid city/state names unless essential.
- Avoid exact dates.
- storyline_description should describe the broader pattern (1-2 sentences), not the single article.

Return schema:
{{"storyline_title":"...","storyline_description":"..."}}

Article title:
{title}

English bullet summary:
{summary_en}

""",
    model_name="gpt-4.1-mini",
    temperature=0.2,
    max_output_tokens=220,
))


# ----------------------------
# STORYLINE_DIGEST_EN v1
# ----------------------------
register(PromptSpec(
    task="storyline_digest_en",
    version="v1",
    name="Storyline digest in English from multiple representative article summaries",
    system_prompt="""
You are a professional news editor.

Synthesize multiple reports covering the same news storyline into a single, publishable English digest.
Write objectively and concisely. Do not speculate or editorialize.
If different sources report slightly different facts, describe the differences neutrally.

Output STRICT JSON only.
""",
    user_prompt_template="""
Below are multiple representative articles from the same news storyline.
Each includes an outlet, original title, and an English bullet summary.

Your tasks:
1) Produce a concise, publishable English headline title_en (8-14 words).
2) Produce a merged English digest summary_en with 4-7 bullet points:
   - Each bullet must start with "- "
   - End each bullet with a period.
   - Deduplicate overlapping facts.
   - Order bullets by importance.
   - If sources differ, use neutral phrasing (e.g., "Some reports say...", "Officials said...").
3) Do NOT mention sources explicitly and do NOT include links.

Context:
- {article_count} articles are related to this storyline in the recent period.
- Coverage spans {source_count} distinct news outlets.

Input articles:
{digest_input}

Return schema:
{{"title_en":"...","summary_en":"- ...\\n- ...\\n- ..."}}
""",
    model_name="gpt-4.1-mini",
    temperature=0.2,
    max_output_tokens=650,
))


# ----------------------------
# STORYLINE_DIGEST_TRANSLATE_ZH v1
# ----------------------------
register(PromptSpec(
    task="storyline_digest_translate_zh",
    version="v1",
    name="Translate + adapt English storyline digest into publishable Simplified Chinese (Red Note style)",
    system_prompt="""
你是一名中文新闻编辑，负责把英文新闻主线摘要改写成适合发布在小红书的中文新闻内容。
要求语言自然、专业、信息密度高，符合中文新闻表达习惯。
不得夸张、不得加入未经文本支持的推测或评论。

输出 STRICT JSON only。
""",
    user_prompt_template="""
请将以下英文新闻主线摘要改写为中文发布版本。

要求：
1) 生成中文主标题 title_zh (20-34 字，新闻风格，避免口语和夸张）。
2) 生成中文摘要 summary_zh (4-7 条项目符号）：
   - 每条以 "- " 开头
   - 句末使用 "。"
   - 忠实原意，但可为中文读者优化表达顺序
   - 若英文中存在不确定或分歧信息，请用“有报道称”“另有消息指出”等中性表述
3) 不要添加链接、来源列表或额外评论。

英文标题：
{title_en}

英文摘要：
{summary_en}

Return schema:
{{"title_zh":"...","summary_zh":"- ...\\n- ...\\n- ..."}}
""",
    # DeepSeek is OpenAI-API compatible; your runner uses OpenAI SDK with base_url.
    model_name="deepseek-chat",
    temperature=0.2,
    max_output_tokens=800,
))


# ----------------------------
# STORYLINE_TITLE_ZH_POLISH v1  (optional but recommended)
# ----------------------------
register(PromptSpec(
    task="storyline_title_zh_polish",
    version="v1",
    name="Polish an existing Chinese headline into tighter news style (no meaning change)",
    system_prompt="""
你是中文新闻编辑，擅长将标题改写得更像“新闻标题”。
要求：不改变事实含义，不夸张，不加入情绪化词语。
""",
    user_prompt_template="""
下面是一个准确但可能略显平淡的中文新闻标题。

请在不改变事实含义的前提下，将其改写为更符合中文新闻标题风格的版本：
- 更紧凑
- 信息密度更高
- 不要夸张或加入情绪化词语
- 长度保持在 20-30 字

原标题：
{title_zh}

只输出改写后的标题文本，不要加任何解释或多余字符。
""",
    model_name="gpt-4.1-mini",
    temperature=0.2,
    max_output_tokens=80,
))

