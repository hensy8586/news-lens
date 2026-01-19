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

