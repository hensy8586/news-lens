from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from supabase import Client as SupabaseClient

from .prompt_versions import PROMPT_SPECS, PromptSpec


@dataclass(frozen=True)
class PromptTemplateRow:
    id: str
    task: str
    version: str
    name: Optional[str]
    system_prompt: Optional[str]
    user_prompt_template: str
    model_name: str
    temperature: float
    max_output_tokens: Optional[int]
    is_active: bool


def _row_to_template(row: Dict[str, Any]) -> PromptTemplateRow:
    return PromptTemplateRow(
        id=row["id"],
        task=row["task"],
        version=row["version"],
        name=row.get("name"),
        system_prompt=row.get("system_prompt"),
        user_prompt_template=row["user_prompt_template"],
        model_name=row["model_name"],
        temperature=float(row.get("temperature") or 0.0),
        max_output_tokens=row.get("max_output_tokens"),
        is_active=bool(row.get("is_active", True)),
    )


def fetch_prompt_from_db(sb: SupabaseClient, task: str, version: str) -> Optional[PromptTemplateRow]:
    res = (
        sb.table("prompt_templates")
        .select("id,task,version,name,system_prompt,user_prompt_template,model_name,temperature,max_output_tokens,is_active")
        .eq("task", task)
        .eq("version", version)
        .limit(1)
        .execute()
    )
    rows = res.data or []
    if not rows:
        return None
    return _row_to_template(rows[0])


def insert_prompt_into_db(sb: SupabaseClient, spec: PromptSpec) -> PromptTemplateRow:
    payload = {
        "task": spec.task,
        "version": spec.version,
        "name": spec.name,
        "system_prompt": spec.system_prompt,
        "user_prompt_template": spec.user_prompt_template,
        "model_name": spec.model_name,
        "temperature": spec.temperature,
        "max_output_tokens": spec.max_output_tokens,
        "is_active": spec.is_active,
    }
    res = sb.table("prompt_templates").insert(payload).execute()
    row = (res.data or [None])[0]
    if not row:
        # Extremely rare; but better error message than later None deref
        raise RuntimeError(f"Failed to insert prompt template into DB for {(spec.task, spec.version)}")
    return _row_to_template(row)


def get_or_seed_prompt_template(sb: SupabaseClient, task: str, version: str) -> PromptTemplateRow:
    """
    DB-first:
      1) Try fetch from prompt_templates
      2) If missing, load from prompt_versions.py registry and insert
      3) If missing from registry, raise error
    """
    existing = fetch_prompt_from_db(sb, task, version)
    if existing is not None:
        return existing

    spec = PROMPT_SPECS.get((task, version))
    if spec is None:
        raise KeyError(
            f"Prompt template not found in DB and not defined in prompt_versions.py for (task={task!r}, version={version!r})."
        )

    return insert_prompt_into_db(sb, spec)
