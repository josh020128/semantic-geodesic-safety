from __future__ import annotations

import json
import os
import re
from typing import Any

from anthropic import Anthropic


def make_claude_client() -> Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY (or CLAUDE_API_KEY).")
    return Anthropic(api_key=api_key)


_CLAUDE_CLIENT = make_claude_client()
_CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)


def _extract_json_array(text: str) -> str:
    """
    Claude sometimes wraps JSON in markdown fences or adds whitespace.
    Strip fences and then fall back to the first JSON array substring.
    """
    cleaned = _FENCE_RE.sub("", text).strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        return cleaned

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1].strip()

    return cleaned


def _response_text(response: Any) -> str:
    """
    Anthropic SDK returns a response with a list of content blocks.
    We concatenate all text blocks.
    """
    text_parts: list[str] = []
    for part in (getattr(response, "content", None) or []):
        if getattr(part, "type", None) == "text":
            text_parts.append(part.text)
    return "\n".join(text_parts).strip()


def claude_batch_callback(
    manipulated_label: str,
    scene_labels: list[str],
    system_instruction: str,
) -> list[dict[str, Any]]:
    """
    Batch callback for SemanticRouter.

    Expected signature:
        llm_batch_callback(manipulated_label, scene_labels, system_instruction)
            -> list[dict[str, Any]]

    Returns a list of dicts following the exact JSON schema expected by router.py.
    """
    if not isinstance(scene_labels, list) or len(scene_labels) == 0:
        return []

    # Deduplicate while preserving order
    deduped_scene_labels: list[str] = []
    seen = set()
    for scene_label in scene_labels:
        s = str(scene_label).strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        deduped_scene_labels.append(s)

    if not deduped_scene_labels:
        return []

    pairs = [
        {
            "manipulated": manipulated_label,
            "scene": scene_label,
        }
        for scene_label in deduped_scene_labels
    ]

    prompt_payload = (
        "Evaluate the following ordered (manipulated object, scene object) pairs.\n"
        "Return only a JSON array following the required schema.\n\n"
        f"{json.dumps(pairs, ensure_ascii=False)}"
    )

    response = _CLAUDE_CLIENT.messages.create(
        model=_CLAUDE_MODEL,
        system=system_instruction,
        messages=[{"role": "user", "content": prompt_payload}],
        temperature=0.1,
        max_tokens=4096,
    )

    text = _response_text(response)
    if not text:
        raise RuntimeError("Claude returned empty response.")

    json_text = _extract_json_array(text)
    try:
        batch_data = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Claude returned invalid JSON: {e}\nExtracted text: {json_text}\nRaw text: {text}"
        ) from e

    if not isinstance(batch_data, list):
        raise RuntimeError("Claude batch callback expected a JSON array.")

    cleaned: list[dict[str, Any]] = []
    for item in batch_data:
        if not isinstance(item, dict):
            continue

        manipulated = str(item.get("manipulated", "")).strip()
        scene = str(item.get("scene", "")).strip()
        if not manipulated or not scene:
            continue

        weights = item.get("weights", {}) or {}

        cleaned.append(
            {
                "manipulated": manipulated,
                "scene": scene,
                "families": list(item.get("families", [])),
                "scene_role": item.get("scene_role", "hazard_target"),
                "topology_template": item.get("topology_template", "isotropic_sphere"),
                "weights": {
                    "w_+x": float(weights.get("w_+x", 0.0)),
                    "w_-x": float(weights.get("w_-x", 0.0)),
                    "w_+y": float(weights.get("w_+y", 0.0)),
                    "w_-y": float(weights.get("w_-y", 0.0)),
                    "w_+z": float(weights.get("w_+z", 0.0)),
                    "w_-z": float(weights.get("w_-z", 0.0)),
                },
                "radius_m": float(item.get("radius_m", 0.2)),
                "vertical_rule": item.get("vertical_rule", "standard_decay"),
                "lateral_decay": item.get("lateral_decay", "moderate"),
                "receptacle_attenuation": float(item.get("receptacle_attenuation", 1.0)),
            }
        )

    return cleaned


def claude_pair_callback(
    manipulated_label: str,
    scene_label: str,
    system_instruction: str,
) -> dict[str, Any]:
    """
    Optional single-pair wrapper for debugging.
    Not needed in the final batch router, but handy for testing.
    """
    results = claude_batch_callback(
        manipulated_label=manipulated_label,
        scene_labels=[scene_label],
        system_instruction=system_instruction,
    )
    if not results:
        raise RuntimeError(
            f"Claude returned no result for pair: {(manipulated_label, scene_label)}"
        )
    return results[0]

