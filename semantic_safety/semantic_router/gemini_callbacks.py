from __future__ import annotations

import json
import os
from typing import Any

from google import genai
from google.genai import types


def make_gemini_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    return genai.Client(api_key=api_key) if api_key else genai.Client()


_GEMINI_CLIENT = make_gemini_client()


def gemini_batch_callback(
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

    response = _GEMINI_CLIENT.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt_payload,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )

    if not response.text:
        raise RuntimeError("Gemini returned empty response.")

    try:
        batch_data = json.loads(response.text.strip())
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Gemini returned invalid JSON: {e}\nRaw text: {response.text}"
        ) from e

    if not isinstance(batch_data, list):
        raise RuntimeError("Gemini batch callback expected a JSON array.")

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
                "vertical_rule": item.get("vertical_rule", "standard_decay"),
                "lateral_decay": item.get("lateral_decay", "moderate"),
                "receptacle_attenuation": float(item.get("receptacle_attenuation", 1.0)),
            }
        )

    return cleaned


def gemini_pair_callback(
    manipulated_label: str,
    scene_label: str,
    system_instruction: str,
) -> dict[str, Any]:
    """
    Optional single-pair wrapper for debugging.
    Not needed in the final batch router, but handy for testing.
    """
    results = gemini_batch_callback(
        manipulated_label=manipulated_label,
        scene_labels=[scene_label],
        system_instruction=system_instruction,
    )
    if not results:
        raise RuntimeError(
            f"Gemini returned no result for pair: {(manipulated_label, scene_label)}"
        )
    return results[0]