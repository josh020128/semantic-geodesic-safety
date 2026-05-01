#!/usr/bin/env python3
"""Batch Phase 0: Claude generates semantic risk entries for (manipulated, scene) pairs → data/ JSON."""

import itertools
import json
import os
import re
from pathlib import Path

from anthropic import Anthropic

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_PATH = DATA_DIR / "semantic_risk_demo_claude.json"

# 1. Define the datasets (kept identical to run_phase0.py)
group_a = ["cup of water", "kitchen knife", "hot soldering iron"]
group_b = ["laptop", "mug", "bowl", "soccer ball", "power drill", "bleach cleanser"]
group_c = ["table", "shelf"]

M = group_a + group_b
S = group_a + group_b + group_c
all_pairs = list(itertools.product(M, S))

SYSTEM_INSTRUCTION = """
Evaluate each ordered pair: (manipulated object, scene object).

You are generating a semantic risk-prior dataset for robotic manipulation safety.

Task:
For every ordered pair (manipulated object, scene object), output exactly one JSON object describing how risky it is to move the manipulated object near the scene object.

Interpretation:
- manipulated object = the object the robot is holding or moving
- scene object = a nearby object in the environment
- the pair is ordered, so (cup of water, power drill) is different from (power drill, cup of water)

Return:
- one JSON array
- one object per ordered pair
- valid JSON only
- no markdown
- no explanations
- no extra fields

Use exactly this schema for every entry:

{
  "manipulated": "string",
  "scene": "string",
  "families": ["string", ...],
  "scene_role": "hazard_target | safe_receptacle | support_context",
  "topology_template": "upward_vertical_cone | isotropic_sphere | forward_directional_cone | planar_half_space",
  "weights": {
    "w_+x": float,
    "w_-x": float,
    "w_+y": float,
    "w_-y": float,
    "w_+z": float,
    "w_-z": float
  },
  "radius_m": float,
  "vertical_rule": "standard_decay | gravity_column",
  "lateral_decay": "low | moderate | high",
  "receptacle_attenuation": float
}

Field meanings:

1) families
Choose zero or more relevant hazard families from:
["liquid", "electrical", "thermal", "sharp", "fragile", "impact", "contamination"]

2) scene_role
Choose exactly one:
- "hazard_target": the scene object is vulnerable or may be harmed
- "safe_receptacle": the scene object safely receives or contains the hazard (choose this when the scene object is safe to interact with)
- "support_context": the scene object is mainly structural context
- Do not output any other scene_role value.

Hard rule for support_context (must follow):
- If scene_role is "support_context" (e.g., table, shelf, floor, wall, counter, desk), then:
  - families must be []
  - topology_template must be "isotropic_sphere"
  - ALL weights must be 0.0 (w_+x, w_-x, w_+y, w_-y, w_+z, w_-z)
  - radius_m must be 0.0
  - vertical_rule must be "standard_decay"
  - lateral_decay must be "moderate"
  - receptacle_attenuation must be 0.1

3) topology_template
Choose the spatial risk shape:
- "upward_vertical_cone": strongest risk above the scene object; good for spill/drip/drop hazards
- "isotropic_sphere": risk spreads in all directions; good for general proximity/collision hazards
- "forward_directional_cone": risk mainly extends in one horizontal direction
- "planar_half_space": risk is concentrated on one side of the scene object

4) weights
Each weight must be in [0.0, 1.0].
The frame is centered on the scene object.
Important convention:
- w_+z always means risk above the scene object
- DO NOT automatically make w_+z high just because the manipulated object is a liquid.
- Make w_+z high only when "from above" creates meaningful consequence to the scene object (i.e., the scene is vulnerable to spill/drip/drop from above).
- If the scene object is water-tolerant, washable, or not meaningfully harmed by getting wet (e.g., bowl, mug, sink, most metal tools like a kitchen knife),
  then liquid-from-above should usually be LOW risk (often "safe_receptacle" or "support_context"), with low w_+z and usually "standard_decay".
- w_-z is often 0.0 when “below the scene object” is not dangerous

Interpretation:
- 1.0 = maximum risk
- 0.7 to 0.9 = strong risk
- 0.4 to 0.6 = moderate risk
- 0.1 to 0.3 = weak risk
- 0.0 = no meaningful risk

5) radius_m
Use a realistic local manipulation radius in meters.
radius_m is a RADIUS (not diameter). It should be small and local.
Avoid unrealistic large radii:
- In most everyday tabletop scenes, radius_m should usually be in [0.06, 0.20]
- Use >0.20 only for truly large/extended hazards (e.g., large fragile area, large swinging tool), and still keep it conservative
- For small hand-sized scene objects (e.g., bleach cleanser bottle), radius_m is often 0.05–0.12
When in doubt, choose a smaller radius.

Calibration example:
- (soccer ball, bleach cleanser): impact hazard only → radius_m around 0.08–0.12 is typically sufficient (e.g., 0.10)

6) vertical_rule
- "standard_decay": normal distance decay
- "gravity_column": persistent danger in the vertical-above direction.
  Use this only when (a) danger mainly comes from above AND (b) the scene object is meaningfully vulnerable to that above-direction hazard
  (e.g., liquids above electronics, liquids above hot electrical tools, dropping heavy objects onto fragile items).
  If the scene object is NOT vulnerable to being wetted (e.g., bowl, sink, mug) then prefer "standard_decay" and low w_+z.

7) lateral_decay
- "low": spreads broadly sideways
- "moderate": standard sideways spread
- "high": falls off quickly sideways

8) receptacle_attenuation
Use a value in [0.0, 1.0].
Interpretation: how vulnerable the scene object is / how much the interaction matters.
- 1.0 = highly vulnerable / meaningful consequence (keep high only when truly harmful)
- 0.7–0.9 = fairly vulnerable / meaningful
- 0.4–0.6 = mild vulnerability / low-to-moderate consequence (use this when the relation exists but isn't severe)
- 0.1–0.3 = safe receptacle or very safe / negligible consequence

Guidance:
- Do NOT leave this at 1.0 by default. It should vary with the pair.
- For "safe_receptacle", receptacle_attenuation should usually be 0.1–0.3.
- For "support_context", receptacle_attenuation must be 0.1 (see hard rule above).
- For "hazard_target", receptacle_attenuation should usually be 1.0.
- If there is no meaningful hazard for the pair, keep weights low (often near 0.0) and keep receptacle_attenuation low (typically 0.1–0.3).

Decision rules:
- Base the answer on common-sense physical interaction.
- Be conservative but physically plausible.
- If the manipulated object can directly damage or may damage the scene object, use "hazard_target".
- If the scene object is completely safe to interact with the manipulated object, use "safe_receptacle".
- If the scene object is mainly a table, desk, wall, floor, counter, or shelf, often use "support_context".

Vulnerability sanity checks (important):
- Liquids: Only treat liquid-from-above as high risk when the scene object is actually vulnerable (electronics, paper/books, open food, sensitive machinery, etc.).
  Do NOT treat "getting wet" as inherently harmful for water-tolerant objects (bowl, mug, sink, plastic tray) or most metal tools (kitchen knife).
- Sharp/thermal/electrical: Only assign high weights when there is a plausible damage mode (cutting, burning, shock, corrosion/shorting, contamination).

Concrete examples to calibrate:
- (cup of water, laptop): hazard_target; families include ["liquid","electrical"]; upward_vertical_cone; w_+z high; vertical_rule gravity_column.
- (cup of water, bowl): safe_receptacle; families may be ["liquid"] or []; low weights (including low w_+z); vertical_rule standard_decay; receptacle_attenuation low (e.g., 0.1–0.3).
- (cup of water, kitchen knife): hazard_target; low weights; vertical_rule standard_decay (wet knife is not a meaningful hazard to the knife).

Example:
[  
    {
    "manipulated": "cup of water",
    "scene": "power drill",
    "families": ["liquid", "electrical"],
    "scene_role": "hazard_target",
    "topology_template": "upward_vertical_cone",
    "weights": {
        "w_+x": 0.3,
        "w_-x": 0.3,
        "w_+y": 0.3,
        "w_-y": 0.3,
        "w_+z": 1.0,
        "w_-z": 0.0
    },
    "radius_m": 0.17,
    "vertical_rule": "gravity_column",
    "lateral_decay": "moderate",
    "receptacle_attenuation": 1.0
    }
    ...
]
""".strip()


def chunk_list(data, chunk_size):
    """Yield successive chunks from the data list."""
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)


def _extract_json_array(text: str) -> str:
    """
    Claude sometimes wraps JSON in markdown fences or adds whitespace.
    We aggressively strip fences and then fall back to the first JSON array substring.
    """
    cleaned = _FENCE_RE.sub("", text).strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        return cleaned

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1].strip()

    return cleaned


def generate_dataset() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY (or CLAUDE_API_KEY).")

    # Use a concrete model id by default. This should be one that is visible in `c.models.list()`
    # for your Anthropic account / API key.
    model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
    client = Anthropic(api_key=api_key)

    final_dataset: list = []
    batches = list(chunk_list(all_pairs, 20))

    print(f"Total pairs to process: {len(all_pairs)}")
    print(f"Processing in {len(batches)} batches...\n")

    for idx, batch in enumerate(batches):
        print(f"Processing batch {idx + 1}/{len(batches)}...")

        prompt_payload = "Calculate risk for the following pairs:\n"
        for manip, scene in batch:
            prompt_payload += f"- Manipulated: '{manip}', Scene: '{scene}'\n"
        prompt_payload += "\nReturn ONLY the JSON array (no extra text)."

        try:
            response = client.messages.create(
                model=model,
                system=SYSTEM_INSTRUCTION,
                messages=[{"role": "user", "content": prompt_payload}],
                temperature=0.1,
                max_tokens=4096,
            )

            text_parts = []
            for part in response.content or []:
                if getattr(part, "type", None) == "text":
                    text_parts.append(part.text)
            text = "\n".join(text_parts).strip()
            if not text:
                raise RuntimeError("Claude returned empty response.")

            json_text = _extract_json_array(text)
            batch_data = json.loads(json_text)
            if not isinstance(batch_data, list):
                print(
                    f"Error processing batch {idx + 1}: expected JSON array, got {type(batch_data)}"
                )
                continue
            final_dataset.extend(batch_data)
        except Exception as e:
            print(f"Error processing batch {idx + 1}: {e}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=4)

    print(f"\nSuccess! Generated {len(final_dataset)} entries.")
    print(f"Dataset saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_dataset()

