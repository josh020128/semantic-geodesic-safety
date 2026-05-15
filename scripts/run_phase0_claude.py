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
group_b = ["laptop", "mug", "bowl", "soccer ball", "power drill", "bleach cleanser", "apple", "banana", "box", "can"]
group_c = ["table", "shelf"]

M = group_a + group_b
S = group_a + group_b + group_c
all_pairs = list(itertools.product(M, S))

SYSTEM_INSTRUCTION = """
Evaluate each ordered pair: (manipulated object, scene object).

You are generating a semantic consequence prior for robotic manipulation.

Important scope:
- This dataset is NOT for collision checking.
- This dataset is NOT for contact mechanics.
- Geometric collision avoidance is handled separately by the occupancy volume.
- Your task is only to describe semantic consequence around the scene object.

Definitions:
- manipulated object = the object the robot is holding or moving
- scene object = a nearby object in the environment
- the pair is ordered, so (cup of water, laptop) is different from (laptop, cup of water)

Return:
- one JSON array
- exactly one object per ordered pair
- valid JSON only
- no markdown
- no explanations
- no comments
- no extra fields

Use exactly this schema for every entry:

{
  "manipulated": "string",
  "scene": "string",
  "weights": {
    "w_+x": number or "inf",
    "w_-x": number or "inf",
    "w_+y": number or "inf",
    "w_-y": number or "inf",
    "w_+z": number or "inf",
    "w_-z": number or "inf"
  },
  "sigma_m": float
}

Field meanings:

1) weights
The frame is centered on the scene object.
The six directional weights describe semantic consequence by direction:
- w_+x = consequence in the +x direction from the scene object
- w_-x = consequence in the -x direction from the scene object
- w_+y = consequence in the +y direction from the scene object
- w_-y = consequence in the -y direction from the scene object
- w_+z = consequence above the scene object
- w_-z = consequence below the scene object

Allowed value for each weight:
- 0.0
- any real number in [0.0, 1.0]
- "inf"

Interpretation:
- 0.0 = no meaningful semantic consequence in that direction
- values in (0.0, 1.0] = finite soft semantic consequence
- "inf" = hard semantic no-go from that direction

Use "inf" sparingly.
Use "inf" only when approaching from that direction should be treated as categorically forbidden in semantic terms, not merely high risk.

Examples where "inf" may be appropriate:
- liquid above vulnerable electronics
- liquid above a device where spill-from-above would be unacceptable
- a direction where the semantic consequence is clearly catastrophic rather than merely undesirable

Do NOT use "inf" for:
- ordinary collision avoidance
- generic proximity discomfort
- structural obstacles such as tables, walls, shelves, floors, desks, or counters

2) sigma_m
sigma_m is the Gaussian spatial spread in meters.
It controls how far the semantic consequence extends away from the scene object.

Interpretation:
- smaller sigma_m = more local consequence
- larger sigma_m = broader consequence

Rules:
- sigma_m must be >= 0.0
- if all six weights are 0.0, set sigma_m = 0.0
- otherwise sigma_m must be > 0.0
- choose a conservative but physically plausible value
- avoid arbitrarily large spreads

Typical tabletop range:
- often between 0.05 and 0.20 meters
- use larger values only when the consequence meaningfully extends farther

Decision rules:
- Base the answer on common-sense physical interaction.
- Be conservative but physically plausible.
- Use non zero weights only when the semantic consequence is meaningful, which means the object in the scene is actually damaged or harmed when the manipulated object is getting near or touching it.
- Use the minimum number of assumptions needed.
- Do not invent hidden categories, roles, or templates.
- Output only manipulated, scene, weights, and sigma_m.

How to think about weights:
- If the semantic consequence is roughly similar in all directions, use similar values for all six weights.
- If the consequence is strongly directional, assign higher values only to the relevant directions.
- If the consequence is mainly from above, make w_+z larger than the others.
- If a direction is not meaningful, set its weight to 0.0.
- Do NOT automatically make w_+z high just because the manipulated object is a liquid.
  Make w_+z high only when the scene object is actually vulnerable to consequence from above.

How to think about structural context:
- If the scene object is mainly structural context and there is no meaningful pair-specific semantic consequence
  (for example: table, wall, floor, shelf, desk, counter),
  then set all six weights to 0.0 and sigma_m to 0.0.

How to think about object tolerance:
- Do not treat “getting wet” as inherently harmful for every object.
- Water-tolerant or receptacle-like objects (for example bowl, sink, mug, tray) often deserve all-zero or very low weights.
- Only assign high weights when there is a plausible semantic damage mode.

Calibration examples:

1)
(manipulated="cup of water", scene="power drill")
Reasoning target:
- spill-from-above onto an electrical tool is highly undesirable
Example output:
{
  "manipulated": "cup of water",
  "scene": "power drill",
  "weights": {
    "w_+x": 0.3,
    "w_-x": 0.3,
    "w_+y": 0.3,
    "w_-y": 0.3,
    "w_+z": "inf",
    "w_-z": 0.0
  },
  "sigma_m": 0.10
}

2)
(manipulated="cup of water", scene="bowl")
Reasoning target:
- bowl usually receives or contains liquid
Example output:
{
  "manipulated": "cup of water",
  "scene": "bowl",
  "weights": {
    "w_+x": 0.0,
    "w_-x": 0.0,
    "w_+y": 0.0,
    "w_-y": 0.0,
    "w_+z": 0.0,
    "w_-z": 0.0
  },
  "sigma_m": 0.0
}

3)
(manipulated="hot soldering iron", scene="laptop")
Reasoning target:
- thermal consequence exists in close proximity
- stronger above / near the laptop, but not necessarily infinite
Example output:
{
  "manipulated": "hot soldering iron",
  "scene": "laptop",
  "weights": {
    "w_+x": 0.6,
    "w_-x": 0.6,
    "w_+y": 0.6,
    "w_-y": 0.6,
    "w_+z": 0.6,
    "w_-z": 0.2
  },
  "sigma_m": 0.10
}

4)
(manipulated="cup of water", scene="table")
Reasoning target:
- table is mainly structural context, not a pair-specific semantic hazard target
Example output:
{
  "manipulated": "cup of water",
  "scene": "table",
  "weights": {
    "w_+x": 0.0,
    "w_-x": 0.0,
    "w_+y": 0.0,
    "w_-y": 0.0,
    "w_+z": 0.0,
    "w_-z": 0.0
  },
  "sigma_m": 0.0
}

Output format reminder:
- return one JSON array
- one object per ordered pair
- valid JSON only
- no markdown
- no prose
- no extra keys
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

