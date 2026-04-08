#!/usr/bin/env python3
"""Batch Phase 0: Gemini generates semantic risk entries for (manipulated, scene) pairs → data/ JSON."""

import itertools
import json
import os
from pathlib import Path

from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_PATH = DATA_DIR / "semantic_risk_demo.json"

# 1. Define the datasets
group_a = ["cup of water", "kitchen knife", "hot soldering iron", "heavy metal wrench"]
group_b = ["open laptop", "wine glass", "balloon", "power strip", "power drill"]
group_c = ["kitchen sink", "plastic tray"]

M = group_a + group_b
S = group_a + group_b + group_c
all_pairs = list(itertools.product(M, S))

SYSTEM_INSTRUCTION = """
You are an expert physical reasoning engine for a robotics lab.
Evaluate each ordered pair:
(manipulated object, scene object)

The coordinate frame is centered on the SCENE object:
- w_+z: above the scene object
- w_-z: below the scene object
- w_+x, w_-x, w_+y, w_-y: horizontal directions around it

Risk should reflect meaningful consequence to the scene object, not mere physical reachability.
If the scene object is only weakly affected, use low weights.

Choose exactly one scene_role:
- "hazard_target": meaningfully threatened; may emit a repulsive semantic risk field
- "safe_receptacle": safely receives or contains the manipulated object
- "support_context": mainly structural/support context (table, wall, floor, shelf, desk, counter)
- "neutral_context": weak or irrelevant interaction

Examples:
- water over laptop -> hazard_target
- water over sink -> safe_receptacle
- water over table -> support_context
- wrench over wine glass -> hazard_target

Use "gravity_column" only when danger mainly comes from above AND the scene object is meaningfully vulnerable to spill/drop from above.
Otherwise use "standard_decay". Use "none" if vertical behavior is not special.

Choose 1 primary hazard family, optionally 1 secondary:
["liquid", "thermal", "sharp", "fragile", "impact", "contamination", "electrical"]

Choose 1 topology_template:
- "upward_vertical_cone"
- "isotropic_sphere"
- "forward_directional_cone"
- "planar_half_space"

Weight rules:
- all weights must be in [0.0, 1.0]
- high weights only if consequence is meaningful
- safe_receptacle, support_context, and weak interactions should usually have low weights
- for upward_vertical_cone, w_+z may dominate only when above-position consequence is meaningful

lateral_decay:
- "narrow", "moderate", or "wide"

receptacle_attenuation in [0.1, 1.0]:
- 1.0 = highly vulnerable target
- 0.4–0.8 = moderate vulnerability
- 0.1–0.3 = safe receptacle or negligible consequence

Output only a JSON array in this schema:

[
  {
    "manipulated": "...",
    "scene": "...",
    "families": ["..."],
    "scene_role": "hazard_target",
    "topology_template": "...",
    "weights": {
      "w_+x": 0.0,
      "w_-x": 0.0,
      "w_+y": 0.0,
      "w_-y": 0.0,
      "w_+z": 0.0,
      "w_-z": 0.0
    },
    "vertical_rule": "standard_decay",
    "lateral_decay": "moderate",
    "receptacle_attenuation": 1.0
  }
]
"""


def chunk_list(data, chunk_size):
    """Yield successive chunks from the data list."""
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


def generate_dataset() -> None:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    final_dataset: list = []
    batches = list(chunk_list(all_pairs, 20))

    print(f"Total pairs to process: {len(all_pairs)}")
    print(f"Processing in {len(batches)} batches...\n")

    for idx, batch in enumerate(batches):
        print(f"Processing batch {idx + 1}/{len(batches)}...")

        prompt_payload = "Calculate risk for the following pairs:\n"
        for manip, scene in batch:
            prompt_payload += f"- Manipulated: '{manip}', Scene: '{scene}'\n"

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt_payload,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            if not response.text:
                raise RuntimeError("Gemini returned empty response.")
            batch_data = json.loads(response.text.strip())
            if not isinstance(batch_data, list):
                print(f"Error processing batch {idx + 1}: expected JSON array, got {type(batch_data)}")
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
