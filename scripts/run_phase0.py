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
group_b = ["open laptop", "wine glass", "balloon", "power strip"]
group_c = ["kitchen sink", "plastic tray"]

M = group_a + group_b
S = group_a + group_b + group_c
all_pairs = list(itertools.product(M, S))

SYSTEM_INSTRUCTION = """
You are an expert physical reasoning engine for a robotics lab.
We are building a semantic risk field. Your task is to evaluate the risk of a robot moving a manipulated object near a scene object.

The pair is ORDERED:
(manipulated object, scene object)

CRITICAL SPATIAL RULES
The coordinate frame is centered on the SCENE object:

* w_+z = space directly ABOVE the scene object
* w_-z = space directly BELOW the scene object
* w_+x, w_-x, w_+y, w_-y = horizontal directions around the scene object

Interpretation:

* If the manipulated object threatens the scene object mainly by being ABOVE it (for example, spilling liquid or dropping onto it), then w_+z should be the dominant direction and w_-z should be low.
* If the scene object is not meaningfully vulnerable to the manipulated object, then assign a weak risk field with low weights.

IMPORTANT CONSEQUENCE RULE
Do not assign high risk merely because both objects are hazard-related.
Risk should reflect whether the manipulated object meaningfully threatens the scene object.
Examples:

* water over laptop = high risk
* water over kitchen knife = low risk
* power strip over plastic tray = low risk
* wrench over wine glass = high risk

Taxonomy:
Hazard Family (choose 1 primary family, optionally 1 secondary family):
["liquid", "thermal", "sharp", "fragile", "impact", "contamination", "electrical"]

Field Topology (choose 1):

* "upward_vertical_cone": danger mainly from above the scene object; use for spills/drops from above
* "isotropic_sphere": danger from all directions; use for fragile or keep-away objects
* "forward_directional_cone": danger concentrated in one horizontal direction; use for pointed or edged tools
* "planar_half_space": danger mainly lies on one side of a boundary or surface; use for restricted zones or surface-based hazards

Weight guidance:

* All weights must be between 0.0 and 1.0
* 1.0 = strongest nominal danger direction
* 0.0 = no meaningful hazard
* Low-risk interactions should use low weights, not strong weights with arbitrary topology
* For upward_vertical_cone cases involving vulnerable scene objects, w_+z should generally be high and w_-z should generally be low

Contextual attenuation:

* receptacle_attenuation in [0.1, 1.0]
* 1.0: scene object is highly vulnerable to the manipulated object
* 0.4–0.8: scene object has limited vulnerability / weak consequence
* 0.1–0.3: safe receptacle or strongly consequence-reducing context
* If the scene object is a safe receptacle/support (e.g. sink, tray in an appropriate context), use low attenuation and usually low weights as well

Output strictly as a JSON array using this exact schema.
Do not use markdown code fences. Do not include extra text.

[
    {
        "manipulated": "...",
        "scene": "...",
        "families": ["..."],
        "topology_template": "...",
        "weights": {
            "w_+x": 0.0,
            "w_-x": 0.0,
            "w_+y": 0.0,
            "w_-y": 0.0,
            "w_+z": 0.0,
            "w_-z": 0.0
        },
        "receptacle_attenuation": 1.0
    },
    ...
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
