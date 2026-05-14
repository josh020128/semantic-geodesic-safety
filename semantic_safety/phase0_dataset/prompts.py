"""Prompt strings for Phase 0 LLM prior and Gemini batch system instruction."""

SYSTEM_INSTRUCTION = """
Evaluate each ordered pair: (manipulated object, scene object).

You are generating a semantic consequence-prior dataset for robotic manipulation.

Important scope:
- This dataset is NOT for collision checking or contact safety.
- Collision and physical contact constraints are handled separately by the occupancy volume.
- Your task is only to describe the semantic consequence field induced by moving the manipulated object near the scene object.

Interpretation:
- manipulated object = the object the robot is holding or moving
- scene object = a nearby object in the environment
- the pair is ordered, so (cup of water, power drill) is different from (power drill, cup of water)

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
  "radius_m": float,
  "attenuation": float
}

Formal meaning of the fields:

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
Examples:
- spilling liquid onto vulnerable electronics from above
- dropping or passing a dangerous object through a direction where consequence would be unacceptable

Do NOT use "inf" for ordinary collision avoidance or generic proximity discomfort.
Those belong to occupancy/collision reasoning, not this dataset.

2) radius_m
radius_m is the spatial decay length scale of the semantic consequence field.
It controls how far the consequence meaningfully extends away from the scene object.

Interpretation:
- larger radius_m = longer spatial influence
- smaller radius_m = more local influence

Rules:
- radius_m must be >= 0.0
- if all six weights are 0.0, set radius_m = 0.0
- otherwise radius_m must be > 0.0
- choose a conservative but physically plausible value
- avoid arbitrarily large radii

Typical tabletop range:
- often between 0.05 and 0.30 meters
- use values outside this range only when truly justified by the pair

3) attenuation
attenuation is a scalar in [0.0, 1.0].
It scales the overall consequence magnitude.

Interpretation:
- 1.0 = no attenuation / Most common case that has at least some semantic hazard consequence
- smaller values = reduced consequence because the scene object tends to absorb, contain, or mitigate the consequence
- 0.0 = effectively negligible semantic consequence

Rules:
- attenuation must be in [0.0, 1.0]
- if all six weights are 0.0, attenuation should usually be 0.0
- do not use attenuation to represent collision safety
- attenuation only represents semantic consequence reduction, not whether contact is mechanically safe

General decision rules:
- Base the answer on common-sense physical interaction.
- Be conservative but physically plausible.
- Use the minimum number of assumptions needed.
- Do not invent hidden categories, roles, or templates.
- Do not output any field other than manipulated, scene, weights, radius_m, attenuation.

How to think about weights:
- If the consequence is roughly symmetric in all directions, use similar values for all six weights.
- If the consequence is strongly directional, assign higher values only to the relevant directions.
- If the consequence is mainly "from above", make w_+z larger than the others.
- If a direction is not meaningful, set its weight to 0.0.
- Do not automatically make w_+z high just because the manipulated object is a liquid.
  Make w_+z high only when the scene object is actually vulnerable to consequence from above.

Important exclusions:
- Do NOT encode collision-only danger here.
- Do NOT treat generic tables, walls, floors, shelves, desks, or counters as risky unless there is a meaningful pair-specific semantic consequence.
- If the scene object is mainly structural context and there is no meaningful semantic consequence for the pair, set all six weights to 0.0, radius_m to 0.0, and attenuation to 0.0.

Output format reminder:
- return one JSON array
- one object per ordered pair
- valid JSON only
- no markdown
- no prose
- no extra keys
"""

RISK_PRIOR_PROMPT = """You are an expert physical reasoning engine for robotics semantic risk.

Evaluate the ordered pair (manipulated object, scene object):
- manipulated: {manipulated}
- scene: {scene}

The coordinate frame is centered on the scene object. Weights encode meaningful hazard along +x, -x, +y, -y, +z (above), -z (below) relative to the scene object. Use values in [0.0, 1.0]. Reflect consequence to the scene object, not mere reachability; if the scene is not meaningfully vulnerable, keep all weights low.

Output strictly one JSON object with these keys (no markdown fences, no extra text):
{{
  "base_risk": <float>,
  "w_plus_x": <float>,
  "w_minus_x": <float>,
  "w_plus_y": <float>,
  "w_minus_y": <float>,
  "w_plus_z": <float>,
  "w_minus_z": <float>
}}
"""