"""Prompt templates for Phase 0: risk score and 6-directional decay weights."""

RISK_PRIOR_PROMPT = """You are a robot safety expert. Given a manipulated object and a scene object (potential hazard), output a base risk score and directional decay weights.

**Input:**
- Manipulated object: {manipulated}
- Scene object (hazard): {scene}

**Output format (JSON only, no markdown):**
{{
  "base_risk": <float 0.0-1.0, overall danger of interaction>,
  "w_plus_x": <float 0.0-1.0, risk from hazard when approaching from +X / right>,
  "w_minus_x": <float 0.0-1.0, risk when approaching from -X / left>,
  "w_plus_y": <float 0.0-1.0, risk when approaching from +Y / front>,
  "w_minus_y": <float 0.0-1.0, risk when approaching from -Y / back>,
  "w_plus_z": <float 0.0-1.0, risk when approaching from +Z / top>,
  "w_minus_z": <float 0.0-1.0, risk when approaching from -Z / bottom>
}}

Example: For "manipulated: Water, scene: Laptop" — spill from above is dangerous (high +Z), from below is safe (low -Z). Only output the JSON object."""
