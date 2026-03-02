"""
Semantic Geodesic Risk Fields: zero-shot topological safety for robot policies.
Phase 0: LLM prior (risk score + 6-directional weights).
Phase 1: SONATA segmentation → occupancy → FMM → directional interpolation → V_risk(x).
Phase 2: Risk-aware trajectory optimization (placeholder).
"""

__version__ = "0.1.0"
