"""
Semantic Geodesic Risk Fields: zero-shot topological safety for robot policies.
Phase 0: offline LLM dataset (risk score + 6-directional weights).
Loop 1: perception → occupancy → FMM → directional interpolation → V_risk(x).
Loop 2: risk-aware trajectory optimization (phase2_control).
"""

__version__ = "0.1.0"
