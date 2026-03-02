#!/usr/bin/env python3
"""
Example: run Phase 0 only (LLM prior) or full pipeline with synthetic/sample data.
Usage:
  python scripts/run_pipeline.py --phase0 --manipulated "Water" --scene "Laptop"
  python scripts/run_pipeline.py --config config/default.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root for imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semantic_safety.config import load_config
from semantic_safety.pipeline import run_phase0, run_pipeline
from semantic_safety.phase0_llm_prior import RiskPrior


def main():
    p = argparse.ArgumentParser(description="Semantic Geodesic Risk Fields pipeline")
    p.add_argument("--config", type=str, default=None, help="Path to config YAML")
    p.add_argument("--phase0", action="store_true", help="Run only Phase 0 (LLM prior)")
    p.add_argument("--manipulated", type=str, default="Water", help="Manipulated object name")
    p.add_argument("--scene", type=str, default="Laptop", help="Scene/hazard object name")
    args = p.parse_args()

    config = load_config(args.config)

    if args.phase0:
        prior = run_phase0(args.manipulated, args.scene, config)
        print("Phase 0 — Risk prior:")
        print(f"  base_risk: {prior.base_risk}")
        print(f"  w_+x, w_-x: {prior.w_plus_x}, {prior.w_minus_x}")
        print(f"  w_+y, w_-y: {prior.w_plus_y}, {prior.w_minus_y}")
        print(f"  w_+z, w_-z: {prior.w_plus_z}, {prior.w_minus_z}")
        return

    # Full pipeline needs point cloud; use minimal synthetic data if none provided
    import numpy as np
    n = 5000
    point_cloud = {
        "coord": np.random.randn(n, 3).astype(np.float32) * 0.5,
        "color": np.clip(np.random.rand(n, 3), 0, 1).astype(np.float32),
        "normal": np.random.randn(n, 3).astype(np.float32),
    }
    point_cloud["normal"] /= np.linalg.norm(point_cloud["normal"], axis=1, keepdims=True)
    hazard_label = 1
    centroid = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    out = run_pipeline(
        point_cloud,
        manipulated_object=args.manipulated,
        scene_object=args.scene,
        hazard_label=hazard_label,
        centroid=centroid,
        config_path=args.config,
    )
    print("Phase 1 output keys:", list(out.keys()))
    print("V_risk shape:", out["V_risk"].shape)
    print("V_risk min/max:", out["V_risk"].min(), out["V_risk"].max())


if __name__ == "__main__":
    main()
