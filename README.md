# Semantic Geodesic Risk Fields

Zero-shot topological safety for learned robot policies: 3D semantic risk from SONATA + LLM prior + geodesic distance.

## Architecture (from proposal)

- **Phase 0 — Offline LLM prior**: Object strings → LLM → base risk score + 6-directional decay weights (`w_+x, w_-x, w_+y, w_-y, w_+z, w_-z`).
- **Phase 1 — Real-time risk field**: RGB-D → SONATA (semantic segmentation) → occupancy grid → boundary seeding → FMM (geodesic + Euclidean) → directional interpolation → occlusion shielding → final cost field `V_risk(x)`.
- **Phase 2 — Trajectory optimization**: Risk-aware control (placeholder).

## Repository layout

```
semantic_safety/
├── README.md
├── requirements.txt
├── config/
│   └── default.yaml
├── semantic_safety/           # Main package
│   ├── __init__.py
│   ├── config.py
│   ├── pipeline.py            # Orchestrates Phase 0 → 1 → 2
│   ├── phase0_llm_prior/      # LLM → risk score + 6-directional weights
│   │   ├── __init__.py
│   │   ├── llm_prior.py
│   │   └── prompts.py
│   ├── sonata_integration/    # SONATA wrapper for 3D semantic segmentation
│   │   ├── __init__.py
│   │   └── segmenter.py
│   ├── occupancy/             # Grid, boundary seeding
│   │   ├── __init__.py
│   │   └── grid.py
│   ├── distance/              # FMM: geodesic + Euclidean
│   │   ├── __init__.py
│   │   └── fast_marching.py
│   ├── risk_field/            # W_hazard(x), A(x), V_risk(x)
│   │   ├── __init__.py
│   │   ├── directional.py
│   │   ├── shielding.py
│   │   └── cost.py
│   └── phase2_optimization/   # Placeholder: risk-aware trajectory opt
│       ├── __init__.py
│       └── optimizer.py
├── scripts/
│   └── run_pipeline.py
└── sonata/                     # Clone from https://github.com/facebookresearch/sonata
```

## Setup

1. **Clone SONATA** (required for Phase 1):

   ```bash
   cd /path/to/semantic_safety
   git clone https://github.com/facebookresearch/sonata.git
   # Then install sonata per its README (conda env or pip + deps).
   ```

2. **Python env** (for this repo):

   ```bash
   pip install -r requirements.txt
   ```

3. **LLM**: Set `OPENAI_API_KEY` (or your LLM provider key) for Phase 0.

## Quick run

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/run_pipeline.py --config config/default.yaml
```

## References

- Proposal: *Semantic Geodesic Risk Fields: Zero-Shot Topological Safety for Learned Robot Policies*
- SONATA: [facebookresearch/sonata](https://github.com/facebookresearch/sonata) (CVPR’25)
