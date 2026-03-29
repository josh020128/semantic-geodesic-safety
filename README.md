# Semantic Geodesic Risk Fields

Zero-shot topological safety for learned robot policies: 3D semantic risk from RGB-D perception, LLM prior, and geodesic distance.

## Architecture (from proposal)

- **Phase 0 — Offline LLM dataset**: Object strings → LLM → base risk score + 6-directional decay weights (`w_+x, w_-x, w_+y, w_-y, w_+z, w_-z`), written to JSON (e.g. under `data/`). Batched generation lives in `semantic_safety.phase0_dataset`.
- **Loop 1 — Real-time risk field**: Perception (`perception_2d3d`: RealSense, 2D grounding, 3D deprojection) → optional fast/slow semantic routing (`semantic_router`) → occupancy grid → boundary seeding → FMM (geodesic + Euclidean) → directional interpolation → shielding → final cost field `V_risk(x)` (`risk_field`, `metric_propagation`).
- **Loop 2 — Trajectory evaluation**: Whole-body kinematics and local trajectory optimization against the risk grid (`phase2_control`).

`semantic_safety.pipeline` orchestrates Phase 0 and Loop 1; Loop 2 is consumed via `phase2_control`.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── config/
│   └── default.yaml
├── data/                           # Offline caches and priors
│   └── semantic_risk_demo.json     # JSON dataset produced by Phase 0 (placeholder)
├── scripts/
│   ├── run_phase0.py               # Generates the JSON dataset (stub)
│   ├── run_loop1_demo.py           # Real-time perception & risk grid (stub)
│   └── run_pipeline.py             # CLI: Phase 0 and/or synthetic Loop 1
└── semantic_safety/                # Main package
    ├── __init__.py
    ├── config.py
    ├── pipeline.py                 # Master orchestrator for Loop 1 & hooks to Loop 2
    ├── phase0_dataset/             # Offline generation only
    │   ├── __init__.py
    │   ├── generator.py            # Batched API logic for dataset creation
    │   └── prompts.py              # Meta-prompts (3-layer taxonomy)
    ├── perception_2d3d/            # Loop 1 “eyes” (stubs)
    │   ├── __init__.py
    │   ├── realsense.py
    │   ├── segment_2d.py
    │   └── deproject_3d.py
    ├── semantic_router/            # Fast/slow brain (stubs)
    │   ├── __init__.py
    │   ├── router.py
    │   ├── embeddings.py
    │   └── slow_brain.py
    ├── metric_propagation/         # Grid and distances
    │   ├── __init__.py
    │   ├── occupancy_grid.py
    │   └── fmm_distance.py
    ├── risk_field/                 # Math engine
    │   ├── __init__.py
    │   ├── interpolation.py        # Discrete 6-directional weights → continuous field
    │   └── superposition.py        # Shielding and V_risk composition
    └── phase2_control/             # Loop 2
        ├── __init__.py
        ├── kinematics.py           # Stub: whole-body / tilt penalty
        └── optimizer.py            # Local trajectory optimizer (placeholder)
```

## Setup

### Conda environment (recommended)

From the project root:

```bash
cd /path/to/semantic-geodesic-safety
conda env create -f environment.yml
conda activate semantic_safety
```

To **recreate** the env (e.g. after removing it):

```bash
conda env remove -n semantic_safety
conda env create -f environment.yml
conda activate semantic_safety
```

### Optional: API key for Phase 0 (LLM)

- **Gemini** (default in config): set `GOOGLE_API_KEY` (see [Google AI Studio](https://aistudio.google.com/apikey)).
- **OpenAI**: set `OPENAI_API_KEY` and use `provider: openai` in `config/default.yaml`.

```bash
conda activate semantic_safety
conda env config vars set GOOGLE_API_KEY=your_key_here
conda activate semantic_safety   # reactivate to apply
```

### Perception stack (Loop 1)

RealSense, Lang-SAM / Grounded-SAM, and Open3D wiring will live under `semantic_safety.perception_2d3d` once implemented. Until then, provide `point_cloud["segment"]` yourself or use synthetic labels in scripts.

## Quick run

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/run_pipeline.py --config config/default.yaml
```

Phase 0 only:

```bash
python scripts/run_pipeline.py --phase0 --manipulated "Water" --scene "Laptop"
```

## References

- Proposal: *Semantic Geodesic Risk Fields: Zero-Shot Topological Safety for Learned Robot Policies*
