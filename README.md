# Semantic Geodesic Risk Fields

Zero-shot topological safety for learned robot policies: 3D semantic risk from RGB-D perception, an offline LLM prior, boundary-conformal Eikonal distances, and a bounded multi-hazard cost field.

## Architecture (from proposal)

- **Phase 0 — Offline LLM dataset**: Object pairs → LLM → JSON (`data/semantic_risk_demo.json`) with hazard families, field topology, six directional weights, and contextual attenuation. Batch generation: `scripts/run_phase0.py`, schema/prompts under `semantic_safety.phase0_dataset`.
- **Loop 1 — Real-time risk field**  
  - *Perception (planned)*: `perception_2d3d` — RealSense, open-vocabulary 2D segmentation, depth → point cloud / boundary.  
  - *Semantic router*: `semantic_router` — Tier 1 O(1) lookup from the JSON cache; Tier 2 isotropic fallback; Tier 3 async LLM refresh (`router.py`).  
  - *Metric propagation*: voxel occupancy from points → **`WorkspaceGrid`** + **`scikit-fmm`** solves the Eikonal equation for **clamped signed distance** from the hazard boundary ∂Ω — **geodesic** with obstacles masked, **unobstructed** baseline for shielding (not centroid-based distance for FMM).  
  - *Risk field*: **`interpolation.py`** expands the six LLM weights into a continuous field **W(x)** using **centroid-relative** directions and L1 blending (stable for non-convex shapes); **`superposition.py`** provides shielding **A(x)**, per-grid **`risk_cost_field`**, and **`compute_logsumexp_superposition`** to merge multiple hazard grids into **V_final(x)** (smooth max, **β** / **v_max** in `config/default.yaml`) without spurious infinite “walls.”
- **Loop 2 — Trajectory evaluation**: Whole-body kinematics and optimization against the dense risk grid (`phase2_control`).

`semantic_safety.pipeline` runs Phase 0 and a single-hazard Loop 1 path; multi-hazard fusion is available via `compute_logsumexp_superposition` when you have several **V_i** fields.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── config/
│   └── default.yaml
├── data/
│   └── semantic_risk_demo.json     # Phase 0 output (regenerate with scripts/run_phase0.py)
├── scripts/
│   ├── run_phase0.py               # Batch Gemini → JSON dataset
│   ├── run_pipeline.py             # CLI: Phase 0 and/or synthetic Loop 1
│   ├── run_loop1_demo.py         # Loop 1 demo (stub)
│   └── test_semantic_router.py     # Router Tier 1/2/3 smoke test
└── semantic_safety/
    ├── config.py
    ├── pipeline.py
    ├── phase0_dataset/
    │   ├── generator.py
    │   ├── prompts.py
    │   └── pair_generator.py
    ├── perception_2d3d/            # Loop 1 perception (stubs)
    ├── semantic_router/
    │   └── router.py               # Tier 1/2/3 routing
    ├── metric_propagation/
    │   ├── occupancy_grid.py
    │   └── fmm_distance.py        # WorkspaceGrid (scikit-fmm)
    ├── risk_field/
    │   ├── interpolation.py       # Centroid + L1 → W(x)
    │   └── superposition.py       # Shielding, risk_cost_field, LogSumExp fusion
    └── phase2_control/
        ├── kinematics.py
        └── optimizer.py
```

## Setup

### Conda environment (recommended)

From the project root:

```bash
cd /path/to/semantic-geodesic-safety
conda env create -f environment.yml
conda activate semantic_safety
```

Core Python deps (including `numpy`, `scipy`, `scikit-fmm`) are installed via the `pip:` section of `environment.yml`; `requirements.txt` mirrors them for pip-only / CI setups.

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

RealSense, Lang-SAM / Grounded-SAM, and Open3D wiring will live under `semantic_safety.perception_2d3d` once implemented. Until then, provide `point_cloud["segment"]` yourself or use synthetic labels in `scripts/run_pipeline.py`.

## Quick run

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/run_pipeline.py --config config/default.yaml
```

Phase 0 only:

```bash
python scripts/run_pipeline.py --phase0 --manipulated "Water" --scene "Laptop"
```

Semantic router smoke test:

```bash
python scripts/test_semantic_router.py
```

## References

- Proposal: *Semantic Geodesic Risk Fields: Zero-Shot Topological Safety for Learned Robot Policies*
