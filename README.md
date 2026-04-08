# Semantic Geodesic Risk Fields

Zero-shot topological safety for learned robot policies: 3D semantic risk from RGB-D perception, an offline LLM prior, boundary-conformal Eikonal distances, and a bounded multi-hazard cost field.

## Architecture (from proposal)

- **Phase 0 — Offline LLM dataset**: Object pairs → LLM → JSON (`data/semantic_risk_demo.json`) with hazard families, field topology, six directional weights, and contextual attenuation. Prompts live in **`semantic_safety/phase0_dataset/prompts.py`**: **`RISK_PRIOR_PROMPT`** (single pair → JSON for **`LLMPrior`** in `generator.py`) and **`GEMINI_SYSTEM_INSTRUCTION`** (batch schema for Gemini). Batch dataset generation: **`scripts/run_phase0.py`**; pair/schema helpers under **`semantic_safety.phase0_dataset`**.
- **Loop 1 — Real-time risk field**  
  - *Perception*: **`perception_2d3d`** — **`SemanticPerception`** in **`lang_sam_wrapper.py`** (SAM mask proposals + Grounding DINO scoring), **`mujoco_camera.py`** for synthetic RGB-D, **`deproject_3d.py`** / **`transform.py`** for camera geometry; **`realsense.py`** for hardware capture when used.  
  - *Semantic router*: **`semantic_router`** — Tier 1 O(1) lookup from the JSON cache; Tier 2 isotropic fallback; Tier 3 async LLM refresh (`router.py`, optional **`gemini_callbacks.py`**).  
  - *Metric propagation*: voxel occupancy from points → **`WorkspaceGrid`** + **`scikit-fmm`** solves the Eikonal equation for **clamped signed distance** from the hazard boundary ∂Ω — **geodesic** with obstacles masked, **unobstructed** baseline for shielding (not centroid-based distance for FMM).  
  - *Risk field*: **`interpolation.py`** expands the six LLM weights into a continuous field **W(x)** using **centroid-relative** directions and L1 blending (stable for non-convex shapes); **`templates.py`** builds analytic template fields from topology / parameters; **`superposition.py`** provides shielding **A(x)**, per-grid **`risk_cost_field`**, and **`compute_logsumexp_superposition`** to merge multiple hazard grids into **V_final(x)** (smooth max, **β** / **v_max** in `config/default.yaml`) without spurious infinite “walls.”
- **Loop 2 — Trajectory evaluation**: Whole-body kinematics and optimization against the dense risk grid (`phase2_control`).
- *Planning utilities*: **`planning/risk_map.py`** — experimental **`GeodesicRiskVolume`** helper (voxel grid, obstacles, risk sampling); separate from the main **`WorkspaceGrid`** path in `metric_propagation`.

`semantic_safety.pipeline` runs Phase 0 and a single-hazard Loop 1 path. Multi-hazard fusion uses **`compute_logsumexp_superposition`** when you have several **V_i** fields. End-to-end MuJoCo → perception → metric → risk demos live under **`scripts/test_full_pipeline.py`** (heavy deps: **MuJoCo**, **OpenCV**, **PyTorch**, **transformers** for SAM / Grounding DINO). **`scripts/test_full_pipeline_viewer.py`** is an optional viewer companion.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── requirements-no-fmm.txt      # Same as requirements.txt minus scikit-fmm
├── environment.yml
├── tabletop.xml                  # MuJoCo scene for synthetic camera tests
├── config/
│   └── default.yaml
├── data/
│   └── semantic_risk_demo.json
├── scripts/
│   ├── run_phase0.py             # Batch Gemini → JSON dataset
│   ├── run_pipeline.py          # CLI: Phase 0 and/or synthetic Loop 1
│   ├── run_loop1_demo.py        # Loop 1 demo (stub)
│   ├── test_semantic_router.py  # Router Tier 1/2/3
│   ├── test_mujoco_camera.py    # MuJoCo RGB-D render
│   ├── test_math_engine.py      # Metric / risk components
│   ├── test_shielding.py
│   ├── test_templates.py         # Risk template fields
│   ├── test_fmm_distance.py
│   ├── test_router.py
│   ├── test_loop1_smoke.py
│   ├── test_perception_candidates.py
│   ├── analyze_loop1_field.py
│   ├── test_full_pipeline.py     # MuJoCo → SAM/DINO → deproject → risk volume (heavy deps)
│   └── test_full_pipeline_viewer.py
└── semantic_safety/
    ├── config.py
    ├── pipeline.py
    ├── phase0_dataset/           # LLM prior, pair generation, prompts.py
    ├── perception_2d3d/
    │   ├── mujoco_camera.py
    │   ├── lang_sam_wrapper.py   # SemanticPerception: SAM + Grounding DINO
    │   ├── realsense.py
    │   ├── deproject_3d.py
    │   └── transform.py
    ├── semantic_router/
    │   ├── router.py             # Tier 1/2/3 routing
    │   └── gemini_callbacks.py   # Optional Gemini batch hook
    ├── metric_propagation/
    │   ├── occupancy_grid.py
    │   └── fmm_distance.py       # WorkspaceGrid (scikit-fmm)
    ├── risk_field/
    │   ├── interpolation.py      # Centroid + L1 → W(x)
    │   ├── templates.py          # Topology-parameterized fields
    │   └── superposition.py      # Shielding, risk_cost_field, LogSumExp fusion
    ├── planning/
    │   └── risk_map.py           # GeodesicRiskVolume (utility / experiments)
    └── phase2_control/
```

## Setup

### Conda environment (recommended)

From the project root:

```bash
cd /path/to/semantic-geodesic-safety
conda env create -f environment.yml
conda activate semantic_safety
```

Core Python deps (including `numpy`, `scipy`, `scikit-fmm`) are installed via the `pip:` section of `environment.yml`; **`requirements.txt`** mirrors them for pip-only / CI setups. If you do not need FMM (**`scikit-fmm`**), you can use **`requirements-no-fmm.txt`** instead and avoid installing that package.

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

**`SemanticPerception`** (`lang_sam_wrapper.py`) uses **PyTorch** and **transformers** (SAM + Grounding DINO). Install those and any camera drivers (e.g. RealSense) for your setup. For minimal pipeline tests without perception, provide `point_cloud["segment"]` yourself or use synthetic labels in **`scripts/run_pipeline.py`**.

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
python scripts/test_mujoco_camera.py
python scripts/test_full_pipeline.py   # needs mujoco, opencv, torch, transformers, etc.
```
