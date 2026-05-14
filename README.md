# Semantic Geodesic Risk Fields

**Semantic risk on a voxel grid from RGB-D (or MuJoCo synthetic camera), geodesic distances around obstacles, and a soft directional hazard field** fused from an offline JSON prior. This repository pairs that **Loop 1** risk volume with **UR5 trajectory / IK tooling** for experiments.

---

## Main entry points (current)

These are the primary scripts to run end-to-end or robot-side workflows:

| Script | Role |
|--------|------|
| **`scripts/test_full_siglip2_pipeline.py`** | **Loop 1 demo:** MuJoCo (or RealSense) → **MobileSAM v2 proposals** + **SigLIP2** labeling → voxel occupancy → FMM geodesics → **soft risk field** + optional **+z “inf” gravity column** → saves `loop1_risk_field.npz` and overlays. Monkey-patches the detector; **all risk math lives in** `scripts/test_full_pipeline.py`. |
| **`scripts/ur5_solve_pyroki.py`** | **UR5 IK (PyRoki):** loads processed Cartesian pose sequences (`*.npy` homogeneous transforms), solves IK with **`semantic_safety/ur5_experiment/pyroki_solver.py`**, writes joint trajectories and JSON metrics under `out/ur5_pyroki_ik/` (default). |
| **`scripts/ur5_solve_ik.py`** | **UR5 IK (MuJoCo damped least squares):** same style of pose files, solves with **`semantic_safety/ur5_experiment/ik_solver.py`** on **`semantic_safety/ur5_experiment/mujoco_ur5_env.py`**, writes `out/ur5_ik/` (default). |

**Typical Loop 1 command:**

```bash
cd /path/to/semantic-geodesic-safety
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/test_full_siglip2_pipeline.py --xml-path tabletop.xml --manipulated "cup of water" --scene-label "power drill"
```

Use `--help` on each script for flags (prior JSON, SAM/SigLIP options, `--time-risk-voxel-map` / `--no-time-risk-voxel-map` for backend timing, etc.).

---

## What the main Loop 1 stack uses (summary)

### `scripts/test_full_siglip2_pipeline.py` (frontend only in-repo)

- **`semantic_safety/perception_2d3d/mobilesamv2_wrapper_v2.py`** — proposals / masks.
- **`semantic_safety/perception_2d3d/siglip2_wrapper_v2.py`**, **`siglip2_labeler_v2.py`**, **`lvis_bank_v2.py`**, **`instance_semantic_siglip2_frontend_v2.py`** — open-vocabulary labeling on crops.

### `scripts/test_full_pipeline.py` (loaded by SigLIP2 script; also runnable standalone)

**Perception / geometry**

- **`semantic_safety/perception_2d3d/mujoco_camera.py`** or **`realsense.py`** — images + depth (+ intrinsics).
- **`semantic_safety/perception_2d3d/transform.py`** — `WorldTransform`, projection helpers.
- **`semantic_safety/perception_2d3d/lang_sam_wrapper.py`** — default **`SemanticPerception`** when you run **`test_full_pipeline.py` directly** (Grounding DINO + SAM-style path), **not** used when SigLIP2 monkey-patches the class.

**Prior & metrics & risk**

- **`semantic_safety/semantic_router/router.py`** — JSON prior lookup for `(manipulated, scene)` → **`weights`** + **`sigma_m`** (minimal schema; legacy keys are not emitted).
- **`semantic_safety/phase0_dataset/prompts.py`** — `SYSTEM_INSTRUCTION` for optional LLM hooks wired in the router (usually disabled in demos).
- **`semantic_safety/metric_propagation/fmm_distance.py`** — **`WorkspaceGrid`**, boundary-seeded **geodesic / Euclidean** distances for shielding.
- **`semantic_safety/risk_field/templates.py`** — **`build_risk_field_from_params`**: directional coefficients, **Gaussian** \(\exp(-d_{\mathrm{geo}}^2/(2\sigma^2))\), finite vs **`"inf"`** axis weights.
- **`semantic_safety/risk_field/superposition.py`** — **`shielding_ratio`**, **`compute_sum_superposition`** (and optional log-sum-exp helper).

**Data / assets**

- MuJoCo XML (e.g. **`tabletop.xml`**, or others you pass with `--xml-path`).
- Prior JSON, default **`data/semantic_risk_demo_claude.json`** (`manipulated`, `scene`, `weights`, `sigma_m`; **`"inf"`** on an axis encodes hard / column semantics handled in the pipeline).

**Artifacts written (working directory)**

- **`loop1_risk_field.npz`** — `risk_field`, grid axes, `occupancy_free`, `semantic_hard_mask` (reserved / zeros in current soft-column mode), metadata.
- **`loop1_scene_objects.json`**, **`test_rgb.png`**, **`test_depth_debug.png`**, **`perception_debug/`**, overlay PNGs — see script help / base pipeline tail.

### UR5 IK scripts

- **`semantic_safety/ur5_experiment/pyroki_solver.py`** — PyRoki-based solver config + sequence IK (`ur5_solve_pyroki.py`).
- **`semantic_safety/ur5_experiment/ik_solver.py`**, **`mujoco_ur5_env.py`** — MuJoCo model wrapper + damped least-squares IK (`ur5_solve_ik.py`).
- **`semantic_safety/ur5_experiment/trajectory.py`**, **`workspace_astar.py`**, **`risk_volume_query.py`** — used by **other** UR5 / planning scripts in `scripts/` (path generation, grid shortest-path search, risk sampling), **not** imported by the two `ur5_solve_*.py` CLIs above (those only need the solver + env modules listed).

Pose inputs for the IK CLIs are usually produced by **`scripts/ur5_process_trajectory.py`** (and related **`ur5_plan_cartesian.py`**, **`ur5_debug_risk_volume.py`**, etc.) — treat those as **pipeline helpers** next to the main three.

---

## Architecture (concise)

The pipeline ultimately produces **two aligned voxel 3D maps** on the same **`WorkspaceGrid`** (same resolution and bounds):

| Map | Role |
|-----|------|
| **Geometry / collision occupancy** | Boolean **free vs occupied** volume (global and, per hazard, local variants). Used for **collision-style reasoning**, obstacle-aware **FMM** geodesics, and shielding. Exported as **`occupancy_free`** in `loop1_risk_field.npz` (and used internally for distance transforms). |
| **Semantic risk field** | Scalar **hazard intensity** per voxel after directional weights, Gaussian decay in \(d_{\mathrm{geo}}\), optional **+\(z\) `"inf"`** column, and **multi-object superposition**. Exported as **`risk_field`** (`V_{\mathrm{final}}`) in `loop1_risk_field.npz`. |

**Processing flow**

1. **Detect / segment** (SigLIP2 path or legacy LangSAM path in `test_full_pipeline.py`).
2. **Back-project** masks + depth → 3D points; **voxelize** on the grid → populate the **occupancy / collision map** (obstacles + table clipping, etc.).
3. **Per-hazard local occupancy** (source vs dilated blockers); **FMM** for \(d_{\mathrm{geo}}\), \(d_{\mathrm{euc}}\) → shielding **\(A(x)\)** — all defined on the same grid as the occupancy map.
4. **Semantic risk** on that grid: \(V_{\mathrm{soft}} \approx W(x)\,A(x)\,\exp(-d_{\mathrm{geo}}^2/(2\sigma^2))\); **+\(z\) weight `"inf"`** adds a **persistent upward column** merged with **`np.maximum`** against the soft field.
5. **Superpose** hazards (sum with cap) → **`risk_field`**; export **both** maps (plus axes, metadata, optional `semantic_hard_mask`) in **`loop1_risk_field.npz`**.

---

## Repository layout (trimmed)

```
.
├── README.md
├── requirements.txt
├── requirements-no-fmm.txt
├── environment.yml
├── *.xml                         # MuJoCo scenes (e.g. tabletop.xml)
├── config/default.yaml
├── data/
│   ├── semantic_risk_demo_claude.json   # primary prior example
│   └── ...
├── scripts/
│   ├── test_full_siglip2_pipeline.py    # MAIN: Loop 1 + SigLIP2
│   ├── test_full_pipeline.py            # MAIN backend for Loop 1 (loaded by SigLIP2 script)
│   ├── ur5_solve_pyroki.py              # MAIN: PyRoki IK
│   ├── ur5_solve_ik.py                  # MAIN: MuJoCo DLS IK
│   └── ...                              # helpers / tests (see below)
└── semantic_safety/
    ├── phase0_dataset/           # prompts, optional LLM dataset tools
    ├── perception_2d3d/          # cameras, transforms, SigLIP2 / MobileSAM, lang_sam
    ├── semantic_router/          # router.py (+ optional callbacks)
    ├── metric_propagation/       # fmm_distance.py (WorkspaceGrid)
    ├── risk_field/               # templates.py, superposition.py, interpolation.py (legacy tests)
    ├── ur5_experiment/           # PyRoki + MuJoCo IK + trajectory / A* / risk query helpers
    ├── planning/                 # experimental risk_map utilities
    ├── phase2_control/           # placeholder / future control
    └── pipeline.py               # older orchestration demo (not the SigLIP2 main path)
```

---

## Debugging, legacy, and auxiliary scripts

The following are **not** required for the three main entry points. They were used for **component tests, older perception stacks, one-off analysis, or UR5 side experiments**:

| Area | Examples |
|------|-----------|
| **Unit / smoke tests** | `scripts/test_semantic_router.py`, `test_mujoco_camera.py`, `test_math_engine.py`, `test_fmm_distance.py`, `test_templates.py`, `test_shielding.py`, `test_router.py`, `test_loop1_smoke.py`, `test_perception_candidates.py`, … |
| **SigLIP2 / frontend-only sandboxes** | `scripts/test_siglip2_labeler_v2.py`, `test_instance_semantic_siglip2_frontend_v2.py`, `test_instance_frontend_v2.py`, `test_semantic_labeler_v2.py` |
| **Alternate or experimental pipelines** | `scripts/test_full_pipeline_v2.py`, `test_full_pipeline_viewer.py`, `_patch_test_full_pipeline.py`, `scripts/run_pipeline.py`, `semantic_safety/pipeline.py` |
| **Loop 1 analysis** | `scripts/analyze_loop1_field.py` |
| **Phase 0 dataset (LLM → JSON)** | `scripts/run_phase0.py`, `run_phase0_claude.py` — regenerate priors, separate from live Loop 1 |
| **UR5 extras** | `ur5_plan_cartesian.py`, `ur5_process_trajectory.py`, `ur5_run_static_ik.py`, `ur5_run_static_pyroki.py`, `ur5_replay_trajectory.py`, `ur5_viewer_replay.py`, `ur5_visualize_paths.py`, `ur5_check_pyroki.py`, `ur5_debug_env.py`, `ur5_debug_risk_volume.py`, `scripts/test.py` |

**`semantic_safety/risk_field/interpolation.py`** — older “centroid + L1” directional field helpers; **current `templates.py` path does not depend on it** for `test_full_siglip2_pipeline` / `test_full_pipeline` risk construction. It remains for **`test_math_engine.py`** and historical notebooks.

---

## Setup

### Conda (recommended)

```bash
cd /path/to/semantic-geodesic-safety
conda env create -f environment.yml
conda activate semantic_safety
```

Core deps include **NumPy**, **SciPy**, **OpenCV**, **MuJoCo**, **scikit-fmm**, **PyTorch** / **transformers** (for SigLIP2 and MobileSAM). Use **`requirements-no-fmm.txt`** only if you intentionally omit FMM.

### Optional API keys (Phase 0 or router LLM refresh)

- **Gemini**: `GOOGLE_API_KEY` (see [Google AI Studio](https://aistudio.google.com/apikey)).
- **OpenAI**: `OPENAI_API_KEY` if you switch provider in config.
- **Anthropic (Claude)**: `ANTHROPIC_API_KEY` when using Claude-backed batch hooks (e.g. `semantic_safety/semantic_router/claude_callbacks.py` or Phase 0 / prior expansion that calls the Anthropic API). See [Anthropic console](https://console.anthropic.com/) for keys.

---

## Optional: regenerate the prior JSON

Batch or single-pair LLM tools live under **`semantic_safety/phase0_dataset/`** and **`scripts/run_phase0.py`** / **`run_phase0_claude.py`**. The live demo reads a checked-in file such as **`data/semantic_risk_demo_claude.json`**; you do **not** need to rerun Phase 0 to try Loop 1.

---

## License / citation

(Add your paper or license text here if applicable.)
