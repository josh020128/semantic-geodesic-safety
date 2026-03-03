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

### Conda environment (recommended)

From the project root:

```bash
cd /Users/giunglee/Documents/semantic_safety   # or your path
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

- **Gemini** (default in config): set `GOOGLE_API_KEY` (get a key at [Google AI Studio](https://aistudio.google.com/apikey)).
- **OpenAI**: set `OPENAI_API_KEY` and use `provider: openai` in `config/default.yaml`.

You can set the key only in this env:

```bash
conda activate semantic_safety
conda env config vars set GOOGLE_API_KEY=your_key_here
conda activate semantic_safety   # reactivate to apply
```

### SONATA (for Phase 1 only)

Clone and install SONATA when you need 3D segmentation:

```bash
git clone https://github.com/facebookresearch/sonata.git
# Then install sonata per its README (separate conda/pip deps).
```

## Quick run

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/run_pipeline.py --config config/default.yaml
```

## References

- Proposal: *Semantic Geodesic Risk Fields: Zero-Shot Topological Safety for Learned Robot Policies*
- SONATA: [facebookresearch/sonata](https://github.com/facebookresearch/sonata) (CVPR’25)
