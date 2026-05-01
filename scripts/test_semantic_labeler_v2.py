#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw

try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_IMPORT_ERROR = e
else:
    _TORCH_IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic_safety.perception_2d3d.semantic_labeler_v2 import (  # noqa: E402
    PrototypeBankV2,
    SemanticLabelerV2,
)


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for test_semantic_labeler_v2.py but could not be imported. "
            f"Original import error: {_TORCH_IMPORT_ERROR}"
        )


def _load_rgb(path: str) -> np.ndarray:
    rgb = np.array(Image.open(path).convert("RGB"))
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image at {path}, got shape {rgb.shape}")
    return rgb


def _load_masks(path: str) -> np.ndarray:
    masks = np.load(path)
    if masks.ndim != 3:
        raise ValueError(f"Expected masks [N,H,W], got shape {masks.shape}")
    return (masks > 0).astype(np.uint8)


def _load_bboxes(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"Expected bboxes [N,4], got shape {arr.shape}")
    return arr.astype(np.float32)


def _instance_similarity(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings [N,C], got {embeddings.shape}")
    if embeddings.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float32)
    embs = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8, None)
    sims = embs @ embs.T
    return sims.astype(np.float32)


def _draw_label_overlay(
    rgb: np.ndarray,
    masks: np.ndarray,
    bboxes: Optional[np.ndarray],
    label_assignments: Optional[list[dict[str, Any]]],
    out_path: str | Path,
) -> None:
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)

    if label_assignments is None:
        label_assignments = [
            {"label": f"obj_{i:02d}", "score": None, "prototype_index": None}
            for i in range(masks.shape[0])
        ]

    for i in range(masks.shape[0]):
        row = label_assignments[i]
        label = str(row.get("label", f"obj_{i:02d}"))
        score = row.get("score", None)
        text = f"{i}:{label}" if score is None else f"{i}:{label} ({float(score):.2f})"

        if bboxes is not None and i < bboxes.shape[0]:
            x1, y1, x2, y2 = [int(round(v)) for v in bboxes[i].tolist()]
        else:
            ys, xs = np.where(masks[i] > 0)
            if xs.size == 0 or ys.size == 0:
                continue
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())

        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        ty = max(0, y1 - 14)
        draw.text((x1, ty), text, fill=(0, 255, 0))

    img.save(out_path)


def main() -> None:
    _require_torch()

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to RGB image")
    ap.add_argument("--masks", required=True, help="Path to sam_masks.npy [N,H,W]")
    ap.add_argument("--out_dir", required=True, help="Output directory")

    ap.add_argument("--bboxes", default=None, help="Optional path to bboxes.npy [N,4]")
    ap.add_argument("--prototype_bank", default=None, help="Optional .npz prototype bank for label assignment")
    ap.add_argument("--device", default=None, help="cuda|cpu|mps (default: auto)")
    ap.add_argument("--dino_target_h", type=int, default=500, help="Resize height for DINO features")
    ap.add_argument("--n_samples", type=int, default=0, help="If >0, save random pooled feature samples per mask")
    ap.add_argument("--similarity_threshold", type=float, default=0.35, help="Prototype assignment threshold")
    ap.add_argument("--unknown_prefix", type=str, default="obj", help="Prefix for unmatched instances")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb = _load_rgb(args.image)
    masks_np = _load_masks(args.masks)
    bboxes_np = _load_bboxes(args.bboxes)

    if bboxes_np is not None and bboxes_np.shape[0] != masks_np.shape[0]:
        raise ValueError(
            f"Number of bboxes ({bboxes_np.shape[0]}) does not match number of masks ({masks_np.shape[0]})"
        )

    masks_t = torch.from_numpy(masks_np.astype(bool))

    t0 = time.time()
    labeler = SemanticLabelerV2(device=args.device)
    init_time = time.time() - t0

    t1 = time.time()
    prototype_bank = PrototypeBankV2.load_npz(args.prototype_bank) if args.prototype_bank else None
    compute_out = labeler.compute_and_assign(
        rgb_np=rgb,
        masks_hw=masks_t,
        prototype_bank=prototype_bank,
        target_h=int(args.dino_target_h),
        n_samples=int(args.n_samples),
        similarity_threshold=float(args.similarity_threshold),
        unknown_prefix=str(args.unknown_prefix),
    )
    compute_time = time.time() - t1

    dino_embeddings = np.asarray(compute_out["dino_embeddings"], dtype=np.float32)
    dino_embedding_samples = np.asarray(compute_out["dino_embedding_samples"], dtype=np.float32)
    label_assignments = compute_out.get("label_assignments", None)

    np.save(out_dir / "dino_embeddings.npy", dino_embeddings)
    if args.n_samples > 0:
        np.save(out_dir / "dino_embedding_samples.npy", dino_embedding_samples)

    sims = _instance_similarity(dino_embeddings)
    np.save(out_dir / "instance_similarity.npy", sims)

    if label_assignments is not None:
        with open(out_dir / "label_assignments.json", "w", encoding="utf-8") as f:
            json.dump(label_assignments, f, indent=2, ensure_ascii=False)

    _draw_label_overlay(
        rgb=rgb,
        masks=masks_np,
        bboxes=bboxes_np,
        label_assignments=label_assignments,
        out_path=out_dir / "semantic_label_overlay.png",
    )

    meta = {
        "image_path": str(args.image),
        "masks_path": str(args.masks),
        "bboxes_path": str(args.bboxes) if args.bboxes else None,
        "prototype_bank": str(args.prototype_bank) if args.prototype_bank else None,
        "num_instances": int(masks_np.shape[0]),
        "device": labeler.device,
        "dino_target_h": int(args.dino_target_h),
        "n_samples": int(args.n_samples),
        "similarity_threshold": float(args.similarity_threshold),
        "timings": {
            "init_labeler": init_time,
            "compute_embeddings_and_assign": compute_time,
            "total": init_time + compute_time,
        },
    }
    with open(out_dir / "semantic_labeler_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved semantic labeler outputs for {masks_np.shape[0]} instance(s) to {out_dir}")
    print(f" - dino_embeddings.npy: {dino_embeddings.shape}")
    print(f" - instance_similarity.npy: {sims.shape}")
    if label_assignments is not None:
        print(" - label_assignments.json")
    print(" - semantic_label_overlay.png")


if __name__ == "__main__":
    main()
