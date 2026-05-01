from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
from PIL import Image

try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_IMPORT_ERROR = e
else:
    _TORCH_IMPORT_ERROR = None

from semantic_safety.perception_2d3d.mobilesamv2_wrapper_v2 import MobileSAMV2WrapperV2
from semantic_safety.perception_2d3d.semantic_labeler_v2 import (
    PrototypeBankV2,
    SemanticLabelerV2,
    labels_from_json_prior_scenes,
)


OpenVocabCallback = Callable[
    [np.ndarray, dict[str, Any], Sequence[str]],
    dict[str, Any],
]


class InstanceSemanticFrontendV2:
    """
    New instance-level perception frontend for semantic_safety.

    Design goals:
      1) keep the current downstream pipeline unchanged
      2) replace only the RGB-image semantic perception front-end
      3) return detections in a schema compatible with test_full_pipeline.py

    Returned detection schema matches what the current pipeline expects:
      {
        "label": str,
        "score": float,
        "box": {"xmin","ymin","xmax","ymax"},
        "mask": np.ndarray[H,W] bool,
        "source": str,
        "top_k": list[[label, score], ...]
      }

    Notes:
    - Closed-set first: if a prototype bank is available, try assigning labels from it.
    - Unknown instances may optionally go through an open-vocabulary fallback callback.
    - If neither applies, detections are returned as unknown_object_* labels.
    """

    def __init__(
        self,
        *,
        prior_json_path: Optional[str] = None,
        prototype_bank_path: Optional[str] = None,
        mobilesam_wrapper: Optional[MobileSAMV2WrapperV2] = None,
        semantic_labeler: Optional[SemanticLabelerV2] = None,
        open_vocab_callback: Optional[OpenVocabCallback] = None,
        device: Optional[str] = None,
        sam_size: int = 1024,
        yolo_conf: float = 0.40,
        yolo_iou: float = 0.90,
        min_mask_size: int = 1000,
        max_batch_size: int = 320,
        enable_post_merge: bool = False,  # reserved for future wrapper sync
        closed_set_similarity_threshold: float = 0.35,
        dino_target_h: int = 500,
        unknown_prefix: str = "unknown_object",
    ) -> None:
        if torch is None:
            raise RuntimeError(
                "PyTorch is required for InstanceSemanticFrontendV2. "
                f"Original import error: {_TORCH_IMPORT_ERROR}"
            )

        self.prior_json_path = prior_json_path
        self.prototype_bank_path = prototype_bank_path
        self.open_vocab_callback = open_vocab_callback
        self.closed_set_similarity_threshold = float(closed_set_similarity_threshold)
        self.dino_target_h = int(dino_target_h)
        self.unknown_prefix = str(unknown_prefix)

        self.mobility_kwargs = {
            "sam_size": int(sam_size),
            "yolo_conf": float(yolo_conf),
            "yolo_iou": float(yolo_iou),
            "min_mask_size": int(min_mask_size),
            "max_batch_size": int(max_batch_size),
            "device": device,
        }
        self.enable_post_merge = bool(enable_post_merge)

        self.mobilesam = mobilesam_wrapper or MobileSAMV2WrapperV2(**self.mobility_kwargs)
        self.semantic_labeler = semantic_labeler or SemanticLabelerV2(device=device)

        self.closed_set_labels: list[str] = []
        if self.prior_json_path is not None and Path(self.prior_json_path).exists():
            self.closed_set_labels = labels_from_json_prior_scenes(self.prior_json_path)

        self.prototype_bank: Optional[PrototypeBankV2] = None
        if self.prototype_bank_path is not None and Path(self.prototype_bank_path).exists():
            self.prototype_bank = PrototypeBankV2.load_npz(self.prototype_bank_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_scene_objects(
        self,
        image_path: str,
        candidate_labels: Sequence[str],
        max_proposals: int = 25,   # kept for compatibility; wrapper currently ignores this
        max_results: int = 30,
        save_debug: bool = False,
        debug_dir: str = "perception_debug_v2",
        include_unknown: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Drop-in replacement for SemanticPerception.detect_scene_objects(...).

        Compatibility notes:
        - candidate_labels is still accepted because downstream code passes it.
        - current v2 frontend uses candidate_labels mainly for open-vocab fallback filtering
          or optional result pruning.
        """
        del max_proposals  # current wrapper does not use proposal count directly

        rgb = np.array(Image.open(image_path).convert("RGB"))
        results = self.mobilesam.process_image(rgb, verbose=False)

        masks_t = results.sam_masks
        boxes_t = results.input_boxes
        scores_t = results.box_scores

        if masks_t.shape[0] == 0:
            return []

        label_info = self._assign_instance_labels(
            rgb_np=rgb,
            masks=masks_t,
            candidate_labels=candidate_labels,
            include_unknown=include_unknown,
        )

        detections: list[dict[str, Any]] = []
        for i in range(masks_t.shape[0]):
            label_row = label_info[i]
            label = str(label_row["label"])
            score = float(label_row["score"])

            # If user provided candidate_labels, prefer results that match that space.
            # Unknowns may still pass through if include_unknown=True.
            if candidate_labels:
                cand_norm = {c.strip().lower() for c in candidate_labels}
                if label.strip().lower() not in cand_norm and not label.startswith(self.unknown_prefix):
                    # keep open-vocab fallback results only if they match the candidate set
                    continue

            if (not include_unknown) and label.startswith(self.unknown_prefix):
                continue

            box = boxes_t[i].detach().cpu().numpy().tolist()
            mask = masks_t[i].detach().cpu().numpy().astype(bool)

            top_k = label_row.get("top_k", [])
            detections.append(
                {
                    "label": label,
                    "score": score,
                    "box": {
                        "xmin": int(round(box[0])),
                        "ymin": int(round(box[1])),
                        "xmax": int(round(box[2])),
                        "ymax": int(round(box[3])),
                    },
                    "mask": mask,
                    "source": str(label_row.get("source", "instance_semantic_frontend_v2")),
                    "top_k": top_k,
                }
            )

        detections = sorted(detections, key=lambda d: float(d["score"]), reverse=True)[:max_results]

        if save_debug:
            self._save_debug_outputs(
                rgb_np=rgb,
                detections=detections,
                debug_dir=debug_dir,
                timings=results.timings,
            )

        return detections

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assign_instance_labels(
        self,
        *,
        rgb_np: np.ndarray,
        masks,
        candidate_labels,
        include_unknown: bool,
    ):
        """
        Closed-set first, open-vocab second.

        If no prototype bank is available, do NOT run DINO at all.
        Just create unknown placeholders, then optionally run open-vocab fallback.
        """
        rows = []

        # ------------------------------------------------------------------
        # FAST PATH: no prototype bank -> skip DINO completely
        # ------------------------------------------------------------------
        if self.prototype_bank is None:
            for i in range(masks.shape[0]):
                rows.append(
                    {
                        "label": f"{self.unknown_prefix}_{i:02d}",
                        "score": 0.0,
                        "source": "instance_semantic_frontend_v2/no_prototype_bank",
                        "top_k": [],
                    }
                )

            # Unknown -> optional open-vocab fallback
            if self.open_vocab_callback is not None:
                for i in range(len(rows)):
                    det_stub = {
                        "index": i,
                        "mask": masks[i].detach().cpu().numpy().astype(bool),
                    }
                    ov = self.open_vocab_callback(
                        rgb_np,
                        det_stub,
                        list(candidate_labels) if candidate_labels is not None else [],
                    )
                    if ov is None:
                        continue

                    rows[i] = {
                        "label": str(ov.get("label", rows[i]["label"])),
                        "score": float(ov.get("score", rows[i]["score"])),
                        "source": str(ov.get("source", "open_vocab_fallback")),
                        "top_k": ov.get("top_k", rows[i].get("top_k", [])),
                    }

            if not include_unknown:
                rows = [r for r in rows if not str(r["label"]).startswith(self.unknown_prefix)]

            return rows

        # ------------------------------------------------------------------
        # SLOW PATH: prototype bank exists -> run DINO closed-set labeling
        # ------------------------------------------------------------------
        out = self.semantic_labeler.compute_and_assign(
            rgb_np=rgb_np,
            masks_hw=masks,
            prototype_bank=self.prototype_bank,
            target_h=self.dino_target_h,
            n_samples=0,
            similarity_threshold=self.closed_set_similarity_threshold,
            unknown_prefix=self.unknown_prefix,
        )

        label_assignments = out.get("label_assignments", None)

        if label_assignments is None:
            for i in range(masks.shape[0]):
                rows.append(
                    {
                        "label": f"{self.unknown_prefix}_{i:02d}",
                        "score": 0.0,
                        "source": "instance_semantic_frontend_v2/no_labels",
                        "top_k": [],
                    }
                )
        else:
            for i, row in enumerate(label_assignments):
                rows.append(
                    {
                        "label": str(row["label"]),
                        "score": float(row["score"]),
                        "source": "json_closed_set",
                        "top_k": [],
                    }
                )

        # Optional fallback for unknowns
        if self.open_vocab_callback is not None:
            for i in range(len(rows)):
                if not rows[i]["label"].startswith(self.unknown_prefix):
                    continue

                det_stub = {
                    "index": i,
                    "mask": masks[i].detach().cpu().numpy().astype(bool),
                }
                ov = self.open_vocab_callback(
                    rgb_np,
                    det_stub,
                    list(candidate_labels) if candidate_labels is not None else [],
                )
                if ov is None:
                    continue

                rows[i] = {
                    "label": str(ov.get("label", rows[i]["label"])),
                    "score": float(ov.get("score", rows[i]["score"])),
                    "source": str(ov.get("source", "open_vocab_fallback")),
                    "top_k": ov.get("top_k", rows[i].get("top_k", [])),
                }

        if not include_unknown:
            rows = [r for r in rows if not str(r["label"]).startswith(self.unknown_prefix)]

        return rows

    def _save_debug_outputs(
        self,
        *,
        rgb_np: np.ndarray,
        detections: list[dict[str, Any]],
        debug_dir: str,
        timings: dict[str, float],
    ) -> None:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)

        try:
            from PIL import ImageDraw
        except Exception:
            ImageDraw = None  # type: ignore

        # Save labeled overlay.
        img = Image.fromarray(rgb_np.copy())
        if ImageDraw is not None:
            draw = ImageDraw.Draw(img)
            for i, det in enumerate(detections):
                box = det["box"]
                label = det["label"]
                score = det["score"]
                draw.rectangle(
                    [(box["xmin"], box["ymin"]), (box["xmax"], box["ymax"])],
                    outline=(0, 255, 0),
                    width=2,
                )
                draw.text(
                    (box["xmin"], max(0, box["ymin"] - 12)),
                    f"{i}:{label} {score:.2f}",
                    fill=(0, 255, 0),
                )
        img.save(Path(debug_dir) / "labeled_overlay.png")

        serializable = []
        for det in detections:
            row = {k: v for k, v in det.items() if k != "mask"}
            serializable.append(row)

        with open(Path(debug_dir) / "scene_detections.json", "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        with open(Path(debug_dir) / "frontend_metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_detections": len(detections),
                    "prior_json_path": self.prior_json_path,
                    "prototype_bank_path": self.prototype_bank_path,
                    "closed_set_labels_count": len(self.closed_set_labels),
                    "enable_post_merge": self.enable_post_merge,
                    "timings": {k: float(v) for k, v in timings.items()},
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
