from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import cv2
import numpy as np
import torch

from semantic_safety.perception_2d3d.lvis_bank_v2 import LVISBankV2
from semantic_safety.perception_2d3d.siglip2_wrapper_v2 import SigLIP2WrapperV2


@dataclass
class InstanceLabelResultV2:
    instance_index: int
    label: str
    canonical_label: str
    score: float
    score_margin: float
    mask_area_px: int
    crop_box_xyxy: list[int]
    topk_canonical: list[tuple[str, float]]
    topk_raw: list[tuple[str, float]]

    def to_dict(self) -> dict:
        return asdict(self)


class SigLIP2LabelerV2:
    """
    SigLIP2-based instance labeler for mask proposals.

    Intended usage
    --------------
    1. MobileSAM / SAM / detector produces masks + boxes.
    2. This labeler extracts per-instance crops.
    3. SigLIP2 scores the crops against a text bank
       (LVIS + custom robotics labels).
    4. Raw alias scores are aggregated to canonical labels.
    5. Top label + top-k alternatives are returned.

    Key design choice
    -----------------
    By default, this labeler does NOT require JSON-derived candidate labels.
    It can optionally use candidate_labels as a restrictive subset, but the
    default behavior is to score against the full LVIS/custom bank.
    """

    def __init__(
        self,
        wrapper: SigLIP2WrapperV2,
        label_bank: LVISBankV2,
        *,
        top_k: int = 5,
        crop_pad_frac: float = 0.08,
        use_masked_crop: bool = True,
        square_crop: bool = False,
        min_mask_area_px: int = 64,
        mask_background_value: int = 0,
        candidate_subset_enabled: bool = False,
        canonical_score_reduce: str = "max",
        low_confidence_mode: str = "unknown",   # "unknown" | "drop" | "keep"
        unknown_score_thresh: float = 0.07,
        unknown_margin_thresh: float = 0.000,
        unknown_label_prefix: str = "unknown_object",
    ) -> None:
        self.wrapper = wrapper
        self.label_bank = label_bank

        self.top_k = int(top_k)
        self.crop_pad_frac = float(crop_pad_frac)
        self.use_masked_crop = bool(use_masked_crop)
        self.square_crop = bool(square_crop)
        self.min_mask_area_px = int(min_mask_area_px)
        self.mask_background_value = int(mask_background_value)
        self.candidate_subset_enabled = bool(candidate_subset_enabled)

        canonical_score_reduce = str(canonical_score_reduce).lower()
        if canonical_score_reduce not in {"max", "mean"}:
            raise ValueError("canonical_score_reduce must be one of {'max', 'mean'}")
        self.canonical_score_reduce = canonical_score_reduce

        self._cached_text_bank_labels: list[str] | None = None
        self._cached_text_embeddings: torch.Tensor | None = None
        self._cached_canonical_labels: list[str] | None = None
        self._cached_alias_to_canonical_index: list[int] | None = None

        low_confidence_mode = str(low_confidence_mode).lower()
        if low_confidence_mode not in {"unknown", "drop", "keep"}:
            raise ValueError("low_confidence_mode must be one of {'unknown', 'drop', 'keep'}")

        self.low_confidence_mode = low_confidence_mode
        self.unknown_score_thresh = float(unknown_score_thresh)
        self.unknown_margin_thresh = float(unknown_margin_thresh)
        self.unknown_label_prefix = str(unknown_label_prefix)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def label_instances(
        self,
        image_rgb: np.ndarray,
        masks: np.ndarray,
        boxes_xyxy: Optional[np.ndarray] = None,
        *,
        candidate_labels: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
        return_debug_tensors: bool = False,
    ) -> dict:
        """
        Parameters
        ----------
        image_rgb:
            HxWx3 uint8 RGB image.
        masks:
            [N,H,W] boolean or binary masks.
        boxes_xyxy:
            Optional [N,4] boxes in xyxy format.
        candidate_labels:
            Optional label hints. Used ONLY if candidate_subset_enabled=True.
        top_k:
            Optional override for number of returned top-k labels.
        return_debug_tensors:
            If True, include similarity tensors in the returned dict.

        Returns
        -------
        dict with:
          - results: list[InstanceLabelResultV2]
          - text_bank_labels_used: list[str]
          - canonical_labels_used: list[str]
          - raw_similarity: np.ndarray [N,M] (optional)
          - canonical_similarity: np.ndarray [N,C] (optional)
          - image_embeddings: np.ndarray [N,D] (optional)
        """
        self._validate_inputs(image_rgb, masks, boxes_xyxy)

        top_k = self.top_k if top_k is None else int(top_k)
        n = int(masks.shape[0])

        if boxes_xyxy is None:
            boxes_xyxy = self._boxes_from_masks(masks)

        bank_labels = self._resolve_text_bank_labels(candidate_labels)
        text_embeddings, canonical_labels, alias_to_canonical_index = self._prepare_bank_embeddings(bank_labels)

        crops: list[np.ndarray] = []
        crop_boxes: list[list[int]] = []
        mask_areas: list[int] = []
        valid_indices: list[int] = []

        for i in range(n):
            mask = self._as_bool_mask(masks[i])
            area = int(mask.sum())
            if area < self.min_mask_area_px:
                continue

            box = boxes_xyxy[i]
            crop, crop_box = self._extract_crop(
                image_rgb=image_rgb,
                mask=mask,
                box_xyxy=box,
                pad_frac=self.crop_pad_frac,
                use_masked_crop=self.use_masked_crop,
                square_crop=self.square_crop,
            )

            if crop.size == 0:
                continue

            crops.append(crop)
            crop_boxes.append(crop_box)
            mask_areas.append(area)
            valid_indices.append(i)

        if len(crops) == 0:
            return {
                "results": [],
                "text_bank_labels_used": bank_labels,
                "canonical_labels_used": canonical_labels,
            }

        raw_similarity, image_embeddings, _ = self.wrapper.score_images_against_texts(
            images=crops,
            texts=bank_labels,
            text_embeddings=text_embeddings,
            return_cpu=True,
        )

        canonical_similarity = self._aggregate_similarity_to_canonical(
            raw_similarity,
            alias_to_canonical_index=alias_to_canonical_index,
            num_canonical=len(canonical_labels),
            reduce=self.canonical_score_reduce,
        )

        raw_topk_all = self.wrapper.topk_labels(raw_similarity, bank_labels, k=min(top_k, len(bank_labels)))
        canonical_topk_all = self._topk_from_similarity(
            canonical_similarity,
            canonical_labels,
            k=min(top_k, len(canonical_labels)),
        )

        results: list[InstanceLabelResultV2] = []
        for row_i, inst_i in enumerate(valid_indices):
            topk_canon = canonical_topk_all[row_i]
            topk_raw = raw_topk_all[row_i]

            best_label = topk_canon[0][0]
            best_score = float(topk_canon[0][1])
            second_score = float(topk_canon[1][1]) if len(topk_canon) > 1 else -1e9
            score_margin = float(best_score - second_score)

            final_label, final_canonical = self._resolve_low_confidence_label(
                instance_index=int(inst_i),
                best_label=best_label,
                best_score=best_score,
                score_margin=score_margin,
            )

            # drop mode
            if final_label is None or final_canonical is None:
                continue

            results.append(
                InstanceLabelResultV2(
                    instance_index=int(inst_i),
                    label=final_label,
                    canonical_label=final_canonical,
                    score=best_score,
                    score_margin=score_margin,
                    mask_area_px=int(mask_areas[row_i]),
                    crop_box_xyxy=list(map(int, crop_boxes[row_i])),
                    topk_canonical=topk_canon,
                    topk_raw=topk_raw,
                )
            )

        out = {
            "results": results,
            "text_bank_labels_used": bank_labels,
            "canonical_labels_used": canonical_labels,
        }

        if return_debug_tensors:
            out["raw_similarity"] = raw_similarity.numpy()
            out["canonical_similarity"] = canonical_similarity.numpy()
            out["image_embeddings"] = image_embeddings.numpy()

        return out

    # ------------------------------------------------------------------
    # Bank / embedding preparation
    # ------------------------------------------------------------------

    def _resolve_text_bank_labels(
        self,
        candidate_labels: Optional[Sequence[str]],
    ) -> list[str]:
        """
        Default behavior:
          use full LVIS/custom bank

        Optional behavior:
          if candidate_subset_enabled=True and candidate_labels are provided,
          use the canonicalized candidate subset only.
        """
        if self.candidate_subset_enabled and candidate_labels:
            subset = []
            seen = set()
            for lbl in candidate_labels:
                canon = self.label_bank.canonicalize(lbl)
                if canon not in seen:
                    seen.add(canon)
                    subset.append(canon)

            if len(subset) > 0:
                return subset

        return self.label_bank.get_text_bank(canonical_only=False, prepend_photo_prompt=False)

    def _prepare_bank_embeddings(
        self,
        bank_labels: list[str],
    ) -> tuple[torch.Tensor, list[str], list[int]]:
        """
        Returns:
          text_embeddings: [M,D]
          canonical_labels: list[str] length C
          alias_to_canonical_index: list[int] length M
        """
        if (
            self._cached_text_bank_labels is not None
            and bank_labels == self._cached_text_bank_labels
            and self._cached_text_embeddings is not None
            and self._cached_canonical_labels is not None
            and self._cached_alias_to_canonical_index is not None
        ):
            return (
                self._cached_text_embeddings,
                self._cached_canonical_labels,
                self._cached_alias_to_canonical_index,
            )

        text_embeddings = self.wrapper.encode_texts(bank_labels, use_cache=True, return_cpu=True)

        canonical_labels: list[str] = []
        canonical_to_idx: dict[str, int] = {}
        alias_to_canonical_index: list[int] = []

        for lbl in bank_labels:
            canon = self.label_bank.canonicalize(lbl)
            if canon not in canonical_to_idx:
                canonical_to_idx[canon] = len(canonical_labels)
                canonical_labels.append(canon)
            alias_to_canonical_index.append(canonical_to_idx[canon])

        self._cached_text_bank_labels = list(bank_labels)
        self._cached_text_embeddings = text_embeddings
        self._cached_canonical_labels = canonical_labels
        self._cached_alias_to_canonical_index = alias_to_canonical_index

        return text_embeddings, canonical_labels, alias_to_canonical_index

    # ------------------------------------------------------------------
    # Similarity aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_similarity_to_canonical(
        raw_similarity: torch.Tensor,
        *,
        alias_to_canonical_index: list[int],
        num_canonical: int,
        reduce: str = "max",
    ) -> torch.Tensor:
        """
        raw_similarity: [N, M]
        alias_to_canonical_index: length M, mapping raw label col -> canonical col
        output: [N, C]
        """
        raw_similarity = raw_similarity.float()
        n, m = raw_similarity.shape

        if m != len(alias_to_canonical_index):
            raise ValueError(
                f"alias_to_canonical_index length mismatch: raw_similarity has {m} cols "
                f"but alias map has {len(alias_to_canonical_index)}"
            )

        if reduce == "max":
            out = torch.full((n, num_canonical), -1e9, dtype=raw_similarity.dtype)
            for raw_j, canon_j in enumerate(alias_to_canonical_index):
                out[:, canon_j] = torch.maximum(out[:, canon_j], raw_similarity[:, raw_j])
            return out

        if reduce == "mean":
            out = torch.zeros((n, num_canonical), dtype=raw_similarity.dtype)
            counts = torch.zeros((num_canonical,), dtype=raw_similarity.dtype)
            for raw_j, canon_j in enumerate(alias_to_canonical_index):
                out[:, canon_j] += raw_similarity[:, raw_j]
                counts[canon_j] += 1.0
            counts = torch.clamp(counts, min=1.0)
            out = out / counts[None, :]
            return out

        raise ValueError(f"Unsupported reduce: {reduce}")

    @staticmethod
    def _topk_from_similarity(
        similarity: torch.Tensor,
        labels: Sequence[str],
        k: int,
    ) -> list[list[tuple[str, float]]]:
        if similarity.ndim != 2:
            raise ValueError(f"Expected similarity [N,C], got {tuple(similarity.shape)}")

        labels = list(labels)
        k = max(1, min(int(k), len(labels)))
        vals, idxs = torch.topk(similarity.float(), k=k, dim=1)

        results: list[list[tuple[str, float]]] = []
        for row_vals, row_idxs in zip(vals, idxs):
            row = []
            for v, j in zip(row_vals.tolist(), row_idxs.tolist()):
                row.append((labels[j], float(v)))
            results.append(row)
        return results

    def _is_low_confidence(self, score: float, margin: float) -> bool:
        return (
            float(score) < self.unknown_score_thresh
            or float(margin) < self.unknown_margin_thresh
        )

    def _resolve_low_confidence_label(
        self,
        *,
        instance_index: int,
        best_label: str,
        best_score: float,
        score_margin: float,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Returns:
        (label, canonical_label)

        Cases:
        - keep mode: return original label
        - unknown mode: return unknown label if low confidence
        - drop mode: return (None, None) if low confidence
        """
        low_conf = self._is_low_confidence(best_score, score_margin)

        if not low_conf or self.low_confidence_mode == "keep":
            canon = self.label_bank.canonicalize(best_label)
            return best_label, canon

        if self.low_confidence_mode == "drop":
            return None, None

        # unknown mode
        unknown_name = f"{self.unknown_label_prefix}_{instance_index:02d}"
        return unknown_name, unknown_name
    # ------------------------------------------------------------------
    # Crop extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        image_rgb: np.ndarray,
        masks: np.ndarray,
        boxes_xyxy: Optional[np.ndarray],
    ) -> None:
        if not isinstance(image_rgb, np.ndarray):
            raise TypeError(f"image_rgb must be np.ndarray, got {type(image_rgb)}")
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected image_rgb HxWx3, got {image_rgb.shape}")

        if not isinstance(masks, np.ndarray):
            raise TypeError(f"masks must be np.ndarray, got {type(masks)}")
        if masks.ndim != 3:
            raise ValueError(f"Expected masks [N,H,W], got {masks.shape}")
        if masks.shape[1] != image_rgb.shape[0] or masks.shape[2] != image_rgb.shape[1]:
            raise ValueError(
                f"Mask/image size mismatch: masks {masks.shape}, image {image_rgb.shape}"
            )

        if boxes_xyxy is not None:
            if not isinstance(boxes_xyxy, np.ndarray):
                raise TypeError(f"boxes_xyxy must be np.ndarray, got {type(boxes_xyxy)}")
            if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[1] != 4:
                raise ValueError(f"Expected boxes_xyxy [N,4], got {boxes_xyxy.shape}")
            if boxes_xyxy.shape[0] != masks.shape[0]:
                raise ValueError(
                    f"Box/mask count mismatch: boxes {boxes_xyxy.shape[0]} vs masks {masks.shape[0]}"
                )

    @staticmethod
    def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
        return np.asarray(mask).astype(bool)

    @staticmethod
    def _boxes_from_masks(masks: np.ndarray) -> np.ndarray:
        boxes = []
        for i in range(masks.shape[0]):
            m = masks[i].astype(bool)
            ys, xs = np.where(m)
            if len(xs) == 0 or len(ys) == 0:
                boxes.append([0, 0, 1, 1])
            else:
                boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
        return np.asarray(boxes, dtype=np.float32)

    @staticmethod
    def _clip_box(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        w: int,
        h: int,
    ) -> tuple[int, int, int, int]:
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        return x1, y1, x2, y2

    def _extract_crop(
        self,
        *,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        box_xyxy: np.ndarray,
        pad_frac: float,
        use_masked_crop: bool,
        square_crop: bool,
    ) -> tuple[np.ndarray, list[int]]:
        h, w = image_rgb.shape[:2]

        x1, y1, x2, y2 = [int(round(float(v))) for v in box_xyxy.tolist()]
        x1, y1, x2, y2 = self._clip_box(x1, y1, x2, y2, w=w, h=h)

        bw = x2 - x1
        bh = y2 - y1

        pad_x = int(round(bw * pad_frac))
        pad_y = int(round(bh * pad_frac))

        x1 -= pad_x
        x2 += pad_x
        y1 -= pad_y
        y2 += pad_y

        if square_crop:
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            side = int(round(max(x2 - x1, y2 - y1)))
            x1 = int(round(cx - side / 2))
            x2 = int(round(cx + side / 2))
            y1 = int(round(cy - side / 2))
            y2 = int(round(cy + side / 2))

        x1, y1, x2, y2 = self._clip_box(x1, y1, x2, y2, w=w, h=h)

        crop = image_rgb[y1:y2, x1:x2].copy()

        if crop.size == 0:
            return crop, [x1, y1, x2, y2]

        if use_masked_crop:
            crop_mask = mask[y1:y2, x1:x2]
            if crop_mask.shape[:2] != crop.shape[:2]:
                return crop, [x1, y1, x2, y2]

            bg_val = np.uint8(self.mask_background_value)
            masked = np.full_like(crop, fill_value=bg_val)
            masked[crop_mask] = crop[crop_mask]
            crop = masked

        return crop, [x1, y1, x2, y2]