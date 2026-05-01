from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.error import HTTPError, URLError

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore
    _TORCH_IMPORT_ERROR = e
else:
    _TORCH_IMPORT_ERROR = None


@dataclass
class MobileSAMV2Results:
    """Container for single-image instance mask extraction results."""

    image_rgb: np.ndarray
    sam_masks: "torch.Tensor"
    input_boxes: "torch.Tensor"
    box_scores: Optional["torch.Tensor"] = None
    timings: dict[str, float] = field(default_factory=dict)
    processed_shape: tuple[int, int, int] | None = None
    original_shape: tuple[int, int, int] | None = None


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError(
            "PyTorch is required for MobileSAMV2WrapperV2 but could not be imported. "
            f"Original import error: {_TORCH_IMPORT_ERROR}"
        )


def resize_image_np(
    image: np.ndarray,
    *,
    mode: str = "image",
    long_edge_size: int = 1024,
    shape: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """
    Resize a numpy image/mask.

    Args:
        image: [H,W,3] RGB image or [H,W] mask.
        mode: "image" -> bilinear, "mask" -> nearest.
        long_edge_size: resize target for the longer side when shape is None.
        shape: optional target (H, W).
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2D/3D numpy array, got shape {image.shape}")

    h, w = image.shape[:2]
    if shape is not None:
        target_h, target_w = int(shape[0]), int(shape[1])
    else:
        if h >= w:
            target_h = int(long_edge_size)
            target_w = max(1, int(round(long_edge_size * w / h)))
        else:
            target_w = int(long_edge_size)
            target_h = max(1, int(round(long_edge_size * h / w)))

    interp = cv2.INTER_LINEAR if mode == "image" else cv2.INTER_NEAREST
    resized = cv2.resize(image, (target_w, target_h), interpolation=interp)
    return resized


def resize_masks_torch(
    masks: "torch.Tensor",
    *,
    shape: tuple[int, int],
) -> "torch.Tensor":
    """Resize masks [N,H,W] to (target_h, target_w) with nearest interpolation."""
    _require_torch()

    if masks.ndim != 3:
        raise ValueError(f"Expected masks [N,H,W], got shape {tuple(masks.shape)}")

    target_h, target_w = int(shape[0]), int(shape[1])
    resized = F.interpolate(masks.unsqueeze(1).float(), size=(target_h, target_w), mode="nearest")
    return resized.squeeze(1) > 0.5


def scale_bounding_boxes(
    boxes: "torch.Tensor",
    *,
    old_size: tuple[int, int],
    new_size: tuple[int, int],
) -> "torch.Tensor":
    """Scale xyxy boxes from old_size=(H,W) to new_size=(H,W)."""
    _require_torch()

    if boxes.numel() == 0:
        return boxes.clone()

    old_h, old_w = old_size
    new_h, new_w = new_size
    scale = torch.tensor(
        [new_w / old_w, new_h / old_h, new_w / old_w, new_h / old_h],
        device=boxes.device,
        dtype=boxes.dtype,
    )
    return boxes * scale


def load_model_with_retry(
    repo: str,
    model_name: str,
    *,
    max_retries: int = 5,
    initial_delay: float = 10.0,
) -> Any:
    """Load a torch.hub model bundle with exponential backoff."""
    _require_torch()

    delay = float(initial_delay)
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            print(f"Loading MobileSAMV2 models (attempt {attempt + 1}/{max_retries})...")
            models = torch.hub.load(repo, model_name)
            print("MobileSAMV2 models loaded successfully.")
            return models
        except (HTTPError, URLError, RuntimeError) as e:
            last_error = e
            if attempt == max_retries - 1:
                break
            jitter = random.uniform(0.1 * delay, 0.2 * delay)
            sleep_time = delay + jitter
            print(
                f"Failed to load MobileSAMV2 models (attempt {attempt + 1}/{max_retries}). "
                f"Retrying in {sleep_time:.1f}s... Error: {e}"
            )
            time.sleep(sleep_time)
            delay *= 2.5

    raise RuntimeError(
        f"Failed to load model bundle repo={repo} model={model_name} after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


class MobileSAMV2WrapperV2:
    """
    Phase-1 wrapper for instance mask extraction using:
      ObjAwareModel -> bounding boxes
      MobileSAMV2 predictor -> box-conditioned instance masks

    This class intentionally does NOT do semantic labeling or DINO pooling yet.
    """

    def __init__(
        self,
        *,
        sam_size: int = 1024,
        mobilesamv2_encoder_name: str = "mobilesamv2_efficientvit_l2",
        yolo_conf: float = 0.40,
        yolo_iou: float = 0.90,
        min_mask_size: int = 1000,
        max_batch_size: int = 320,
        device: Optional[str] = None,
        enable_post_merge: bool = False,
        merge_box_iou_thresh: float = 0.85,
        merge_mask_iou_thresh: float = 0.80,
        merge_containment_thresh: float = 0.90,
        enable_geometry_cleanup: bool = True,
        cleanup_duplicate_box_iou_thresh: float = 0.75,
        cleanup_duplicate_mask_iou_thresh: float = 0.60,
        cleanup_inner_box_containment_thresh: float = 0.90,
        cleanup_inner_mask_containment_thresh: float = 0.90,
        cleanup_inner_area_ratio_thresh: float = 0.65,
        enable_container_suppression: bool = True,
        container_child_box_containment_thresh: float = 0.85,
        container_child_mask_containment_thresh: float = 0.85,
        container_child_area_ratio_thresh: float = 0.75,
        container_combined_child_area_ratio_thresh: float = 0.60,
        container_children_pairwise_iou_max: float = 0.25,
        postprocess_debug: bool = True,
        enable_post_box_score_filter: bool = True,
        post_box_score_thresh: float = 0.80,
    ) -> None:
        _require_torch()

        if device is not None:
            self.device = str(device)
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.sam_size = int(sam_size)
        self.yolo_conf = float(yolo_conf)
        self.yolo_iou = float(yolo_iou)
        self.min_mask_size = int(min_mask_size)
        self.max_batch_size = int(max_batch_size)
        self.mobilesamv2_encoder_name = str(mobilesamv2_encoder_name)

        self.models = self._setup_models()

        self.enable_post_merge = bool(enable_post_merge)
        self.merge_box_iou_thresh = float(merge_box_iou_thresh)
        self.merge_mask_iou_thresh = float(merge_mask_iou_thresh)
        self.merge_containment_thresh = float(merge_containment_thresh)

        self.enable_geometry_cleanup = bool(enable_geometry_cleanup)
        self.cleanup_duplicate_box_iou_thresh = float(cleanup_duplicate_box_iou_thresh)
        self.cleanup_duplicate_mask_iou_thresh = float(cleanup_duplicate_mask_iou_thresh)
        self.cleanup_inner_box_containment_thresh = float(cleanup_inner_box_containment_thresh)
        self.cleanup_inner_mask_containment_thresh = float(cleanup_inner_mask_containment_thresh)
        self.cleanup_inner_area_ratio_thresh = float(cleanup_inner_area_ratio_thresh)

        self.enable_container_suppression = bool(enable_container_suppression)
        self.container_child_box_containment_thresh = float(container_child_box_containment_thresh)
        self.container_child_mask_containment_thresh = float(container_child_mask_containment_thresh)
        self.container_child_area_ratio_thresh = float(container_child_area_ratio_thresh)
        self.container_combined_child_area_ratio_thresh = float(container_combined_child_area_ratio_thresh)
        self.container_children_pairwise_iou_max = float(container_children_pairwise_iou_max)

        self.postprocess_debug = bool(postprocess_debug)
        self.enable_post_box_score_filter = bool(enable_post_box_score_filter)
        self.post_box_score_thresh = float(post_box_score_thresh)

    def _setup_models(self) -> dict[str, Any]:
        mobilesamv2, obj_aware_model, predictor = load_model_with_retry(
            "RogerQi/MobileSAMV2",
            self.mobilesamv2_encoder_name,
            max_retries=5,
            initial_delay=10.0,
        )
        mobilesamv2 = mobilesamv2.to(device=self.device)
        mobilesamv2.eval()

        return {
            "mobilesamv2": mobilesamv2,
            "ObjAwareModel": obj_aware_model,
            "predictor": predictor,
        }

    def _predict_boxes(self, image_rgb: np.ndarray) -> tuple["torch.Tensor", Optional["torch.Tensor"]]:
        obj_results = self.models["ObjAwareModel"](
            image_rgb,
            device=self.device,
            imgsz=self.sam_size,
            conf=self.yolo_conf,
            iou=self.yolo_iou,
            verbose=False,
        )

        if len(obj_results) == 0 or obj_results[0].boxes is None or obj_results[0].boxes.xyxy.numel() == 0:
            device = torch.device(self.device)
            return torch.empty((0, 4), device=device), torch.empty((0,), device=device)

        boxes_xyxy = obj_results[0].boxes.xyxy
        box_scores = getattr(obj_results[0].boxes, "conf", None)
        return boxes_xyxy, box_scores

    def _predict_masks(self, image_rgb: np.ndarray, boxes_xyxy: "torch.Tensor") -> "torch.Tensor":
        predictor = self.models["predictor"]
        mobilesamv2 = self.models["mobilesamv2"]

        predictor.set_image(image_rgb)

        input_boxes_np = boxes_xyxy.detach().cpu().numpy()
        input_boxes_np = predictor.transform.apply_boxes(input_boxes_np, predictor.original_size)
        input_boxes = torch.from_numpy(input_boxes_np).to(self.device).detach()

        if input_boxes.shape[0] == 0:
            return torch.empty((0, image_rgb.shape[0], image_rgb.shape[1]), dtype=torch.bool, device=self.device)

        batch_size = min(self.max_batch_size, int(input_boxes.shape[0]))

        while batch_size > 0:
            try:
                masks_out = []
                image_embedding = predictor.features
                prompt_embedding = mobilesamv2.prompt_encoder.get_dense_pe()

                image_embedding = torch.repeat_interleave(image_embedding, batch_size, dim=0)
                prompt_embedding = torch.repeat_interleave(prompt_embedding, batch_size, dim=0)

                for start in range(0, input_boxes.shape[0], batch_size):
                    boxes_batch = input_boxes[start : start + batch_size]
                    if boxes_batch.shape[0] == 0:
                        continue

                    with torch.no_grad():
                        sparse_embeddings, dense_embeddings = mobilesamv2.prompt_encoder(
                            points=None,
                            boxes=boxes_batch,
                            masks=None,
                        )

                        low_res_masks, _ = mobilesamv2.mask_decoder(
                            image_embeddings=image_embedding[: boxes_batch.shape[0]],
                            image_pe=prompt_embedding[: boxes_batch.shape[0]],
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False,
                            simple_type=True,
                        )

                        full_res_masks = predictor.model.postprocess_masks(
                            low_res_masks,
                            predictor.input_size,
                            predictor.original_size,
                        )
                        masks_bool = (full_res_masks > mobilesamv2.mask_threshold).squeeze(1)
                        masks_out.append(masks_bool)

                return torch.cat(masks_out, dim=0).bool()

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    batch_size = batch_size // 2
                    print(f"OOM during mask generation. Retrying with batch_size={batch_size}.")
                    continue
                raise

        raise RuntimeError("Failed to generate MobileSAMV2 masks.")

    def _post_filter_by_box_score(
        self,
        masks: "torch.Tensor",
        boxes: "torch.Tensor",
        scores: Optional["torch.Tensor"],
    ) -> tuple["torch.Tensor", "torch.Tensor", Optional["torch.Tensor"]]:
        """
        Remove low-confidence proposals after geometry cleanup.

        Useful for synthetic topology scenes where giant false proposals
        often have clearly lower box scores than real object proposals.
        """
        if scores is None or masks.shape[0] == 0:
            return masks, boxes, scores

        keep = scores >= self.post_box_score_thresh
        keep_idx = torch.where(keep)[0]

        if self.postprocess_debug:
            print("\n--- POST BOX-SCORE FILTER DEBUG ---")
            print(f"threshold = {self.post_box_score_thresh:.3f}")
            print(f"before    = {int(masks.shape[0])}")
            print(f"after     = {int(keep_idx.numel())}")

            dropped_idx = torch.where(~keep)[0]
            for i in dropped_idx.tolist():
                box = boxes[i].detach().cpu().tolist()
                print(
                    f"[DROP low-score] i={i} "
                    f"score={float(scores[i].item()):.3f} "
                    f"box={[round(x, 1) for x in box]}"
                )

        if keep_idx.numel() == 0:
            # fail-safe: keep the highest-score proposal instead of returning empty
            best_idx = int(torch.argmax(scores).item())
            keep_idx = torch.tensor([best_idx], dtype=torch.long, device=masks.device)

            if self.postprocess_debug:
                print(
                    f"[POST BOX-SCORE FILTER] all proposals were below threshold; "
                    f"keeping best idx={best_idx} score={float(scores[best_idx].item()):.3f}"
                )

        return masks[keep_idx], boxes[keep_idx], scores[keep_idx]

    def process_image(self, image_rgb: np.ndarray, *, verbose: bool = False) -> MobileSAMV2Results:
        _require_torch()

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image [H,W,3], got shape {image_rgb.shape}")

        timings: dict[str, float] = {}
        t_total = time.time()

        original_shape = image_rgb.shape

        t0 = time.time()
        processed_rgb = resize_image_np(image_rgb, mode="image", long_edge_size=self.sam_size)
        processed_shape = processed_rgb.shape
        timings["image_preprocessing"] = time.time() - t0

        t0 = time.time()
        boxes_processed, box_scores = self._predict_boxes(processed_rgb)
        timings["box_detection"] = time.time() - t0

        t0 = time.time()
        masks_processed = self._predict_masks(processed_rgb, boxes_processed)
        timings["mask_generation"] = time.time() - t0

        t0 = time.time()
        if masks_processed.shape[0] > 0:
            flat = masks_processed.view(masks_processed.shape[0], -1)
            valid = flat.any(dim=1)
            masks_processed = masks_processed[valid]
            boxes_processed = boxes_processed[valid]
            if box_scores is not None:
                box_scores = box_scores[valid]

            areas = masks_processed.view(masks_processed.shape[0], -1).sum(dim=1)
            keep = areas >= int(self.min_mask_size)
            masks_processed = masks_processed[keep]
            boxes_processed = boxes_processed[keep]
            if box_scores is not None:
                box_scores = box_scores[keep]

        timings["mask_filtering"] = time.time() - t0

        if self.enable_geometry_cleanup and masks_processed.shape[0] > 1:
            t_cleanup = time.time()
            masks_processed, boxes_processed, box_scores = self._geometry_cleanup_instances(
                masks_processed,
                boxes_processed,
                box_scores,
            )
            timings["geometry_cleanup"] = time.time() - t_cleanup
        else:
            timings["geometry_cleanup"] = 0.0

        if self.enable_post_box_score_filter and box_scores is not None and masks_processed.shape[0] > 0:
            t_score_filter = time.time()
            masks_processed, boxes_processed, box_scores = self._post_filter_by_box_score(
                masks_processed,
                boxes_processed,
                box_scores,
            )
            timings["post_box_score_filter"] = time.time() - t_score_filter
        else:
            timings["post_box_score_filter"] = 0.0

        if self.enable_post_merge and masks_processed.shape[0] > 1:
            t_merge = time.time()
            masks_processed, boxes_processed, box_scores = self._post_merge_instances(
                masks_processed,
                boxes_processed,
                box_scores,
            )
            timings["post_merge"] = time.time() - t_merge
        else:
            timings["post_merge"] = 0.0

        t0 = time.time()
        masks_original = resize_masks_torch(masks_processed, shape=original_shape[:2]).cpu()
        boxes_original = scale_bounding_boxes(
            boxes_processed.detach().cpu(),
            old_size=processed_shape[:2],
            new_size=original_shape[:2],
        )
        box_scores_cpu = box_scores.detach().cpu() if box_scores is not None else None
        timings["resize_to_original"] = time.time() - t0

        timings["total_time"] = time.time() - t_total

        if verbose:
            print("\n[MobileSAMV2WrapperV2] detailed timings")
            for key, value in timings.items():
                print(f"  {key:24s}: {value:.3f}s")
            print(f"  masks kept: {int(masks_original.shape[0])}")

        return MobileSAMV2Results(
            image_rgb=image_rgb,
            sam_masks=masks_original.bool(),
            input_boxes=boxes_original.to(torch.float32),
            box_scores=box_scores_cpu.to(torch.float32) if box_scores_cpu is not None else None,
            timings=timings,
            processed_shape=processed_shape,
            original_shape=original_shape,
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _bbox_area_xyxy(self, box: "torch.Tensor") -> float:
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _bbox_intersection_xyxy(self, a: "torch.Tensor", b: "torch.Tensor") -> float:
        ax1, ay1, ax2, ay2 = [float(v) for v in a.tolist()]
        bx1, by1, bx2, by2 = [float(v) for v in b.tolist()]

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        return inter_w * inter_h

    def _box_iou_xyxy(self, a: "torch.Tensor", b: "torch.Tensor") -> float:
        inter_area = self._bbox_intersection_xyxy(a, b)
        area_a = self._bbox_area_xyxy(a)
        area_b = self._bbox_area_xyxy(b)
        union = area_a + area_b - inter_area

        if union <= 0.0:
            return 0.0
        return inter_area / union

    def _bbox_containment_xyxy(self, inner: "torch.Tensor", outer: "torch.Tensor") -> float:
        inter_area = self._bbox_intersection_xyxy(inner, outer)
        inner_area = self._bbox_area_xyxy(inner)
        if inner_area <= 0.0:
            return 0.0
        return inter_area / inner_area

    def _mask_area(self, a: "torch.Tensor") -> float:
        return float(a.bool().sum().item())

    def _mask_iou(self, a: "torch.Tensor", b: "torch.Tensor") -> float:
        a = a.bool()
        b = b.bool()
        inter = (a & b).sum().item()
        union = (a | b).sum().item()
        if union <= 0:
            return 0.0
        return float(inter / union)

    def _containment_ratio(self, small: "torch.Tensor", large: "torch.Tensor") -> float:
        small = small.bool()
        large = large.bool()
        denom = small.sum().item()
        if denom <= 0:
            return 0.0
        inter = (small & large).sum().item()
        return float(inter / denom)

    # ------------------------------------------------------------------
    # New minimal cleanup
    # ------------------------------------------------------------------

    def _suppress_duplicate_and_inner_instances(
        self,
        masks: "torch.Tensor",
        boxes: "torch.Tensor",
        scores: Optional["torch.Tensor"],
    ) -> tuple["torch.Tensor", "torch.Tensor", Optional["torch.Tensor"]]:
        if masks.shape[0] <= 1:
            return masks, boxes, scores

        if scores is not None:
            order = torch.argsort(scores, descending=True)
        else:
            areas = masks.view(masks.shape[0], -1).sum(dim=1)
            order = torch.argsort(areas, descending=True)

        masks = masks[order]
        boxes = boxes[order]
        scores = scores[order] if scores is not None else None

        used = torch.zeros(masks.shape[0], dtype=torch.bool, device=masks.device)
        keep_indices: list[int] = []
        debug_logs: list[str] = []

        areas = masks.view(masks.shape[0], -1).sum(dim=1).float()

        for i in range(masks.shape[0]):
            if used[i]:
                continue

            used[i] = True
            keep_indices.append(i)

            for j in range(i + 1, masks.shape[0]):
                if used[j]:
                    continue

                box_iou = self._box_iou_xyxy(boxes[i], boxes[j])
                mask_iou = self._mask_iou(masks[i], masks[j])

                j_in_i_mask = self._containment_ratio(masks[j], masks[i])
                i_in_j_mask = self._containment_ratio(masks[i], masks[j])

                j_in_i_box = self._bbox_containment_xyxy(boxes[j], boxes[i])
                i_in_j_box = self._bbox_containment_xyxy(boxes[i], boxes[j])

                area_ratio_ji = float(areas[j].item() / max(float(areas[i].item()), 1e-8))
                area_ratio_ij = float(areas[i].item() / max(float(areas[j].item()), 1e-8))

                # A) near duplicate
                if (
                    box_iou >= self.cleanup_duplicate_box_iou_thresh
                    or mask_iou >= self.cleanup_duplicate_mask_iou_thresh
                ):
                    used[j] = True
                    debug_logs.append(
                        f"[DROP duplicate] j={j} <- i={i} | "
                        f"box_iou={box_iou:.3f}, mask_iou={mask_iou:.3f}"
                    )
                    continue

                # B) j is a small inner proposal inside i
                if (
                    (j_in_i_mask >= self.cleanup_inner_mask_containment_thresh
                     or j_in_i_box >= self.cleanup_inner_box_containment_thresh)
                    and area_ratio_ji <= self.cleanup_inner_area_ratio_thresh
                ):
                    used[j] = True
                    debug_logs.append(
                        f"[DROP inner-small] j={j} inside i={i} | "
                        f"mask_cont={j_in_i_mask:.3f}, box_cont={j_in_i_box:.3f}, "
                        f"area_ratio={area_ratio_ji:.3f}"
                    )
                    continue

        keep_indices_t = torch.tensor(keep_indices, dtype=torch.long, device=masks.device)

        if self.postprocess_debug:
            print("\n--- GEOMETRY CLEANUP DEBUG | duplicate + inner suppression ---")
            print(f"before = {int(masks.shape[0])}")
            print(f"after  = {int(keep_indices_t.numel())}")
            for line in debug_logs:
                print(line)

        out_masks = masks[keep_indices_t]
        out_boxes = boxes[keep_indices_t]
        out_scores = scores[keep_indices_t] if scores is not None else None
        return out_masks, out_boxes, out_scores

    def _suppress_multi_object_containers(
        self,
        masks: "torch.Tensor",
        boxes: "torch.Tensor",
        scores: Optional["torch.Tensor"],
    ) -> tuple["torch.Tensor", "torch.Tensor", Optional["torch.Tensor"]]:
        """
        Suppress overly large container proposals that mostly cover two or more
        smaller children with low pairwise overlap.

        This helps remove 'one big box covering two nearby drills'.
        """
        if masks.shape[0] <= 2:
            return masks, boxes, scores

        n = masks.shape[0]
        keep = torch.ones(n, dtype=torch.bool, device=masks.device)
        areas = masks.view(n, -1).sum(dim=1).float()
        debug_logs: list[str] = []

        for i in range(n):
            child_candidates: list[int] = []

            for j in range(n):
                if i == j:
                    continue

                child_mask_cont = self._containment_ratio(masks[j], masks[i])
                child_box_cont = self._bbox_containment_xyxy(boxes[j], boxes[i])
                area_ratio = float(areas[j].item() / max(float(areas[i].item()), 1e-8))

                if (
                    (child_mask_cont >= self.container_child_mask_containment_thresh
                     or child_box_cont >= self.container_child_box_containment_thresh)
                    and area_ratio <= self.container_child_area_ratio_thresh
                ):
                    child_candidates.append(j)

            if len(child_candidates) < 2:
                continue

            combined_child_area = float(areas[child_candidates].sum().item())
            combined_ratio = combined_child_area / max(float(areas[i].item()), 1e-8)

            if combined_ratio < self.container_combined_child_area_ratio_thresh:
                continue

            low_overlap_children = True
            for a_idx in range(len(child_candidates)):
                for b_idx in range(a_idx + 1, len(child_candidates)):
                    a = child_candidates[a_idx]
                    b = child_candidates[b_idx]
                    pair_iou = self._mask_iou(masks[a], masks[b])
                    if pair_iou > self.container_children_pairwise_iou_max:
                        low_overlap_children = False
                        break
                if not low_overlap_children:
                    break

            if not low_overlap_children:
                continue

            # If the container score is clearly worse than the best child, drop it.
            if scores is not None:
                best_child_score = float(scores[child_candidates].max().item())
                this_score = float(scores[i].item())
                if this_score > best_child_score + 1e-6:
                    continue

            keep[i] = False
            debug_logs.append(
                f"[DROP container] i={i} | children={child_candidates}, "
                f"combined_child_area_ratio={combined_ratio:.3f}"
            )

        keep_idx = torch.where(keep)[0]

        if self.postprocess_debug:
            print("\n--- GEOMETRY CLEANUP DEBUG | multi-object container suppression ---")
            print(f"before = {int(n)}")
            print(f"after  = {int(keep_idx.numel())}")
            for line in debug_logs:
                print(line)

        out_masks = masks[keep_idx]
        out_boxes = boxes[keep_idx]
        out_scores = scores[keep_idx] if scores is not None else None
        return out_masks, out_boxes, out_scores

    def _geometry_cleanup_instances(
        self,
        masks: "torch.Tensor",
        boxes: "torch.Tensor",
        scores: Optional["torch.Tensor"],
    ) -> tuple["torch.Tensor", "torch.Tensor", Optional["torch.Tensor"]]:
        """
        Minimal wrapper-level proposal cleanup:
        1) remove duplicates and small inner proposals
        2) remove oversized container proposals spanning multiple children
        """
        masks, boxes, scores = self._suppress_duplicate_and_inner_instances(masks, boxes, scores)

        if self.enable_container_suppression and masks.shape[0] > 2:
            masks, boxes, scores = self._suppress_multi_object_containers(masks, boxes, scores)

        return masks, boxes, scores

    # ------------------------------------------------------------------
    # Existing conservative merge
    # ------------------------------------------------------------------

    def _merge_group(
        self,
        masks: "torch.Tensor",
        boxes: "torch.Tensor",
        scores: Optional["torch.Tensor"],
        group: list[int],
    ) -> tuple["torch.Tensor", "torch.Tensor", Optional["torch.Tensor"]]:
        merged_mask = masks[group].any(dim=0)

        ys, xs = torch.where(merged_mask)
        if xs.numel() == 0 or ys.numel() == 0:
            merged_box = boxes[group[0]].clone()
        else:
            merged_box = torch.tensor(
                [
                    int(xs.min().item()),
                    int(ys.min().item()),
                    int(xs.max().item()),
                    int(ys.max().item()),
                ],
                dtype=boxes.dtype,
                device=boxes.device,
            )

        merged_score = None
        if scores is not None:
            merged_score = scores[group].max()

        return merged_mask, merged_box, merged_score

    def _post_merge_instances(
        self,
        masks: "torch.Tensor",
        boxes: "torch.Tensor",
        scores: Optional["torch.Tensor"],
    ) -> tuple["torch.Tensor", "torch.Tensor", Optional["torch.Tensor"]]:
        """
        Conservative optional post-merge:
        only merge near-duplicate instances, not merely adjacent parts.
        """
        if masks.shape[0] <= 1:
            return masks, boxes, scores

        order = torch.arange(masks.shape[0], device=masks.device)
        if scores is not None:
            order = torch.argsort(scores, descending=True)

        masks = masks[order]
        boxes = boxes[order]
        scores = scores[order] if scores is not None else None

        used = torch.zeros(masks.shape[0], dtype=torch.bool, device=masks.device)

        merged_masks = []
        merged_boxes = []
        merged_scores = []

        for i in range(masks.shape[0]):
            if used[i]:
                continue

            used[i] = True
            group = [i]

            for j in range(i + 1, masks.shape[0]):
                if used[j]:
                    continue

                box_iou = self._box_iou_xyxy(boxes[i], boxes[j])
                mask_iou = self._mask_iou(masks[i], masks[j])

                contain_ij = self._containment_ratio(masks[i], masks[j])
                contain_ji = self._containment_ratio(masks[j], masks[i])
                max_contain = max(contain_ij, contain_ji)

                should_merge = (
                    (box_iou >= self.merge_box_iou_thresh)
                    or (mask_iou >= self.merge_mask_iou_thresh)
                    or (max_contain >= self.merge_containment_thresh)
                )

                if should_merge:
                    used[j] = True
                    group.append(j)

            mm, mb, ms = self._merge_group(masks, boxes, scores, group)
            merged_masks.append(mm)
            merged_boxes.append(mb)
            if scores is not None:
                merged_scores.append(ms)

        out_masks = torch.stack(merged_masks, dim=0)
        out_boxes = torch.stack(merged_boxes, dim=0)
        out_scores = torch.stack(merged_scores, dim=0) if scores is not None else None

        return out_masks, out_boxes, out_scores