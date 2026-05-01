from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

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

try:
    from semantic_safety.perception_2d3d.dino_features_v2 import DinoFeaturesV2
except Exception:  # pragma: no cover
    from dino_features_v2 import DinoFeaturesV2  # type: ignore


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError(
            "PyTorch is required for SemanticLabelerV2 but could not be imported. "
            f"Original import error: {_TORCH_IMPORT_ERROR}"
        )


@dataclass
class PrototypeBankV2:
    labels: list[str]
    prototypes: np.ndarray  # [K, C], assumed L2 normalized

    def validate(self) -> None:
        if self.prototypes.ndim != 2:
            raise ValueError(f"Expected prototypes [K,C], got shape {self.prototypes.shape}")
        if len(self.labels) != self.prototypes.shape[0]:
            raise ValueError(
                f"labels length {len(self.labels)} does not match prototype rows {self.prototypes.shape[0]}"
            )

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, labels=np.asarray(self.labels, dtype=object), prototypes=self.prototypes.astype(np.float32))

    @staticmethod
    def load_npz(path: str | Path) -> "PrototypeBankV2":
        arr = np.load(path, allow_pickle=True)
        labels = [str(x) for x in arr["labels"].tolist()]
        prototypes = np.asarray(arr["prototypes"], dtype=np.float32)
        bank = PrototypeBankV2(labels=labels, prototypes=prototypes)
        bank.validate()
        return bank


class SemanticLabelerV2:
    """
    Phase-2 semantic module:
      1) compute mask-pooled DINO embeddings
      2) optionally assign labels via cosine similarity against a prototype bank

    Important: DINO is not a text-image aligned embedding space like CLIP.
    Therefore label assignment here is prototype-bank based, not text-prompt based.
    """

    def __init__(self, dino: Optional[DinoFeaturesV2] = None, *, device: Optional[str] = None) -> None:
        _require_torch()
        self.dino = dino if dino is not None else DinoFeaturesV2(device=device)
        self.device = self.dino.device

    @torch.inference_mode()
    def compute_mask_pooled_embeddings(
        self,
        *,
        rgb_np: np.ndarray,
        masks_hw: torch.Tensor | np.ndarray,
        target_h: int = 500,
        n_samples: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
          rgb_np: [H,W,3]
          masks_hw: [N,H,W] bool/0-1
        Returns:
          avg_emb_np: [N,C] float32, L2-normalized
          samples_np: [N,n_samples,C] float32 (or empty [0,0,0])
        """
        if isinstance(masks_hw, np.ndarray):
            masks = torch.from_numpy(masks_hw)
        else:
            masks = masks_hw

        if masks.ndim != 3:
            raise ValueError(f"Expected masks [N,H,W], got shape {tuple(masks.shape)}")
        masks = masks.bool()

        feats, (Hf, Wf) = self.dino.compute_dense_features_from_numpy(rgb_np, target_h=target_h)
        C, _, _ = feats.shape

        masks_resized = F.interpolate(
            masks.unsqueeze(1).float(),
            size=(Hf, Wf),
            mode="nearest",
        ).squeeze(1).bool().to(feats.device)

        avg_list: list[torch.Tensor] = []
        sample_list: list[torch.Tensor] = []

        for i in range(masks_resized.shape[0]):
            m = masks_resized[i]
            idx = torch.nonzero(m, as_tuple=True)
            if idx[0].numel() == 0:
                avg = torch.zeros((C,), device=feats.device, dtype=torch.float32)
                avg_list.append(avg)
                if n_samples > 0:
                    sample_list.append(torch.zeros((n_samples, C), device=feats.device, dtype=torch.float32))
                continue

            pix = feats[:, idx[0], idx[1]]  # [C,Npix]
            avg = pix.mean(dim=1).to(torch.float32)
            avg = F.normalize(avg.unsqueeze(0), dim=1).squeeze(0)
            avg_list.append(avg)

            if n_samples > 0:
                num_pix = pix.shape[1]
                take = min(int(n_samples), int(num_pix))
                rp = torch.randperm(num_pix, device=feats.device)[:take]
                samp = pix[:, rp].T.to(torch.float32)
                samp = F.normalize(samp, dim=1)
                if take < n_samples:
                    pad = torch.zeros((n_samples - take, C), device=feats.device, dtype=torch.float32)
                    samp = torch.cat([samp, pad], dim=0)
                sample_list.append(samp)

        avg_emb = torch.stack(avg_list, dim=0).detach().cpu().numpy().astype(np.float32)
        if n_samples > 0:
            samples = torch.stack(sample_list, dim=0).detach().cpu().numpy().astype(np.float32)
        else:
            samples = np.empty((0, 0, 0), dtype=np.float32)

        return avg_emb, samples

    @staticmethod
    def build_prototype_bank_from_examples(
        labels: Sequence[str],
        embeddings: np.ndarray,
        *,
        reduction: str = "mean",
    ) -> PrototypeBankV2:
        """
        Build one prototype per label from example embeddings.

        Args:
          labels: length N
          embeddings: [N,C]
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"Expected embeddings [N,C], got {embeddings.shape}")
        if len(labels) != embeddings.shape[0]:
            raise ValueError("labels length must match embedding rows")

        grouped: dict[str, list[np.ndarray]] = {}
        for label, emb in zip(labels, embeddings):
            grouped.setdefault(str(label), []).append(emb)

        proto_labels: list[str] = []
        proto_vecs: list[np.ndarray] = []
        for label, vecs in grouped.items():
            mat = np.stack(vecs, axis=0)
            if reduction == "mean":
                proto = mat.mean(axis=0)
            elif reduction == "medoid":
                sims = mat @ mat.T
                proto = mat[np.argmax(sims.sum(axis=1))]
            else:
                raise ValueError(f"Unsupported reduction: {reduction}")

            norm = np.linalg.norm(proto)
            if norm > 1e-8:
                proto = proto / norm
            proto_labels.append(label)
            proto_vecs.append(proto.astype(np.float32))

        bank = PrototypeBankV2(labels=proto_labels, prototypes=np.stack(proto_vecs, axis=0))
        bank.validate()
        return bank

    @staticmethod
    def cosine_assign_labels(
        embeddings: np.ndarray,
        bank: PrototypeBankV2,
        *,
        similarity_threshold: float = 0.35,
        unknown_prefix: str = "obj",
    ) -> list[dict[str, object]]:
        """
        Assign labels by nearest prototype.

        Returns per-instance dicts with:
          label, score, prototype_index
        """
        bank.validate()
        embs = np.asarray(embeddings, dtype=np.float32)
        if embs.ndim != 2:
            raise ValueError(f"Expected embeddings [N,C], got {embs.shape}")
        if embs.shape[1] != bank.prototypes.shape[1]:
            raise ValueError(
                f"Embedding dim {embs.shape[1]} does not match bank dim {bank.prototypes.shape[1]}"
            )

        # Re-normalize defensively.
        embs = embs / np.clip(np.linalg.norm(embs, axis=1, keepdims=True), 1e-8, None)
        protos = bank.prototypes / np.clip(np.linalg.norm(bank.prototypes, axis=1, keepdims=True), 1e-8, None)

        sims = embs @ protos.T  # [N,K]
        best_idx = np.argmax(sims, axis=1)
        best_score = sims[np.arange(sims.shape[0]), best_idx]

        out: list[dict[str, object]] = []
        for i in range(embs.shape[0]):
            idx = int(best_idx[i])
            score = float(best_score[i])
            if score >= similarity_threshold:
                label = bank.labels[idx]
            else:
                label = f"{unknown_prefix}_{i:02d}"
            out.append(
                {
                    "label": label,
                    "score": score,
                    "prototype_index": idx,
                }
            )
        return out

    def compute_and_assign(
        self,
        *,
        rgb_np: np.ndarray,
        masks_hw: torch.Tensor | np.ndarray,
        prototype_bank: Optional[PrototypeBankV2] = None,
        target_h: int = 500,
        n_samples: int = 0,
        similarity_threshold: float = 0.35,
        unknown_prefix: str = "obj",
    ) -> dict[str, object]:
        avg_emb, samples = self.compute_mask_pooled_embeddings(
            rgb_np=rgb_np,
            masks_hw=masks_hw,
            target_h=target_h,
            n_samples=n_samples,
        )

        labels: Optional[list[dict[str, object]]] = None
        if prototype_bank is not None:
            labels = self.cosine_assign_labels(
                avg_emb,
                prototype_bank,
                similarity_threshold=similarity_threshold,
                unknown_prefix=unknown_prefix,
            )

        return {
            "dino_embeddings": avg_emb,
            "dino_embedding_samples": samples,
            "label_assignments": labels,
        }


def labels_from_json_prior_scenes(path: str | Path) -> list[str]:
    """Small convenience helper: extract unique scene labels from semantic prior JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected prior JSON to be a list of entries.")
    labels = sorted({str(row["scene"]) for row in data if isinstance(row, dict) and "scene" in row})
    return labels
