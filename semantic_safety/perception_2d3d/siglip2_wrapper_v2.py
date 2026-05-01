from __future__ import annotations

import hashlib
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor


@dataclass
class SigLIP2EncodeOutput:
    embeddings: torch.Tensor
    items: list[str] | None = None


class SigLIP2WrapperV2:
    """
    Thin wrapper around a Hugging Face SigLIP 2 model.

    Main responsibilities:
      - load processor + model
      - encode text labels
      - encode RGB crops / masked crops
      - compute image-text similarity

    Notes
    -----
    - This wrapper does NOT perform canonicalization by itself.
    - This wrapper does NOT require a candidate list from prior JSON.
    - It can be used with any vocabulary bank (e.g. LVIS + custom labels).
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        device: Optional[str] = None,
        dtype: Optional[str] = "auto",
        batch_size: int = 32,
        normalize: bool = True,
        prepend_photo_prompt: bool = False,
    ) -> None:
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.normalize = bool(normalize)
        self.prepend_photo_prompt = bool(prepend_photo_prompt)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval().to(self.device)

        self.compute_dtype = self._resolve_dtype(dtype)
        self._text_cache: dict[str, torch.Tensor] = {}

    def _resolve_dtype(self, dtype: Optional[str]) -> torch.dtype:
        if dtype is None or dtype == "auto":
            if self.device.type == "cuda":
                return torch.float16
            return torch.float32

        dtype = str(dtype).lower()
        if dtype in {"fp16", "float16", "half"}:
            return torch.float16
        if dtype in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype in {"fp32", "float32"}:
            return torch.float32

        raise ValueError(f"Unsupported dtype spec: {dtype}")

    def _autocast_context(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.compute_dtype)
        return nullcontext()

    @staticmethod
    def _to_pil_rgb(image: np.ndarray | Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected np.ndarray or PIL.Image, got {type(image)}")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got shape {image.shape}")

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        return Image.fromarray(image, mode="RGB")

    def _prepare_texts(self, texts: Sequence[str]) -> list[str]:
        cleaned: list[str] = []
        for t in texts:
            s = str(t).strip()
            if self.prepend_photo_prompt:
                s = f"a photo of {s}"
            cleaned.append(s)
        return cleaned

    @staticmethod
    def _normalize_embeddings(x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1)

    @staticmethod
    def _hash_texts(texts: Sequence[str]) -> str:
        joined = "\n".join(texts)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def _extract_feature_tensor(self, output, kind: str) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output

        if kind == "text":
            for attr in ["text_embeds", "pooler_output"]:
                val = getattr(output, attr, None)
                if isinstance(val, torch.Tensor):
                    return val
        elif kind == "image":
            for attr in ["image_embeds", "pooler_output"]:
                val = getattr(output, attr, None)
                if isinstance(val, torch.Tensor):
                    return val

        last_hidden = getattr(output, "last_hidden_state", None)
        if isinstance(last_hidden, torch.Tensor):
            if last_hidden.ndim == 3:
                return last_hidden[:, 0, :]
            return last_hidden

        raise RuntimeError(
            f"Could not extract {kind} features from model output of type {type(output)}"
        )

    def encode_texts(
        self,
        texts: Sequence[str],
        use_cache: bool = True,
        return_cpu: bool = True,
    ) -> torch.Tensor:
        """
        Encode a sequence of text labels into a [N, D] tensor.
        """
        texts = self._prepare_texts(texts)
        if len(texts) == 0:
            raise ValueError("encode_texts received an empty text list.")

        cache_key = self._hash_texts(texts)
        if use_cache and cache_key in self._text_cache:
            out = self._text_cache[cache_key]
            return out.cpu() if return_cpu else out

        outputs: list[torch.Tensor] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            inputs = self.processor(
                text=batch,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                with self._autocast_context():
                    if hasattr(self.model, "get_text_features"):
                        raw_out = self.model.get_text_features(**inputs)
                    else:
                        raw_out = self.model(**inputs)

            feats = self._extract_feature_tensor(raw_out, kind="text").float()
            if self.normalize:
                feats = self._normalize_embeddings(feats)
            outputs.append(feats)

        text_embeds = torch.cat(outputs, dim=0)
        if use_cache:
            self._text_cache[cache_key] = text_embeds.detach().cpu()

        return text_embeds.cpu() if return_cpu else text_embeds

    def encode_images(
        self,
        images: Sequence[np.ndarray | Image.Image],
        return_cpu: bool = True,
    ) -> torch.Tensor:
        """
        Encode a sequence of RGB images into a [N, D] tensor.
        """
        if len(images) == 0:
            raise ValueError("encode_images received an empty image list.")

        pil_images = [self._to_pil_rgb(img) for img in images]
        outputs: list[torch.Tensor] = []

        for start in range(0, len(pil_images), self.batch_size):
            batch = pil_images[start : start + self.batch_size]
            inputs = self.processor(
                images=batch,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                with self._autocast_context():
                    if hasattr(self.model, "get_image_features"):
                        raw_out = self.model.get_image_features(**inputs)
                    else:
                        raw_out = self.model(**inputs)

            feats = self._extract_feature_tensor(raw_out, kind="image").float()
            if self.normalize:
                feats = self._normalize_embeddings(feats)
            outputs.append(feats)

        image_embeds = torch.cat(outputs, dim=0)
        return image_embeds.cpu() if return_cpu else image_embeds

    def compute_similarity(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity matrix [N_images, N_texts].
        """
        if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D embeddings, got image {tuple(image_embeddings.shape)} "
                f"and text {tuple(text_embeddings.shape)}"
            )

        img = image_embeddings.float()
        txt = text_embeddings.float()

        if not self.normalize:
            img = self._normalize_embeddings(img)
            txt = self._normalize_embeddings(txt)

        return img @ txt.T

    def score_images_against_texts(
        self,
        images: Sequence[np.ndarray | Image.Image],
        texts: Sequence[str],
        text_embeddings: Optional[torch.Tensor] = None,
        return_cpu: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """
        Convenience method:
          - encode images
          - encode texts (or reuse precomputed text embeddings)
          - return similarity matrix

        Returns
        -------
        sim: [N_images, N_texts]
        image_embeds: [N_images, D]
        processed_texts: list[str]
        """
        processed_texts = self._prepare_texts(texts)
        image_embeds = self.encode_images(images, return_cpu=False).float()
        target_device = image_embeds.device

        if text_embeddings is None:
            text_embeddings = self.encode_texts(
                processed_texts,
                use_cache=True,
                return_cpu=False,
            )
        text_embeddings = text_embeddings.to(target_device).float()

        if self.normalize:
            text_embeddings = self._normalize_embeddings(text_embeddings)

        sim = self.compute_similarity(image_embeds, text_embeddings)

        if return_cpu:
            return sim.cpu(), image_embeds.cpu(), processed_texts
        return sim, image_embeds, processed_texts

    def topk_labels(
        self,
        similarity: torch.Tensor,
        texts: Sequence[str],
        k: int = 5,
    ) -> list[list[tuple[str, float]]]:
        """
        Convert similarity matrix into top-k label predictions per image.
        """
        if similarity.ndim != 2:
            raise ValueError(f"Expected similarity [N, M], got {tuple(similarity.shape)}")

        texts = list(texts)
        k = max(1, min(int(k), len(texts)))

        sim = similarity.float()
        vals, idxs = torch.topk(sim, k=k, dim=1)

        results: list[list[tuple[str, float]]] = []
        for row_vals, row_idxs in zip(vals, idxs):
            row: list[tuple[str, float]] = []
            for v, j in zip(row_vals.tolist(), row_idxs.tolist()):
                row.append((texts[j], float(v)))
            results.append(row)
        return results