from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore
    TF = None  # type: ignore
    _TORCH_IMPORT_ERROR = e
else:
    _TORCH_IMPORT_ERROR = None

try:
    from safetensors.torch import load_file as safetensors_load
except Exception:  # pragma: no cover
    safetensors_load = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _require_torch() -> None:
    if torch is None or F is None or TF is None:
        raise RuntimeError(
            "PyTorch/torchvision are required for DinoFeaturesV2 but could not be imported. "
            f"Original import error: {_TORCH_IMPORT_ERROR}"
        )


def strip_prefixes(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("model.", "module.", "state_dict.", "backbone.")
    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        kk = k
        for p in prefixes:
            if kk.startswith(p):
                kk = kk[len(p) :]
        out[kk] = v
    return out


def load_local_state_dict(path: str) -> dict[str, torch.Tensor]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        if safetensors_load is None:
            raise RuntimeError("Install safetensors to load .safetensors DINO weights.")
        state = safetensors_load(path, device="cpu")
    else:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    return strip_prefixes(state)


def resize_to_multiple_of_numpy(
    img_np: np.ndarray,
    multiple: int,
    target_h: Optional[int] = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Resize image to target_h while preserving aspect ratio, then snap H/W down to a multiple.
    Returns resized image and (H, W).
    """
    h, w = img_np.shape[:2]
    if target_h is not None:
        scale = target_h / float(h)
        new_h = int(h * scale)
        new_w = int(w * scale)
        H = max(multiple, (new_h // multiple) * multiple)
        W = max(multiple, (new_w // multiple) * multiple)
    else:
        H = max(multiple, (h // multiple) * multiple)
        W = max(multiple, (w // multiple) * multiple)

    img_pil = Image.fromarray(img_np)
    img_resized = np.array(img_pil.resize((W, H), Image.BILINEAR))
    return img_resized, (H, W)


class _GetLast(torch.nn.Module):
    """Small wrapper so we can optionally torch.compile the 'last layer only' path."""

    def __init__(self, model: torch.nn.Module, norm: bool):
        super().__init__()
        self.model = model
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.get_intermediate_layers(
            x,
            n=[len(self.model.blocks) - 1],
            reshape=True,
            norm=self.norm,
        )[0]


class DinoFeaturesV2:
    """
    Lightweight DINOv3 dense feature wrapper for the semantic_safety repo.

    Outputs dense normalized feature maps [C, H, W] suitable for mask pooling.
    This follows the same overall idea as the SMBRF single-frame wrapper:
    DINO dense map -> resize masks to feature resolution -> average features in each mask.
    """

    def __init__(
        self,
        *,
        stride: int = 4,
        norm: bool = True,
        patch_size: int = 16,
        device: Optional[str] = None,
        weights_path: Optional[str] = None,
        compile_last_block: bool = False,
        dtype_out: "torch.dtype" = torch.bfloat16 if torch is not None else None,
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

        self.stride = int(stride)
        self.norm = bool(norm)
        self.patch_size = int(patch_size)
        self.dtype_out = dtype_out
        assert self.patch_size % self.stride == 0, "stride must divide patch_size"

        if weights_path is None:
            weights_path = os.environ.get("DINO_WEIGHTS_PATH")
        if weights_path is None:
            weights_path = str(
                Path(__file__).resolve().parent / "pretrained_models" / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
            )

        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"DINOv3 weights not found at: {weights_path}\n"
                "Set DINO_WEIGHTS_PATH or pass weights_path=..."
            )

        self.model = torch.hub.load(
            "facebookresearch/dinov3",
            "dinov3_vitl16",
            source="github",
            pretrained=False,
        ).to(self.device).eval()

        state = load_local_state_dict(weights_path)
        self.model.load_state_dict(state, strict=False)

        self.compiled_get_last = None
        if compile_last_block and hasattr(torch, "compile"):
            try:
                self.compiled_get_last = torch.compile(_GetLast(self.model, self.norm), mode="reduce-overhead")
            except Exception:
                self.compiled_get_last = _GetLast(self.model, self.norm)

    def rgb_image_preprocessing(self, rgb_im: np.ndarray, *, target_h: int = 500) -> torch.Tensor:
        resized_np, _ = resize_to_multiple_of_numpy(rgb_im, self.patch_size, target_h)
        return TF.to_tensor(resized_np)  # [3,H,W]

    def _infer_last(self, x: torch.Tensor) -> torch.Tensor:
        if self.compiled_get_last is not None:
            t = self.compiled_get_last(x)
        else:
            t = self.model.get_intermediate_layers(
                x,
                n=[len(self.model.blocks) - 1],
                reshape=True,
                norm=self.norm,
            )[0]

        if t.dim() == 4 and t.size(0) == 1:
            t = t.squeeze(0)

        s = t.shape
        c_axis = max(range(3), key=lambda i: s[i])
        if c_axis == 0:
            t = t.permute(1, 2, 0)
        elif c_axis == 1:
            t = t.permute(0, 2, 1)
        return t.contiguous()  # [H,W,C]

    @torch.inference_mode()
    def dense_embeddings(self, img_rgb: torch.Tensor) -> torch.Tensor:
        """
        Multi-shift dense embeddings on GPU. Returns [C,H,W].
        """
        Cimg, H, W = img_rgb.shape

        mean = torch.tensor(IMAGENET_MEAN, dtype=img_rgb.dtype, device=self.device)[:, None, None]
        std = torch.tensor(IMAGENET_STD, dtype=img_rgb.dtype, device=self.device)[:, None, None]
        x0 = ((img_rgb.to(self.device, non_blocking=True) - mean) / std).unsqueeze(0)

        shifts = [
            (dy, dx)
            for dy in range(0, self.patch_size, self.stride)
            for dx in range(0, self.patch_size, self.stride)
        ]

        dense = None
        for dy, dx in shifts:
            Hp = math.ceil((H + dy) / self.patch_size) * self.patch_size
            Wp = math.ceil((W + dx) / self.patch_size) * self.patch_size
            pad_bottom = Hp - (H + dy)
            pad_right = Wp - (W + dx)

            x = F.pad(x0, (dx, pad_right, dy, pad_bottom))
            feats = self._infer_last(x)  # [Ht,Wt,C]
            Ht, Wt, C = feats.shape

            if dense is None:
                dense = feats.new_full((H, W, C), float("nan"))

            ys = (torch.arange(Ht, device=feats.device) * self.patch_size - dy)
            xs = (torch.arange(Wt, device=feats.device) * self.patch_size - dx)
            vy = (ys >= 0) & (ys < H)
            vx = (xs >= 0) & (xs < W)
            if not (vy.any() and vx.any()):
                continue

            ys = ys[vy]
            xs = xs[vx]
            block = feats[vy][:, vx, :].reshape(-1, C)
            yy = ys[:, None].repeat(1, xs.numel()).reshape(-1)
            xx = xs[None, :].repeat(ys.numel(), 1).reshape(-1)
            flat_idx = yy * W + xx
            dense.view(-1, C).index_copy_(0, flat_idx, block)

        if dense is None:
            raise RuntimeError("DINO dense_embeddings failed to initialize output tensor.")

        if torch.isnan(dense).any():
            step = self.stride
            small = dense[::step, ::step, :].permute(2, 0, 1).unsqueeze(0)
            up = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
            mask = ~torch.isnan(dense).any(dim=2, keepdim=True)
            dense = torch.where(mask, dense, up)

        return dense.permute(2, 0, 1).to(self.dtype_out).contiguous()

    @torch.inference_mode()
    def get_dense_features_rgb(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized dense features [C,H,W]."""
        feats = self.dense_embeddings(rgb_tensor)
        return F.normalize(feats, dim=0)

    @torch.inference_mode()
    def compute_dense_features_from_numpy(
        self,
        rgb_np: np.ndarray,
        *,
        target_h: int = 500,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        Convenience wrapper: RGB numpy -> dense normalized features [C,Hf,Wf] and feature spatial size.
        """
        rgb_t = self.rgb_image_preprocessing(rgb_np, target_h=target_h)
        feats = self.get_dense_features_rgb(rgb_t)
        _, Hf, Wf = feats.shape
        return feats, (Hf, Wf)
