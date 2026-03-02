"""
SONATA integration: semantic 3D segmentation from point clouds.
Expects SONATA to be installed or available at repo_path (clone from facebookresearch/sonata).
Output: point cloud with segment labels for building occupancy grid and hazard masks.
"""

from pathlib import Path
from typing import Any

import numpy as np


class SonataSegmenter:
    """
    Wrapper around SONATA for semantic segmentation.
    Load point cloud dict (coord, color, normal), run SONATA, return same dict with 'segment' and optionally 'feat'.
    """

    def __init__(
        self,
        repo_path: str | Path | None = None,
        checkpoint: str = "sonata",
        repo_id: str = "facebook/sonata",
        device: str = "cuda",
    ):
        self.repo_path = Path(repo_path) if repo_path else None
        self.checkpoint = checkpoint
        self.repo_id = repo_id
        self.device = device
        self._model = None
        self._transform = None

    def _ensure_sonata(self) -> None:
        if self._model is not None:
            return
        import os
        path = self.repo_path or os.environ.get("SONATA_PATH")
        if path:
            path = Path(path)
            if (path / "sonata").is_dir():
                import sys
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
        try:
            import sonata.model
            import sonata.transform
            import torch
        except ImportError as e:
            raise ImportError(
                "SONATA not found. Clone https://github.com/facebookresearch/sonata and set "
                "SONATA_PATH or pass repo_path to SonataSegmenter, then install sonata deps."
            ) from e
        self._model = sonata.model.load(
            self.checkpoint,
            repo_id=self.repo_id,
        )
        if hasattr(self._model, "cuda") and self.device == "cuda":
            self._model = self._model.cuda()
        self._transform = sonata.transform.default()

    def segment(
        self,
        point: dict[str, np.ndarray],
        return_features: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Run SONATA on point dict with keys: coord (N,3), color (N,3), normal (N,3).
        Returns same dict with 'segment' (N,) labels; if return_features, adds 'feat' (N, D).
        """
        self._ensure_sonata()
        import torch

        transformed = self._transform(point)
        dev = next(self._model.parameters(), torch.tensor(0)).device
        for key in list(transformed.keys()):
            if isinstance(transformed[key], torch.Tensor):
                transformed[key] = transformed[key].to(dev, non_blocking=True)

        out = self._model(transformed)

        # Map features back to original point count (see SONATA README)
        for _ in range(2):
            assert "pooling_parent" in out.keys()
            assert "pooling_inverse" in out.keys()
            parent = out.pop("pooling_parent")
            inverse = out.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, out.feat[inverse]], dim=-1)
            out = parent
        while "pooling_parent" in out.keys():
            assert "pooling_inverse" in out.keys()
            parent = out.pop("pooling_parent")
            inverse = out.pop("pooling_inverse")
            parent.feat = out.feat[inverse]
            out = parent
        feat = out.feat[out.inverse].detach().cpu().numpy()

        result = dict(point)
        if return_features:
            result["feat"] = feat
        # If you have a linear probe / classifier for ScanNet-style labels, apply it to feat here.
        # For now we don't assume a trained head; segment can be placeholder or from another source.
        if "segment" not in result:
            result["segment"] = np.zeros(len(point["coord"]), dtype=np.int32)
        return result
