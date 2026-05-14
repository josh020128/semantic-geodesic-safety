from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class GridIndex:
    ix: int
    iy: int
    iz: int


class RiskVolumeQuery:
    """
    Query API for Loop-1 risk field artifacts.

    This does NOT generate risk maps.
    It only loads loop1_risk_field.npz and exposes:

      - world point -> voxel index
      - voxel index -> world point
      - nearest risk sampling
      - trilinear risk sampling
      - occupancy/free-space query
      - margin-inflated occupancy query

    Convention:
      x, y, z axes are assumed to be MuJoCo/world-frame coordinates.
    """

    def __init__(
        self,
        risk_field: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        occupancy_free: np.ndarray,
        table_top_z: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.risk_field = np.asarray(risk_field, dtype=np.float64)
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.z = np.asarray(z, dtype=np.float64)
        self.occupancy_free = np.asarray(occupancy_free, dtype=bool)
        self.table_top_z = table_top_z
        self.metadata = metadata or {}

        self._validate()

        self.shape = self.risk_field.shape
        self.origin = np.array([self.x[0], self.y[0], self.z[0]], dtype=np.float64)
        self.spacing = np.array(
            [
                self._axis_spacing(self.x, "x"),
                self._axis_spacing(self.y, "y"),
                self._axis_spacing(self.z, "z"),
            ],
            dtype=np.float64,
        )
        self.bounds_min = np.array([self.x[0], self.y[0], self.z[0]], dtype=np.float64)
        self.bounds_max = np.array([self.x[-1], self.y[-1], self.z[-1]], dtype=np.float64)

    @classmethod
    def from_npz(cls, path: str | Path) -> "RiskVolumeQuery":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Risk field artifact not found: {path}")

        data = np.load(path, allow_pickle=True)

        required = ["risk_field", "x", "y", "z", "occupancy_free"]
        missing = [k for k in required if k not in data.files]
        if missing:
            raise KeyError(f"Missing keys in {path}: {missing}")

        table_top_z = None
        if "table_top_z" in data.files:
            raw = np.asarray(data["table_top_z"]).reshape(-1)
            if raw.size > 0 and float(raw[0]) >= 0.0:
                table_top_z = float(raw[0])

        metadata: dict[str, Any] = {}
        if "metadata_json" in data.files:
            raw_meta = data["metadata_json"]
            try:
                if raw_meta.shape == ():
                    meta_str = str(raw_meta.item())
                else:
                    meta_str = str(raw_meta.reshape(-1)[0])
                metadata = json.loads(meta_str)
            except Exception:
                metadata = {"metadata_parse_error": str(raw_meta)}

        return cls(
            risk_field=data["risk_field"],
            x=data["x"],
            y=data["y"],
            z=data["z"],
            occupancy_free=data["occupancy_free"],
            table_top_z=table_top_z,
            metadata=metadata,
        )

    def _validate(self) -> None:
        if self.risk_field.ndim != 3:
            raise ValueError(f"risk_field must be 3D, got shape {self.risk_field.shape}")

        expected_shape = (len(self.x), len(self.y), len(self.z))
        if self.risk_field.shape != expected_shape:
            raise ValueError(
                f"risk_field shape {self.risk_field.shape} does not match axes {expected_shape}"
            )

        if self.occupancy_free.shape != self.risk_field.shape:
            raise ValueError(
                f"occupancy_free shape {self.occupancy_free.shape} "
                f"does not match risk_field shape {self.risk_field.shape}"
            )

        for name, axis in [("x", self.x), ("y", self.y), ("z", self.z)]:
            if axis.ndim != 1:
                raise ValueError(f"{name} axis must be 1D.")
            if len(axis) < 2:
                raise ValueError(f"{name} axis must have at least 2 elements.")
            if not np.all(np.diff(axis) > 0):
                raise ValueError(f"{name} axis must be strictly increasing.")

    @staticmethod
    def _axis_spacing(axis: np.ndarray, name: str) -> float:
        diffs = np.diff(axis)
        spacing = float(np.median(diffs))
        if not np.allclose(diffs, spacing, rtol=1e-3, atol=1e-6):
            raise ValueError(
                f"{name} axis is not uniformly spaced enough for fast query. "
                f"min diff={diffs.min()}, max diff={diffs.max()}"
            )
        return spacing

    def is_inside_bounds(self, world_pt: np.ndarray, pad: float = 0.0) -> bool:
        p = np.asarray(world_pt, dtype=np.float64)
        if p.shape != (3,):
            raise ValueError(f"world_pt must have shape (3,), got {p.shape}")

        return bool(
            np.all(p >= self.bounds_min - pad)
            and np.all(p <= self.bounds_max + pad)
        )

    def world_to_grid_float(self, world_pt: np.ndarray) -> np.ndarray:
        p = np.asarray(world_pt, dtype=np.float64)
        if p.shape != (3,):
            raise ValueError(f"world_pt must have shape (3,), got {p.shape}")

        return (p - self.origin) / self.spacing

    def world_to_grid_idx(
        self,
        world_pt: np.ndarray,
        *,
        clip: bool = False,
        nearest: bool = True,
    ) -> GridIndex:
        g = self.world_to_grid_float(world_pt)

        if nearest:
            idx = np.rint(g).astype(int)
        else:
            idx = np.floor(g).astype(int)

        if clip:
            idx = np.clip(idx, 0, np.array(self.shape, dtype=int) - 1)
        else:
            if np.any(idx < 0) or np.any(idx >= np.array(self.shape, dtype=int)):
                raise IndexError(
                    f"world_pt {np.round(world_pt, 4)} maps outside grid: idx={idx}, "
                    f"shape={self.shape}"
                )

        return GridIndex(int(idx[0]), int(idx[1]), int(idx[2]))

    def grid_idx_to_world(self, idx: GridIndex | tuple[int, int, int] | np.ndarray) -> np.ndarray:
        if isinstance(idx, GridIndex):
            ix, iy, iz = idx.ix, idx.iy, idx.iz
        else:
            arr = np.asarray(idx, dtype=int)
            if arr.shape != (3,):
                raise ValueError(f"idx must have shape (3,), got {arr.shape}")
            ix, iy, iz = int(arr[0]), int(arr[1]), int(arr[2])

        if not (0 <= ix < self.shape[0] and 0 <= iy < self.shape[1] and 0 <= iz < self.shape[2]):
            raise IndexError(f"idx {(ix, iy, iz)} outside grid shape {self.shape}")

        return np.array([self.x[ix], self.y[iy], self.z[iz]], dtype=np.float64)

    def sample_risk_nearest(
        self,
        world_pt: np.ndarray,
        *,
        outside_value: float = np.inf,
    ) -> float:
        if not self.is_inside_bounds(world_pt):
            return float(outside_value)

        idx = self.world_to_grid_idx(world_pt, clip=True, nearest=True)
        return float(self.risk_field[idx.ix, idx.iy, idx.iz])

    def sample_risk_trilinear(
        self,
        world_pt: np.ndarray,
        *,
        outside_value: float = np.inf,
    ) -> float:
        if not self.is_inside_bounds(world_pt):
            return float(outside_value)

        g = self.world_to_grid_float(world_pt)

        i0 = np.floor(g).astype(int)
        t = g - i0

        # Need i0 + 1 to be valid for trilinear.
        # If on the top boundary, clamp to the nearest valid cell.
        max_i0 = np.array(self.shape, dtype=int) - 2
        i0 = np.clip(i0, 0, max_i0)
        t = np.clip(t, 0.0, 1.0)

        ix, iy, iz = int(i0[0]), int(i0[1]), int(i0[2])
        tx, ty, tz = float(t[0]), float(t[1]), float(t[2])

        c000 = self.risk_field[ix,     iy,     iz]
        c100 = self.risk_field[ix + 1, iy,     iz]
        c010 = self.risk_field[ix,     iy + 1, iz]
        c110 = self.risk_field[ix + 1, iy + 1, iz]
        c001 = self.risk_field[ix,     iy,     iz + 1]
        c101 = self.risk_field[ix + 1, iy,     iz + 1]
        c011 = self.risk_field[ix,     iy + 1, iz + 1]
        c111 = self.risk_field[ix + 1, iy + 1, iz + 1]

        c00 = c000 * (1 - tx) + c100 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c11 = c011 * (1 - tx) + c111 * tx

        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty

        c = c0 * (1 - tz) + c1 * tz
        return float(c)

    def is_free(self, world_pt: np.ndarray, *, outside_is_free: bool = False) -> bool:
        if not self.is_inside_bounds(world_pt):
            return bool(outside_is_free)

        idx = self.world_to_grid_idx(world_pt, clip=True, nearest=True)
        return bool(self.occupancy_free[idx.ix, idx.iy, idx.iz])

    def is_free_idx(self, idx: GridIndex | tuple[int, int, int] | np.ndarray) -> bool:
        if isinstance(idx, GridIndex):
            ix, iy, iz = idx.ix, idx.iy, idx.iz
        else:
            arr = np.asarray(idx, dtype=int)
            ix, iy, iz = int(arr[0]), int(arr[1]), int(arr[2])

        if not (0 <= ix < self.shape[0] and 0 <= iy < self.shape[1] and 0 <= iz < self.shape[2]):
            return False

        return bool(self.occupancy_free[ix, iy, iz])

    def is_free_with_margin(self, world_pt: np.ndarray, margin_m: float) -> bool:
        """
        Conservative free-space query.

        margin_m inflates obstacles by approximately margin_m.
        This is useful for treating the end-effector as a small sphere
        instead of a zero-radius point.
        """
        if margin_m <= 0.0:
            return self.is_free(world_pt)

        if not self.is_inside_bounds(world_pt):
            return False

        margin_voxels = int(np.ceil(margin_m / float(np.min(self.spacing))))
        idx = self.world_to_grid_idx(world_pt, clip=True, nearest=True)

        ix0 = max(0, idx.ix - margin_voxels)
        ix1 = min(self.shape[0], idx.ix + margin_voxels + 1)
        iy0 = max(0, idx.iy - margin_voxels)
        iy1 = min(self.shape[1], idx.iy + margin_voxels + 1)
        iz0 = max(0, idx.iz - margin_voxels)
        iz1 = min(self.shape[2], idx.iz + margin_voxels + 1)

        local_free = self.occupancy_free[ix0:ix1, iy0:iy1, iz0:iz1]
        if local_free.size == 0:
            return False

        return bool(np.all(local_free))

    def build_inflated_free_mask(self, margin_m: float) -> np.ndarray:
        """
        Return free mask after obstacle inflation.

        occupancy_free=True means free.
        Inflating obstacles means dilating occupied voxels, then taking inverse.
        """
        if margin_m <= 0.0:
            return self.occupancy_free.copy()

        margin_voxels = int(np.ceil(margin_m / float(np.min(self.spacing))))
        occupied = ~self.occupancy_free

        structure = ndimage.generate_binary_structure(3, 1)
        inflated_occupied = ndimage.binary_dilation(
            occupied,
            structure=structure,
            iterations=margin_voxels,
        )
        return ~inflated_occupied

    @staticmethod
    def _normalize_label(label: str | None) -> str:
        if label is None:
            return ""
        return " ".join(str(label).lower().replace("_", " ").replace("-", " ").split())

    def bbox_to_mask(
        self,
        bbox_3d: list[float] | tuple[float, float, float, float, float, float] | np.ndarray,
        *,
        pad_xy_m: float = 0.0,
        pad_z_m: float = 0.0,
    ) -> np.ndarray:
        """
        Convert an axis-aligned 3D bbox into a solid voxel mask.

        bbox_3d format:
            [xmin, xmax, ymin, ymax, zmin, zmax]

        This is intentionally more conservative than the perception occupancy mask,
        which may only contain sparse observed surface voxels.
        """
        bbox = np.asarray(bbox_3d, dtype=np.float64).reshape(-1)
        if bbox.shape != (6,):
            raise ValueError(f"bbox_3d must have 6 values, got shape {bbox.shape}")

        xmin, xmax, ymin, ymax, zmin, zmax = bbox.tolist()

        xmin -= pad_xy_m
        xmax += pad_xy_m
        ymin -= pad_xy_m
        ymax += pad_xy_m
        zmin -= pad_z_m
        zmax += pad_z_m

        x_mask = (self.x >= xmin) & (self.x <= xmax)
        y_mask = (self.y >= ymin) & (self.y <= ymax)
        z_mask = (self.z >= zmin) & (self.z <= zmax)

        return x_mask[:, None, None] & y_mask[None, :, None] & z_mask[None, None, :]

    def build_planner_free_mask_from_bboxes(
        self,
        scene_objects: list[dict],
        *,
        base_free_mask: np.ndarray | None = None,
        pad_xy_m: float = 0.08,
        pad_z_m: float = 0.10,
        exclude_label_substrings: tuple[str, ...] = ("floor", "ground"),
        skip_low_table_like_support: bool = True,
        table_support_z_tolerance_m: float = 0.04,
        return_debug: bool = False,
    ):
        """
        Build a conservative planner collision mask from detected object bboxes.

        Returns:
            planner_free_mask

        or, if return_debug=True:
            planner_free_mask, debug_rows

        Why this exists:
            The saved occupancy_free from Loop 1 is good for risk generation,
            but object interiors can remain free because perception occupancy is
            based on observed surface/depth points. For planning, we want a
            conservative hard obstacle mask.

        Rule:
            - start from saved occupancy_free
            - fill each object bbox as solid obstacle
            - optionally skip the low tabletop bbox, because table_top_z already
              handles the table support plane
            - keep shelves/walls/objects as obstacles
        """
        if base_free_mask is None:
            planner_free = self.occupancy_free.copy()
        else:
            planner_free = np.asarray(base_free_mask, dtype=bool).copy()
            if planner_free.shape != self.shape:
                raise ValueError(
                    f"base_free_mask shape {planner_free.shape} does not match grid shape {self.shape}"
                )

        debug_rows: list[dict] = []

        for obj in scene_objects:
            label = self._normalize_label(obj.get("label", "unknown"))
            bbox = obj.get("bbox_3d", None)

            if bbox is None:
                bbox = obj.get("occupancy_bbox_3d", None)

            row = {
                "label": label,
                "skipped": False,
                "reason": None,
                "blocked_voxels": 0,
            }

            if bbox is None:
                row["skipped"] = True
                row["reason"] = "missing_bbox_3d"
                debug_rows.append(row)
                continue

            bbox_arr = np.asarray(bbox, dtype=np.float64).reshape(-1)
            if bbox_arr.shape != (6,):
                row["skipped"] = True
                row["reason"] = f"invalid_bbox_shape_{bbox_arr.shape}"
                debug_rows.append(row)
                continue

            if any(substr in label for substr in exclude_label_substrings):
                row["skipped"] = True
                row["reason"] = "excluded_label"
                debug_rows.append(row)
                continue

            # Important:
            # A true tabletop should not be filled as a solid obstacle above the table,
            # otherwise the entire planning workspace may become blocked.
            #
            # But a shelf may be mislabeled as "table" by open-vocabulary perception.
            # So only skip table-like objects whose bbox is close to the known table top.
            if skip_low_table_like_support and ("table" in label or "tabletop" in label):
                if self.table_top_z is not None:
                    zmax = float(bbox_arr[5])
                    if zmax <= self.table_top_z + table_support_z_tolerance_m:
                        row["skipped"] = True
                        row["reason"] = "low_table_like_support"
                        debug_rows.append(row)
                        continue

            mask = self.bbox_to_mask(
                bbox_arr,
                pad_xy_m=pad_xy_m,
                pad_z_m=pad_z_m,
            )

            blocked_voxels = int(mask.sum())
            planner_free[mask] = False

            row["blocked_voxels"] = blocked_voxels
            row["bbox_3d"] = [float(v) for v in bbox_arr.tolist()]
            debug_rows.append(row)

        if return_debug:
            return planner_free, debug_rows

        return planner_free

    def print_planner_mask_debug(
        self,
        planner_free_mask: np.ndarray,
        debug_rows: list[dict] | None = None,
    ) -> None:
        """
        Print quick comparison between original saved occupancy and planner mask.
        """
        planner_free_mask = np.asarray(planner_free_mask, dtype=bool)
        if planner_free_mask.shape != self.shape:
            raise ValueError(
                f"planner_free_mask shape {planner_free_mask.shape} does not match grid shape {self.shape}"
            )

        original_free_frac = float(self.occupancy_free.mean())
        planner_free_frac = float(planner_free_mask.mean())

        newly_blocked = self.occupancy_free & (~planner_free_mask)

        print("\n=== Planner Free Mask Debug ===")
        print(f"original free fraction : {original_free_frac:.4f}")
        print(f"planner free fraction  : {planner_free_frac:.4f}")
        print(f"newly blocked voxels   : {int(newly_blocked.sum())}")
        print(f"newly blocked fraction : {float(newly_blocked.mean()):.4f}")

        if debug_rows is not None:
            print("\nBBox obstacle rows:")
            for row in debug_rows:
                label = row.get("label", "unknown")
                skipped = row.get("skipped", False)
                reason = row.get("reason", None)
                blocked = row.get("blocked_voxels", 0)

                if skipped:
                    print(f"  - {label:<20s} skipped | reason={reason}")
                else:
                    print(f"  - {label:<20s} blocked_voxels={blocked}")

    def summary(self) -> str:
        risk_positive = self.risk_field[self.risk_field > 0.0]
        max_risk = float(self.risk_field.max())
        mean_positive = float(risk_positive.mean()) if risk_positive.size > 0 else 0.0

        lines = [
            "=== RiskVolumeQuery Summary ===",
            f"shape              : {self.shape}",
            f"spacing            : {np.round(self.spacing, 6)}",
            f"x range            : {self.x[0]:.4f} .. {self.x[-1]:.4f}",
            f"y range            : {self.y[0]:.4f} .. {self.y[-1]:.4f}",
            f"z range            : {self.z[0]:.4f} .. {self.z[-1]:.4f}",
            f"occupancy free frac: {float(self.occupancy_free.mean()):.4f}",
            f"risk max           : {max_risk:.4f}",
            f"risk positive mean : {mean_positive:.4f}",
            f"table_top_z        : {self.table_top_z}",
        ]

        if self.metadata:
            lines.append("metadata:")
            for key in [
                "manipulated_obj",
                "camera_type",
                "xml_path",
                "target_label_mode",
                "prior_json_path",
                "detection_pass_used",
                "fallback_used",
            ]:
                if key in self.metadata:
                    lines.append(f"  {key}: {self.metadata[key]}")

        return "\n".join(lines)