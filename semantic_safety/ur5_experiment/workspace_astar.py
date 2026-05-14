from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Literal

import numpy as np

from semantic_safety.ur5_experiment.risk_volume_query import (
    GridIndex,
    RiskVolumeQuery,
)


RiskReduceMode = Literal["center", "mean", "max"]


@dataclass
class AStarConfig:
    """
    Configuration for Cartesian voxel A*.

    risk_weight:
        0.0 means collision-only baseline.
        >0 means semantic risk-aware planning.

    stride:
        Coarsening factor relative to the original risk grid.
        Your current risk grid resolution is around 0.004 m, so stride=5 gives
        approximately 0.020 m planning resolution.

    risk_reduce:
        How to downsample risk into the coarse planner grid.
        - center: risk at the center voxel
        - mean: average risk inside the coarse block
        - max: conservative max risk inside the coarse block
    """

    stride: int = 5
    risk_weight: float = 0.0
    risk_normalizer: float | None = None
    risk_power: float = 1.0
    risk_reduce: RiskReduceMode = "mean"

    allow_diagonal: bool = True
    max_expansions: int = 300_000
    nearest_free_search_radius: int = 10

    # Usually keep this None for the first experiment.
    # If set, voxels with risk above this raw score are treated as blocked.
    risk_block_threshold: float | None = None

    # Table/floor clearance: when table_top_z exists and table_clearance_m > 0,
    # contributes min_z = table_top_z + table_clearance_m.
    table_clearance_m: float = 0.12

    # Optional absolute lower bound on EE z for the planner workspace (meters).
    planning_min_z_floor_m: float | None = None

    # EE point inflated as a sphere for validity (meters); 0.0 disables.
    ee_radius_m: float = 0.05

    # Segment collision check sampling spacing for edge validity.
    # Only used when ee_radius_m > 0.
    edge_check_spacing_m: float = 0.01


@dataclass
class AStarResult:
    success: bool
    message: str

    world_path: np.ndarray
    voxel_path: np.ndarray
    coarse_path: np.ndarray

    total_cost: float
    path_length_m: float
    integrated_risk: float
    max_risk: float

    num_expanded: int
    risk_weight: float
    stride: int


class WorkspaceAStar:
    """
    3D Cartesian end-effector point planner over a Loop-1 risk volume.

    State:
        coarse voxel index (i, j, k)

    Hard constraint:
        planner_free_mask == True

    Soft cost:
        semantic risk_field value

    Edge cost:
        distance_m * (1 + risk_weight * normalized_risk)

    When risk_weight = 0:
        this becomes the collision-only baseline.
    """

    def __init__(
        self,
        risk_volume: RiskVolumeQuery,
        planner_free_mask: np.ndarray | None = None,
    ) -> None:
        self.rv = risk_volume

        if planner_free_mask is None:
            self.planner_free_mask = self.rv.occupancy_free.copy()
        else:
            self.planner_free_mask = np.asarray(planner_free_mask, dtype=bool)
            if self.planner_free_mask.shape != self.rv.shape:
                raise ValueError(
                    f"planner_free_mask shape {self.planner_free_mask.shape} "
                    f"does not match risk volume shape {self.rv.shape}"
                )

    def _make_working_free_mask(self, config: AStarConfig) -> np.ndarray:
        """
        Return a copy of planner_free_mask with per-plan constraints applied.

        Important:
          Do not mutate self.planner_free_mask in-place.
        """
        working = np.asarray(self.planner_free_mask, dtype=bool).copy()

        min_z_candidates: list[float] = []

        if config.planning_min_z_floor_m is not None:
            min_z_candidates.append(float(config.planning_min_z_floor_m))

        if getattr(self.rv, "table_top_z", None) is not None and float(
            config.table_clearance_m
        ) > 0.0:
            min_z_candidates.append(
                float(self.rv.table_top_z) + float(config.table_clearance_m)
            )

        if len(min_z_candidates) > 0:
            min_z = max(min_z_candidates)
            low_z = self.rv.z < min_z
            working[:, :, low_z] = False

            if getattr(config, "verbose", False):
                print(f"[A*] Applied min_z constraint: z >= {min_z:.4f} m")

        return working

    def plan(
        self,
        start_world: np.ndarray,
        goal_world: np.ndarray,
        config: AStarConfig,
    ) -> AStarResult:
        if config.stride <= 0:
            raise ValueError("config.stride must be positive.")

        if config.risk_weight < 0.0:
            raise ValueError("config.risk_weight must be nonnegative.")

        start_world = np.asarray(start_world, dtype=np.float64)
        goal_world = np.asarray(goal_world, dtype=np.float64)

        if start_world.shape != (3,) or goal_world.shape != (3,):
            raise ValueError("start_world and goal_world must both have shape (3,).")

        working_free_mask = self._make_working_free_mask(config)

        free_c, risk_c = self._build_coarse_fields(
            stride=config.stride,
            risk_reduce=config.risk_reduce,
            free_mask=working_free_mask,
        )

        risk_normalizer = config.risk_normalizer
        if risk_normalizer is None:
            risk_normalizer = float(max(np.max(risk_c), 1.0))
        risk_normalizer = float(max(risk_normalizer, 1e-8))

        start_c_raw = self._world_to_coarse_idx(start_world, config.stride, free_c.shape)
        goal_c_raw = self._world_to_coarse_idx(goal_world, config.stride, free_c.shape)

        start_c = self._nearest_free_coarse(
            start_c_raw,
            free_c,
            risk_c,
            stride=config.stride,
            max_radius=config.nearest_free_search_radius,
            ee_margin_m=float(config.ee_radius_m),
        )
        goal_c = self._nearest_free_coarse(
            goal_c_raw,
            free_c,
            risk_c,
            stride=config.stride,
            max_radius=config.nearest_free_search_radius,
            ee_margin_m=float(config.ee_radius_m),
        )

        if start_c is None:
            return self._failure(
                "Could not find nearby free start voxel.",
                config=config,
            )

        if goal_c is None:
            return self._failure(
                "Could not find nearby free goal voxel.",
                config=config,
            )

        if start_c != start_c_raw:
            print(f"[A*] start snapped from {start_c_raw} to nearest free {start_c}")
        if goal_c != goal_c_raw:
            print(f"[A*] goal snapped from {goal_c_raw} to nearest free {goal_c}")

        neighbor_offsets = self._neighbor_offsets(config.allow_diagonal)

        open_heap: list[tuple[float, float, tuple[int, int, int]]] = []
        parent: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        g_score: dict[tuple[int, int, int], float] = {start_c: 0.0}

        h0 = self._heuristic_world(start_c, goal_c, config.stride)
        heapq.heappush(open_heap, (h0, 0.0, start_c))

        closed: set[tuple[int, int, int]] = set()
        num_expanded = 0

        while open_heap:
            _, g_current_heap, current = heapq.heappop(open_heap)

            if current in closed:
                continue

            g_current = g_score.get(current, np.inf)
            if g_current_heap > g_current + 1e-12:
                continue

            closed.add(current)
            num_expanded += 1

            if current == goal_c:
                coarse_path = self._reconstruct_path(parent, current)
                return self._make_result(
                    coarse_path=coarse_path,
                    total_cost=g_score[current],
                    risk_c=risk_c,
                    config=config,
                    num_expanded=num_expanded,
                    message="success",
                )

            if num_expanded >= config.max_expansions:
                return self._failure(
                    f"Reached max_expansions={config.max_expansions}.",
                    config=config,
                    num_expanded=num_expanded,
                )

            for offset in neighbor_offsets:
                nbr = (
                    current[0] + offset[0],
                    current[1] + offset[1],
                    current[2] + offset[2],
                )

                if not self._in_coarse_bounds(nbr, free_c.shape):
                    continue

                if not free_c[nbr]:
                    continue

                if config.risk_block_threshold is not None:
                    if float(risk_c[nbr]) > float(config.risk_block_threshold):
                        continue

                p_cur = self._coarse_to_world(current, config.stride)
                p_nbr = self._coarse_to_world(nbr, config.stride)

                if float(config.ee_radius_m) > 0.0:
                    if not self._edge_margin_ok(
                        p_cur,
                        p_nbr,
                        margin_m=float(config.ee_radius_m),
                        sample_spacing_m=float(config.edge_check_spacing_m),
                    ):
                        continue

                edge_dist = float(np.linalg.norm(p_nbr - p_cur))

                if edge_dist <= 0.0:
                    continue

                r_cur = float(risk_c[current])
                r_nbr = float(risk_c[nbr])
                avg_risk = 0.5 * (r_cur + r_nbr)
                avg_risk_norm = max(0.0, avg_risk / risk_normalizer)

                risk_multiplier = avg_risk_norm ** float(config.risk_power)
                edge_cost = edge_dist * (1.0 + float(config.risk_weight) * risk_multiplier)

                tentative_g = g_current + edge_cost

                if tentative_g < g_score.get(nbr, np.inf):
                    parent[nbr] = current
                    g_score[nbr] = tentative_g
                    h = self._heuristic_world(nbr, goal_c, config.stride)
                    f = tentative_g + h
                    heapq.heappush(open_heap, (f, tentative_g, nbr))

        return self._failure(
            "Open set exhausted; no path found.",
            config=config,
            num_expanded=num_expanded,
        )

    # ------------------------------------------------------------------
    # Coarse grid construction
    # ------------------------------------------------------------------

    def _build_coarse_fields(
        self,
        *,
        stride: int,
        risk_reduce: RiskReduceMode,
        free_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build coarse free/risk grids.

        free_c is conservative:
            a coarse cell is free only if all fine voxels in the block are free.

        risk_c:
            downsampled semantic risk.
        """
        if free_mask is None:
            free_mask = self.planner_free_mask

        fine_shape = np.array(self.rv.shape, dtype=int)
        coarse_shape = tuple(np.ceil(fine_shape / stride).astype(int).tolist())

        free_c = np.zeros(coarse_shape, dtype=bool)
        risk_c = np.zeros(coarse_shape, dtype=np.float64)

        for i in range(coarse_shape[0]):
            sx = slice(i * stride, min((i + 1) * stride, fine_shape[0]))

            for j in range(coarse_shape[1]):
                sy = slice(j * stride, min((j + 1) * stride, fine_shape[1]))

                for k in range(coarse_shape[2]):
                    sz = slice(k * stride, min((k + 1) * stride, fine_shape[2]))

                    free_block = free_mask[sx, sy, sz]
                    risk_block = self.rv.risk_field[sx, sy, sz]

                    free_c[i, j, k] = bool(np.all(free_block))

                    if risk_reduce == "center":
                        center_orig = self._coarse_to_orig_idx((i, j, k), stride)
                        risk_c[i, j, k] = float(
                            self.rv.risk_field[
                                center_orig[0],
                                center_orig[1],
                                center_orig[2],
                            ]
                        )
                    elif risk_reduce == "mean":
                        risk_c[i, j, k] = float(np.mean(risk_block))
                    elif risk_reduce == "max":
                        risk_c[i, j, k] = float(np.max(risk_block))
                    else:
                        raise ValueError(f"Unsupported risk_reduce: {risk_reduce}")

        return free_c, risk_c

    # ------------------------------------------------------------------
    # Index conversion
    # ------------------------------------------------------------------

    def _world_to_coarse_idx(
        self,
        world_pt: np.ndarray,
        stride: int,
        coarse_shape: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        idx = self.rv.world_to_grid_idx(world_pt, clip=True, nearest=True)
        fine = np.array([idx.ix, idx.iy, idx.iz], dtype=int)
        coarse = fine // stride
        coarse = np.clip(coarse, 0, np.array(coarse_shape, dtype=int) - 1)
        return int(coarse[0]), int(coarse[1]), int(coarse[2])

    def _coarse_to_orig_idx(
        self,
        coarse_idx: tuple[int, int, int],
        stride: int,
    ) -> tuple[int, int, int]:
        # Use the center fine voxel of the coarse block.
        fine = np.array(coarse_idx, dtype=int) * stride + stride // 2
        fine = np.clip(fine, 0, np.array(self.rv.shape, dtype=int) - 1)
        return int(fine[0]), int(fine[1]), int(fine[2])

    def _coarse_to_world(
        self,
        coarse_idx: tuple[int, int, int],
        stride: int,
    ) -> np.ndarray:
        fine_idx = self._coarse_to_orig_idx(coarse_idx, stride)
        return self.rv.grid_idx_to_world(np.array(fine_idx, dtype=int))

    @staticmethod
    def _in_coarse_bounds(
        idx: tuple[int, int, int],
        shape: tuple[int, int, int],
    ) -> bool:
        return (
            0 <= idx[0] < shape[0]
            and 0 <= idx[1] < shape[1]
            and 0 <= idx[2] < shape[2]
        )

    # ------------------------------------------------------------------
    # Free voxel snapping
    # ------------------------------------------------------------------

    def _nearest_free_coarse(
        self,
        requested: tuple[int, int, int],
        free_c: np.ndarray,
        risk_c: np.ndarray,
        *,
        stride: int,
        max_radius: int,
        ee_margin_m: float = 0.0,
    ) -> tuple[int, int, int] | None:
        def _coarse_margin_ok(coarse_idx: tuple[int, int, int]) -> bool:
            if float(ee_margin_m) <= 0.0:
                return True
            p = self._coarse_to_world(coarse_idx, stride)
            return bool(self.rv.is_free_with_margin(p, margin_m=float(ee_margin_m)))

        if self._in_coarse_bounds(requested, free_c.shape) and bool(free_c[requested]):
            if _coarse_margin_ok(requested):
                return requested

        req = np.array(requested, dtype=int)
        best: tuple[int, int, int] | None = None
        best_score: tuple[float, float] | None = None

        for radius in range(1, max_radius + 1):
            i0 = max(0, requested[0] - radius)
            i1 = min(free_c.shape[0], requested[0] + radius + 1)
            j0 = max(0, requested[1] - radius)
            j1 = min(free_c.shape[1], requested[1] + radius + 1)
            k0 = max(0, requested[2] - radius)
            k1 = min(free_c.shape[2], requested[2] + radius + 1)

            for i in range(i0, i1):
                for j in range(j0, j1):
                    for k in range(k0, k1):
                        cand = (i, j, k)
                        if not bool(free_c[cand]):
                            continue
                        if not _coarse_margin_ok(cand):
                            continue

                        delta = np.array(cand, dtype=int) - req
                        dist = float(np.linalg.norm(delta))
                        risk = float(risk_c[cand])
                        score = (dist, risk)

                        if best_score is None or score < best_score:
                            best = cand
                            best_score = score

            if best is not None:
                return best

        return None

    def _edge_margin_ok(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        *,
        margin_m: float,
        sample_spacing_m: float = 0.01,
    ) -> bool:
        """
        Check that the entire straight segment p0 -> p1 is free with margin.

        This is stricter than checking only the neighbor endpoint. It matters
        especially for diagonal moves, where the segment can pass close to an
        obstacle even if both endpoint voxel centers are valid.

        Args:
            p0: current world point, shape (3,)
            p1: neighbor world point, shape (3,)
            margin_m: inflated EE radius / safety margin.
            sample_spacing_m: distance between samples along the edge.

        Returns:
            True if all sampled points are free with margin.
        """
        margin_m = float(margin_m)

        if margin_m <= 0.0:
            return True

        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)

        dist = float(np.linalg.norm(p1 - p0))

        if dist <= 1e-12:
            return bool(self.rv.is_free_with_margin(p0, margin_m=margin_m))

        sample_spacing_m = max(float(sample_spacing_m), 1e-4)

        # Include both endpoints. This also covers the old p_nbr endpoint check.
        num_samples = int(np.ceil(dist / sample_spacing_m)) + 1
        num_samples = max(num_samples, 2)

        for t in np.linspace(0.0, 1.0, num_samples):
            p = (1.0 - t) * p0 + t * p1

            if not self.rv.is_free_with_margin(p, margin_m=margin_m):
                return False

        return True

    # ------------------------------------------------------------------
    # A* utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _neighbor_offsets(allow_diagonal: bool) -> list[tuple[int, int, int]]:
        offsets: list[tuple[int, int, int]] = []

        if allow_diagonal:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        offsets.append((dx, dy, dz))
        else:
            offsets = [
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            ]

        return offsets

    def _heuristic_world(
        self,
        current: tuple[int, int, int],
        goal: tuple[int, int, int],
        stride: int,
    ) -> float:
        p_cur = self._coarse_to_world(current, stride)
        p_goal = self._coarse_to_world(goal, stride)
        return float(np.linalg.norm(p_goal - p_cur))

    @staticmethod
    def _reconstruct_path(
        parent: dict[tuple[int, int, int], tuple[int, int, int]],
        current: tuple[int, int, int],
    ) -> np.ndarray:
        path = [current]
        while current in parent:
            current = parent[current]
            path.append(current)

        path.reverse()
        return np.asarray(path, dtype=np.int32)

    # ------------------------------------------------------------------
    # Result construction
    # ------------------------------------------------------------------

    def _make_result(
        self,
        *,
        coarse_path: np.ndarray,
        total_cost: float,
        risk_c: np.ndarray,
        config: AStarConfig,
        num_expanded: int,
        message: str,
    ) -> AStarResult:
        voxel_path = np.asarray(
            [
                self._coarse_to_orig_idx(
                    (int(c[0]), int(c[1]), int(c[2])),
                    config.stride,
                )
                for c in coarse_path
            ],
            dtype=np.int32,
        )

        world_path = np.asarray(
            [self.rv.grid_idx_to_world(idx) for idx in voxel_path],
            dtype=np.float64,
        )

        path_length_m = self._compute_path_length(world_path)
        integrated_risk = self._compute_integrated_risk(world_path)
        max_risk = self._compute_max_risk(world_path)

        return AStarResult(
            success=True,
            message=message,
            world_path=world_path,
            voxel_path=voxel_path,
            coarse_path=coarse_path,
            total_cost=float(total_cost),
            path_length_m=float(path_length_m),
            integrated_risk=float(integrated_risk),
            max_risk=float(max_risk),
            num_expanded=int(num_expanded),
            risk_weight=float(config.risk_weight),
            stride=int(config.stride),
        )

    def _failure(
        self,
        message: str,
        *,
        config: AStarConfig,
        num_expanded: int = 0,
    ) -> AStarResult:
        return AStarResult(
            success=False,
            message=message,
            world_path=np.zeros((0, 3), dtype=np.float64),
            voxel_path=np.zeros((0, 3), dtype=np.int32),
            coarse_path=np.zeros((0, 3), dtype=np.int32),
            total_cost=np.inf,
            path_length_m=np.inf,
            integrated_risk=np.inf,
            max_risk=np.inf,
            num_expanded=int(num_expanded),
            risk_weight=float(config.risk_weight),
            stride=int(config.stride),
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_path_length(world_path: np.ndarray) -> float:
        if len(world_path) < 2:
            return 0.0
        diffs = np.diff(world_path, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def _compute_integrated_risk(self, world_path: np.ndarray) -> float:
        if len(world_path) < 2:
            return 0.0

        risks = np.asarray(
            [self.rv.sample_risk_trilinear(p, outside_value=0.0) for p in world_path],
            dtype=np.float64,
        )

        diffs = np.diff(world_path, axis=0)
        dists = np.linalg.norm(diffs, axis=1)

        avg_risks = 0.5 * (risks[:-1] + risks[1:])
        return float(np.sum(avg_risks * dists))

    def _compute_max_risk(self, world_path: np.ndarray) -> float:
        if len(world_path) == 0:
            return 0.0

        risks = [
            self.rv.sample_risk_trilinear(p, outside_value=0.0)
            for p in world_path
        ]
        return float(np.max(risks))