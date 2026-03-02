"""Occupancy grid and boundary seeding for hazard surface S."""

from .grid import build_occupancy_grid, extract_boundary_seeds

__all__ = ["build_occupancy_grid", "extract_boundary_seeds"]
