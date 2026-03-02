"""Geodesic (FMM) and Euclidean distance fields."""

from .fast_marching import geodesic_distance_field, euclidean_distance_field

__all__ = ["geodesic_distance_field", "euclidean_distance_field"]
