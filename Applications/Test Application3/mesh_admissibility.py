"""Mesh-based source-center admissibility and sampling for App3."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import matplotlib.tri as mtri

    _MTRI_IMPORT_ERROR = None
except Exception as exc:  # An optional Matplotlib install may also fail at import.
    mtri = None
    _MTRI_IMPORT_ERROR = exc


class MeshAdmissibility:
    """Locate and sample points in the fluid triangles of an App3 mesh."""

    def __init__(self, mesh_data, tolerance: float = 1.0e-12):
        coordinates = np.asarray(mesh_data["coord"], dtype=np.float64)
        connections = np.asarray(mesh_data["connections"], dtype=int)

        if coordinates.ndim != 2 or coordinates.shape[1] < 2:
            raise ValueError("mesh_data['coord'] must have shape (num_nodes, >=2)")
        if connections.ndim != 2 or connections.shape[1] != 3:
            raise ValueError("mesh_data['connections'] must contain triangles")
        if not np.all(np.isfinite(coordinates[:, :2])):
            raise ValueError("Mesh coordinates must be finite")
        if np.any(connections < 0) or np.any(connections >= len(coordinates)):
            raise ValueError("Triangle connectivity contains an invalid node index")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and nonnegative")

        self.node_coordinates = coordinates
        self.coordinates = coordinates
        self.connections = connections
        self.x = coordinates[:, 0].copy()
        self.y = coordinates[:, 1].copy()
        self.xmin = float(np.min(self.x))
        self.xmax = float(np.max(self.x))
        self.ymin = float(np.min(self.y))
        self.ymax = float(np.max(self.y))
        self.bounding_box = (self.xmin, self.xmax, self.ymin, self.ymax)
        self.tolerance = float(tolerance)

        self.triangle_vertices = coordinates[connections, :2]
        vertex0 = self.triangle_vertices[:, 0]
        edge01 = self.triangle_vertices[:, 1] - vertex0
        edge02 = self.triangle_vertices[:, 2] - vertex0
        self._vertex0 = vertex0
        self._edge01 = edge01
        self._edge02 = edge02
        self._determinants = self._cross2d(edge01, edge02)
        self.triangle_areas = 0.5 * np.abs(self._determinants)

        scale = max(
            self.bounding_box[1] - self.bounding_box[0],
            self.bounding_box[3] - self.bounding_box[2],
            1.0,
        )
        self._coordinate_tolerance = max(
            self.tolerance * scale,
            10.0 * np.finfo(np.float64).eps * scale,
        )
        self._triangle_min = np.min(self.triangle_vertices, axis=1)
        self._triangle_max = np.max(self.triangle_vertices, axis=1)

        valid_area = self.triangle_areas > self._coordinate_tolerance**2
        if not np.any(valid_area):
            raise ValueError("Mesh contains no nondegenerate triangles")
        self._sample_triangle_indices = np.flatnonzero(valid_area)
        sample_areas = self.triangle_areas[valid_area]
        self._sample_probabilities = sample_areas / np.sum(sample_areas)

        self._triangulation = None
        self._trifinder = None
        self.trifinder_error = None
        if mtri is not None:
            try:
                self._triangulation = mtri.Triangulation(
                    self.x, self.y, triangles=self.connections
                )
                self._trifinder = self._triangulation.get_trifinder()
            except Exception as exc:  # Matplotlib may reject malformed meshes.
                self.trifinder_error = exc
        else:
            self.trifinder_error = _MTRI_IMPORT_ERROR

    @staticmethod
    def _cross2d(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    @property
    def locator_name(self) -> str:
        return "matplotlib_trifinder" if self._trifinder is not None else "brute_force"

    def _inside_bounding_box(self, x: float, y: float) -> bool:
        xmin, xmax, ymin, ymax = self.bounding_box
        tol = self._coordinate_tolerance
        return xmin - tol <= x <= xmax + tol and ymin - tol <= y <= ymax + tol

    def _find_triangle_brute_force(self, x: float, y: float) -> Optional[int]:
        point = np.array([x, y], dtype=np.float64)
        tol_xy = self._coordinate_tolerance
        candidates = np.flatnonzero(
            np.all(point >= self._triangle_min - tol_xy, axis=1)
            & np.all(point <= self._triangle_max + tol_xy, axis=1)
            & (np.abs(self._determinants) > tol_xy**2)
        )
        if candidates.size == 0:
            return None

        relative = point - self._vertex0[candidates]
        alpha = self._cross2d(relative, self._edge02[candidates]) / self._determinants[
            candidates
        ]
        beta = self._cross2d(self._edge01[candidates], relative) / self._determinants[
            candidates
        ]
        gamma = 1.0 - alpha - beta
        barycentric_tolerance = max(self.tolerance, 1.0e-12)
        inside = (
            (alpha >= -barycentric_tolerance)
            & (beta >= -barycentric_tolerance)
            & (gamma >= -barycentric_tolerance)
        )
        if not np.any(inside):
            return None
        return int(candidates[np.flatnonzero(inside)[0]])

    def find_triangle(self, x: float, y: float) -> Optional[int]:
        """Return the containing fluid-triangle index, or ``None``."""
        x = float(x)
        y = float(y)
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        if not self._inside_bounding_box(x, y):
            return None

        if self._trifinder is not None:
            try:
                triangle_index = int(self._trifinder(x, y))
                if triangle_index >= 0:
                    return triangle_index
            except Exception as exc:
                self.trifinder_error = exc
                self._trifinder = None

        # Also check misses so points on mesh edges receive tolerant handling.
        return self._find_triangle_brute_force(x, y)

    def is_point_in_domain(self, x: float, y: float) -> bool:
        """Return whether a point lies in at least one fluid mesh triangle."""
        return self.find_triangle(x, y) is not None

    def validate_sources_in_domain(
        self, control, num_sources: Optional[int] = None
    ) -> bool:
        """Validate source centers in a flat ``[x, y, A, ...]`` control."""
        try:
            values = np.asarray(control, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return False
        if num_sources is None:
            if values.size == 0 or values.size % 3 != 0:
                return False
            num_sources = values.size // 3
        try:
            num_sources = int(num_sources)
        except (TypeError, ValueError, OverflowError):
            return False
        if num_sources <= 0 or values.size != 3 * num_sources:
            return False
        if not np.all(np.isfinite(values)):
            return False
        sources = values.reshape(num_sources, 3)
        return all(self.is_point_in_domain(x, y) for x, y in sources[:, :2])

    @staticmethod
    def _coerce_rng(rng):
        if rng is None or isinstance(rng, (int, np.integer)):
            return np.random.default_rng(rng)
        if not all(hasattr(rng, method) for method in ("choice", "random", "uniform")):
            raise TypeError("rng must be a NumPy random generator or integer seed")
        return rng

    def sample_point_in_domain(self, rng=None) -> np.ndarray:
        """Area-weight a triangle, then sample a barycentric interior point."""
        rng = self._coerce_rng(rng)
        for _attempt in range(100):
            local_index = int(
                rng.choice(
                    len(self._sample_triangle_indices), p=self._sample_probabilities
                )
            )
            triangle_index = int(self._sample_triangle_indices[local_index])
            vertices = self.triangle_vertices[triangle_index]
            root_r1 = np.sqrt(float(rng.random()))
            r2 = float(rng.random())
            weights = np.array(
                [1.0 - root_r1, root_r1 * (1.0 - r2), root_r1 * r2]
            )
            point = weights @ vertices
            if self.is_point_in_domain(point[0], point[1]):
                return np.asarray(point, dtype=np.float64)
        raise RuntimeError("Failed to sample a point inside the mesh after 100 attempts")

    def sample_control_in_domain(
        self,
        num_sources: int,
        amplitude_bounds,
        rng=None,
        min_separation: Optional[float] = None,
    ) -> np.ndarray:
        """Sample ``[x1, y1, A1, ...]`` from mesh triangles and amplitude bounds."""
        num_sources = int(num_sources)
        if num_sources <= 0:
            raise ValueError("num_sources must be positive")

        bounds = np.asarray(amplitude_bounds, dtype=np.float64)
        if bounds.shape == (2,):
            bounds = np.tile(bounds, (num_sources, 1))
        if bounds.shape != (num_sources, 2):
            raise ValueError(
                "amplitude_bounds must have shape (2,) or (num_sources, 2)"
            )
        if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError("Amplitude bounds must be finite and ordered")

        if min_separation is None:
            min_separation = 0.0
        min_separation = float(min_separation)
        if not np.isfinite(min_separation) or min_separation < 0.0:
            raise ValueError("min_separation must be finite and nonnegative")

        rng = self._coerce_rng(rng)
        centers = []
        max_attempts_per_source = 10000
        for source_id in range(num_sources):
            for _attempt in range(max_attempts_per_source):
                candidate = self.sample_point_in_domain(rng)
                if all(
                    np.linalg.norm(candidate - center) >= min_separation
                    for center in centers
                ):
                    centers.append(candidate)
                    break
            else:
                raise RuntimeError(
                    "Could not satisfy min_separation while sampling source "
                    f"{source_id + 1} after {max_attempts_per_source} attempts"
                )

        control = np.empty(3 * num_sources, dtype=np.float64)
        for source_id, center in enumerate(centers):
            control[3 * source_id : 3 * source_id + 2] = center
            control[3 * source_id + 2] = rng.uniform(
                bounds[source_id, 0], bounds[source_id, 1]
            )
        return control
