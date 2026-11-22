"""Mesh-based source-center admissibility and sampling for Application1.

This mirrors the App4 ``MeshAdmissibility`` API (point location,
area-weighted sampling, control sampling with amplitude bounds and minimum
separation) but drops App4's analytic building/cutout exclusion.  App4 needed an
analytic guard because its buildings were *circles*, whose curved boundaries can
leave admissible-looking linear-mesh slivers.  The DC Navy Yard buildings are
exact *polygonal* mesh holes, so a source center is admissible if and only if it
lies inside some fluid triangle.  The mesh point locator is therefore the
authoritative admissibility check: points inside any of the 21 building holes,
or outside the outer DC boundary, are not inside any triangle and are rejected.
"""

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
    """Locate and sample points in the fluid triangles of the DC mesh."""

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
        # Unique fluid-triangle vertices, used only to build a smooth restoring
        # penalty for invalid (out-of-domain) source centers. Building-hole
        # interiors contain no nodes, so the nearest fluid vertex to a center
        # inside a hole lies on the hole boundary; the restoring direction there
        # points back into the fluid street.
        self._fluid_vertex_coords = coordinates[
            np.unique(connections.reshape(-1)), :2
        ]
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

        # Lazily built domain-boundary geometry (see _ensure_boundary_geometry).
        self._boundary_a = None
        self._boundary_b = None
        self._boundary_ab = None
        self._boundary_len_sq = None
        self._triangle_centroids = None

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

    def nearest_domain_distance_sq(self, x: float, y: float):
        """Squared distance from ``(x, y)`` to the fluid mesh, with gradient.

        Returns ``(dist_sq, (dx, dy))`` where ``dist_sq`` is the squared distance
        to the nearest fluid-triangle vertex and ``(dx, dy)`` is the gradient of
        ``dist_sq`` with respect to the point (``2 * (p - nearest)``).  When the
        point already lies in the fluid domain this returns ``(0.0, (0.0, 0.0))``
        so a distance-based penalty and its gradient both vanish on admissible
        centers.  This supports the restoring invalid-iterate penalty in the
        optimization driver; it never affects admissibility decisions or valid
        objective/gradient evaluations.
        """
        x = float(x)
        y = float(y)
        if not (np.isfinite(x) and np.isfinite(y)):
            return float("inf"), (0.0, 0.0)
        if self.is_point_in_domain(x, y):
            return 0.0, (0.0, 0.0)
        point = np.array([x, y], dtype=np.float64)
        diffs = self._fluid_vertex_coords - point
        dist_sq_all = np.einsum("ij,ij->i", diffs, diffs)
        nearest = int(np.argmin(dist_sq_all))
        dist_sq = float(dist_sq_all[nearest])
        grad = 2.0 * (point - self._fluid_vertex_coords[nearest])
        return dist_sq, (float(grad[0]), float(grad[1]))

    # ------------------------------------------------------------------
    # Domain-boundary geometry: interior clearance, projection, visibility.
    #
    # ``nearest_domain_distance_sq`` above answers "how far OUTSIDE the fluid
    # domain is this point" and returns 0 inside, so it cannot measure the
    # clearance of an interior center from the nearest wall.  The helpers below
    # use the true domain boundary -- the mesh edges belonging to exactly one
    # triangle, i.e. the outer DC boundary plus every building hole -- and work
    # both inside and outside the domain.  They add no physical assumption and
    # never change what is admissible.
    # ------------------------------------------------------------------
    def _ensure_boundary_geometry(self) -> None:
        if self._boundary_a is not None:
            return
        edges = np.concatenate(
            [
                self.connections[:, [0, 1]],
                self.connections[:, [1, 2]],
                self.connections[:, [2, 0]],
            ],
            axis=0,
        )
        edges = np.sort(edges, axis=1)
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        boundary_edges = unique_edges[counts == 1]
        if boundary_edges.size == 0:
            raise ValueError("Mesh has no boundary edges; cannot build geometry")
        self._boundary_a = self.coordinates[boundary_edges[:, 0], :2]
        self._boundary_b = self.coordinates[boundary_edges[:, 1], :2]
        self._boundary_ab = self._boundary_b - self._boundary_a
        length_sq = np.einsum("ij,ij->i", self._boundary_ab, self._boundary_ab)
        length_sq[length_sq <= 0.0] = 1.0
        self._boundary_len_sq = length_sq
        self._triangle_centroids = np.mean(self.triangle_vertices, axis=1)

    @property
    def boundary_edge_count(self) -> int:
        """Number of mesh edges on the domain boundary (outer plus holes)."""
        self._ensure_boundary_geometry()
        return int(self._boundary_a.shape[0])

    def nearest_boundary_point(self, x: float, y: float):
        """Return ``(distance, point)`` for the closest point on the boundary.

        Works for points inside the fluid domain (interior wall clearance) and
        outside it (how far a rejected center is from re-entering the mesh).
        """
        self._ensure_boundary_geometry()
        point = np.array([float(x), float(y)], dtype=np.float64)
        if not np.all(np.isfinite(point)):
            return float("inf"), point
        t = np.einsum("ij,ij->i", point - self._boundary_a, self._boundary_ab)
        t = np.clip(t / self._boundary_len_sq, 0.0, 1.0)
        feet = self._boundary_a + t[:, None] * self._boundary_ab
        diffs = feet - point
        dist_sq = np.einsum("ij,ij->i", diffs, diffs)
        nearest = int(np.argmin(dist_sq))
        return float(np.sqrt(dist_sq[nearest])), feet[nearest].copy()

    def distance_to_boundary(self, x: float, y: float) -> float:
        """Distance from ``(x, y)`` to the nearest fluid-domain wall."""
        return self.nearest_boundary_point(x, y)[0]

    def signed_clearance(self, x: float, y: float) -> float:
        """Signed wall clearance: positive inside the fluid, negative outside.

        Unlike ``nearest_domain_distance_sq`` -- which is identically zero
        everywhere inside the fluid domain and so cannot express "how much
        room is left before the nearest wall" -- this uses the true boundary
        geometry (``is_point_in_domain`` plus ``distance_to_boundary`` /
        ``nearest_boundary_point``) on BOTH sides of the wall: it equals
        ``+distance_to_boundary(x, y)`` strictly inside the fluid, ``0.0`` on
        the boundary, and ``-distance_to_boundary(x, y)`` outside it (in a
        building hole or off the mesh entirely). It is continuous across the
        wall and suitable as a smooth inequality-constraint quantity (e.g.
        ``signed_clearance(x, y) - margin >= 0``) for a feasibility repair.
        """
        x = float(x)
        y = float(y)
        if not (np.isfinite(x) and np.isfinite(y)):
            return float("-inf")
        distance = self.distance_to_boundary(x, y)
        if not np.isfinite(distance):
            return float("-inf")
        return distance if self.is_point_in_domain(x, y) else -distance

    def project_into_domain(self, x: float, y: float):
        """Project a point onto the fluid domain.

        Returns ``(point, was_projected, distance_moved)``.  Admissible points
        are returned unchanged with ``was_projected=False``.  An inadmissible
        point is moved to the closest point of the domain boundary and nudged a
        hair inside so the mesh locator accepts it; the fallback for pathological
        geometry is the nearest fluid-triangle centroid, which is always strictly
        interior.  The map is continuous: as an outside point approaches the
        wall, its projection approaches the same wall point.
        """
        point = np.array([float(x), float(y)], dtype=np.float64)
        if not np.all(np.isfinite(point)):
            self._ensure_boundary_geometry()
            fallback = self._triangle_centroids[0].copy()
            return fallback, True, float("inf")
        if self.is_point_in_domain(point[0], point[1]):
            return point, False, 0.0

        distance, foot = self.nearest_boundary_point(point[0], point[1])
        inward = foot - point
        norm = float(np.hypot(inward[0], inward[1]))
        if norm > 0.0:
            direction = inward / norm
            for nudge in (0.0, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2, 0.25):
                candidate = foot + nudge * direction
                if self.is_point_in_domain(candidate[0], candidate[1]):
                    moved = float(np.hypot(*(candidate - point)))
                    return candidate, True, moved
        else:
            if self.is_point_in_domain(foot[0], foot[1]):
                return foot.copy(), True, 0.0

        self._ensure_boundary_geometry()
        diffs = self._triangle_centroids - point
        nearest = int(np.argmin(np.einsum("ij,ij->i", diffs, diffs)))
        candidate = self._triangle_centroids[nearest].copy()
        return candidate, True, float(np.hypot(*(candidate - point)))

    def segment_in_domain(
        self, start, end, spacing: float = 1.0, max_samples: int = 512
    ) -> bool:
        """Whether the straight segment ``start``-``end`` stays in the fluid mesh.

        Sampled at ``spacing`` metres (endpoints included).  Used to tell apart
        two centers that are merely close from two centers that are close but
        separated by a building: the second pair is NOT the same basin.
        """
        p = np.asarray(start, dtype=np.float64)[:2]
        q = np.asarray(end, dtype=np.float64)[:2]
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(q))):
            return False
        length = float(np.hypot(*(q - p)))
        if not (np.isfinite(spacing) and spacing > 0.0):
            raise ValueError("spacing must be finite and positive")
        samples = int(np.ceil(length / float(spacing))) + 1
        samples = int(np.clip(samples, 2, int(max_samples)))
        for t in np.linspace(0.0, 1.0, samples):
            r = p + t * (q - p)
            if not self.is_point_in_domain(r[0], r[1]):
                return False
        return True

    def segment_crosses_boundary(
        self, start, end, endpoint_tol: float = 1.0e-6
    ) -> bool:
        """Whether the OPEN segment ``start``-``end`` properly crosses a wall.

        Exact (non-sampled) segment-segment intersection against every domain
        boundary edge -- the mesh edges belonging to exactly one triangle, i.e.
        the outer DC boundary AND every building hole.  Only crossings strictly
        inside the query segment count: contact within ``endpoint_tol`` metres of
        either endpoint is ignored, so a candidate sitting exactly ON a wall
        (this mesh has a 10th-percentile wall clearance of 0 m) does not report a
        crossing against the very edge it lies on.

        This is a crossing test, not a containment test.  Because the domain is
        polygonal, "no proper crossing" implies the open segment lies entirely on
        one side of every wall -- entirely inside the fluid, or entirely inside a
        building / outside the mesh -- so a single interior point decides which.
        See ``segment_connects_in_domain``, which combines the two.
        """
        self._ensure_boundary_geometry()
        p = np.asarray(start, dtype=np.float64)[:2]
        q = np.asarray(end, dtype=np.float64)[:2]
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(q))):
            return False
        d1 = q - p
        length = float(np.hypot(d1[0], d1[1]))
        if length <= 0.0:
            return False

        d2 = self._boundary_ab
        denom = d1[0] * d2[:, 1] - d1[1] * d2[:, 0]
        # Parallel / collinear edges give denom == 0; a collinear overlap is a
        # measure-zero grazing contact and is left to the interior-point test.
        usable = np.abs(denom) > 1.0e-14
        if not np.any(usable):
            return False
        ap = self._boundary_a - p
        t = np.full(denom.shape, np.nan)
        u = np.full(denom.shape, np.nan)
        t[usable] = (
            ap[usable, 0] * d2[usable, 1] - ap[usable, 1] * d2[usable, 0]
        ) / denom[usable]
        u[usable] = (
            ap[usable, 0] * d1[1] - ap[usable, 1] * d1[0]
        ) / denom[usable]

        t_eps = min(0.49, max(float(endpoint_tol), 0.0) / length)
        hit = (
            usable
            & (t > t_eps)
            & (t < 1.0 - t_eps)
            & (u >= -1.0e-12)
            & (u <= 1.0 + 1.0e-12)
        )
        return bool(np.any(hit))

    def segment_connects_in_domain(self, start, end) -> tuple:
        """Whether two points are joined by a straight path inside the fluid.

        Returns ``(connected, info)``.  ``connected`` is True only when the
        straight segment stays in the admissible fluid domain for its whole
        length; leaving through a BUILDING wall and leaving through the OUTER
        domain boundary are both disconnections, because both boundaries are
        walls of the same fluid region.

        Combines the exact crossing test with a small number of interior sample
        points.  The crossing test alone cannot see the case where both endpoints
        sit exactly on opposite faces of one building (both crossings land on the
        endpoints and are excluded); an interior sample settles it, and by the
        polygonal argument in ``segment_crosses_boundary`` one interior point is
        already decisive.  Three are used for numerical safety in case a sample
        lands on a wall.
        """
        p = np.asarray(start, dtype=np.float64)[:2]
        q = np.asarray(end, dtype=np.float64)[:2]
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(q))):
            return False, {"reason": "non_finite_endpoint", "crossed_wall": False}
        if self.segment_crosses_boundary(p, q):
            return False, {"reason": "crosses_wall", "crossed_wall": True}
        for t in (0.25, 0.5, 0.75):
            r = p + t * (q - p)
            if not self.is_point_in_domain(r[0], r[1]):
                return False, {
                    "reason": "interior_point_outside_fluid",
                    "crossed_wall": True,
                }
        return True, {"reason": "clear_path", "crossed_wall": False}

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
