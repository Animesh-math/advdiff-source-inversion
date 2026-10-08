"""Discrete observation operators for the transient tracking objective.

The historical (and still default) observation is a boundary integral over a
named measurement boundary.  This module adds the second supported observation:
DISCRETE MESH-NODE SENSORS.  Both are consumed through one small interface so
the objective, the RK4 discrete adjoint, and the sensitivity initializer never
branch on "which kind of measurement is this" beyond a single lookup.

Interface
---------
``observe(state)``                   C U, i.e. the observed rows of every frame
``residual(state, ud)``              C U - d
``weighted_norm_squared(r)``         per-frame series r_n^T W r_n
``objective_series(state, ud)``      the two combined (what the objective sums)
``transpose_apply(r)``               C^T W r scattered back to all mesh nodes
``adjoint_rhs_all_times(...)``       eta1 * C^T W (C U - d), what the adjoint needs
``metadata()``                       JSON-ready record of ids / coords / weights

Discrete node-sensor objective
------------------------------
For sensor nodes ``S = {j_1, ..., j_Ns}`` the observation operator extracts the
FE nodal values,

    C_s U^n = (U^n_{j_1}, ..., U^n_{j_Ns}),   r_s^n = C_s U^n - d_s^n,

and the objective uses the SAME temporal quadrature convention as the existing
boundary implementation (``gradient_test._time_reduce`` / ``_time_weights``):

    J_h = (eta1 * dt / 2) * sum_n omega_n * || C_s U^n - d_s^n ||_{W_s}^2 .

Its exact state derivative, which is what the discrete adjoint is forced with,
is therefore

    dJ_h/dU^n = eta1 * dt * omega_n * C_s^T W_s (C_s U^n - d_s^n),

so ``adjoint_rhs_all_times`` returns ``eta1 * C_s^T W_s r_s^n`` and the caller
applies the per-frame time weight exactly as it already does for the boundary
operator.  No boundary-specific logic is reused and nothing is approximated.

Spatial weighting W_s
---------------------
Two conventions are supported and are always represented EXPLICITLY (never
hidden in indexing):

* ``diagonal``  -- ordinary point sensors.  ``W_s = diag(w)`` with equal
  positive weights by default, i.e. the plain Euclidean sensor sum.
* ``matrix``    -- a general symmetric positive semi-definite ``W_s``.  This
  exists because the existing P1 boundary quadrature is NOT representable by a
  diagonal nodal weighting: expanding

      int_e (u - u_d)^2 ds ,   u|_e = n1 u_a + n2 u_b

  over a boundary edge ``e = (a, b)`` gives the CONSISTENT edge mass matrix
  ``|e| * [[1/3, 1/6], [1/6, 1/3]]``, which couples the two edge nodes.  A
  mass-lumped/trapezoidal half-edge nodal weight would change the discrete
  objective.  ``boundary_quadrature_weights`` therefore assembles the exact
  symmetric ``W_s`` from the same 3-point rule the boundary code uses, so a node
  observation on the boundary nodes reproduces the boundary objective (and its
  gradient) to floating-point roundoff.
"""

from __future__ import annotations

import numpy as np

try:  # scipy is already a hard dependency of the optimizer / adjoint solvers.
    from scipy import sparse as _sparse
except Exception:  # pragma: no cover - dense fallback keeps this importable
    _sparse = None


DIAGONAL_WEIGHTS = "diagonal"
MATRIX_WEIGHTS = "matrix"


class NodeObservation:
    """Row-selection observation operator on discrete FE mesh nodes.

    Parameters
    ----------
    node_ids:
        Solver mesh-node indices, in the order the sensors are reported.  Must
        be unique and inside the mesh; use :func:`resolve_node_ids` to validate
        and de-duplicate user input before constructing.
    weights:
        Either a 1-D array of ``Ns`` positive diagonal weights, or a symmetric
        ``(Ns, Ns)`` matrix (dense array or SciPy sparse matrix).
    coords:
        Optional ``(Ns, 2)`` sensor coordinates, recorded in the metadata.
    description / selection / extra:
        Free-form provenance recorded in the metadata for reproducibility.
    """

    kind = "nodes"

    def __init__(
        self,
        node_ids,
        weights,
        *,
        coords=None,
        description: str = "",
        selection: str = "",
        extra: dict | None = None,
    ):
        node_ids = np.asarray(node_ids, dtype=int).reshape(-1)
        if node_ids.size == 0:
            raise ValueError("A node observation needs at least one sensor node")
        if np.unique(node_ids).size != node_ids.size:
            raise ValueError("Duplicate sensor node ids in a NodeObservation")
        self.node_ids = node_ids

        weights = self._normalize_weights(weights, node_ids.size)
        self.weights = weights
        self.weight_kind = (
            DIAGONAL_WEIGHTS if isinstance(weights, np.ndarray) and weights.ndim == 1
            else MATRIX_WEIGHTS
        )
        self.coords = (
            None if coords is None
            else np.asarray(coords, dtype=np.float64).reshape(node_ids.size, -1)[:, :2]
        )
        self.description = str(description)
        self.selection = str(selection)
        self.extra = dict(extra) if extra else {}

    # -- construction helpers ------------------------------------------------
    @staticmethod
    def _normalize_weights(weights, n: int):
        if _sparse is not None and _sparse.issparse(weights):
            if weights.shape != (n, n):
                raise ValueError(
                    f"Sparse sensor weight matrix must be {(n, n)}; got {weights.shape}"
                )
            return weights.tocsr()
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim == 0:
            return np.full(n, float(weights), dtype=np.float64)
        if weights.ndim == 1:
            if weights.size != n:
                raise ValueError(
                    f"Diagonal sensor weights must have {n} entries; got {weights.size}"
                )
            if np.any(weights <= 0.0):
                raise ValueError("Diagonal sensor weights must be strictly positive")
            return weights.astype(np.float64, copy=True)
        if weights.ndim == 2:
            if weights.shape != (n, n):
                raise ValueError(
                    f"Dense sensor weight matrix must be {(n, n)}; got {weights.shape}"
                )
            return weights.astype(np.float64, copy=True)
        raise ValueError("Sensor weights must be scalar, 1-D, or 2-D")

    @property
    def num_sensors(self) -> int:
        return int(self.node_ids.size)

    # -- the observation interface ------------------------------------------
    def observe(self, state: np.ndarray) -> np.ndarray:
        """``C U``: the ``(nframes, Ns)`` sensor trace of a full state."""
        return np.asarray(state, dtype=np.float64)[:, self.node_ids]

    def residual(self, state: np.ndarray, ud: np.ndarray) -> np.ndarray:
        """``C U - d``.  ``ud`` is a full nodal state; ``d = C U_true``."""
        if callable(ud):
            raise TypeError(
                "Node-sensor observations require nodal observation data "
                "(nframes, npoin); a callable u_d(x, y, t) is boundary-only."
            )
        return self.observe(state) - self.observe(ud)

    def apply_weights(self, r: np.ndarray) -> np.ndarray:
        """``W r`` applied row-wise to an ``(nframes, Ns)`` residual block."""
        r = np.asarray(r, dtype=np.float64)
        if self.weight_kind == DIAGONAL_WEIGHTS:
            return r * self.weights[np.newaxis, :]
        if _sparse is not None and _sparse.issparse(self.weights):
            return np.asarray(self.weights.dot(r.T).T)
        return r @ self.weights

    def weighted_norm_squared(self, r: np.ndarray) -> np.ndarray:
        """Per-frame series ``r_n^T W r_n`` of an ``(nframes, Ns)`` block."""
        r = np.asarray(r, dtype=np.float64)
        # Early-time sensor values underflow to subnormals; the guard only
        # silences the benign BLAS subnormal warnings (same convention as
        # sensitivity_initialization._ObsTraceContext).
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            return np.einsum("ns,ns->n", self.apply_weights(r), r)

    def weighted_inner(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """Per-frame series ``r1_n^T W r2_n`` (bilinear form of two blocks)."""
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            return np.einsum("ns,ns->n", self.apply_weights(r1), np.asarray(r2))

    def transpose_apply(self, r: np.ndarray, npoin: int) -> np.ndarray:
        """``C^T W r``: scatter the weighted residual back to all mesh nodes."""
        out = np.zeros((np.asarray(r).shape[0], int(npoin)), dtype=np.float64)
        out[:, self.node_ids] = self.apply_weights(r)
        return out

    def objective_series(self, state: np.ndarray, ud: np.ndarray) -> np.ndarray:
        """Per-frame ``|| C U^n - d^n ||_W^2`` (the objective's spatial term)."""
        return self.weighted_norm_squared(self.residual(state, ud))

    def adjoint_rhs_all_times(
        self, state: np.ndarray, ud: np.ndarray, eta1: float, npoin: int
    ) -> np.ndarray:
        """``eta1 * C^T W (C U^n - d^n)`` for every frame.

        This is the exact transpose of the observation operator applied to the
        weighted residual, i.e. ``dJ/dU^n`` without the per-frame time weight
        (the caller applies the same time weights it uses for boundary mode).
        """
        return float(eta1) * self.transpose_apply(self.residual(state, ud), npoin)

    # -- provenance ----------------------------------------------------------
    def metadata(self) -> dict:
        record = {
            "observation_mode": self.kind,
            "num_sensors": self.num_sensors,
            "sensor_node_ids": [int(i) for i in self.node_ids],
            "weight_kind": self.weight_kind,
            "weight_convention": self.description,
            "selection": self.selection,
        }
        if self.coords is not None:
            record["sensor_coordinates"] = [
                [float(x), float(y)] for x, y in self.coords
            ]
        if self.weight_kind == DIAGONAL_WEIGHTS:
            record["sensor_weights"] = [float(w) for w in self.weights]
        else:
            diag = (
                self.weights.diagonal() if _sparse is not None
                and _sparse.issparse(self.weights) else np.diag(self.weights)
            )
            nnz = (
                int(self.weights.nnz) if _sparse is not None
                and _sparse.issparse(self.weights)
                else int(np.count_nonzero(self.weights))
            )
            record["sensor_weight_matrix"] = {
                "shape": [self.num_sensors, self.num_sensors],
                "nonzeros": nnz,
                "diagonal": [float(v) for v in np.asarray(diag).reshape(-1)],
                "row_sums": [
                    float(v)
                    for v in np.asarray(
                        self.weights.sum(axis=1)
                        if _sparse is not None and _sparse.issparse(self.weights)
                        else self.weights.sum(axis=1)
                    ).reshape(-1)
                ],
            }
        if self.extra:
            record.update(self.extra)
        return record

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"NodeObservation(num_sensors={self.num_sensors}, "
            f"weight_kind={self.weight_kind!r}, selection={self.selection!r})"
        )


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------
def resolve_node_ids(requested, npoin: int, print_fn=print):
    """Validate user-supplied solver node ids.

    Every index must exist in the mesh; duplicates are removed (first occurrence
    kept) with a clear message.  Returns the resolved ``int`` array.
    """
    requested = [int(value) for value in np.asarray(requested, dtype=int).reshape(-1)]
    if not requested:
        raise ValueError("No sensor node ids supplied")

    bad = [i for i in requested if i < 0 or i >= int(npoin)]
    if bad:
        raise ValueError(
            f"Sensor node id(s) outside the mesh (valid range 0..{int(npoin) - 1}): "
            f"{sorted(set(bad))}"
        )

    seen = {}
    for index in requested:
        seen.setdefault(index, None)
    resolved = np.asarray(list(seen.keys()), dtype=int)
    if resolved.size != len(requested):
        duplicates = sorted(
            {i for i in requested if requested.count(i) > 1}
        )
        print_fn(
            f"note: removed {len(requested) - resolved.size} duplicate sensor node "
            f"id(s) {duplicates}; keeping the first occurrence of each"
        )
    return resolved


def snap_coordinates_to_nodes(
    coords_requested,
    mesh_coord,
    tolerance: float,
    candidate_ids=None,
):
    """Snap requested ``(x, y)`` locations to the nearest mesh nodes.

    Returns ``(node_ids, records)`` where each record holds the requested
    coordinate, the resolved node id, the actual mesh-node coordinate, and the
    Euclidean snapping distance.  Raises if any snapping distance exceeds
    ``tolerance`` rather than silently snapping across the domain.

    A nearest-neighbour scan over a few thousand nodes is trivial, so no spatial
    index is built.
    """
    mesh_xy = np.asarray(mesh_coord, dtype=np.float64)[:, :2]
    candidate_ids = (
        np.arange(mesh_xy.shape[0], dtype=int)
        if candidate_ids is None
        else np.asarray(candidate_ids, dtype=int)
    )
    requested = np.asarray(coords_requested, dtype=np.float64).reshape(-1, 2)

    records = []
    node_ids = []
    for x, y in requested:
        distances = np.hypot(
            mesh_xy[candidate_ids, 0] - x, mesh_xy[candidate_ids, 1] - y
        )
        local = int(np.argmin(distances))
        node = int(candidate_ids[local])
        records.append(
            {
                "requested": [float(x), float(y)],
                "node_id": node,
                "node_coordinate": [
                    float(mesh_xy[node, 0]), float(mesh_xy[node, 1])
                ],
                "snap_distance": float(distances[local]),
            }
        )
        node_ids.append(node)

    too_far = [r for r in records if r["snap_distance"] > float(tolerance)]
    if too_far:
        detail = "; ".join(
            f"({r['requested'][0]:.4f}, {r['requested'][1]:.4f}) -> node "
            f"{r['node_id']} at ({r['node_coordinate'][0]:.4f}, "
            f"{r['node_coordinate'][1]:.4f}), {r['snap_distance']:.4f} away"
            for r in too_far
        )
        raise ValueError(
            f"Requested sensor coordinate(s) are further than the snapping "
            f"tolerance {float(tolerance)} from any mesh node: {detail}. "
            "Pick coordinates inside the fluid mesh or raise "
            "--sensor-snap-tolerance deliberately."
        )
    return np.asarray(node_ids, dtype=int), records


def boundary_quadrature_weights(coord, edges, st_weight, st_xgauss):
    """Assemble the exact P1 boundary observation weights.

    Returns ``(node_ids, W)`` where ``node_ids`` are the unique nodes of
    ``edges`` (ascending) and ``W`` is the symmetric matrix with

        r^T W r = sum_e sum_g (n1 r_a + n2 r_b)^2 w_g |e| ,

    i.e. exactly the spatial term the boundary objective and the boundary
    adjoint forcing already evaluate.  The same interval rule is passed in by
    the caller so the two implementations cannot drift apart.
    """
    coord = np.asarray(coord, dtype=np.float64)
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    if edges.size == 0:
        raise ValueError("No boundary edges supplied for the observation weights")

    node_ids = np.unique(edges.reshape(-1))
    local = {int(node): i for i, node in enumerate(node_ids)}
    n = node_ids.size

    st_weight = np.asarray(st_weight, dtype=np.float64)
    st_xgauss = np.asarray(st_xgauss, dtype=np.float64)

    rows, cols, vals = [], [], []
    for a_global, b_global in edges:
        a, b = local[int(a_global)], local[int(b_global)]
        xa, ya = coord[int(a_global), 0], coord[int(a_global), 1]
        xb, yb = coord[int(b_global), 0], coord[int(b_global), 1]
        elen = float(np.hypot(xb - xa, yb - ya))
        w_aa = w_ab = w_bb = 0.0
        for g, s in enumerate(st_xgauss):
            n1 = 1.0 - float(s)
            n2 = float(s)
            factor = float(st_weight[g]) * elen
            w_aa += n1 * n1 * factor
            w_ab += n1 * n2 * factor
            w_bb += n2 * n2 * factor
        rows.extend([a, a, b, b])
        cols.extend([a, b, a, b])
        vals.extend([w_aa, w_ab, w_ab, w_bb])

    if _sparse is not None:
        weights = _sparse.coo_matrix(
            (vals, (rows, cols)), shape=(n, n), dtype=np.float64
        ).tocsr()
    else:  # pragma: no cover - dense fallback
        weights = np.zeros((n, n), dtype=np.float64)
        for r, c, v in zip(rows, cols, vals):
            weights[r, c] += v
    return node_ids, weights


__all__ = [
    "DIAGONAL_WEIGHTS",
    "MATRIX_WEIGHTS",
    "NodeObservation",
    "boundary_quadrature_weights",
    "resolve_node_ids",
    "snap_coordinates_to_nodes",
]
