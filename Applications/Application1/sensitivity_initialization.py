r"""Greedy adjoint-sensitivity initialization for Application1.

This is a completely optional initialization strategy (``--init-mode sensitivity``)
that greedily places the requested number of unit-amplitude Gaussian sources by
following the adjoint source-insertion sensitivity field, then hands the assembled
control to the existing full optimizer for joint refinement.  It adds NO new
forward/adjoint/objective machinery: it reuses

  * ``source_insertion_sensitivity`` (``sis``) for the exact discrete insertion
    sensitivity field  G(c) = dJ/dA_new  (one forward + one adjoint per greedy
    step; every candidate center is an inexpensive Gaussian-weighted quadrature
    sum, NOT a PDE solve),
  * ``gradient_test`` (``gt``) for the forward solver and the exact discrete
    tracking objective (boundary interval quadrature + time reduction), used to
    form the amplitude curvature, and
  * ``MeshAdmissibility`` for the physical admissibility mask.

Discrete formulas actually implemented (matching the optimizer's own discrete
inner products; see ``source_insertion_sensitivity`` and
``Test Application2/gradient_test.py``):

  Insertion sensitivity (searched/plotted field), for a unit Gaussian phi_c at
  candidate center c and the current residual r = C u(q_current) - y_d:

      G(c) = < source_adjoint , inv_mass * assemble_load_vector(phi_c) >
           = eta1 * T_reduce( \int_Gamma_obs r * delta_u_c ds )               (1)

  where source_adjoint is the exact RK4 discrete source adjoint driven by r on
  the (possibly union) measurement boundary.  This equals dJ/dA_new at amplitude
  0, carrying the objective's sign convention (G(c) < 0 => a positive-amplitude
  insertion at c descends J).

  Amplitude along the one-dimensional insertion direction (closed form):

      a*(c) = - G(c) / kappa(c),                                              (2)
      kappa(c) = eta1 * T_reduce( \int_Gamma_obs |delta_u_c|^2 ds ) + lambda  (3)
               = 2 * J_track(delta_u_c ; u_d = 0)  +  lambda,

  with delta_u_c = C times the state response to the UNIT Gaussian at c (one
  forward solve, using the SAME boundary/time discrete norm the objective uses).
  ``lambda`` is the source Tikhonov weight (Application1 uses LAMBDA_REG = 0, so
  the regularization contributes nothing, but it is carried for correctness).
  a* is then projected onto the existing amplitude bounds
  [amplitude_floor, amplitude_upper].

Because the transport PDE is linear in the source with homogeneous initial and
left-Dirichlet conditions, delta_u_c is independent of the current background
q_current and is obtained from the existing forward solver without any duplicate
solver.

Scoring modes (``score`` parameter / ``--sensitivity-score``):

  * ``raw_g`` (original behavior): c* = argmin_c G(c) over the masked candidates.
    KNOWN LIMITATION: G(c) rewards candidate observability (large kappa near
    measurement boundaries), not explanatory power; on Application1 it selects
    hyper-observable near-boundary nodes over the true source basins.

  * ``predicted_reduction``: because the PDE is linear in the source, the exact
    objective change of inserting amplitude a at c is the quadratic

        DeltaJ(c, a) = a G(c) + 0.5 a^2 kappa(c),                             (4)

    minimized over the bounded amplitude a in [floor, upper].  A top-K screen
    first selects K basin-diverse minima of G(c) (pairwise distance >= a
    diversity radius), then one unit forward solve per screened candidate gives
    kappa(c) and the EXACT bounded DeltaJ(c); the candidate with the smallest
    DeltaJ is selected.  This equals the exact 1-D objective reduction, not an
    approximation.

  Beam search (``beam_width`` > 1, requires ``predicted_reduction``): retains
  the best B first-source candidates, expands each with its own
  residual/adjoint recomputation and second-source screen, JOINTLY refits all
  amplitudes (small box-constrained QP in the exact weighted observation norm),
  scores every complete control by its exact objective (evaluated from cached
  unit boundary traces by linearity -- no extra PDE solves), and returns the
  best admissible control.

Joint amplitude solver (``amplitude_solver`` / ``--amplitude-solver``): for
fixed source centers the joint amplitude fit is the box-constrained QP

    min_{A_min <= a <= A_max}  0.5 a^T H a - a^T h,                          (5)

    H[i, j] = < delta_u_{c_i} , delta_u_{c_j} >_obs + lambda delta_ij,
    h[i]    = < delta_u_{c_i} , u_d >_obs,

in the SAME weighted observation/time inner product the objective uses.  Two
solvers solve this identical problem:

  * ``enumeration``: the original exact 3^n active-set enumeration
    (``_solve_box_qp``).  Exact, dependency-free, exponential, and refused for
    n > ``_ENUMERATION_MAX_SOURCES``.
  * ``osqp`` (default): the same QP handed to OSQP as P = H (symmetrized),
    q = -h, A = I, l = A_min, u = A_max.  No source-count cap, so it covers the
    intended 8-10 source problems.  Non-convergence, a bad termination status or
    a bound violation RAISE; there is no silent fallback and no silently added
    regularization.

Both receive byte-identical H and h, including the caller's ``lambda``
Tikhonov term; the solver choice does not touch the single-source ranking
formula (2)-(3), which stays in closed form.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

import gradient_test as gt  # noqa: F401  (Test Application2 forward/objective; on path via run_optimization)
import source_insertion_sensitivity as sis
from mesh_admissibility import MeshAdmissibility


# Placeholder control passed to the forward solver / objective when the source
# field is supplied through ``source_func`` (background sum) or a precomputed
# ``state_out``, and to the objective when the regularization weight is 0.  In
# those paths the control vector is never read, so its length is irrelevant.
_DUMMY_Z = np.zeros(3, dtype=np.float64)


@dataclass
class SensitivityStepDiagnostics:
    """Concise per-greedy-source diagnostic record.

    The trailing optional fields are populated only by the
    ``predicted_reduction`` scoring path; the original ``raw_g`` path leaves
    them at their defaults so existing consumers are unaffected.
    """

    greedy_index: int
    center: List[float]
    min_masked_sensitivity: float
    unconstrained_amplitude: float
    projected_amplitude: float
    observation_curvature: float
    num_admissible_candidates: int
    mask_changed_selection: bool
    separation_constraint_active: bool
    amplitude_clipped: bool
    score: str = "raw_g"
    raw_g_at_center: Optional[float] = None
    normalized_score: Optional[float] = None
    predicted_reduction: Optional[float] = None
    objective_before: Optional[float] = None
    objective_after: Optional[float] = None
    rank_among_screened: Optional[int] = None
    distance_to_previous: Optional[float] = None
    selection_changed_by_score: Optional[bool] = None
    n_forward_solves: Optional[int] = None
    n_adjoint_solves: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "greedy_index": self.greedy_index,
            "center": self.center,
            "min_masked_sensitivity": self.min_masked_sensitivity,
            "unconstrained_amplitude": self.unconstrained_amplitude,
            "projected_amplitude": self.projected_amplitude,
            "observation_curvature": self.observation_curvature,
            "num_admissible_candidates": self.num_admissible_candidates,
            "mask_changed_selection": self.mask_changed_selection,
            "separation_constraint_active": self.separation_constraint_active,
            "amplitude_clipped": self.amplitude_clipped,
            "score": self.score,
            "raw_g_at_center": self.raw_g_at_center,
            "normalized_score": self.normalized_score,
            "predicted_reduction": self.predicted_reduction,
            "objective_before": self.objective_before,
            "objective_after": self.objective_after,
            "rank_among_screened": self.rank_among_screened,
            "distance_to_previous": self.distance_to_previous,
            "selection_changed_by_score": self.selection_changed_by_score,
            "n_forward_solves": self.n_forward_solves,
            "n_adjoint_solves": self.n_adjoint_solves,
        }


class SensitivityInitializationError(RuntimeError):
    """Raised when the greedy placement cannot continue gracefully."""


def _coherence_from_norms(
    inner_t1t2: float, norm2_t1: float, norm2_t2: float
) -> float:
    """Safe dimensionless coherence ``|<t1,t2>| / (||t1|| ||t2||)``.

    Takes the (already computed) observation-space inner product ``<t1,t2>`` and
    the squared observation norms ``||t1||^2``, ``||t2||^2``.  A zero / near-zero
    response norm returns ``0.0`` -- a vanishing response is not coherent with
    anything -- so the coherence cap never divides by zero or yields a NaN.
    Module-level (not a closure) so it is directly unit-testable.
    """
    denom = np.sqrt(max(float(norm2_t1), 0.0)) * np.sqrt(max(float(norm2_t2), 0.0))
    if not np.isfinite(denom) or denom <= 1.0e-300:
        return 0.0
    val = abs(float(inner_t1t2)) / denom
    return float(val) if np.isfinite(val) else 0.0


def centers_same_basin(
    center_a,
    center_b,
    dedup_radius: float,
    admissibility: MeshAdmissibility,
    geometry_aware: bool = False,
) -> bool:
    """Whether two placed centers represent the same placement basin.

    Distance alone is not enough in a street network: two centers a few metres
    apart can sit against opposite faces of the same building, which are
    different streets and genuinely different hypotheses.  With
    ``geometry_aware`` the centers must ALSO be mutually visible through the
    fluid domain.  Visibility is stricter than connectivity, so this can only
    keep more branches, never merge branches that the distance rule kept apart.
    """
    close = float(
        np.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])
    ) < float(dedup_radius)
    if not close or not geometry_aware:
        return bool(close)
    return bool(admissibility.segment_in_domain(center_a, center_b))


def branches_same_basin(
    centers_a,
    centers_b,
    dedup_radius: float,
    admissibility: MeshAdmissibility,
    geometry_aware: bool = False,
) -> bool:
    """Whether two branches are duplicates (every paired center same basin)."""
    if len(centers_a) != len(centers_b) or not centers_a:
        return False
    return all(
        centers_same_basin(a, b, dedup_radius, admissibility, geometry_aware)
        for a, b in zip(centers_a, centers_b)
    )


def _outer_boundary_node_coords(mesh_data, coord: np.ndarray) -> np.ndarray:
    """Coordinates of all non-building (outer) boundary nodes, for the buffer mask."""
    nodes_by_name = mesh_data.get("boundary_nodes_by_name", {})
    idx: List[int] = []
    for name, nodes in nodes_by_name.items():
        if name.startswith("__") or name.startswith("building_"):
            continue
        idx.extend(int(v) for v in np.asarray(nodes, dtype=int).reshape(-1))
    if not idx:
        return np.empty((0, 2), dtype=np.float64)
    return coord[np.unique(np.asarray(idx, dtype=int)), :2]


def _building_boundary_node_coords(mesh_data, coord: np.ndarray) -> np.ndarray:
    """Coordinates of all building-wall boundary nodes, for the building buffer mask."""
    nodes_by_name = mesh_data.get("boundary_nodes_by_name", {})
    idx: List[int] = []
    for name, nodes in nodes_by_name.items():
        if name.startswith("building_"):
            idx.extend(int(v) for v in np.asarray(nodes, dtype=int).reshape(-1))
    if not idx:
        return np.empty((0, 2), dtype=np.float64)
    return coord[np.unique(np.asarray(idx, dtype=int)), :2]


def _base_admissibility_mask(
    coord: np.ndarray,
    admissibility: MeshAdmissibility,
    outer_boundary_xy: np.ndarray,
    building_boundary_xy: np.ndarray,
    boundary_buffer: float,
    building_buffer: float,
) -> np.ndarray:
    """Static (separation-independent) admissibility mask over candidate nodes.

    A candidate node is admissible iff it is inside the fluid mesh domain (which
    already excludes building interiors and the exterior, i.e. non-fluid nodes),
    and, when the optional buffers are positive, it is at least that far from the
    outer boundary / building walls.  With both buffers at their default 0.0 no
    extra physical assumption is imposed beyond the mesh admissibility itself.
    """
    n = coord.shape[0]
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        mask[i] = admissibility.is_point_in_domain(float(coord[i, 0]), float(coord[i, 1]))

    if boundary_buffer > 0.0 and outer_boundary_xy.shape[0] > 0:
        for i in np.flatnonzero(mask):
            d = np.min(np.hypot(outer_boundary_xy[:, 0] - coord[i, 0],
                                outer_boundary_xy[:, 1] - coord[i, 1]))
            if d < boundary_buffer:
                mask[i] = False

    if building_buffer > 0.0 and building_boundary_xy.shape[0] > 0:
        for i in np.flatnonzero(mask):
            d = np.min(np.hypot(building_boundary_xy[:, 0] - coord[i, 0],
                                building_boundary_xy[:, 1] - coord[i, 1]))
            if d < building_buffer:
                mask[i] = False

    return mask


def _separation_mask(
    coord: np.ndarray,
    placed_centers: Sequence[np.ndarray],
    min_separation: Optional[float],
) -> np.ndarray:
    """Nodes at least ``min_separation`` from every already-placed center."""
    n = coord.shape[0]
    if not placed_centers or not min_separation or min_separation <= 0.0:
        return np.ones(n, dtype=bool)
    keep = np.ones(n, dtype=bool)
    for c in placed_centers:
        dist = np.hypot(coord[:, 0] - c[0], coord[:, 1] - c[1])
        keep &= dist >= float(min_separation)
    return keep


class _ObsTraceContext:
    """Vectorized evaluator of the EXACT discrete observation norm.

    Boundary mode reproduces ``gradient_test.compute_objective_parts``
    bit-for-bit: for each measurement-boundary edge (a, b) with length ``elen``
    and the 3-point Gauss rule on [0, 1],

        ||x||_obs^2 = eta1 * T_reduce_n( sum_e sum_g (x_g)^2 w_g elen_e ),

    where ``x_g = n1 x[n, a] + n2 x[n, b]``.  States are reduced to their
    boundary-node traces once (``extract``), after which norms, inner products,
    and exact objectives of linear source combinations need NO further PDE data.

    Node-sensor mode (``mesh_data["observation"]``) keeps the identical
    structure: ``extract`` selects the sensor rows and the per-step series is the
    weighted form ``x_n^T W x_n`` of the SAME observation operator the objective
    and the discrete adjoint use, so every cached-trace shortcut in the
    initializer stays exact.
    """

    def __init__(self, mesh_data, eta1: float, dt_out: float, time_rule: str):
        self.eta1 = float(eta1)
        self.dt_out = float(dt_out)
        self.time_rule = str(time_rule)
        self.observation = gt.observation_operator(mesh_data)

        if self.observation is not None:
            self.bnodes = np.asarray(self.observation.node_ids, dtype=int)
            return

        coord = np.asarray(mesh_data["coord"], dtype=np.float64)
        edges = np.asarray(
            mesh_data["boundary_edges_by_name"][
                mesh_data.get("measurement_boundary", "right")
            ],
            dtype=int,
        )
        # Unique boundary nodes and local edge-endpoint indices into the trace.
        self.bnodes = np.unique(edges.reshape(-1))
        local = {int(n): i for i, n in enumerate(self.bnodes)}
        self._ia = np.array([local[int(e[0])] for e in edges], dtype=int)
        self._ib = np.array([local[int(e[1])] for e in edges], dtype=int)
        xa, ya = coord[edges[:, 0], 0], coord[edges[:, 0], 1]
        xb, yb = coord[edges[:, 1], 0], coord[edges[:, 1], 1]
        self._elen = np.hypot(xb - xa, yb - ya)  # (n_edges,)
        st_weight, st_x = gt.interval_quadrature_0_to_1(3)
        self._n1 = 1.0 - st_x  # (3,)
        self._n2 = st_x
        self._wg = st_weight

    def extract(self, state: np.ndarray) -> np.ndarray:
        """Observed trace (nt, n_obs) of a full state (nt, npoin)."""
        return np.asarray(state, dtype=np.float64)[:, self.bnodes].copy()

    def _series(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        """Per-time-step spatial observation form of t1 * t2."""
        if self.observation is not None:
            return self.observation.weighted_inner(t1, t2)
        ua1, ub1 = t1[:, self._ia], t1[:, self._ib]  # (nt, ne)
        ua2, ub2 = t2[:, self._ia], t2[:, self._ib]
        series = np.zeros(t1.shape[0], dtype=np.float64)
        # Early-time boundary values underflow to subnormals; the errstate guard
        # only silences the benign BLAS subnormal warnings (cf.
        # source_insertion_sensitivity.sensitivity_at_candidates).
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            for g in range(self._wg.size):
                v1 = self._n1[g] * ua1 + self._n2[g] * ub1
                v2 = self._n1[g] * ua2 + self._n2[g] * ub2
                series += self._wg[g] * ((v1 * v2) @ self._elen)
        return series

    def inner(self, t1: np.ndarray, t2: np.ndarray) -> float:
        """eta1-weighted space-time inner product of two boundary traces."""
        return self.eta1 * gt._time_reduce(
            self._series(t1, t2), self.dt_out, self.time_rule
        )

    def norm2(self, t: np.ndarray) -> float:
        return self.inner(t, t)

    def tracking_objective(self, t_model: np.ndarray, t_data: np.ndarray) -> float:
        """0.5 * ||t_model - t_data||_obs^2 (the exact discrete tracking term)."""
        diff = t_model - t_data
        return 0.5 * self.norm2(diff)


#: ``--amplitude-solver`` values. Both solve the SAME joint amplitude problem
#: (same H, h, observation/time weights, amplitude bounds and caller-applied
#: Tikhonov term); they differ only in how the box-constrained QP is solved.
AMPLITUDE_SOLVER_ENUMERATION = "enumeration"
AMPLITUDE_SOLVER_OSQP = "osqp"
AMPLITUDE_SOLVERS = (AMPLITUDE_SOLVER_ENUMERATION, AMPLITUDE_SOLVER_OSQP)

#: Default joint amplitude solver for the Application1 initialization pipeline
#: (``build_sensitivity_initial_control`` and the ``--amplitude-solver`` CLI).
#: OSQP is the default because the 3^n enumeration is exponential and caps out
#: at ``_ENUMERATION_MAX_SOURCES``, while the intended 8-10 source problems need
#: a solver whose cost does not grow with the number of active-set assignments.
#: The low-level ``_solve_box_qp`` helper still defaults to the exact
#: enumeration so direct (diagnostic / reference) callers are unchanged.
DEFAULT_AMPLITUDE_SOLVER = AMPLITUDE_SOLVER_OSQP

#: Largest ``n`` the 3^n active-set enumeration is allowed to attempt. This is a
#: property of the ENUMERATION solver only -- it does not constrain OSQP.
_ENUMERATION_MAX_SOURCES = 8

# OSQP settings. The amplitude QP is tiny (n = number of sources), so a tight
# tolerance costs microseconds. ``polishing`` runs OSQP's exact active-set
# polish step on the converged iterate, which is what turns an ADMM solution
# into one that satisfies the KKT system to ~1e-15.
#
# eps_abs is a floor for a well-scaled problem; eps_rel is what actually carries
# the scale here, since the Gram entries of a production problem reach ~1e8.
# A tiny QP that has not converged in a few thousand ADMM iterations is not
# going to, so the iteration cap stays modest rather than burning 2e5 sweeps on
# a hopeless problem (measured: the production two-source case converges in 100).
OSQP_EPS_ABS = 1.0e-10
OSQP_EPS_REL = 1.0e-10
OSQP_MAX_ITER = 20000

#: OSQP statuses that may carry a usable solution. Both are then VERIFIED
#: independently against ``OSQP_KKT_TOL`` below -- the status alone is not the
#: acceptance criterion, in either direction. Measured on badly scaled synthetic
#: Gram matrices: OSQP can report ``solved`` with a scaled box-KKT residual of
#: ~1e-3, and can report ``solved inaccurate`` for an answer that is in fact
#: optimal to ~1e-11 (because the requested tolerance was deliberately tight).
#: Any other status (``maximum iterations reached``, ``primal infeasible``, ...)
#: is rejected outright.
OSQP_ACCEPTED_STATUSES = ("solved", "solved inaccurate")

#: Independent acceptance test applied to whatever OSQP returns: the scaled
#: box-constrained first-order residual
#:     ||a - P_[lo,hi](a - (G a + b))||_inf / max(1, ||b||_inf)
#: which is zero exactly at a KKT point of the amplitude QP. This is a
#: VERIFICATION, not a repair: a point that fails it is rejected with an error,
#: never patched, re-solved with added regularization, or passed to enumeration.
#: 1e-7 is ~5 orders looser than the worst residual measured over 1200 synthetic
#: problems with the real H = T'T, b = -T'u_d structure (1.6e-12 noiseless,
#: 1.2e-10 at 1% noise), so it rejects genuine failures without being brittle.
OSQP_KKT_TOL = 1.0e-7

#: Absolute+relative slack allowed on the amplitude bounds before OSQP's answer
#: is rejected as infeasible. ADMM enforces constraints only to its tolerance,
#: so a converged solution can sit a few eps outside the box; anything inside
#: this slack is clipped onto the box, anything outside is an error.
OSQP_BOUND_TOL = 1.0e-8


def _box_kkt_residual(
    G: np.ndarray, b: np.ndarray, a: np.ndarray, lo: float, hi: float
) -> float:
    """Scaled first-order residual of ``min 0.5 a'Ga + b'a`` on ``[lo, hi]^n``.

    ``||a - P_[lo,hi](a - grad)||_inf / max(1, ||b||_inf)``, zero exactly at a
    KKT point: an interior component needs a zero gradient, one at ``lo`` needs
    grad >= 0, one at ``hi`` needs grad <= 0.
    """
    grad = G @ a + b
    residual = float(np.max(np.abs(a - np.clip(a - grad, lo, hi))))
    return residual / max(1.0, float(np.max(np.abs(b))))
#: OSQP 1.x writes polish notes (e.g. "Polishing not needed - no active set
#: detected at optimal point") to the C-level stdout REGARDLESS of
#: ``verbose=False``. A beam initialization performs hundreds of joint refits,
#: so that line would dominate every run log. Suppress it at the file-descriptor
#: level around setup/solve; the termination status and iteration count are read
#: programmatically and reported in the error message, so nothing actionable is
#: lost. Set to False to see OSQP's raw output while debugging.
OSQP_SUPPRESS_SOLVER_STDOUT = True


@contextmanager
def _suppressed_stdout_fd(enabled: bool = True):
    """Redirect file descriptor 1 to os.devnull for the duration of the block.

    ``contextlib.redirect_stdout`` only rebinds ``sys.stdout`` and therefore does
    not reach output written by an extension module, which is what OSQP does.
    """
    if not enabled:
        yield
        return
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def _require_osqp():
    """Import OSQP (and ``scipy.sparse``) or raise an actionable error.

    Imported lazily so the enumeration solver -- and every run that does not
    select OSQP -- keeps working in an environment without OSQP installed.
    """
    try:
        import osqp  # noqa: PLC0415  (optional dependency, imported on demand)
        import scipy.sparse as sp  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SensitivityInitializationError(
            "The 'osqp' joint amplitude solver was selected but OSQP is not "
            f"importable in this interpreter ({exc}). Install it into the "
            "interpreter that runs the driver:\n"
            f"    {sys.executable} -m pip install 'osqp>=1.0'\n"
            "then verify with:\n"
            f"    {sys.executable} tools/check_inverse_env.py\n"
            "(see Applications/Application1/README.md -> Dependencies). "
            "Alternatively select the dependency-free exact solver with "
            "--amplitude-solver enumeration."
        ) from exc
    return osqp, sp


def validate_amplitude_solver(solver: str) -> str:
    """Normalize/validate an ``--amplitude-solver`` selection."""
    name = str(solver)
    if name not in AMPLITUDE_SOLVERS:
        raise ValueError(
            f"Unknown amplitude solver {solver!r}; expected one of "
            f"{AMPLITUDE_SOLVERS}"
        )
    return name


def amplitude_solver_available(solver: str) -> bool:
    """True when ``solver`` can actually run in this interpreter."""
    try:
        require_amplitude_solver(solver)
    except SensitivityInitializationError:
        return False
    return True


def require_amplitude_solver(solver: str) -> str:
    """Validate ``solver`` AND prove it can run here, with an actionable error.

    Called by the driver before any PDE solve so a missing OSQP install fails at
    startup rather than after the first forward solves.
    """
    name = validate_amplitude_solver(solver)
    if name == AMPLITUDE_SOLVER_OSQP:
        _require_osqp()
    return name


def _solve_box_qp_osqp(
    G: np.ndarray, b: np.ndarray, lo: float, hi: float
) -> np.ndarray:
    """Minimize 0.5 a^T G a + b^T a over [lo, hi]^n with OSQP.

    Maps the joint amplitude problem onto OSQP's ``min 0.5 x^T P x + q^T x``
    s.t. ``l <= A x <= u`` with ``P = G`` (symmetrized), ``q = b``, ``A = I``
    and ``l / u`` the amplitude bounds.  ``G`` and ``b`` arrive EXACTLY as the
    enumeration solver receives them, so the observation/time weights and any
    Tikhonov term the caller already folded into ``G`` are untouched: no
    additional regularization is applied here.

    Acceptance is decided by a VERIFIED optimality residual, not by OSQP's
    status label (which measurement showed to be unreliable in both directions
    on badly scaled Gram matrices).  Raises
    ``SensitivityInitializationError`` when the status is outside
    ``OSQP_ACCEPTED_STATUSES``, the iterate is non-finite, the bound violation
    exceeds ``OSQP_BOUND_TOL``, or the scaled box-KKT residual exceeds
    ``OSQP_KKT_TOL``.  It never falls back to another solver and never adds
    regularization.
    """
    osqp, sp = _require_osqp()
    G = np.asarray(G, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = b.size
    if G.shape != (n, n):
        raise ValueError(f"G has shape {G.shape}, expected ({n}, {n})")
    if not (np.isfinite(G).all() and np.isfinite(b).all()):
        raise SensitivityInitializationError(
            "Joint amplitude QP has non-finite data (G or b); refusing to solve."
        )
    if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
        raise ValueError(f"Invalid amplitude box [{lo}, {hi}]")

    # G is a Gram matrix plus lam*I, so it is symmetric in exact arithmetic;
    # it is assembled by mirroring an upper triangle, so it is symmetric to the
    # last bit here too. Symmetrize explicitly anyway -- OSQP reads only the
    # upper triangle of P, so an asymmetric P would silently change the problem.
    P = 0.5 * (G + G.T)
    settings = _osqp_settings()
    with _suppressed_stdout_fd(OSQP_SUPPRESS_SOLVER_STDOUT):
        problem = osqp.OSQP()
        problem.setup(
            P=sp.triu(sp.csc_matrix(P), format="csc"),
            q=b,
            A=sp.identity(n, format="csc"),
            l=np.full(n, float(lo), dtype=np.float64),
            u=np.full(n, float(hi), dtype=np.float64),
            **settings,
        )
        result = problem.solve()
    status = str(getattr(result.info, "status", "unknown"))
    iterations = getattr(result.info, "iter", "?")
    if status not in OSQP_ACCEPTED_STATUSES:
        raise SensitivityInitializationError(
            f"OSQP did not solve the joint amplitude QP (n={n}): status "
            f"{status!r} after {iterations} iterations. Re-run with "
            "--amplitude-solver enumeration to use the exact active-set solver "
            "(no automatic fallback is applied)."
        )
    a = np.asarray(result.x, dtype=np.float64).reshape(-1)
    if a.size != n or not np.isfinite(a).all():
        raise SensitivityInitializationError(
            f"OSQP reported status {status!r} but returned a non-finite or "
            f"wrongly sized amplitude vector for the joint QP (n={n})."
        )

    # Bound feasibility. ADMM enforces constraints only to its tolerance, so a
    # converged iterate can sit a few eps outside the box.
    scale = max(1.0, abs(float(lo)), abs(float(hi)))
    violation = float(max(np.max(lo - a), np.max(a - hi), 0.0))
    if violation > OSQP_BOUND_TOL * scale:
        raise SensitivityInitializationError(
            f"OSQP amplitude solution violates the bounds [{lo}, {hi}] by "
            f"{violation:.3e} (> {OSQP_BOUND_TOL * scale:.3e}); refusing it "
            "rather than clipping a materially infeasible iterate."
        )
    a = np.clip(a, lo, hi)

    # Independent optimality VERIFICATION on the clipped, box-feasible point.
    # OSQP's own status is necessary but not sufficient: on badly scaled Gram
    # matrices it can report `solved` for a point whose scaled box-KKT residual
    # is ~1e-3, and `solved inaccurate` for one that is optimal to ~1e-11. The
    # residual below is the quantity that actually matters, so it -- not the
    # label -- decides. A point that fails is REJECTED with the measured number;
    # nothing is patched, re-solved with added regularization, or quietly handed
    # to the enumeration solver.
    # Use P, not G: the gradient of 0.5 a'Ga + b'a is 0.5(G + G')a + b = Pa + b,
    # and P is exactly the matrix OSQP was given. For this module's symmetric
    # Gram input the two agree to the last bit; using P keeps the check exact
    # for any caller.
    kkt = _box_kkt_residual(P, b, a, lo, hi)
    if not (np.isfinite(kkt) and kkt <= OSQP_KKT_TOL):
        raise SensitivityInitializationError(
            f"OSQP returned status {status!r} after {iterations} iterations for "
            f"the joint amplitude QP (n={n}), but the solution is not "
            f"first-order optimal on the box: scaled projected-gradient "
            f"residual {kkt:.3e} > {OSQP_KKT_TOL:.0e}. This is usually a "
            f"severely ill-conditioned Gram matrix (cond = "
            f"{np.linalg.cond(P):.3e}), i.e. one or more amplitudes are not "
            "identifiable from the current observations. Re-run with "
            "--amplitude-solver enumeration for the exact active-set solver "
            "(no automatic fallback is applied, and no regularization is added)."
        )
    return a


def _osqp_settings() -> dict:
    """OSQP settings, tolerant of the 0.6 (``polish``) / 1.x (``polishing``) rename."""
    osqp, _sp = _require_osqp()
    try:
        major = int(str(getattr(osqp, "__version__", "0")).split(".", 1)[0])
    except ValueError:  # pragma: no cover - defensive
        major = 1
    return {
        "verbose": False,
        "eps_abs": OSQP_EPS_ABS,
        "eps_rel": OSQP_EPS_REL,
        "max_iter": OSQP_MAX_ITER,
        ("polishing" if major >= 1 else "polish"): True,
    }


def _solve_box_qp(
    G: np.ndarray,
    b: np.ndarray,
    lo: float,
    hi: float,
    *,
    solver: str = AMPLITUDE_SOLVER_ENUMERATION,
) -> np.ndarray:
    """Minimize 0.5 a^T G a + b^T a over the box [lo, hi]^n.

    ``solver='enumeration'`` (the default here, so every existing direct caller
    is unchanged) enumerates all 3^n active-set assignments (free / at lo / at
    hi), solves the reduced system for the free variables (least-squares if
    singular, e.g. for coherent columns), keeps box-feasible candidates, and
    returns the feasible candidate with the smallest objective.  For a convex QP
    the true minimizer's active set is among the enumerated assignments, so this
    is exact -- but the cost is exponential, and it refuses
    ``n > _ENUMERATION_MAX_SOURCES``.

    ``solver='osqp'`` solves the SAME problem with OSQP (see
    ``_solve_box_qp_osqp``); it has no source-count limit and is the default of
    the Application1 initialization pipeline.
    """
    solver = validate_amplitude_solver(solver)
    if solver == AMPLITUDE_SOLVER_OSQP:
        return _solve_box_qp_osqp(G, b, lo, hi)
    G = np.asarray(G, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = b.size
    if n > _ENUMERATION_MAX_SOURCES:
        # 3^n enumeration is exact but exponential; it is intended for the small
        # source counts this application runs (27 assignments at M=3, 81 at
        # M=4). Fail loudly rather than silently burning 3^n solves. This limit
        # is specific to the enumeration solver: --amplitude-solver osqp has no
        # such cap and is the route to the intended 8-10 source problems.
        raise ValueError(
            f"_solve_box_qp enumerates 3^n active sets and is intended for "
            f"small n; got n={n}. Use solver='{AMPLITUDE_SOLVER_OSQP}' "
            f"(--amplitude-solver {AMPLITUDE_SOLVER_OSQP}) before raising the "
            "source count this far."
        )
    best_a, best_q = None, np.inf
    for code in range(3**n):
        status = []
        c = code
        for _ in range(n):
            status.append(c % 3)  # 0=free, 1=lo, 2=hi
            c //= 3
        a = np.empty(n, dtype=np.float64)
        fixed_mask = np.array([s != 0 for s in status])
        a[np.array([s == 1 for s in status])] = lo
        a[np.array([s == 2 for s in status])] = hi
        free = np.flatnonzero(~fixed_mask)
        fixed = np.flatnonzero(fixed_mask)
        if free.size:
            rhs = -(b[free] + G[np.ix_(free, fixed)] @ a[fixed])
            Gff = G[np.ix_(free, free)]
            try:
                af = np.linalg.solve(Gff, rhs)
            except np.linalg.LinAlgError:
                af, *_ = np.linalg.lstsq(Gff, rhs, rcond=None)
            a[free] = af
        if np.any(a < lo - 1.0e-12) or np.any(a > hi + 1.0e-12):
            continue
        a = np.clip(a, lo, hi)
        q = float(0.5 * a @ G @ a + b @ a)
        if q < best_q:
            best_q, best_a = q, a.copy()
    if best_a is None:  # numerically impossible for a nonempty box; be safe
        best_a = np.full(n, lo, dtype=np.float64)
    return best_a


def _topk_diverse(
    g_vals: np.ndarray,
    candidate_idx: np.ndarray,
    candidate_xy: np.ndarray,
    k: int,
    radius: float,
    segment_is_admissible: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None,
    log: Optional[dict] = None,
) -> List[int]:
    """Indices of up to ``k`` most-negative-G candidates, pairwise >= radius apart.

    Walking candidates in ascending G order and skipping any within ``radius``
    of an already-picked one yields one representative per sensitivity basin
    instead of ``k`` adjacent mesh nodes of the deepest basin.

    With ``segment_is_admissible`` supplied the proximity rule becomes
    OBSTACLE-AWARE: being within ``radius`` of a retained candidate only makes
    the new candidate redundant if the straight segment joining them also stays
    inside the admissible fluid domain.  Two candidates metres apart on opposite
    faces of one building are different streets, and Euclidean distance alone
    cannot tell that apart from two nodes of the same street.  A candidate is
    suppressed iff SOME retained candidate is both within ``radius`` and
    fluid-connected to it; if every close retained candidate is separated by a
    wall the new candidate survives as an obstacle-separated basin.  This can
    only ever retain more candidates than the Euclidean rule, never fewer.

    ``segment_is_admissible=None`` (the default) reproduces the Euclidean-only
    behavior exactly.  ``log``, when a dict is supplied, is filled with the
    per-comparison records and the summary counts described in the module docs.
    """
    order = candidate_idx[np.argsort(g_vals[candidate_idx])]
    picked: List[int] = []
    events: List[dict] = []
    n_suppressed = 0
    n_preserved = 0
    n_comparisons = 0
    preserved_nodes: List[int] = []
    suppressed_nodes: List[int] = []

    for i in order:
        c = candidate_xy[i]
        suppress = False
        separated_from: List[int] = []
        for j in picked:
            distance = float(
                np.hypot(c[0] - candidate_xy[j][0], c[1] - candidate_xy[j][1])
            )
            if distance >= radius:
                continue
            if segment_is_admissible is None:
                suppress = True
                break
            n_comparisons += 1
            connected = bool(segment_is_admissible(candidate_xy[j], c))
            events.append(
                {
                    "candidate_node": int(i),
                    "candidate_xy": [float(c[0]), float(c[1])],
                    "retained_node": int(j),
                    "retained_xy": [
                        float(candidate_xy[j][0]),
                        float(candidate_xy[j][1]),
                    ],
                    "distance_m": distance,
                    "radius_m": float(radius),
                    "segment_in_fluid": connected,
                    "obstacle_encountered": (not connected),
                    "action": (
                        "suppressed_same_region" if connected
                        else "retained_obstacle_separated"
                    ),
                }
            )
            if connected:
                suppress = True
                break
            separated_from.append(int(j))

        if suppress:
            n_suppressed += 1
            if len(suppressed_nodes) < 60:
                suppressed_nodes.append(int(i))
        else:
            if separated_from:
                n_preserved += 1
                preserved_nodes.append(int(i))
            picked.append(int(i))
        if len(picked) >= k:
            break

    if log is not None:
        log.clear()
        log.update(
            {
                "obstacle_aware": segment_is_admissible is not None,
                "radius_m": float(radius),
                "events": events,
                "n_close_comparisons": int(n_comparisons),
                "n_suppressed_by_distance": int(n_suppressed),
                "n_preserved_obstacle_separated": int(n_preserved),
                "n_retained": int(len(picked)),
                "preserved_nodes": preserved_nodes,
                "suppressed_nodes_sample": suppressed_nodes,
                "picked": list(picked),
            }
        )
    return picked


def _log_diversity(log: dict, print_fn: Callable[[str], None], label: str) -> None:
    """Print the obstacle-aware diversity decisions and the summary counts."""
    if not log:
        return
    if not log.get("obstacle_aware", False):
        print_fn(
            f"    [diversity {label}] euclidean-only: retained "
            f"{log['n_retained']}, suppressed {log['n_suppressed_by_distance']}"
        )
        return
    preserved = [
        event for event in log.get("events", [])
        if event["action"] == "retained_obstacle_separated"
    ]
    shown = preserved[:40]
    for event in shown:
        cx, cy = event["candidate_xy"]
        rx, ry = event["retained_xy"]
        print_fn(
            f"    [diversity {label}] new ({cx:7.2f},{cy:7.2f}) vs retained "
            f"({rx:7.2f},{ry:7.2f})  d={event['distance_m']:6.2f} m "
            f"< r={event['radius_m']:.1f} m  segment_in_fluid=False  "
            f"obstacle=True  -> retained as obstacle-separated candidate"
        )
    if len(preserved) > len(shown):
        print_fn(
            f"    [diversity {label}] ... {len(preserved) - len(shown)} further "
            "obstacle-separated comparisons not shown"
        )
    print_fn(
        f"    [diversity {label}] summary: close comparisons "
        f"{log['n_close_comparisons']}, suppressed as same-region "
        f"{log['n_suppressed_by_distance']}, preserved because an obstacle "
        f"separated them {log['n_preserved_obstacle_separated']}, final retained "
        f"{log['n_retained']}"
    )


def _gaussian_sum_source(centers_amps, sigma2: float):
    """Return a source_func summing unit-shaped Gaussians for placed sources.

    This is the same Gaussian source model the optimizer uses
    (``exp(-|x - c|^2 / sigma2)``), assembled directly as a ``source_func`` for
    ``gt.run_forward`` so the number of greedily placed sources is decoupled from
    the optimizer's fixed control length.  No new forward solver is introduced.
    """
    ca = [(float(cx), float(cy), float(a)) for cx, cy, a in centers_amps]
    inv_sigma2 = 1.0 / float(sigma2)

    def source(x: float, y: float) -> float:
        total = 0.0
        for cx, cy, a in ca:
            total += a * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) * inv_sigma2))
        return float(total)

    return source


def _unit_response_curvature(
    mesh_data,
    pde: dict,
    center: np.ndarray,
    eta1: float,
    lam: float,
    objective_time_rule: str,
):
    r"""kappa(c) via ONE unit-Gaussian forward solve and the exact objective norm.

    Runs one forward solve for the UNIT-amplitude Gaussian at ``center`` (through
    the existing solver's ``source_func`` hook) to get delta_u_c, and returns
    ``(kappa, delta_u)`` with
    kappa(c) = eta1 * T_reduce(\int_Gamma_obs |delta_u_c|^2) + lambda, computed as
    2 * J_track(delta_u_c ; u_d = 0) using the repository's exact discrete
    boundary/time tracking norm (compute_objective with lam = 0).
    """
    unit_source = _gaussian_sum_source([(center[0], center[1], 1.0)], pde["sigma2"])
    delta_u = gt.run_forward(mesh_data, pde, _DUMMY_Z, source_func=unit_source)
    dt_out = pde["dt"] * pde["jprint"]
    j_track_unit = float(
        gt.compute_objective(
            delta_u,
            np.zeros_like(delta_u),
            mesh_data,
            eta1=eta1,
            lam=0.0,
            z=_DUMMY_Z,
            dt_out=dt_out,
            time_rule=objective_time_rule,
        )
    )
    return 2.0 * j_track_unit + float(lam), delta_u


def build_sensitivity_initial_control(
    mesh_data,
    pde: dict,
    observation_state: np.ndarray,
    num_sources: int,
    amplitude_bounds,
    admissibility: MeshAdmissibility,
    *,
    eta1: float,
    lam: float,
    objective_time_rule: str,
    min_separation: Optional[float] = None,
    min_value_tol: float = 0.0,
    boundary_buffer: float = 0.0,
    building_buffer: float = 0.0,
    candidate_xy: Optional[np.ndarray] = None,
    dirichlet_tags: Sequence[str] = ("left",),
    save_plots: bool = False,
    plot_dir: Optional[Path] = None,
    plot_context: Optional[str] = None,
    verbose: bool = True,
    print_fn: Callable[[str], None] = print,
    score: str = "raw_g",
    top_k: int = 24,
    beam_width: int = 1,
    screen_radius: Optional[float] = None,
    true_centers_for_plots: Optional[np.ndarray] = None,
    refine_centers: bool = False,
    coherence_cap: Optional[float] = None,
    refine_max_fev: int = 50,
    refine_pair: bool = False,
    pair_refine_max_fev: Optional[int] = None,
    geometry_aware_branches: bool = False,
    obstacle_aware_diversity: bool = False,
    reselect_second: bool = False,
    reselect_threshold: float = 0.25,
    reselect_max_passes: int = 2,
    reselect_refine_max_fev: Optional[int] = None,
    reselect_sigma2: Optional[float] = None,
    reselect_observation_state: Optional[np.ndarray] = None,
    amplitude_solver: str = DEFAULT_AMPLITUDE_SOLVER,
):
    """Greedily build an initial control from the adjoint insertion sensitivity.

    Parameters mirror the optimizer's own configuration so the initialization
    uses the SAME physics, observation set, and discrete norms.

    ``observation_state`` is the data u_d of the inverse problem (the forward
    state of the true control), exactly as the optimizer generates it; it is used
    only as observation data, never for candidate selection.
    ``true_centers_for_plots`` (optional, (n, 2)) is used EXCLUSIVELY to mark the
    reference source centers on saved diagnostic plots; it never participates in
    selection, scoring, or masking.

    ``score`` selects the placement rule (see the module docstring): ``raw_g``
    is the original argmin-G behavior; ``predicted_reduction`` screens the
    ``top_k`` basin-diverse G minima (pairwise >= ``screen_radius``, default
    ``max(min_separation, 2 * sqrt(sigma2))``) and selects by exact bounded 1-D
    objective reduction.  ``beam_width`` > 1 (predicted_reduction only) keeps
    several first-source hypotheses, jointly refits amplitudes, and returns the
    complete control with the smallest exact objective.

    Returns ``(control, diagnostics, extras)`` where ``control`` has length
    ``3 * num_sources`` in the optimizer's ``[x1, y1, A1, x2, y2, A2, ...]``
    ordering, ``diagnostics`` is a list of :class:`SensitivityStepDiagnostics`,
    and ``extras`` carries plot paths and solve counts.
    """
    if score not in ("raw_g", "predicted_reduction"):
        raise ValueError(f"Unknown sensitivity score {score!r}")
    # Validated (and, for OSQP, proven importable) up front so a bad or
    # unavailable selection fails before any PDE solve. The raw_g path does no
    # joint amplitude fitting, so the selection is inert there; it is still
    # validated so both paths reject the same inputs.
    amplitude_solver = require_amplitude_solver(amplitude_solver)
    if int(beam_width) > 1 and score != "predicted_reduction":
        raise ValueError("beam_width > 1 requires score='predicted_reduction'")
    if score == "predicted_reduction":
        return _predicted_reduction_beam(
            mesh_data,
            pde,
            observation_state,
            num_sources,
            amplitude_bounds,
            admissibility,
            eta1=eta1,
            lam=lam,
            objective_time_rule=objective_time_rule,
            min_separation=min_separation,
            min_value_tol=min_value_tol,
            boundary_buffer=boundary_buffer,
            building_buffer=building_buffer,
            candidate_xy=candidate_xy,
            dirichlet_tags=dirichlet_tags,
            save_plots=save_plots,
            plot_dir=plot_dir,
            plot_context=plot_context,
            verbose=verbose,
            print_fn=print_fn,
            top_k=int(top_k),
            beam_width=int(beam_width),
            screen_radius=screen_radius,
            true_centers_for_plots=true_centers_for_plots,
            refine_centers=bool(refine_centers),
            coherence_cap=coherence_cap,
            refine_max_fev=int(refine_max_fev),
            refine_pair=bool(refine_pair),
            pair_refine_max_fev=pair_refine_max_fev,
            geometry_aware_branches=bool(geometry_aware_branches),
            obstacle_aware_diversity=bool(obstacle_aware_diversity),
            reselect_second=bool(reselect_second),
            reselect_threshold=float(reselect_threshold),
            reselect_max_passes=int(reselect_max_passes),
            reselect_refine_max_fev=reselect_refine_max_fev,
            reselect_sigma2=reselect_sigma2,
            reselect_observation_state=reselect_observation_state,
            amplitude_solver=amplitude_solver,
        )
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)[:, :2]
    connections = np.asarray(mesh_data["connections"], dtype=int)
    num_sources = int(num_sources)
    if num_sources < 1:
        raise ValueError("num_sources must be >= 1")

    amp_floor = float(amplitude_bounds[0])
    amp_upper = float(amplitude_bounds[1])
    if not (np.isfinite(amp_floor) and np.isfinite(amp_upper) and amp_floor < amp_upper):
        raise ValueError(f"Invalid amplitude bounds: {amplitude_bounds}")

    if candidate_xy is None:
        candidate_xy = coord
    candidate_xy = np.asarray(candidate_xy, dtype=np.float64)[:, :2]

    outer_boundary_xy = _outer_boundary_node_coords(mesh_data, coord)
    building_boundary_xy = _building_boundary_node_coords(mesh_data, coord)
    static_mask = _base_admissibility_mask(
        candidate_xy,
        admissibility,
        outer_boundary_xy,
        building_boundary_xy,
        boundary_buffer,
        building_buffer,
    )
    if not np.any(static_mask):
        raise SensitivityInitializationError(
            "No admissible candidate centers remain after the fluid-mesh / buffer "
            "mask; cannot run sensitivity initialization."
        )

    sigma2 = float(pde["sigma2"])
    # q_current starts as the zero source field (all amplitudes 0).
    control = np.zeros(3 * num_sources, dtype=np.float64)
    placed_centers: List[np.ndarray] = []
    diagnostics: List[SensitivityStepDiagnostics] = []
    plot_paths: List[str] = []

    if verbose:
        print_fn("\n=== Sensitivity Initialization (greedy adjoint insertion) ===")
        print_fn(f"requested sources: {num_sources}")
        print_fn(f"candidate centers: {candidate_xy.shape[0]} mesh nodes")
        print_fn(f"admissible candidates (static mask): {int(np.sum(static_mask))}")
        print_fn(f"insertion Gaussian sigma2: {sigma2} m^2")
        print_fn(f"amplitude bounds: [{amp_floor}, {amp_upper}]")
        print_fn(
            "min source separation: "
            + (f"{min_separation} m" if min_separation else "disabled")
        )
        print_fn(f"meaningful-descent tolerance: G(c*) < -{min_value_tol}")

    placed_amps: List[float] = []
    for greedy_index in range(1, num_sources + 1):
        # 1-4: one forward + one adjoint for the current background -> weight field
        #       W = source_adjoint * inv_mass (the discrete time-integrated adjoint
        #       scaled by the inverse lumped mass). The background source field is
        #       the sum of already-placed Gaussians, assembled as a source_func so
        #       the greedy count is decoupled from the optimizer control length.
        background_source = _gaussian_sum_source(
            [
                (c[0], c[1], a)
                for c, a in zip(placed_centers, placed_amps)
            ],
            sigma2,
        )
        background_state = gt.run_forward(
            mesh_data, pde, _DUMMY_Z, source_func=background_source
        )
        state_out, source_adjoint, inv_mass, W = sis.compute_adjoint_weight_field(
            mesh_data,
            pde,
            _DUMMY_Z,
            observation_state,
            eta1,
            objective_time_rule,
            dirichlet_tags=tuple(dirichlet_tags),
            state_out=background_state,
        )
        # 5: G(c) for every candidate center (cheap Gaussian-weighted quadrature).
        qx, qy, ag = sis.precompute_quadrature_weights(mesh_data, pde, W)
        g_field = sis.sensitivity_at_candidates(candidate_xy, qx, qy, ag, sigma2)

        # 6-7: apply the admissibility + separation mask, then argmin over it.
        sep_mask = _separation_mask(candidate_xy, placed_centers, min_separation)
        mask = static_mask & sep_mask
        if not np.any(mask):
            raise SensitivityInitializationError(
                f"greedy source {greedy_index}: all candidates are masked "
                "(fluid/buffer/separation); stopping placement."
            )
        masked_idx = np.flatnonzero(mask)
        local_argmin = int(np.argmin(g_field[masked_idx]))
        sel = int(masked_idx[local_argmin])
        g_min = float(g_field[sel])

        # Did the mask change the selection vs the unconstrained global argmin?
        global_sel = int(np.argmin(g_field))
        mask_changed_selection = bool(global_sel != sel)
        separation_active = bool(
            placed_centers and not sep_mask[global_sel] and static_mask[global_sel]
        )

        if not (g_min < -float(min_value_tol)):
            raise SensitivityInitializationError(
                f"greedy source {greedy_index}: minimum masked insertion "
                f"sensitivity {g_min:.6e} is not meaningfully negative "
                f"(tol {min_value_tol:.6e}); no descending admissible insertion. "
                f"Placed {len(placed_centers)} source(s) so far."
            )

        center = candidate_xy[sel].copy()

        # 8-9: one unit-Gaussian forward solve -> observation curvature kappa(c*),
        #      then the closed-form 1-D amplitude minimizer.
        kappa, _delta_u = _unit_response_curvature(
            mesh_data, pde, center, eta1, lam, objective_time_rule
        )
        if not (np.isfinite(kappa) and kappa > 0.0):
            raise SensitivityInitializationError(
                f"greedy source {greedy_index}: non-positive observation curvature "
                f"{kappa:.6e} at c*={center.tolist()}; cannot compute amplitude."
            )
        a_unconstrained = -g_min / kappa
        # 10: project onto the amplitude bounds.
        a_proj = float(np.clip(a_unconstrained, amp_floor, amp_upper))
        amplitude_clipped = bool(a_proj != a_unconstrained)

        # 11: q_current <- q_current + a* phi_c*
        offset = 3 * (greedy_index - 1)
        control[offset] = float(center[0])
        control[offset + 1] = float(center[1])
        control[offset + 2] = a_proj
        placed_centers.append(center)
        placed_amps.append(a_proj)

        step = SensitivityStepDiagnostics(
            greedy_index=greedy_index,
            center=[float(center[0]), float(center[1])],
            min_masked_sensitivity=g_min,
            unconstrained_amplitude=float(a_unconstrained),
            projected_amplitude=a_proj,
            observation_curvature=float(kappa),
            num_admissible_candidates=int(masked_idx.size),
            mask_changed_selection=mask_changed_selection,
            separation_constraint_active=separation_active,
            amplitude_clipped=amplitude_clipped,
        )
        diagnostics.append(step)

        if verbose:
            print_fn(
                f"[greedy source {greedy_index}] center=({center[0]:.4f}, "
                f"{center[1]:.4f}) m"
            )
            print_fn(
                f"    min masked G(c*) = {g_min:.6e}   "
                f"admissible candidates = {masked_idx.size}"
            )
            print_fn(
                f"    curvature kappa  = {kappa:.6e}   "
                f"a_unconstrained = {a_unconstrained:.6e}   "
                f"a_projected = {a_proj:.6e}"
            )
            print_fn(
                f"    mask changed selection = {mask_changed_selection}   "
                f"separation active = {separation_active}   "
                f"amplitude clipped = {amplitude_clipped}"
            )

        if save_plots and plot_dir is not None:
            path = _save_step_plots(
                plot_dir,
                greedy_index,
                coord,
                connections,
                source_adjoint,
                g_field,
                mask,
                center,
                placed_centers,
                plot_context,
            )
            if path is not None:
                plot_paths.append(str(path))

    extras = {"plot_paths": plot_paths}
    return control, diagnostics, extras


def _predicted_reduction_beam(
    mesh_data,
    pde: dict,
    observation_state: np.ndarray,
    num_sources: int,
    amplitude_bounds,
    admissibility: MeshAdmissibility,
    *,
    eta1: float,
    lam: float,
    objective_time_rule: str,
    min_separation: Optional[float],
    min_value_tol: float,
    boundary_buffer: float,
    building_buffer: float,
    candidate_xy: Optional[np.ndarray],
    dirichlet_tags: Sequence[str],
    save_plots: bool,
    plot_dir: Optional[Path],
    plot_context: Optional[str],
    verbose: bool,
    print_fn: Callable[[str], None],
    top_k: int,
    beam_width: int,
    screen_radius: Optional[float],
    true_centers_for_plots: Optional[np.ndarray],
    refine_centers: bool = False,
    coherence_cap: Optional[float] = None,
    refine_max_fev: int = 50,
    refine_pair: bool = False,
    pair_refine_max_fev: Optional[int] = None,
    geometry_aware_branches: bool = False,
    obstacle_aware_diversity: bool = False,
    reselect_second: bool = False,
    reselect_threshold: float = 0.25,
    reselect_max_passes: int = 2,
    reselect_refine_max_fev: Optional[int] = None,
    reselect_sigma2: Optional[float] = None,
    reselect_observation_state: Optional[np.ndarray] = None,
    amplitude_solver: str = DEFAULT_AMPLITUDE_SOLVER,
):
    """Exact bounded predicted-reduction placement with optional beam search.

    Per beam level: one forward (branch background) + one adjoint give the
    branch G field; a top-K basin-diverse screen selects candidates; one unit
    forward per NEW candidate (boundary traces are cached across branches and
    levels) gives kappa(c) and the exact bounded 1-D reduction; each retained
    expansion jointly refits ALL amplitudes (exact box QP in the weighted trace
    norm) and is ranked by its exact objective, evaluated from the cached traces
    by source-linearity with no additional PDE solve.

    Optional (default-off, behavior-preserving) refinements:

    * ``refine_centers``: before a retained branch's residual field is
      recomputed at level >= 2, locally refine the branch's already-placed
      source center(s) by VARIABLE PROJECTION -- the spatial coordinates are the
      only optimized variables; for every trial center the amplitude(s) are
      eliminated by the exact bounded joint fit, and the exact reduced objective
      (same forward solver / observation norm) is minimized (Nelder-Mead, at
      most ``refine_max_fev`` forward evaluations per branch), constrained to the
      admissible mesh domain.  The refined center's amplitude, forward state,
      residual, adjoint, and branch-specific sensitivity field are all
      recomputed.  Branches that refine to effectively the same center are then
      deduplicated, keeping the best-objective representative.
    * ``coherence_cap``: reject a screened NEW-source candidate whose
      dimensionless observation-space coherence with ANY already-placed source
      on that branch exceeds the cap, i.e. the candidate is admitted only when
      ``max_j mu(candidate, source_j) <= cap`` with
      ``mu = <t1, t2> / (||t1|| ||t2||)`` (same weighted inner product as the
      objective).  Zero/near-zero response norms yield ``mu=0``
      (never a divide-by-zero).  This is an ADDITIONAL optional admissibility
      check layered on top of the existing minimum-distance screening, not a
      replacement for it.
    * ``refine_pair``: after ALL sources are placed, jointly refine every center
      of each retained branch in ONE variable-projected Nelder-Mead search over
      the full center vector (x1, y1, x2, y2, ...) with all amplitudes eliminated
      by the same exact bounded box QP at every trial.  This is joint refinement,
      not alternating single-source refinement: with a misplaced strong source,
      moving the weak source alone can be uphill in every direction, so a
      one-source-at-a-time scheme takes a zero-length step.  ``pair_refine_max_fev``
      overrides the budget; note one evaluation costs one uncached forward solve
      PER SOURCE.
    * ``geometry_aware_branches``: make branch handling obstacle-aware.  Branch
      deduplication additionally requires the two branches' centers to be
      mutually VISIBLE through the fluid domain, so two centers a few metres
      apart on opposite sides of a building are kept as distinct basins; and each
      branch's refinement is confined to the region visible from its own starting
      center, so refinement cannot migrate a branch into a different street and
      collapse the beam's geometric diversity.  Visibility is a straight-segment
      test, which is stricter than true connectivity: an L-shaped street counts
      as two basins.  That is deliberately conservative -- it can only keep more
      branches, never merge distinct ones.
      It also changes how the beam is CUT at each level: ranking expansions by
      objective alone hands every slot to second-source variants of the single
      best first-source basin, so a distinct street found by refinement is
      dropped before it is ever handed to the optimizer.  One slot is therefore
      reserved for each distinct first-source basin before the remaining slots
      are filled by objective as before.
    * ``reselect_second``: the last source was screened against the residual of
      the placement as it stood BEFORE joint pair refinement.  If refinement then
      moved the leading source(s) by at least ``reselect_threshold`` Gaussian
      radii, or changed their fitted amplitudes by at least that relative amount,
      that residual is stale and so is the candidate it selected.  This stage
      rebuilds the residual at the current placement (forward solve of the
      leading component, adjoint, insertion-sensitivity field), rescreens the
      last source, refits all amplitudes by variable projection, re-refines every
      center jointly, and keeps whichever control has the smaller EXACT
      objective.  At most ``reselect_max_passes`` passes; a pass also stops early
      once the reselected center is within a dedup radius of the previous one or
      the relative objective gain falls below 1e-3, so passes cannot cycle.

    ``extras['branch_controls']`` always reports every retained branch as a full
    control with its exact objective (best first), so the caller can run a
    multistart over the retained geometric basins instead of committing to the
    single best branch.
    """
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)[:, :2]
    connections = np.asarray(mesh_data["connections"], dtype=int)
    num_sources = int(num_sources)
    if num_sources < 1:
        raise ValueError("num_sources must be >= 1")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")
    # Every joint amplitude fit below (level expansion, variable-projected
    # center refinement, and second-source re-selection, at the init width and
    # at the re-selection width) goes through this one selection.
    amplitude_solver = require_amplitude_solver(amplitude_solver)

    amp_floor = float(amplitude_bounds[0])
    amp_upper = float(amplitude_bounds[1])
    if not (np.isfinite(amp_floor) and np.isfinite(amp_upper) and amp_floor < amp_upper):
        raise ValueError(f"Invalid amplitude bounds: {amplitude_bounds}")

    if candidate_xy is None:
        candidate_xy = coord
    candidate_xy = np.asarray(candidate_xy, dtype=np.float64)[:, :2]

    outer_xy = _outer_boundary_node_coords(mesh_data, coord)
    building_xy = _building_boundary_node_coords(mesh_data, coord)
    static_mask = _base_admissibility_mask(
        candidate_xy, admissibility, outer_xy, building_xy,
        boundary_buffer, building_buffer,
    )
    if not np.any(static_mask):
        raise SensitivityInitializationError(
            "No admissible candidate centers remain after the fluid-mesh / buffer "
            "mask; cannot run sensitivity initialization."
        )

    sigma2 = float(pde["sigma2"])
    dt_out = float(pde["dt"] * pde["jprint"])
    if screen_radius is None:
        # Two e-folding radii ~ the Gaussian's effective footprint diameter:
        # with radius sqrt(sigma2) the top-K walk tiles the dominant plume with
        # near-duplicates and weakly observed basins never enter the screen.
        screen_radius = max(float(min_separation or 0.0), 2.0 * float(np.sqrt(sigma2)))
    ctx = _ObsTraceContext(mesh_data, eta1, dt_out, objective_time_rule)
    ud_trace = ctx.extract(observation_state)
    j_zero = ctx.tracking_objective(np.zeros_like(ud_trace), ud_trace)

    wall_start = time.perf_counter()
    # "candidate_traces" counts every screened-candidate trace request (cache
    # hits included); "forward"/"adjoint" count actual PDE solves.
    solves = {"forward": 0, "adjoint": 0, "candidate_traces": 0}
    trace_cache: Dict[int, np.ndarray] = {}
    kappa_cache: Dict[int, float] = {}
    _CACHE_CAP = 96
    bbox_min = coord.min(axis=0)
    bbox_max = coord.max(axis=0)

    # Obstacle-aware diversity: two candidates closer than the screen radius are
    # redundant only when a straight path between them stays in the fluid.  The
    # callback is memoized on the unordered center pair because the top-K walk
    # asks about the same retained candidates thousands of times.
    _segment_cache: Dict[tuple, bool] = {}

    def segment_is_admissible(p, q) -> bool:
        key = (
            round(float(p[0]), 6), round(float(p[1]), 6),
            round(float(q[0]), 6), round(float(q[1]), 6),
        )
        if key[:2] > key[2:]:
            key = key[2:] + key[:2]
        hit = _segment_cache.get(key)
        if hit is None:
            hit = bool(admissibility.segment_connects_in_domain(p, q)[0])
            _segment_cache[key] = hit
        return hit

    diversity_callback = segment_is_admissible if obstacle_aware_diversity else None
    diversity_logs: List[dict] = []

    def run_screen(g_field_in, mask_in, k_in, radius_in, label):
        """Top-K diverse screen with the configured diversity rule + logging."""
        log: dict = {}
        picked = _topk_diverse(
            g_field_in, np.flatnonzero(mask_in), candidate_xy, k_in, radius_in,
            segment_is_admissible=diversity_callback, log=log,
        )
        log["label"] = str(label)
        diversity_logs.append(log)
        if verbose and obstacle_aware_diversity:
            _log_diversity(log, print_fn, str(label))
        return picked, log

    def unit_trace(node: int) -> np.ndarray:
        solves["candidate_traces"] += 1
        if node not in trace_cache:
            src = _gaussian_sum_source(
                [(candidate_xy[node][0], candidate_xy[node][1], 1.0)], sigma2
            )
            w = gt.run_forward(mesh_data, pde, _DUMMY_Z, source_func=src)
            solves["forward"] += 1
            if len(trace_cache) >= _CACHE_CAP:
                trace_cache.pop(next(iter(trace_cache)))
            trace_cache[node] = ctx.extract(w)
        return trace_cache[node]

    def trace_for_center(cx: float, cy: float) -> np.ndarray:
        """Observation trace of a UNIT Gaussian at a CONTINUOUS center (1 forward).

        Used for locally refined centers, which are off the mesh-node dictionary
        and therefore cannot use the node-indexed cache.
        """
        src = _gaussian_sum_source([(float(cx), float(cy), 1.0)], sigma2)
        w = gt.run_forward(mesh_data, pde, _DUMMY_Z, source_func=src)
        solves["forward"] += 1
        return ctx.extract(w)

    def regularization(centers: Sequence[np.ndarray], amps: np.ndarray) -> float:
        if lam == 0.0:
            return 0.0
        z = []
        for c, a in zip(centers, amps):
            z.extend([float(c[0]), float(c[1]), float(a)])
        z = np.asarray(z, dtype=np.float64)
        return 0.5 * float(lam) * float(z @ z)

    def joint_refit_traces(traces: Sequence[np.ndarray]) -> np.ndarray:
        """Bounded joint amplitude fit (box QP) for a list of unit traces.

        H = Gram of the unit traces in the exact weighted observation norm,
        h[i] = <t_i, u_d>, plus the caller's Tikhonov weight on the diagonal.
        Solved over [amp_floor, amp_upper]^n by ``amplitude_solver``.
        """
        n = len(traces)
        Gram = np.empty((n, n), dtype=np.float64)
        rhs = np.empty(n, dtype=np.float64)
        for i in range(n):
            for j in range(i, n):
                Gram[i, j] = Gram[j, i] = ctx.inner(traces[i], traces[j])
            rhs[i] = -ctx.inner(traces[i], ud_trace)
        Gram = Gram + float(lam) * np.eye(n)
        return _solve_box_qp(
            Gram, rhs, amp_floor, amp_upper, solver=amplitude_solver
        )

    def exact_objective_traces(centers, traces, amps) -> float:
        model = np.zeros_like(ud_trace)
        for a, t in zip(amps, traces):
            model = model + float(a) * t
        return ctx.tracking_objective(model, ud_trace) + regularization(centers, amps)

    def coherence(t1: np.ndarray, n1: float, t2: np.ndarray, n2: float) -> float:
        """Dimensionless coherence in the observation norm (see module helper)."""
        return _coherence_from_norms(ctx.inner(t1, t2), n1, n2)

    def _in_domain(cx: float, cy: float) -> bool:
        if not (bbox_min[0] <= cx <= bbox_max[0]
                and bbox_min[1] <= cy <= bbox_max[1]):
            return False
        return bool(admissibility.is_point_in_domain(float(cx), float(cy)))

    def _same_basin(c1, c2) -> bool:
        """Are two centers in the same basin (close, and not split by a wall)?"""
        return centers_same_basin(
            c1, c2, dedup_radius, admissibility, geometry_aware_branches
        )

    def wall_distance(cx: float, cy: float) -> float:
        return float(admissibility.distance_to_boundary(float(cx), float(cy)))

    def refine_centers_vp(
        centers: Sequence[np.ndarray],
        max_fev: Optional[int] = None,
        anchors: Optional[Sequence[np.ndarray]] = None,
    ):
        """Variable-projected local refinement of a branch's placed centers.

        Optimizes ONLY the spatial coordinates, jointly over every center passed
        in (2 * len(centers) variables in one search).  For each trial center set
        the amplitudes are eliminated by the exact bounded joint fit (variable
        projection), and the exact reduced objective is minimized with
        Nelder-Mead (<= ``max_fev`` forward evaluations, default
        ``refine_max_fev``).  Trial centers outside the admissible mesh domain --
        or, when ``anchors`` is given, not visible through the fluid domain from
        their anchor -- are rejected with a finite penalty so the search stays
        feasible and inside its own basin; an infeasible final iterate falls back
        to the start.  Returns ``(refined_centers, traces, amps, objective,
        nfev)``.
        """
        from scipy.optimize import minimize

        m = len(centers)
        budget = int(refine_max_fev if max_fev is None else max_fev)
        x0 = np.concatenate(
            [np.asarray(c, dtype=np.float64)[:2] for c in centers]
        )
        penalty = float(j_zero) * 1.0e3 + 1.0e30

        def feasible(index: int, c) -> bool:
            if not _in_domain(c[0], c[1]):
                return False
            if anchors is None or index >= len(anchors):
                return True
            return bool(admissibility.segment_in_domain(anchors[index], c))

        def projected(flat: np.ndarray) -> float:
            cs = flat.reshape(m, 2)
            for k, c in enumerate(cs):
                if not feasible(k, c):
                    return penalty
            trs = [trace_for_center(c[0], c[1]) for c in cs]
            amps = joint_refit_traces(trs)
            return exact_objective_traces(cs, trs, amps)

        res = minimize(
            projected, x0, method="Nelder-Mead",
            options={"xatol": 0.5, "fatol": 1.0, "maxfev": budget},
        )
        cs = np.asarray(res.x, dtype=np.float64).reshape(m, 2)
        if any(not feasible(k, c) for k, c in enumerate(cs)):
            cs = x0.reshape(m, 2)  # keep the feasible coarse start
        trs = [trace_for_center(c[0], c[1]) for c in cs]
        amps = joint_refit_traces(trs)
        obj = exact_objective_traces(cs, trs, amps)
        return ([cs[i].copy() for i in range(m)], trs, amps,
                float(obj), int(res.nfev))

    def screen_next_source(lead_centers, lead_amps, lead_traces):
        """One residual-sensitivity screen for the NEXT source, given a placement.

        Same chain as the level loop -- background forward solve, adjoint of the
        residual against the observations, insertion-sensitivity field over the
        candidate dictionary, admissibility/separation mask, top-K basin-diverse
        screen, optional coherence cap, exact bounded 1-D predicted reduction --
        but callable at any time, so the residual can be rebuilt after the
        already-placed sources have moved.  Returns the best candidate record, or
        ``None`` when nothing admissible descends.
        """
        if lead_centers:
            src = _gaussian_sum_source(
                [(c[0], c[1], float(a)) for c, a in zip(lead_centers, lead_amps)],
                sigma2,
            )
            bg_state = gt.run_forward(mesh_data, pde, _DUMMY_Z, source_func=src)
            solves["forward"] += 1
        else:
            bg_state = np.zeros_like(observation_state)
        _, _source_adjoint, _inv_mass, W = sis.compute_adjoint_weight_field(
            mesh_data, pde, _DUMMY_Z, observation_state, eta1,
            objective_time_rule, dirichlet_tags=tuple(dirichlet_tags),
            state_out=bg_state,
        )
        solves["adjoint"] += 1
        qx, qy, ag = sis.precompute_quadrature_weights(mesh_data, pde, W)
        g_field = sis.sensitivity_at_candidates(candidate_xy, qx, qy, ag, sigma2)

        mask = static_mask & _separation_mask(
            candidate_xy, list(lead_centers), min_separation
        )
        if not np.any(mask):
            return None
        screened, _log = run_screen(
            g_field, mask, top_k, screen_radius, "reselect"
        )
        screened = [i for i in screened if g_field[i] < -float(min_value_tol)]
        if not screened:
            return None

        existing_norms = [ctx.norm2(t) for t in lead_traces]
        best = None
        for i in screened:
            t2 = unit_trace(i)
            n2 = ctx.norm2(t2)
            kappa = n2 + float(lam)
            if not (np.isfinite(kappa) and kappa > 0.0):
                continue
            mu = None
            if coherence_cap is not None and lead_traces:
                mu = max(
                    coherence(t1, n1, t2, n2)
                    for t1, n1 in zip(lead_traces, existing_norms)
                )
                if mu > float(coherence_cap):
                    continue
            g_i = float(g_field[i])
            a_1d = float(np.clip(-g_i / kappa, amp_floor, amp_upper))
            dj = a_1d * g_i + 0.5 * a_1d * a_1d * kappa
            record = {
                "node": int(i), "g": g_i, "kappa": float(kappa),
                "a_unconstrained": float(-g_i / kappa), "a_1d": a_1d,
                "dj_pred": float(dj),
                "norm_score": float(-g_i / np.sqrt(kappa)),
                "coherence": (float(mu) if mu is not None else None),
                "trace": t2,
            }
            if best is None or record["dj_pred"] < best["dj_pred"]:
                best = record
        return best

    def build_width_ops(sigma2_w: float, observation_w: np.ndarray):
        """Width-local copies of the screening/fitting operations.

        The initialization runs at the first continuation width, where two
        candidate streets closer than a Gaussian radius produce nearly identical
        observation traces and cannot be told apart.  Re-selection can therefore
        be asked to run at a NARROWER width, where they can.  This returns the
        same operations bound to that width and its own observation data and
        trace cache; the init-width path above is untouched.
        """
        pde_w = dict(pde)
        pde_w["sigma2"] = float(sigma2_w)
        ud_w = ctx.extract(observation_w)
        cache_w: Dict[int, np.ndarray] = {}

        def unit_trace_w(node: int) -> np.ndarray:
            solves["candidate_traces"] += 1
            if node not in cache_w:
                src = _gaussian_sum_source(
                    [(candidate_xy[node][0], candidate_xy[node][1], 1.0)], sigma2_w
                )
                w = gt.run_forward(mesh_data, pde_w, _DUMMY_Z, source_func=src)
                solves["forward"] += 1
                if len(cache_w) >= _CACHE_CAP:
                    cache_w.pop(next(iter(cache_w)))
                cache_w[node] = ctx.extract(w)
            return cache_w[node]

        def trace_for_center_w(cx: float, cy: float) -> np.ndarray:
            src = _gaussian_sum_source([(float(cx), float(cy), 1.0)], sigma2_w)
            w = gt.run_forward(mesh_data, pde_w, _DUMMY_Z, source_func=src)
            solves["forward"] += 1
            return ctx.extract(w)

        def joint_refit_w(traces: Sequence[np.ndarray]) -> np.ndarray:
            n = len(traces)
            Gram = np.empty((n, n), dtype=np.float64)
            rhs = np.empty(n, dtype=np.float64)
            for i in range(n):
                for j in range(i, n):
                    Gram[i, j] = Gram[j, i] = ctx.inner(traces[i], traces[j])
                rhs[i] = -ctx.inner(traces[i], ud_w)
            Gram = Gram + float(lam) * np.eye(n)
            return _solve_box_qp(
                Gram, rhs, amp_floor, amp_upper, solver=amplitude_solver
            )

        def exact_objective_w(centers, traces, amps) -> float:
            model = np.zeros_like(ud_w)
            for a, t in zip(amps, traces):
                model = model + float(a) * t
            return ctx.tracking_objective(model, ud_w) + regularization(centers, amps)

        def evaluate_w(centers):
            """Traces, VP amplitudes and exact objective of a placement at w."""
            traces = [trace_for_center_w(c[0], c[1]) for c in centers]
            amps = joint_refit_w(traces)
            return traces, amps, exact_objective_w(centers, traces, amps)

        def screen_w(lead_centers, lead_amps, lead_traces):
            if lead_centers:
                src = _gaussian_sum_source(
                    [(c[0], c[1], float(a))
                     for c, a in zip(lead_centers, lead_amps)],
                    sigma2_w,
                )
                bg_state = gt.run_forward(
                    mesh_data, pde_w, _DUMMY_Z, source_func=src
                )
                solves["forward"] += 1
            else:
                bg_state = np.zeros_like(observation_w)
            _, _sa, _im, W = sis.compute_adjoint_weight_field(
                mesh_data, pde_w, _DUMMY_Z, observation_w, eta1,
                objective_time_rule, dirichlet_tags=tuple(dirichlet_tags),
                state_out=bg_state,
            )
            solves["adjoint"] += 1
            qx, qy, ag = sis.precompute_quadrature_weights(mesh_data, pde_w, W)
            g_field = sis.sensitivity_at_candidates(
                candidate_xy, qx, qy, ag, sigma2_w
            )
            mask = static_mask & _separation_mask(
                candidate_xy, list(lead_centers), min_separation
            )
            if not np.any(mask):
                return None
            radius_w = (
                float(screen_radius) if screen_radius is not None
                else max(float(min_separation or 0.0), 2.0 * np.sqrt(sigma2_w))
            )
            screened, _log_w = run_screen(
                g_field, mask, top_k, radius_w, f"reselect@sigma2={sigma2_w:g}"
            )
            screened = [i for i in screened if g_field[i] < -float(min_value_tol)]
            if not screened:
                return None
            existing_norms = [ctx.norm2(t) for t in lead_traces]
            best = None
            for i in screened:
                t2 = unit_trace_w(i)
                n2 = ctx.norm2(t2)
                kappa = n2 + float(lam)
                if not (np.isfinite(kappa) and kappa > 0.0):
                    continue
                mu = None
                if coherence_cap is not None and lead_traces:
                    mu = max(
                        coherence(t1, n1, t2, n2)
                        for t1, n1 in zip(lead_traces, existing_norms)
                    )
                    if mu > float(coherence_cap):
                        continue
                g_i = float(g_field[i])
                a_1d = float(np.clip(-g_i / kappa, amp_floor, amp_upper))
                dj = a_1d * g_i + 0.5 * a_1d * a_1d * kappa
                record = {
                    "node": int(i), "g": g_i, "kappa": float(kappa),
                    "a_1d": a_1d, "dj_pred": float(dj),
                    "coherence": (float(mu) if mu is not None else None),
                    "trace": t2,
                }
                if best is None or record["dj_pred"] < best["dj_pred"]:
                    best = record
            return best

        def refine_w(centers, max_fev, anchors):
            from scipy.optimize import minimize

            m = len(centers)
            x0 = np.concatenate(
                [np.asarray(c, dtype=np.float64)[:2] for c in centers]
            )
            penalty = float(j_zero) * 1.0e3 + 1.0e30

            def feasible(index, c):
                if not _in_domain(c[0], c[1]):
                    return False
                if anchors is None or index >= len(anchors):
                    return True
                return bool(admissibility.segment_in_domain(anchors[index], c))

            def projected(flat):
                cs = flat.reshape(m, 2)
                for k, c in enumerate(cs):
                    if not feasible(k, c):
                        return penalty
                trs = [trace_for_center_w(c[0], c[1]) for c in cs]
                amps = joint_refit_w(trs)
                return exact_objective_w(cs, trs, amps)

            res = minimize(
                projected, x0, method="Nelder-Mead",
                options={"xatol": 0.5, "fatol": 1.0, "maxfev": int(max_fev)},
            )
            cs = np.asarray(res.x, dtype=np.float64).reshape(m, 2)
            if any(not feasible(k, c) for k, c in enumerate(cs)):
                cs = x0.reshape(m, 2)
            trs, amps, obj = evaluate_w([cs[i] for i in range(m)])
            return ([cs[i].copy() for i in range(m)], trs, amps,
                    float(obj), int(res.nfev))

        return {
            "sigma2": float(sigma2_w),
            "evaluate": evaluate_w,
            "screen": screen_w,
            "refine": refine_w,
        }

    if verbose:
        print_fn("\n=== Sensitivity Initialization (predicted-reduction"
                 + (f", beam width {beam_width}" if beam_width > 1 else "")
                 + ") ===")
        print_fn(f"requested sources: {num_sources} "
                 f"(control dimension {3 * num_sources})")
        print_fn(f"candidate centers: {candidate_xy.shape[0]} mesh nodes "
                 f"({int(np.sum(static_mask))} admissible)")
        print_fn(f"insertion Gaussian sigma2: {sigma2} m^2")
        print_fn(f"amplitude bounds: [{amp_floor}, {amp_upper}]")
        print_fn(f"joint amplitude solver: {amplitude_solver}"
                 + (" (exact 3^n active-set enumeration)"
                    if amplitude_solver == AMPLITUDE_SOLVER_ENUMERATION
                    else f" (eps_abs/eps_rel {OSQP_EPS_ABS:g}, "
                         f"max_iter {OSQP_MAX_ITER})"))
        print_fn(f"top-K screen: {top_k} basin-diverse candidates, "
                 f"diversity radius {screen_radius:.1f} m")
        print_fn("diversity rule: "
                 + ("OBSTACLE-AWARE (close candidates are redundant only when a "
                    "straight path between them stays in the fluid)"
                    if obstacle_aware_diversity else "euclidean distance only"))
        print_fn("min source separation: "
                 + (f"{min_separation} m" if min_separation else "disabled"))
        print_fn(f"meaningful-descent tolerance: G(c*) < -{min_value_tol}")
        print_fn("local center refinement: "
                 + (f"ON (variable projection, <= {refine_max_fev} fwd/branch)"
                    if refine_centers else "OFF"))
        print_fn("new-source coherence cap (max over ALL sources already "
                 "placed on the branch): "
                 + (f"{coherence_cap}" if coherence_cap is not None else "disabled"))
        print_fn("joint pair refinement: "
                 + (f"ON (budget "
                    f"{int(refine_max_fev if not pair_refine_max_fev else pair_refine_max_fev)}"
                    f" evals/branch)" if refine_pair else "OFF"))
        print_fn("geometry-aware branches: "
                 + ("ON (visibility-based dedup and refinement confinement)"
                    if geometry_aware_branches else "OFF"))
        print_fn(f"J(zero source field) = {j_zero:.6e}")

    # A branch is one placement hypothesis: placed source CENTERS (continuous;
    # equal to dictionary-node coordinates until refined) and their unit-amplitude
    # observation TRACES, the jointly refit amplitudes, exact objective, and
    # per-level records (for diagnostics/plots).  ``refined`` marks whether the
    # branch's centers have already been locally refined.
    branches = [{
        "centers": [], "traces": [], "amps": np.zeros(0),
        "objective": j_zero, "levels": [], "refined": False,
    }]
    dedup_radius = max(2.0, 0.25 * float(np.sqrt(sigma2)))
    total_dedup_removed = 0
    # One record per source level: branches in / out, expansions considered, and
    # the solve budget spent, so the sequential placement can be accounted for
    # at any M without parsing stdout.
    source_level_records: List[dict] = []

    for level in range(1, num_sources + 1):
        level_forward_before = int(solves["forward"])
        level_adjoint_before = int(solves["adjoint"])
        level_traces_before = int(solves["candidate_traces"])
        level_branches_before = len(branches)
        if verbose:
            print_fn(f"\nSource level {level}/{num_sources}")
        # (Optional) local variable-projected refinement of each retained
        # branch's placed center(s) BEFORE its residual field is recomputed,
        # then deduplication of branches that refined to the same basin.
        if refine_centers and any(b["centers"] for b in branches):
            for branch in branches:
                if branch["refined"] or not branch["centers"]:
                    continue
                coarse = [np.asarray(c, dtype=np.float64).copy()
                          for c in branch["centers"]]
                obj_before = float(branch["objective"])
                rc, rtr, ramp, robj, nfev = refine_centers_vp(
                    branch["centers"],
                    anchors=(coarse if geometry_aware_branches else None),
                )
                branch["centers"] = rc
                branch["traces"] = rtr
                branch["amps"] = ramp
                branch["objective"] = robj
                branch["refined"] = True
                # Reference the refined center in the stored per-source records.
                for k, rec in enumerate(branch["levels"]):
                    if k < len(rc):
                        rec["refined_center_xy"] = [float(rc[k][0]), float(rc[k][1])]
                if verbose:
                    moves = ", ".join(
                        f"({c0[0]:.1f},{c0[1]:.1f})->({cr[0]:.1f},{cr[1]:.1f})"
                        for c0, cr in zip(coarse, rc)
                    )
                    print_fn(
                        f"  [level {level}] refine branch ({len(rc)} center(s)): "
                        f"{moves}  J {obj_before:.6e} -> {robj:.6e}  "
                        "amps [" + ", ".join(f"{float(a):.4f}" for a in ramp) + "]  "
                        f"refine evals={nfev}"
                    )
            # Deduplicate refined branches, keeping the best-objective one.
            kept: List[dict] = []
            removed = 0
            for b in sorted(branches, key=lambda bb: bb["objective"]):
                is_dup = False
                for kb in kept:
                    if b["centers"] and len(b["centers"]) == len(kb["centers"]):
                        if all(
                            _same_basin(bc, kc)
                            for bc, kc in zip(b["centers"], kb["centers"])
                        ):
                            is_dup = True
                            break
                if is_dup:
                    removed += 1
                else:
                    kept.append(b)
            if removed and verbose:
                print_fn(
                    f"  [level {level}] deduplicated {removed} refined branch(es); "
                    f"{len(kept)} distinct first-source basin(s) remain"
                )
            total_dedup_removed += removed
            branches = kept

        expansions = []
        seen_sigs = set()
        for branch in branches:
            # Branch background: fitted sum of already-placed Gaussians (centers
            # are dictionary-node coordinates until refined).
            if branch["centers"]:
                src = _gaussian_sum_source(
                    [(c[0], c[1], float(a))
                     for c, a in zip(branch["centers"], branch["amps"])],
                    sigma2,
                )
                bg_state = gt.run_forward(mesh_data, pde, _DUMMY_Z, source_func=src)
                solves["forward"] += 1
            else:
                bg_state = np.zeros_like(observation_state)
            _, source_adjoint, _inv_mass, W = sis.compute_adjoint_weight_field(
                mesh_data, pde, _DUMMY_Z, observation_state, eta1,
                objective_time_rule, dirichlet_tags=tuple(dirichlet_tags),
                state_out=bg_state,
            )
            solves["adjoint"] += 1
            qx, qy, ag = sis.precompute_quadrature_weights(mesh_data, pde, W)
            g_field = sis.sensitivity_at_candidates(candidate_xy, qx, qy, ag, sigma2)

            placed_xy = list(branch["centers"])
            sep_mask = _separation_mask(candidate_xy, placed_xy, min_separation)
            mask = static_mask & sep_mask
            if not np.any(mask):
                continue
            screened, diversity_log = run_screen(
                g_field, mask, top_k, screen_radius, f"level {level}"
            )
            screened = [i for i in screened if g_field[i] < -float(min_value_tol)]
            if not screened:
                continue

            raw_argmin = int(np.flatnonzero(mask)[
                np.argmin(g_field[np.flatnonzero(mask)])])
            # Precomputed observation-norms of already-placed sources (for the
            # optional coherence cap); safe/empty for the root branch.
            existing_norms = [ctx.norm2(t) for t in branch["traces"]]

            cand_records = []
            rejected_coh = []
            for i in screened:
                t2 = unit_trace(i)
                n2 = ctx.norm2(t2)
                kappa = n2 + float(lam)
                g_i = float(g_field[i])
                if not (np.isfinite(kappa) and kappa > 0.0):
                    continue
                kappa_cache.setdefault(int(i), float(kappa))
                mu = None
                if coherence_cap is not None and branch["traces"]:
                    mu = max(
                        coherence(t1, n1, t2, n2)
                        for t1, n1 in zip(branch["traces"], existing_norms)
                    )
                    if mu > float(coherence_cap):
                        rejected_coh.append((int(i), float(mu)))
                        continue
                a_unc = -g_i / kappa
                a_1d = float(np.clip(a_unc, amp_floor, amp_upper))
                dj = a_1d * g_i + 0.5 * a_1d * a_1d * kappa
                cand_records.append({
                    "node": int(i), "g": g_i, "kappa": float(kappa),
                    "a_unconstrained": float(a_unc), "a_1d": a_1d,
                    "dj_pred": float(dj),
                    "norm_score": float(-g_i / np.sqrt(kappa)),
                    "coherence": (float(mu) if mu is not None else None),
                })
            if not cand_records:
                if verbose and rejected_coh:
                    print_fn(
                        f"    all {len(rejected_coh)} screened level-{level} "
                        f"candidate(s) rejected by coherence cap {coherence_cap}; "
                        "branch not expanded"
                    )
                continue
            cand_records.sort(key=lambda r: r["dj_pred"])
            for rank, rec in enumerate(cand_records, start=1):
                rec["rank"] = rank

            if verbose:
                tag = (("branch ["
                        + ", ".join(f"({c[0]:.1f},{c[1]:.1f})"
                                    for c in branch["centers"]) + "]")
                       if branch["centers"] else "root branch")
                print_fn(f"[level {level}] {tag}: J_before={branch['objective']:.6e}"
                         f"  raw argmin node at "
                         f"({candidate_xy[raw_argmin][0]:.1f},"
                         f"{candidate_xy[raw_argmin][1]:.1f})")
                if coherence_cap is not None and rejected_coh:
                    print_fn(
                        f"    coherence cap {coherence_cap} (max over "
                        f"{len(branch['traces'])} placed source(s)): rejected "
                        f"{len(rejected_coh)} candidate(s) "
                        + ", ".join(
                            f"({candidate_xy[n][0]:.0f},{candidate_xy[n][1]:.0f}):"
                            f"mu={m:.3f}" for n, m in rejected_coh[:5])
                    )
                for rec in cand_records:
                    c = candidate_xy[rec["node"]]
                    coh_txt = (f" coh={rec['coherence']:.3f}"
                               if rec["coherence"] is not None else "")
                    print_fn(
                        f"    cand ({c[0]:6.1f},{c[1]:6.1f}) G={rec['g']: .4e} "
                        f"kappa={rec['kappa']:.4e} a1d={rec['a_1d']:.4f} "
                        f"dJ_pred={rec['dj_pred']: .4e} rank={rec['rank']}{coh_txt}"
                    )

            for rec in cand_records[:beam_width]:
                cand_xy = np.asarray(candidate_xy[rec["node"]], dtype=np.float64).copy()
                new_centers = list(branch["centers"]) + [cand_xy]
                new_traces = list(branch["traces"]) + [unit_trace(rec["node"])]
                sig = tuple(sorted(
                    (round(float(c[0]), 6), round(float(c[1]), 6))
                    for c in new_centers
                ))
                if sig in seen_sigs:
                    continue
                seen_sigs.add(sig)
                amps = joint_refit_traces(new_traces)
                j_exact = exact_objective_traces(new_centers, new_traces, amps)
                dist_prev = (
                    min(float(np.hypot(cand_xy[0] - p[0], cand_xy[1] - p[1]))
                        for p in placed_xy) if placed_xy else None
                )
                level_record = {
                    **rec,
                    "objective_before": float(branch["objective"]),
                    "objective_after": float(j_exact),
                    "distance_to_previous": dist_prev,
                    "selection_changed_by_score": bool(rec["node"] != raw_argmin),
                    "num_admissible": int(np.sum(mask)),
                    "g_field": g_field,
                    "source_adjoint": source_adjoint,
                    "mask": mask,
                    "center_xy": [float(cand_xy[0]), float(cand_xy[1])],
                    "alternatives_xy": [
                        candidate_xy[r["node"]].copy()
                        for r in cand_records if r["node"] != rec["node"]
                    ],
                    "diversity_preserved_xy": [
                        candidate_xy[n].copy()
                        for n in diversity_log.get("preserved_nodes", [])
                    ],
                    "diversity_suppressed_xy": [
                        candidate_xy[n].copy()
                        for n in diversity_log.get("suppressed_nodes_sample", [])
                    ],
                }
                expansions.append({
                    "centers": new_centers,
                    "traces": new_traces,
                    "amps": amps,
                    "objective": float(j_exact),
                    "levels": branch["levels"] + [level_record],
                    "refined": False,
                    # The placement whose residual produced this candidate. If
                    # the already-placed sources later move (pair refinement),
                    # this record is what makes the staleness test possible.
                    "residual_context": {
                        "centers": [np.asarray(c, dtype=np.float64).copy()
                                    for c in branch["centers"]],
                        "amps": [float(a) for a in branch["amps"]],
                    },
                })

        if not expansions:
            raise SensitivityInitializationError(
                f"beam level {level}/{num_sources}: none of the "
                f"{level_branches_before} retained branch(es) admits a masked "
                f"candidate with insertion sensitivity < -{min_value_tol} (all "
                "masked, no meaningful descent, or all rejected by the "
                f"coherence cap {coherence_cap}); stopping placement with "
                f"{level - 1} of {num_sources} source(s) placed."
            )
        expansions.sort(key=lambda b: b["objective"])
        if geometry_aware_branches and len(expansions) > beam_width:
            # Ranking expansions by objective alone destroys the beam's geometric
            # diversity: every slot goes to a second-source variant of the single
            # best FIRST-source basin, so a distinct street that refinement found
            # is dropped even though it is the only hypothesis that could reach a
            # different basin. Reserve one slot per distinct first-source basin
            # first, then fill any remaining slots by objective as before.
            selected: List[dict] = []
            basin_anchors: List[np.ndarray] = []
            for expansion in expansions:
                if len(selected) >= beam_width:
                    break
                anchor = np.asarray(expansion["centers"][0], dtype=np.float64)
                if any(
                    centers_same_basin(anchor, other, dedup_radius,
                                       admissibility, True)
                    for other in basin_anchors
                ):
                    continue
                selected.append(expansion)
                basin_anchors.append(anchor)
            for expansion in expansions:
                if len(selected) >= beam_width:
                    break
                if not any(expansion is chosen for chosen in selected):
                    selected.append(expansion)
            selected.sort(key=lambda b: b["objective"])
            if verbose and len(basin_anchors) > 1:
                print_fn(
                    f"  [level {level}] beam diversity: reserved a slot for each "
                    f"of {len(basin_anchors)} distinct first-source basin(s)"
                )
            branches = selected
        else:
            branches = expansions[:beam_width]
        if verbose:
            for b_i, b in enumerate(branches, start=1):
                centers = ", ".join(
                    f"({c[0]:.1f},{c[1]:.1f})" for c in b["centers"]
                )
                amps_txt = ", ".join(f"{a:.4f}" for a in b["amps"])
                print_fn(f"  [level {level}] kept branch {b_i}: centers [{centers}] "
                         f"amps [{amps_txt}] J_exact={b['objective']:.6e}")
        source_level_records.append({
            "level": int(level),
            "num_sources": int(num_sources),
            "branches_in": int(level_branches_before),
            "expansions_considered": int(len(expansions)),
            "branches_surviving": int(len(branches)),
            "best_objective": float(branches[0]["objective"]) if branches else None,
            "n_forward_solves": int(solves["forward"] - level_forward_before),
            "n_adjoint_solves": int(solves["adjoint"] - level_adjoint_before),
            "n_candidate_traces": int(
                solves["candidate_traces"] - level_traces_before
            ),
        })
        if verbose:
            record = source_level_records[-1]
            print_fn(
                f"Source level {level}/{num_sources} complete: "
                f"{record['branches_surviving']} branch(es) surviving "
                f"(from {record['expansions_considered']} expansion(s) of "
                f"{record['branches_in']} branch(es)); level cost "
                f"{record['n_forward_solves']} forward / "
                f"{record['n_adjoint_solves']} adjoint solves, "
                f"{record['n_candidate_traces']} candidate trace(s)"
            )

    # Optional joint (all centers at once) variable-projected refinement of the
    # COMPLETE placement, after every source has been selected. The level-loop
    # refinement above only ever sees a branch's earlier sources, so without this
    # the last-placed source is handed to the optimizer at its coarse dictionary
    # node.
    pair_refinement_records: List[dict] = []
    if refine_pair and num_sources > 1:
        pair_budget = int(
            refine_max_fev if not pair_refine_max_fev else pair_refine_max_fev
        )
        if verbose:
            print_fn(
                f"\n[pair refinement] joint variable projection over "
                f"{2 * num_sources} center variables per branch, budget "
                f"{pair_budget} evaluations "
                f"({num_sources} forward solves each)"
            )
        for branch_index, branch in enumerate(branches, start=1):
            pre_centers = [np.asarray(c, dtype=np.float64).copy()
                           for c in branch["centers"]]
            pre_amps = np.asarray(branch["amps"], dtype=np.float64).copy()
            pre_objective = float(branch["objective"])
            rc, rtr, ramp, robj, nfev = refine_centers_vp(
                branch["centers"],
                max_fev=pair_budget,
                anchors=(pre_centers if geometry_aware_branches else None),
            )
            accepted = bool(robj <= pre_objective)
            if accepted:
                branch["centers"] = rc
                branch["traces"] = rtr
                branch["amps"] = ramp
                branch["objective"] = robj
                for k, rec in enumerate(branch["levels"]):
                    if k < len(rc):
                        rec["pair_refined_center_xy"] = [
                            float(rc[k][0]), float(rc[k][1])
                        ]
            post_centers = [np.asarray(c, dtype=np.float64).copy()
                            for c in branch["centers"]]
            post_amps = np.asarray(branch["amps"], dtype=np.float64).copy()
            record = {
                "branch_index": branch_index,
                "accepted": accepted,
                "n_forward_evaluations": int(nfev),
                "objective_before": pre_objective,
                "objective_after": float(branch["objective"]),
                "candidate_objective": float(robj),
                "centers_before": [[float(c[0]), float(c[1])] for c in pre_centers],
                "centers_after": [[float(c[0]), float(c[1])] for c in post_centers],
                "amplitudes_before": [float(a) for a in pre_amps],
                "amplitudes_after": [float(a) for a in post_amps],
                "moved_m": [
                    float(np.hypot(a[0] - b[0], a[1] - b[1]))
                    for a, b in zip(post_centers, pre_centers)
                ],
                "wall_distance_before_m": [
                    wall_distance(c[0], c[1]) for c in pre_centers
                ],
                "wall_distance_after_m": [
                    wall_distance(c[0], c[1]) for c in post_centers
                ],
            }
            pair_refinement_records.append(record)
            if verbose:
                fmt = lambda cs: ", ".join(f"({c[0]:.2f},{c[1]:.2f})" for c in cs)
                amp_fmt = lambda aa: ", ".join(f"{float(a):.4f}" for a in aa)
                print_fn(
                    f"  branch {branch_index}: "
                    f"[{fmt(pre_centers)}] amps [{amp_fmt(pre_amps)}] "
                    f"J={pre_objective:.6e}"
                )
                print_fn(
                    f"           -> [{fmt(post_centers)}] amps "
                    f"[{amp_fmt(post_amps)}] J={float(branch['objective']):.6e}  "
                    f"{'accepted' if accepted else 'REJECTED (kept pre-refinement)'}"
                    f"  evals={nfev}"
                )
                print_fn(
                    "           moved "
                    + ", ".join(f"s{k + 1} {d:.2f} m"
                                for k, d in enumerate(record["moved_m"]))
                    + "; wall distance "
                    + ", ".join(
                        f"s{k + 1} {before:.3f} -> {after:.3f} m"
                        for k, (before, after) in enumerate(zip(
                            record["wall_distance_before_m"],
                            record["wall_distance_after_m"],
                        ))
                    )
                )
        # Two branches can land in the same basin only after this joint move.
        kept_pairs: List[dict] = []
        pair_dedup_removed = 0
        for b in sorted(branches, key=lambda bb: bb["objective"]):
            if any(
                len(b["centers"]) == len(kb["centers"])
                and all(_same_basin(bc, kc)
                        for bc, kc in zip(b["centers"], kb["centers"]))
                for kb in kept_pairs
            ):
                pair_dedup_removed += 1
                continue
            kept_pairs.append(b)
        if pair_dedup_removed and verbose:
            print_fn(
                f"  deduplicated {pair_dedup_removed} pair-refined branch(es); "
                f"{len(kept_pairs)} distinct basin(s) remain"
            )
        total_dedup_removed += pair_dedup_removed
        branches = kept_pairs
    # Optional second-source RE-SELECTION. The last source was screened against
    # the residual of the placement as it stood BEFORE joint pair refinement. If
    # refinement then moved the leading source(s) or changed their fitted
    # amplitudes materially, that residual -- and therefore the selected
    # candidate -- is stale. Rebuild the residual at the current placement,
    # rescreen, refit, re-refine, and keep whichever control has the smaller
    # exact objective.
    reselection_records: List[dict] = []
    # Width on which the reported branch objectives are measured; changes only
    # when re-selection runs at a different width (see below).
    objective_width_sigma2 = float(sigma2)
    if reselect_second and num_sources > 1:
        reselect_budget = int(
            (refine_max_fev if not pair_refine_max_fev else pair_refine_max_fev)
            if not reselect_refine_max_fev else reselect_refine_max_fev
        )
        max_passes = max(0, int(reselect_max_passes))
        # Optionally re-select at a NARROWER width than the initialization
        # width. Two candidate streets closer than one Gaussian radius are
        # indistinguishable at the init width; at the target width they are not.
        width_ops = None
        reselect_width = float(sigma2)
        if (
            reselect_sigma2 is not None
            and float(reselect_sigma2) != float(sigma2)
        ):
            if reselect_observation_state is None:
                raise ValueError(
                    "reselect_sigma2 differs from the initialization width, so "
                    "reselect_observation_state (the observation data at that "
                    "width) must be supplied"
                )
            reselect_width = float(reselect_sigma2)
            width_ops = build_width_ops(reselect_width, reselect_observation_state)
            objective_width_sigma2 = reselect_width
        if verbose:
            print_fn(
                f"\n[second-source re-selection] threshold "
                f"{reselect_threshold:g} (fraction of a Gaussian radius "
                f"{np.sqrt(sigma2):.1f} m, or relative amplitude change), "
                f"<= {max_passes} pass(es), refine budget {reselect_budget}"
            )
            if width_ops is not None:
                print_fn(
                    f"  re-selection runs at sigma2={reselect_width:g} "
                    f"(radius {np.sqrt(reselect_width):.1f} m) instead of the "
                    f"initialization width {sigma2:g}; every retained branch is "
                    "re-evaluated at that width so the comparison and the final "
                    "ranking stay on one scale"
                )
        if width_ops is not None:
            # Put every branch on the re-selection width BEFORE any pass, so
            # "before" and "after" objectives are comparable and the final
            # ranking is consistent.
            for branch in branches:
                trs, amps, obj = width_ops["evaluate"](branch["centers"])
                branch["traces"] = trs
                branch["amps"] = amps
                branch["objective"] = float(obj)
            branches.sort(key=lambda b: b["objective"])
            if verbose:
                for b_i, b in enumerate(branches, start=1):
                    centers = ", ".join(f"({c[0]:.2f},{c[1]:.2f})"
                                        for c in b["centers"])
                    print_fn(f"  branch {b_i} at sigma2={reselect_width:g}: "
                             f"[{centers}] J_exact={b['objective']:.6e}")
        for branch_index, branch in enumerate(branches, start=1):
            for pass_index in range(1, max_passes + 1):
                context = branch.get("residual_context")
                if context is None or not context["centers"]:
                    break
                lead_centers = [np.asarray(c, dtype=np.float64).copy()
                                for c in branch["centers"][:-1]]
                lead_amps = [float(a) for a in branch["amps"][:-1]]
                lead_traces = list(branch["traces"][:-1])
                if len(lead_centers) != len(context["centers"]):
                    break

                # Staleness: how far have the leading sources drifted from the
                # configuration whose residual selected the last source?
                center_shift = max(
                    float(np.hypot(now[0] - was[0], now[1] - was[1]))
                    for now, was in zip(lead_centers, context["centers"])
                )
                amplitude_shift = max(
                    abs(now - was) / max(abs(was), amp_floor)
                    for now, was in zip(lead_amps, context["amps"])
                )
                center_shift_sigma = center_shift / float(np.sqrt(sigma2))
                if width_ops is not None and pass_index == 1:
                    # The original selection was screened at the initialization
                    # width. Re-selecting at a different width makes it stale by
                    # construction, and the drift numbers above are not
                    # comparable across widths (the amplitude refit changes with
                    # the width alone), so do not gate on them here.
                    stale = True
                    stale_reason = (
                        f"selection was screened at sigma2={sigma2:g}, "
                        f"re-selecting at sigma2={reselect_width:g}"
                    )
                else:
                    stale = (
                        center_shift_sigma >= float(reselect_threshold)
                        or amplitude_shift >= float(reselect_threshold)
                    )
                    stale_reason = (
                        f"leading source moved {center_shift:.2f} m "
                        f"({center_shift_sigma:.2f} sigma), amplitude changed "
                        f"{amplitude_shift * 100.0:.1f}%"
                    )
                if verbose:
                    print_fn(
                        f"  branch {branch_index} pass {pass_index}: "
                        f"{stale_reason} -> "
                        f"{'STALE, rescreening' if stale else 'still fresh, skipping'}"
                    )
                if not stale:
                    break

                forward_before = int(solves["forward"])
                adjoint_before = int(solves["adjoint"])
                objective_before = float(branch["objective"])
                old_center = np.asarray(branch["centers"][-1], dtype=np.float64)
                old_amp = float(branch["amps"][-1])

                candidate = (
                    width_ops["screen"](lead_centers, lead_amps, lead_traces)
                    if width_ops is not None
                    else screen_next_source(lead_centers, lead_amps, lead_traces)
                )
                if candidate is None:
                    if verbose:
                        print_fn("      no admissible descending candidate; "
                                 "keeping the current pair")
                    break

                new_center = np.asarray(
                    candidate_xy[candidate["node"]], dtype=np.float64
                ).copy()
                new_centers = lead_centers + [new_center]
                new_traces = lead_traces + [candidate["trace"]]
                if width_ops is not None:
                    # One evaluation: the candidate trace is already at this
                    # width, so only the leading centers need re-tracing.
                    new_traces, refit_amps, objective_refit = width_ops["evaluate"](
                        new_centers
                    )
                    rc, rtr, ramp, robj, nfev = width_ops["refine"](
                        new_centers,
                        reselect_budget,
                        (new_centers if geometry_aware_branches else None),
                    )
                else:
                    refit_amps = joint_refit_traces(new_traces)
                    objective_refit = exact_objective_traces(
                        new_centers, new_traces, refit_amps
                    )
                    rc, rtr, ramp, robj, nfev = refine_centers_vp(
                        new_centers,
                        max_fev=reselect_budget,
                        anchors=(new_centers if geometry_aware_branches else None),
                    )
                accepted = bool(robj < objective_before)
                moved_from_old = float(np.hypot(*(rc[-1] - old_center)))

                record = {
                    "branch_index": branch_index,
                    "pass": pass_index,
                    "residual_first_source_centers": [
                        [float(c[0]), float(c[1])] for c in lead_centers
                    ],
                    "residual_first_source_amplitudes": lead_amps,
                    "stale_center_shift_m": center_shift,
                    "stale_center_shift_sigma": center_shift_sigma,
                    "stale_amplitude_shift": amplitude_shift,
                    "stale_reason": stale_reason,
                    "screen_sigma2": float(reselect_width),
                    "old_second_center": [float(old_center[0]), float(old_center[1])],
                    "old_second_amplitude": old_amp,
                    "new_second_candidate": [float(new_center[0]),
                                             float(new_center[1])],
                    "candidate_coherence": candidate["coherence"],
                    "candidate_predicted_reduction": candidate["dj_pred"],
                    "objective_before": objective_before,
                    "objective_after_refit": float(objective_refit),
                    "objective_after_refinement": float(robj),
                    "accepted": accepted,
                    "second_source_moved_m": moved_from_old,
                    "center_moves_m": [
                        float(np.hypot(a[0] - b[0], a[1] - b[1]))
                        for a, b in zip(rc, new_centers)
                    ],
                    "wall_distance_after_m": [
                        wall_distance(c[0], c[1]) for c in rc
                    ],
                    "n_refine_evaluations": int(nfev),
                    "n_forward_solves": int(solves["forward"] - forward_before),
                    "n_adjoint_solves": int(solves["adjoint"] - adjoint_before),
                }
                reselection_records.append(record)

                if verbose:
                    coherence_txt = (
                        f"{candidate['coherence']:.3f}"
                        if candidate["coherence"] is not None else "n/a"
                    )
                    print_fn(
                        "      residual built at "
                        + ", ".join(f"({c[0]:.2f},{c[1]:.2f}) a={a:.4f}"
                                    for c, a in zip(lead_centers, lead_amps))
                    )
                    print_fn(
                        f"      second source ({old_center[0]:.2f},"
                        f"{old_center[1]:.2f}) a={old_amp:.4f}  ->  candidate "
                        f"({new_center[0]:.2f},{new_center[1]:.2f})  "
                        f"coherence={coherence_txt}  "
                        f"dJ_pred={candidate['dj_pred']:.4e}"
                    )
                    print_fn(
                        f"      J {objective_before:.6e} -> "
                        f"{float(objective_refit):.6e} (joint amplitude refit) -> "
                        f"{float(robj):.6e} (joint {2 * len(new_centers)}-D "
                        "refinement)  "
                        f"{'ACCEPTED' if accepted else 'rejected, keeping previous'}"
                    )
                    print_fn(
                        f"      second source moved {moved_from_old:.2f} m; "
                        f"refine evals={nfev}  forward solves="
                        f"{record['n_forward_solves']}  adjoint solves="
                        f"{record['n_adjoint_solves']}"
                    )

                if not accepted:
                    break
                branch["centers"] = rc
                branch["traces"] = rtr
                branch["amps"] = ramp
                branch["objective"] = robj
                branch["residual_context"] = {
                    "centers": [np.asarray(c, dtype=np.float64).copy()
                                for c in lead_centers],
                    "amps": list(lead_amps),
                }
                for k, rec in enumerate(branch["levels"]):
                    if k < len(rc):
                        rec["reselected_center_xy"] = [float(rc[k][0]),
                                                       float(rc[k][1])]
                # Stop early once the selection is stable or the gain is
                # immaterial, so passes cannot cycle between two candidates.
                relative_gain = (objective_before - robj) / max(
                    abs(objective_before), 1.0
                )
                if moved_from_old < dedup_radius or relative_gain < 1.0e-3:
                    if verbose:
                        print_fn(
                            "      selection stable / gain immaterial; "
                            "stopping re-selection for this branch"
                        )
                    break

    branches.sort(key=lambda b: b["objective"])

    def branch_control(branch) -> np.ndarray:
        flat = np.zeros(3 * num_sources, dtype=np.float64)
        for index, (c, a) in enumerate(zip(branch["centers"], branch["amps"])):
            flat[3 * index] = float(c[0])
            flat[3 * index + 1] = float(c[1])
            flat[3 * index + 2] = float(a)
        return flat

    branch_controls = [
        {
            "branch_index": index,
            "control": branch_control(b).tolist(),
            "objective": float(b["objective"]),
            "centers": [[float(c[0]), float(c[1])] for c in b["centers"]],
            "amplitudes": [float(a) for a in b["amps"]],
            "wall_distances_m": [
                wall_distance(c[0], c[1]) for c in b["centers"]
            ],
        }
        for index, b in enumerate(branches, start=1)
    ]
    if verbose and len(branch_controls) > 1:
        print_fn(f"\n[retained branches] {len(branch_controls)} distinct basin(s)"
                 + (" (geometry-aware)" if geometry_aware_branches else ""))
        for record in branch_controls:
            centers = ", ".join(f"({c[0]:.2f},{c[1]:.2f})"
                                for c in record["centers"])
            print_fn(f"  branch {record['branch_index']}: [{centers}] "
                     f"J_exact={record['objective']:.6e}")

    best = branches[0]
    control = np.zeros(3 * num_sources, dtype=np.float64)
    diagnostics: List[SensitivityStepDiagnostics] = []
    plot_paths: List[str] = []
    for level_index, (center, amp, rec) in enumerate(
        zip(best["centers"], best["amps"], best["levels"]), start=1
    ):
        c = np.asarray(center, dtype=np.float64)
        offset = 3 * (level_index - 1)
        control[offset] = float(c[0])
        control[offset + 1] = float(c[1])
        control[offset + 2] = float(amp)
        diagnostics.append(SensitivityStepDiagnostics(
            greedy_index=level_index,
            center=[float(c[0]), float(c[1])],
            min_masked_sensitivity=rec["g"],
            unconstrained_amplitude=rec["a_unconstrained"],
            projected_amplitude=float(amp),
            observation_curvature=rec["kappa"],
            num_admissible_candidates=rec["num_admissible"],
            mask_changed_selection=rec["selection_changed_by_score"],
            separation_constraint_active=bool(
                rec["distance_to_previous"] is not None
                and min_separation
                and rec["distance_to_previous"] < 2.0 * float(min_separation)
            ),
            amplitude_clipped=bool(
                float(amp) in (amp_floor, amp_upper)
            ),
            score="predicted_reduction",
            raw_g_at_center=rec["g"],
            normalized_score=rec["norm_score"],
            predicted_reduction=rec["dj_pred"],
            objective_before=rec["objective_before"],
            objective_after=rec["objective_after"],
            rank_among_screened=rec["rank"],
            distance_to_previous=rec["distance_to_previous"],
            selection_changed_by_score=rec["selection_changed_by_score"],
            n_forward_solves=solves["forward"],
            n_adjoint_solves=solves["adjoint"],
        ))
        if verbose:
            print_fn(
                f"[selected source {level_index}] center=({c[0]:.4f}, {c[1]:.4f}) m  "
                f"amp={float(amp):.6f} (joint refit; 1-D a={rec['a_1d']:.4f})"
            )
            print_fn(
                f"    G={rec['g']: .6e}  kappa={rec['kappa']:.6e}  "
                f"-G/sqrt(kappa)={rec['norm_score']:.4e}  "
                f"dJ_pred={rec['dj_pred']: .6e}"
            )
            print_fn(
                f"    J_before={rec['objective_before']:.6e} -> "
                f"J_after(joint refit)={rec['objective_after']:.6e}  "
                f"rank={rec['rank']}/{top_k}  "
                f"dist_prev="
                + (f"{rec['distance_to_previous']:.1f} m"
                   if rec["distance_to_previous"] is not None else "n/a")
            )
        if save_plots and plot_dir is not None:
            path = _save_step_plots(
                plot_dir,
                level_index,
                coord,
                connections,
                rec["source_adjoint"],
                rec["g_field"],
                rec["mask"],
                c,
                [np.asarray(cc, dtype=np.float64)
                 for cc in best["centers"][:level_index]],
                plot_context,
                alternatives=rec["alternatives_xy"],
                true_centers=true_centers_for_plots,
                obstacle_separated=rec.get("diversity_preserved_xy"),
                suppressed_duplicates=rec.get("diversity_suppressed_xy"),
            )
            if path is not None:
                plot_paths.append(str(path))

    diversity_summary = {
        "obstacle_aware": bool(obstacle_aware_diversity),
        "n_screens": len(diversity_logs),
        "n_close_comparisons": int(
            sum(log.get("n_close_comparisons", 0) for log in diversity_logs)
        ),
        "n_suppressed_by_distance": int(
            sum(log.get("n_suppressed_by_distance", 0) for log in diversity_logs)
        ),
        "n_preserved_obstacle_separated": int(
            sum(log.get("n_preserved_obstacle_separated", 0)
                for log in diversity_logs)
        ),
        "n_retained_last_screen": int(
            diversity_logs[-1].get("n_retained", 0) if diversity_logs else 0
        ),
        "per_screen": [
            {
                "label": log.get("label"),
                "radius_m": log.get("radius_m"),
                "n_close_comparisons": log.get("n_close_comparisons"),
                "n_suppressed_by_distance": log.get("n_suppressed_by_distance"),
                "n_preserved_obstacle_separated": log.get(
                    "n_preserved_obstacle_separated"
                ),
                "n_retained": log.get("n_retained"),
                "preserved_xy": [
                    [float(candidate_xy[n][0]), float(candidate_xy[n][1])]
                    for n in log.get("preserved_nodes", [])
                ],
            }
            for log in diversity_logs
        ],
    }

    if verbose:
        print_fn(
            "[diversity summary] rule="
            + ("obstacle-aware" if obstacle_aware_diversity else "euclidean-only")
            + f"  screens={diversity_summary['n_screens']}"
            + f"  suppressed as same-region="
            + f"{diversity_summary['n_suppressed_by_distance']}"
            + f"  preserved obstacle-separated="
            + f"{diversity_summary['n_preserved_obstacle_separated']}"
            + f"  retained (last screen)="
            + f"{diversity_summary['n_retained_last_screen']}"
        )
        print_fn(
            f"[solve budget] forward solves: {solves['forward']}  "
            f"adjoint solves: {solves['adjoint']}  "
            f"candidate traces evaluated: {solves['candidate_traces']}  "
            f"wall time: {time.perf_counter() - wall_start:.1f} s  "
            f"(exact branch objectives evaluated from cached traces)"
        )
        print_fn(
            "[source levels] "
            + "; ".join(
                f"level {record['level']}/{num_sources}: "
                f"{record['branches_surviving']} branch(es) surviving"
                for record in source_level_records
            )
        )
        print_fn(
            f"initialized exact objective: {best['objective']:.6e} "
            f"(zero-source J0 = {j_zero:.6e})"
        )

    extras = {
        "plot_paths": plot_paths,
        "num_sources": int(num_sources),
        "n_forward_solves": solves["forward"],
        "n_adjoint_solves": solves["adjoint"],
        "n_candidate_traces": solves["candidate_traces"],
        "wall_time_seconds": float(time.perf_counter() - wall_start),
        "source_levels": source_level_records,
        "initial_exact_objective": float(best["objective"]),
        "zero_source_objective": float(j_zero),
        "refine_centers": bool(refine_centers),
        "coherence_cap": (float(coherence_cap)
                          if coherence_cap is not None else None),
        "n_dedup_removed": int(total_dedup_removed),
        "refine_pair": bool(refine_pair),
        "geometry_aware_branches": bool(geometry_aware_branches),
        "obstacle_aware_diversity": bool(obstacle_aware_diversity),
        "amplitude_solver": str(amplitude_solver),
        "diversity": diversity_summary,
        "pair_refinement": pair_refinement_records,
        "reselect_second": bool(reselect_second),
        "reselect_sigma2": (float(reselect_sigma2)
                            if reselect_sigma2 is not None else None),
        "objective_width_sigma2": float(objective_width_sigma2),
        "second_source_reselection": reselection_records,
        "branch_controls": branch_controls,
    }
    return control, diagnostics, extras


def _save_step_plots(
    plot_dir: Path,
    greedy_index: int,
    coord: np.ndarray,
    connections: np.ndarray,
    source_adjoint: np.ndarray,
    g_field: np.ndarray,
    mask: np.ndarray,
    selected_center: np.ndarray,
    placed_centers: Sequence[np.ndarray],
    plot_context: Optional[str],
    alternatives: Optional[Sequence[np.ndarray]] = None,
    true_centers: Optional[np.ndarray] = None,
    obstacle_separated: Optional[Sequence[np.ndarray]] = None,
    suppressed_duplicates: Optional[Sequence[np.ndarray]] = None,
):
    """Two-panel PNG: integrated adjoint field S(x) and insertion field G(c).

    The selection is made from G(c) (right panel), NOT directly from S(x).
    ``alternatives`` marks the screened-but-not-selected candidate centers and
    ``true_centers`` marks the reference source centers -- both purely
    diagnostic overlays.  Returns the PNG path, or ``None`` if matplotlib is
    unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
    except ModuleNotFoundError:  # pragma: no cover
        return None

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    triang = mtri.Triangulation(coord[:, 0], coord[:, 1], connections)
    g_masked = np.where(mask, g_field, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 8.5))
    title = f"greedy source {greedy_index}"
    if plot_context:
        title += f"  ({plot_context})"

    tpc0 = axes[0].tripcolor(triang, source_adjoint, shading="gouraud", cmap="coolwarm")
    fig.colorbar(tpc0, ax=axes[0], label="S(x) = time-integrated discrete adjoint")
    axes[0].set_title(f"Integrated adjoint field S(x)\n{title}")

    tpc1 = axes[1].tripcolor(triang, g_masked, shading="gouraud", cmap="viridis")
    fig.colorbar(tpc1, ax=axes[1], label="G(c) = dJ/dA_new  ( < 0 = descent )")
    if suppressed_duplicates:
        sup = np.asarray(suppressed_duplicates, dtype=np.float64).reshape(-1, 2)
        axes[1].plot(sup[:, 0], sup[:, 1], ".", color="dimgray",
                     markersize=4, linestyle="none",
                     label="suppressed same-region duplicate")
    if alternatives:
        alt = np.asarray(alternatives, dtype=np.float64).reshape(-1, 2)
        axes[1].plot(alt[:, 0], alt[:, 1], "D", color="orange",
                     markeredgecolor="black", markersize=7, linestyle="none",
                     label="screened alternatives")
    if obstacle_separated:
        obs = np.asarray(obstacle_separated, dtype=np.float64).reshape(-1, 2)
        axes[1].plot(obs[:, 0], obs[:, 1], "s", color="cyan",
                     markeredgecolor="black", markersize=9, linestyle="none",
                     label="retained: obstacle-separated")
    if true_centers is not None:
        tc = np.asarray(true_centers, dtype=np.float64).reshape(-1, 2)
        for ax in axes:
            ax.plot(tc[:, 0], tc[:, 1], "P", color="lime",
                    markeredgecolor="black", markersize=12, linestyle="none",
                    label="true source (plot only)" if ax is axes[1] else None)
    axes[1].plot(
        selected_center[0], selected_center[1], "*", color="red",
        markeredgecolor="black", markersize=18, label="selected c*",
    )
    for c in placed_centers[:-1]:
        axes[1].plot(c[0], c[1], "o", color="white", markeredgecolor="black",
                     markersize=9)
    axes[1].set_title(f"Insertion sensitivity G(c) (masked)\n{title}")
    axes[1].legend(loc="upper right", fontsize=8)

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("x (m, local)")
        ax.set_ylabel("y (m, local)")

    fig.tight_layout()
    path = plot_dir / f"sensitivity_init_source_{greedy_index:02d}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
