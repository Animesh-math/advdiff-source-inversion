"""Adaptive source-count (model-order) inference for Application1.

This adds NO new physics.  It drives the EXISTING sequential adjoint-sensitivity
machinery one source at a time and wraps it in a birth test that decides whether
the next source is real:

  * ``source_insertion_sensitivity`` (``sis``) for the exact discrete insertion
    sensitivity ``G(c) = dJ/dA_new`` of the CURRENT residual,
  * ``sensitivity_initialization`` internals -- ``_ObsTraceContext`` (the exact
    discrete observation norm, boundary and node-sensor alike),
    ``_base_admissibility_mask`` / ``_separation_mask`` / ``_topk_diverse``
    (the same candidate screen), ``_gaussian_sum_source``, ``_solve_box_qp``,
  * a caller-supplied ``polish_fn`` built from the driver's own evaluator and
    ``--post-init-method`` plumbing, so the per-level polish is the same
    optimizer the production run uses.

Why the usual "add a source, J went down" test is invalid
---------------------------------------------------------
The m-source model is NESTED inside the (m+1)-source model: setting the new
amplitude to zero recovers the m-source model exactly.  Therefore

    J_{m+1}^* <= J_m^*      ALWAYS,

for every data set, every noise realization, and every spurious candidate.  A
decrease is a theorem, not evidence.  Two things follow, and both are enforced
here:

1. **The amplitude lower bound must be 0 during inference.**  Running the birth
   test against the production amplitude floor (``--amplitude-floor``, e.g. 0.1)
   destroys the nesting: a spurious source cannot shrink to zero, so it PINS at
   the floor and its non-zero amplitude looks like proof of a source.  It is an
   artefact of the constraint.  ``infer_source_count`` therefore polishes every
   candidate model with amplitude lower bound ``0`` and reports the production
   floor only as provenance.  This is the single most important detail in the
   module.

2. **Acceptance needs a threshold, plus a non-degeneracy check.**  A candidate is
   accepted only when the ACTUAL objective decrease after a bounded polish
   exceeds ``max(abs_reduction_tol, rel_reduction_tol * J_m)`` AND no source in
   the polished (m+1)-model has collapsed to amplitude ``<= amp_tol``.  The
   amplitude check is label-invariant (it looks at the minimum over all m+1
   amplitudes, not at "the new one") because the polish is free to permute and
   move sources.  A model that collapses back onto m effective sources is
   reported as exactly that.

The predicted decrease
----------------------
Because the transport PDE is linear in the source, inserting amplitude ``a`` of a
unit Gaussian at ``c`` changes the objective by the EXACT quadratic

    DeltaJ(c, a) = a G(c) + 0.5 a^2 kappa(c),   kappa(c) = ||delta_u_c||_obs^2 + lam,

so the best bounded 1-D insertion at ``c`` is available in closed form with
``a`` restricted to ``[0, amplitude_upper]`` -- the nested interval.  This is the
same formula ``sensitivity_initialization`` uses for ``predicted_reduction``
scoring; it is an exact 1-D reduction, not a linearization.  The screen reports
it as ``predicted_decrease`` so a level can be abandoned BEFORE paying for a
polish when the best possible birth is already negligible.

Stop reasons
------------
``discrepancy_floor``                   J is at/below the discrepancy tolerance
``max_sources_reached``                 hit ``--max-sources``
``no_admissible_candidate``             screen found nothing that descends
``negligible_predicted_reduction``      best possible birth below tolerance
``rejected_insufficient_actual_reduction``  polish did not deliver the threshold
``rejected_amplitude_collapse``         a source collapsed to <= amp_tol
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

# ``gradient_test`` lives in Test Application2 and is normally put on the path by
# run_optimization. Doing it here as well makes this module importable on its own
# (the test scripts import it directly), and is a no-op when run_optimization has
# already run.
_APP_DIR = Path(__file__).resolve().parent
_APP2_DIR = _APP_DIR.parents[1] / "Applications" / "Test Application2"
for _path in (str(_APP_DIR), str(_APP2_DIR)):
    if _path not in sys.path:
        sys.path.append(_path)

import gradient_test as gt
import sensitivity_initialization as sinit
import source_insertion_sensitivity as sis
from mesh_admissibility import MeshAdmissibility


DEFAULT_MAX_SOURCES = 4
DEFAULT_DISCREPANCY_TOL = 0.0          # 0 disables the absolute-J stop
DEFAULT_REL_REDUCTION_TOL = 1.0e-3     # fraction of J_m
DEFAULT_ABS_REDUCTION_TOL = 0.0
DEFAULT_AMP_TOL = 1.0e-3               # amplitude below this = collapsed source
DEFAULT_TOP_K = 12
#: Amplitude lower bound used for EVERY candidate model during inference. Zero
#: is required for the nesting argument; see the module docstring.
NESTING_AMPLITUDE_LOWER = 0.0

STOP_DISCREPANCY = "discrepancy_floor"
STOP_MAX_SOURCES = "max_sources_reached"
STOP_NO_CANDIDATE = "no_admissible_candidate"
STOP_NEGLIGIBLE_PREDICTED = "negligible_predicted_reduction"
STOP_INSUFFICIENT_ACTUAL = "rejected_insufficient_actual_reduction"
STOP_AMPLITUDE_COLLAPSE = "rejected_amplitude_collapse"

ACCEPT = "accepted"

# ---------------------------------------------------------------------------
# Decision-status semantics for the STOPPED model order (reporting only; does
# NOT change the numerical acceptance rule above). Every stop reason except
# ``STOP_MAX_SOURCES`` reflects a candidate that was actually tested (and
# either the residual hit its floor, or the next-source candidate was
# rejected on its merits) -- these are settled decisions. ``STOP_MAX_SOURCES``
# means the loop was cut off by the ``--max-sources`` budget before the
# acceptance rule could run at level ``max_sources + 1``, so the reported M is
# only a LOWER BOUND: a real additional source may exist and was never
# tested. That must not be reported as a clean decision.
# ---------------------------------------------------------------------------
DECISION_ACCEPTED = "accepted"
DECISION_REJECTED_STOP = "rejected_stop"
DECISION_UNRESOLVED_MAX_SOURCES = "unresolved_max_sources"

#: stop reasons for which the model order is a settled, converged decision.
_CONVERGED_STOP_REASONS = {
    STOP_DISCREPANCY,
    STOP_NO_CANDIDATE,
    STOP_NEGLIGIBLE_PREDICTED,
    STOP_INSUFFICIENT_ACTUAL,
    STOP_AMPLITUDE_COLLAPSE,
}


def decision_status_for_stop_reason(stop_reason: Optional[str]) -> tuple[str, bool]:
    """Map a stop reason to ``(source_count_decision_status, converged)``.

    Reporting/status semantics only -- does not change which model order is
    selected. ``STOP_MAX_SOURCES`` is always unresolved: the run was cut off
    by the source-count budget, not by the acceptance rule, so a real
    additional source may exist and was never tested.
    """
    if stop_reason == STOP_MAX_SOURCES:
        return DECISION_UNRESOLVED_MAX_SOURCES, False
    if stop_reason == STOP_DISCREPANCY:
        return DECISION_ACCEPTED, True
    if stop_reason in _CONVERGED_STOP_REASONS:
        return DECISION_REJECTED_STOP, True
    # Unknown/None stop reason (should not happen in practice): do not claim
    # a converged decision.
    return DECISION_UNRESOLVED_MAX_SOURCES, False


def acceptance_threshold(
    j_current: float, rel_reduction_tol: float, abs_reduction_tol: float
) -> float:
    """Minimum objective DECREASE that counts as evidence for a new source.

    Scale-aware by construction: a relative fraction of the current objective,
    floored by an absolute tolerance.  Exposed (and unit-tested) separately
    because it is the whole stop rule.
    """
    return max(float(abs_reduction_tol), float(rel_reduction_tol) * float(j_current))


def amplitudes_of(control: np.ndarray) -> np.ndarray:
    """Amplitudes ``[A1, ..., AM]`` of a flat ``[x, y, A]``-per-source control."""
    values = np.asarray(control, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.zeros(0, dtype=np.float64)
    return values.reshape(-1, 3)[:, 2].copy()


def append_source(control: np.ndarray, cx: float, cy: float, amplitude: float):
    """Return the control with one more source appended."""
    values = np.asarray(control, dtype=np.float64).reshape(-1)
    return np.concatenate(
        [values, np.asarray([float(cx), float(cy), float(amplitude)], dtype=np.float64)]
    )


class _BirthScreen:
    """One-source birth test on the residual of the current model.

    Holds the candidate dictionary, the admissibility mask and the unit-trace
    cache.  Because the PDE is linear in the source, a unit Gaussian's
    observation trace does NOT depend on the current background, so the cache is
    valid across every level -- which is what keeps the inference affordable.
    """

    def __init__(
        self,
        mesh_data,
        pde: dict,
        observation_state: np.ndarray,
        admissibility: MeshAdmissibility,
        *,
        eta1: float,
        lam: float,
        objective_time_rule: str,
        amplitude_upper: float,
        top_k: int,
        min_separation: Optional[float],
        boundary_buffer: float,
        building_buffer: float,
        screen_radius: Optional[float],
        obstacle_aware_diversity: bool,
        min_value_tol: float,
        dirichlet_tags: Sequence[str],
        print_fn: Callable[[str], None] = print,
    ):
        self.mesh_data = mesh_data
        self.pde = pde
        self.observation_state = np.asarray(observation_state, dtype=np.float64)
        self.admissibility = admissibility
        self.eta1 = float(eta1)
        self.lam = float(lam)
        self.objective_time_rule = str(objective_time_rule)
        self.amplitude_upper = float(amplitude_upper)
        self.top_k = int(top_k)
        self.min_separation = min_separation
        self.min_value_tol = float(min_value_tol)
        self.dirichlet_tags = tuple(dirichlet_tags)
        self.print_fn = print_fn

        coord = np.asarray(mesh_data["coord"], dtype=np.float64)[:, :2]
        self.candidate_xy = coord
        outer_xy = sinit._outer_boundary_node_coords(mesh_data, coord)
        building_xy = sinit._building_boundary_node_coords(mesh_data, coord)
        self.static_mask = sinit._base_admissibility_mask(
            coord, admissibility, outer_xy, building_xy,
            float(boundary_buffer), float(building_buffer),
        )
        if not np.any(self.static_mask):
            raise sinit.SensitivityInitializationError(
                "No admissible candidate centers remain after the fluid-mesh / "
                "buffer mask; cannot run source-count inference."
            )

        self.sigma2 = float(pde["sigma2"])
        dt_out = float(pde["dt"] * pde["jprint"])
        self.screen_radius = (
            float(screen_radius)
            if screen_radius is not None
            else max(float(min_separation or 0.0), 2.0 * float(np.sqrt(self.sigma2)))
        )
        self.ctx = sinit._ObsTraceContext(
            mesh_data, eta1, dt_out, objective_time_rule
        )
        self.ud_trace = self.ctx.extract(self.observation_state)
        self.zero_source_objective = self.ctx.tracking_objective(
            np.zeros_like(self.ud_trace), self.ud_trace
        )

        self.solves = {"forward": 0, "adjoint": 0, "candidate_traces": 0}
        self._trace_cache: dict[int, np.ndarray] = {}
        self._segment_cache: dict[tuple, bool] = {}
        self._diversity_callback = (
            self._segment_is_admissible if obstacle_aware_diversity else None
        )

    # -- helpers -------------------------------------------------------------
    def _segment_is_admissible(self, p, q) -> bool:
        key = (
            round(float(p[0]), 6), round(float(p[1]), 6),
            round(float(q[0]), 6), round(float(q[1]), 6),
        )
        if key[:2] > key[2:]:
            key = key[2:] + key[:2]
        hit = self._segment_cache.get(key)
        if hit is None:
            hit = bool(self.admissibility.segment_connects_in_domain(p, q)[0])
            self._segment_cache[key] = hit
        return hit

    def unit_trace(self, node: int) -> np.ndarray:
        """Observation trace of a UNIT Gaussian at candidate ``node`` (cached)."""
        self.solves["candidate_traces"] += 1
        node = int(node)
        if node not in self._trace_cache:
            src = sinit._gaussian_sum_source(
                [(self.candidate_xy[node][0], self.candidate_xy[node][1], 1.0)],
                self.sigma2,
            )
            state = gt.run_forward(
                self.mesh_data, self.pde, sinit._DUMMY_Z, source_func=src
            )
            self.solves["forward"] += 1
            self._trace_cache[node] = self.ctx.extract(state)
        return self._trace_cache[node]

    def model_state(self, control: np.ndarray) -> np.ndarray:
        """Forward state of the current m-source model (zero state when m = 0)."""
        control = np.asarray(control, dtype=np.float64).reshape(-1)
        if control.size == 0:
            return np.zeros_like(self.observation_state)
        state = gt.run_forward(self.mesh_data, self.pde, control)
        self.solves["forward"] += 1
        return state

    # -- the birth test ------------------------------------------------------
    def best_candidate(self, control: np.ndarray, model_state=None) -> Optional[dict]:
        """Best next-source candidate for the residual of ``control``.

        Chain: background forward state -> adjoint of the residual against the
        observations -> insertion-sensitivity field over the candidate
        dictionary -> admissibility/separation mask -> top-K basin-diverse
        screen -> exact bounded 1-D predicted reduction with amplitude in
        ``[0, amplitude_upper]``.  Returns ``None`` when nothing admissible
        descends.
        """
        control = np.asarray(control, dtype=np.float64).reshape(-1)
        centers = (
            [] if control.size == 0
            else [row[:2] for row in control.reshape(-1, 3)]
        )
        if model_state is None:
            model_state = self.model_state(control)

        _s, _source_adjoint, _inv_mass, weight_field = sis.compute_adjoint_weight_field(
            self.mesh_data, self.pde, sinit._DUMMY_Z, self.observation_state,
            self.eta1, self.objective_time_rule,
            dirichlet_tags=self.dirichlet_tags, state_out=model_state,
        )
        self.solves["adjoint"] += 1
        qx, qy, ag = sis.precompute_quadrature_weights(
            self.mesh_data, self.pde, weight_field
        )
        g_field = sis.sensitivity_at_candidates(
            self.candidate_xy, qx, qy, ag, self.sigma2
        )

        mask = self.static_mask & sinit._separation_mask(
            self.candidate_xy, centers, self.min_separation
        )
        if not np.any(mask):
            return None
        screened = sinit._topk_diverse(
            g_field, np.flatnonzero(mask), self.candidate_xy,
            self.top_k, self.screen_radius,
            segment_is_admissible=self._diversity_callback,
        )
        # G(c) < 0 means a POSITIVE-amplitude insertion at c descends J. With the
        # nested lower bound 0 a candidate with G >= 0 can only pick a = 0, i.e.
        # it predicts no reduction at all, so it is not a birth candidate.
        screened = [i for i in screened if g_field[i] < -float(self.min_value_tol)]
        if not screened:
            return None

        best = None
        for i in screened:
            trace = self.unit_trace(i)
            kappa = float(self.ctx.norm2(trace)) + self.lam
            if not (np.isfinite(kappa) and kappa > 0.0):
                continue
            g_i = float(g_field[i])
            # NESTED bounded 1-D minimizer: a in [0, upper], NOT [floor, upper].
            a_1d = float(
                np.clip(-g_i / kappa, NESTING_AMPLITUDE_LOWER, self.amplitude_upper)
            )
            delta_j = a_1d * g_i + 0.5 * a_1d * a_1d * kappa
            record = {
                "node": int(i),
                "center": [float(self.candidate_xy[i][0]), float(self.candidate_xy[i][1])],
                "g": g_i,
                "kappa": kappa,
                "a_unconstrained": float(-g_i / kappa),
                "a_1d": a_1d,
                "delta_j": float(delta_j),
                "predicted_decrease": float(-delta_j),
            }
            if best is None or record["delta_j"] < best["delta_j"]:
                best = record
        return best


def infer_source_count(
    mesh_data,
    pde: dict,
    observation_state: np.ndarray,
    admissibility: MeshAdmissibility,
    *,
    amplitude_upper: float,
    eta1: float,
    lam: float,
    objective_time_rule: str,
    polish_fn: Callable[[np.ndarray, float], tuple],
    max_sources: int = DEFAULT_MAX_SOURCES,
    discrepancy_tol: float = DEFAULT_DISCREPANCY_TOL,
    rel_reduction_tol: float = DEFAULT_REL_REDUCTION_TOL,
    abs_reduction_tol: float = DEFAULT_ABS_REDUCTION_TOL,
    amp_tol: float = DEFAULT_AMP_TOL,
    top_k: int = DEFAULT_TOP_K,
    min_separation: Optional[float] = None,
    boundary_buffer: float = 0.0,
    building_buffer: float = 0.0,
    screen_radius: Optional[float] = None,
    obstacle_aware_diversity: bool = True,
    min_value_tol: float = 0.0,
    dirichlet_tags: Sequence[str] = ("left",),
    production_amplitude_floor: Optional[float] = None,
    print_fn: Callable[[str], None] = print,
) -> dict:
    """Infer the number of sources by sequential birth testing.

    ``polish_fn(control, amplitude_lower) -> (polished_control, objective, info)``
    is supplied by the driver so the per-level polish uses the driver's own
    evaluator, bounds, feasibility mode and ``--post-init-method``.  It is ALWAYS
    called with ``amplitude_lower = 0`` here; see the module docstring.

    Returns the JSON-ready inference summary.
    """
    started = time.perf_counter()
    max_sources = int(max_sources)
    if max_sources < 1:
        raise ValueError("max_sources must be >= 1")

    screen = _BirthScreen(
        mesh_data, pde, observation_state, admissibility,
        eta1=eta1, lam=lam, objective_time_rule=objective_time_rule,
        amplitude_upper=float(amplitude_upper), top_k=int(top_k),
        min_separation=min_separation, boundary_buffer=boundary_buffer,
        building_buffer=building_buffer, screen_radius=screen_radius,
        obstacle_aware_diversity=bool(obstacle_aware_diversity),
        min_value_tol=min_value_tol, dirichlet_tags=dirichlet_tags,
        print_fn=print_fn,
    )

    control = np.zeros(0, dtype=np.float64)
    j_current = float(screen.zero_source_objective)
    levels: list[dict] = []
    stop_reason = None
    m = 0

    print_fn("\n=== Source-Count Inference ===")
    print_fn(f"max sources: {max_sources}")
    print_fn(f"zero-source objective J_0: {j_current:.10e}")
    print_fn(
        f"tolerances: discrepancy={discrepancy_tol:g}  "
        f"rel_reduction={rel_reduction_tol:g}  abs_reduction={abs_reduction_tol:g}  "
        f"amplitude={amp_tol:g}"
    )
    print_fn(
        "nesting: candidate models are polished with amplitude lower bound "
        f"{NESTING_AMPLITUDE_LOWER:g} (production floor "
        f"{production_amplitude_floor!r} is NOT used as evidence of a source)"
    )

    while True:
        if discrepancy_tol > 0.0 and j_current <= float(discrepancy_tol):
            stop_reason = STOP_DISCREPANCY
            print_fn(
                f"stop: J_{m} = {j_current:.6e} <= discrepancy tolerance "
                f"{float(discrepancy_tol):.6e}"
            )
            break
        if m >= max_sources:
            stop_reason = STOP_MAX_SOURCES
            print_fn(f"stop: reached --max-sources {max_sources}")
            break

        threshold = acceptance_threshold(
            j_current, rel_reduction_tol, abs_reduction_tol
        )
        print_fn(
            f"\n--- level {m + 1}: testing a birth on the {m}-source residual ---"
        )
        print_fn(f"J_before = {j_current:.10e}   acceptance threshold = {threshold:.6e}")

        model_state = screen.model_state(control)
        candidate = screen.best_candidate(control, model_state=model_state)
        record: dict = {
            "level": m + 1,
            "m_before": m,
            "J_before": float(j_current),
            "acceptance_threshold": float(threshold),
            "candidate": candidate,
            "predicted_decrease": (
                None if candidate is None else float(candidate["predicted_decrease"])
            ),
        }

        if candidate is None:
            record.update(
                {"accepted": False, "reason": STOP_NO_CANDIDATE,
                 "J_after_candidate": None, "actual_decrease": None}
            )
            levels.append(record)
            stop_reason = STOP_NO_CANDIDATE
            print_fn("stop: no admissible candidate descends the residual")
            break

        predicted = float(candidate["predicted_decrease"])
        print_fn(
            f"best candidate node {candidate['node']} at "
            f"({candidate['center'][0]:.3f}, {candidate['center'][1]:.3f})  "
            f"G={candidate['g']:.6e}  kappa={candidate['kappa']:.6e}  "
            f"a*={candidate['a_1d']:.6f}"
        )
        print_fn(f"predicted decrease = {predicted:.6e}")

        if predicted <= threshold:
            record.update(
                {"accepted": False, "reason": STOP_NEGLIGIBLE_PREDICTED,
                 "J_after_candidate": None, "actual_decrease": None}
            )
            levels.append(record)
            stop_reason = STOP_NEGLIGIBLE_PREDICTED
            print_fn(
                "stop: the best possible birth is below the acceptance threshold; "
                "no polish attempted"
            )
            break

        candidate_control = append_source(
            control, candidate["center"][0], candidate["center"][1], candidate["a_1d"]
        )
        polished, j_after, polish_info = polish_fn(
            candidate_control, NESTING_AMPLITUDE_LOWER
        )
        polished = np.asarray(polished, dtype=np.float64).reshape(-1)
        j_after = float(j_after)
        actual = j_current - j_after
        amps = amplitudes_of(polished)
        min_amp = float(np.min(amps)) if amps.size else 0.0

        record.update(
            {
                "candidate_control": candidate_control.tolist(),
                "J_after_candidate": j_after,
                "actual_decrease": float(actual),
                "polished_control": polished.tolist(),
                "amplitudes_after": [float(a) for a in amps],
                "min_amplitude_after": min_amp,
                "amplitude_tolerance": float(amp_tol),
                "polish": polish_info,
            }
        )
        print_fn(
            f"J_after = {j_after:.10e}   actual decrease = {actual:.6e}   "
            f"min amplitude = {min_amp:.6e}"
        )

        # Nested-model acceptance: a decrease is guaranteed, so it must clear the
        # threshold AND leave every source with a non-degenerate amplitude.
        if actual <= threshold:
            record.update({"accepted": False, "reason": STOP_INSUFFICIENT_ACTUAL})
            levels.append(record)
            stop_reason = STOP_INSUFFICIENT_ACTUAL
            print_fn(
                f"REJECT: actual decrease {actual:.6e} <= threshold {threshold:.6e} "
                "(a decrease alone is not evidence -- the models are nested)"
            )
            break
        if min_amp <= float(amp_tol):
            record.update({"accepted": False, "reason": STOP_AMPLITUDE_COLLAPSE})
            levels.append(record)
            stop_reason = STOP_AMPLITUDE_COLLAPSE
            print_fn(
                f"REJECT: a source collapsed to amplitude {min_amp:.6e} <= "
                f"{float(amp_tol):.6e}; the {m + 1}-source model degenerated back "
                f"to {m} effective source(s)"
            )
            break

        record.update({"accepted": True, "reason": ACCEPT, "m_after": m + 1})
        levels.append(record)
        control = polished
        j_current = j_after
        m += 1
        print_fn(f"ACCEPT: source count is now {m}")

    wall = time.perf_counter() - started
    decision_status, decision_converged = decision_status_for_stop_reason(stop_reason)
    summary = {
        "inferred_num_sources": int(m),
        "max_sources": int(max_sources),
        "stop_reason": stop_reason,
        # Reporting/status semantics only (see decision_status_for_stop_reason):
        # "unresolved_max_sources" means the reported M is a LOWER BOUND, not a
        # decision the acceptance rule actually reached.
        "source_count_decision_status": decision_status,
        "source_count_decision_converged": bool(decision_converged),
        "zero_source_objective": float(screen.zero_source_objective),
        "final_objective": float(j_current),
        "final_control": control.tolist(),
        "tolerances": {
            "discrepancy_tol": float(discrepancy_tol),
            "rel_reduction_tol": float(rel_reduction_tol),
            "abs_reduction_tol": float(abs_reduction_tol),
            "amp_tol": float(amp_tol),
            "min_value_tol": float(min_value_tol),
        },
        "screen": {
            "top_k": int(top_k),
            "screen_radius": float(screen.screen_radius),
            "min_separation": (
                None if min_separation is None else float(min_separation)
            ),
            "obstacle_aware_diversity": bool(obstacle_aware_diversity),
            "amplitude_upper": float(amplitude_upper),
        },
        "nesting": {
            "amplitude_lower_bound_during_inference": float(NESTING_AMPLITUDE_LOWER),
            "production_amplitude_floor": (
                None if production_amplitude_floor is None
                else float(production_amplitude_floor)
            ),
            "note": (
                "candidate models are polished with amplitude lower bound 0 so the "
                "m-source model stays nested inside the (m+1)-source model; the "
                "production amplitude floor is never used as evidence of a source"
            ),
        },
        "levels": levels,
        "solves": dict(screen.solves),
        "wall_time_seconds": float(wall),
    }

    print_fn(f"\ninferred number of sources: {m}  (stop reason: {stop_reason})")
    print_fn(
        f"source-count decision: {decision_status}  "
        f"(converged={decision_converged})"
    )
    if not decision_converged:
        print_fn(
            f"  WARNING: --max-sources {max_sources} was reached before the "
            "acceptance rule ran again; M is a LOWER BOUND, not a settled "
            "decision -- a real additional source may exist and was never "
            "tested"
        )
    print_fn(f"final inference objective: {j_current:.10e}")
    print_fn(
        f"inference solves: forward={screen.solves['forward']}  "
        f"adjoint={screen.solves['adjoint']}  "
        f"candidate traces={screen.solves['candidate_traces']}"
    )
    print_fn(f"inference wall time: {wall:.2f} s")
    return summary


__all__ = [
    "ACCEPT",
    "DECISION_ACCEPTED",
    "DECISION_REJECTED_STOP",
    "DECISION_UNRESOLVED_MAX_SOURCES",
    "DEFAULT_ABS_REDUCTION_TOL",
    "DEFAULT_AMP_TOL",
    "DEFAULT_DISCREPANCY_TOL",
    "DEFAULT_MAX_SOURCES",
    "DEFAULT_REL_REDUCTION_TOL",
    "DEFAULT_TOP_K",
    "NESTING_AMPLITUDE_LOWER",
    "STOP_AMPLITUDE_COLLAPSE",
    "STOP_DISCREPANCY",
    "STOP_INSUFFICIENT_ACTUAL",
    "STOP_MAX_SOURCES",
    "STOP_NEGLIGIBLE_PREDICTED",
    "STOP_NO_CANDIDATE",
    "acceptance_threshold",
    "amplitudes_of",
    "append_source",
    "decision_status_for_stop_reason",
    "infer_source_count",
]
