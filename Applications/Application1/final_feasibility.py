"""Final feasibility repair/certification for accepted Application1 source
controls.

``--feasibility-mode raw`` (see ``run_optimization.py``) is a useful search
mode: it lets a source center cross a building wall during intermediate
iterations so the optimizer can escape a wall-pinned stall. But nothing
guarantees the FINAL accepted control it hands back is inside the fluid mesh,
and other feasibility modes have their own final-admissibility gaps (see
``Applications/Application1/README.md``, "Known weaknesses"). This module is
the final, short, certification/repair step: given a flat control, decide
whether every source center clears the fluid boundary by the requested
margin and, if not, repair it with a brief SciPy-constrained polish.

This is deliberately NOT a general optimizer. It is meant to run once, after
the real optimization (and any raw-mode search) has already produced a
candidate control, with a small iteration budget.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize

from mesh_admissibility import MeshAdmissibility
from multi_source_utils import flatten_control, reshape_control

DEFAULT_FINAL_FEASIBILITY_MARGIN = 0.5
DEFAULT_FINAL_FEASIBILITY_MAX_ITER = 50
FINAL_FEASIBILITY_METHODS = ("slsqp",)
DEFAULT_FINAL_FEASIBILITY_METHOD = "slsqp"


def signed_clearances(
    control: np.ndarray, admissibility: MeshAdmissibility
) -> np.ndarray:
    """Per-source ``signed_clearance`` for a flat ``[x1, y1, A1, ...]`` control."""
    return np.array(
        [
            admissibility.signed_clearance(float(x), float(y))
            for x, y, _amplitude in reshape_control(control)
        ],
        dtype=np.float64,
    )


def _project_infeasible_centers(control: np.ndarray, admissibility: MeshAdmissibility):
    """Project every source center OUTSIDE the fluid mesh once into the domain.

    Amplitudes are left untouched. Centers already inside the fluid mesh
    (even if closer to the wall than the requested margin) are left alone --
    the constrained solve below is responsible for pushing those the rest of
    the way to the margin. Returns ``(new_control, was_projected)``.
    """
    Z = reshape_control(control).copy()
    was_projected = False
    for source_id in range(Z.shape[0]):
        x, y = float(Z[source_id, 0]), float(Z[source_id, 1])
        if admissibility.is_point_in_domain(x, y):
            continue
        new_point, moved, _distance = admissibility.project_into_domain(x, y)
        Z[source_id, 0] = new_point[0]
        Z[source_id, 1] = new_point[1]
        was_projected = was_projected or bool(moved)
    return flatten_control(Z, num_sources=int(Z.shape[0])), was_projected


def final_feasibility_polish(
    control: np.ndarray,
    admissibility: MeshAdmissibility,
    bounds,
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    *,
    clearance_margin: float = DEFAULT_FINAL_FEASIBILITY_MARGIN,
    max_iter: int = DEFAULT_FINAL_FEASIBILITY_MAX_ITER,
    method: str = DEFAULT_FINAL_FEASIBILITY_METHOD,
) -> dict:
    """Certify or repair a flat source control so every center clears the
    fluid boundary by ``clearance_margin``.

    Parameters
    ----------
    control:
        Flat ``[x1, y1, A1, ..., xM, yM, AM]`` control to certify/repair.
    admissibility:
        A ``MeshAdmissibility`` built from the run's mesh data.
    bounds:
        Sequence of ``(lo, hi)`` pairs, one per entry of ``control`` (i.e. 3
        per source: x, y, amplitude), matching the layout SciPy expects for
        ``scipy.optimize.minimize(..., bounds=...)``.
    objective, gradient:
        Callables ``f(z) -> float`` / ``g(z) -> np.ndarray`` that accept the
        SAME flat ``[x1, y1, A1, ...]`` layout as ``control`` and return the
        true reduced objective/gradient at any point (including centers
        outside the fluid mesh -- callers should build these with
        ``feasibility_mode="raw"`` so intermediate SLSQP iterates, which are
        not guaranteed feasible, are still well-defined).
    clearance_margin:
        Required signed clearance (meters) for every source center.
    max_iter:
        Iteration cap for the repair solve. This is a short final polish, not
        the main optimizer.
    method:
        SciPy constrained method to use. Only ``"slsqp"`` is supported today.

    Returns
    -------
    dict with keys ``repaired_control``, ``feasible``, ``objective_before``,
    ``objective_after``, ``clearance_before_min/max``,
    ``clearance_after_min/max``, ``iterations``, ``message``,
    ``was_projected``, ``method``.
    """
    if method not in FINAL_FEASIBILITY_METHODS:
        raise ValueError(
            f"Unknown final feasibility method {method!r}; expected one of "
            f"{FINAL_FEASIBILITY_METHODS}"
        )
    if not np.isfinite(clearance_margin):
        raise ValueError("clearance_margin must be finite")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")

    control = np.asarray(control, dtype=np.float64).reshape(-1)
    num_sources = control.size // 3
    if control.size != 3 * num_sources or num_sources < 1:
        raise ValueError(
            f"Control of length {control.size} does not describe a whole "
            "number of sources"
        )
    bounds = list(bounds)
    if len(bounds) != control.size:
        raise ValueError(
            f"Expected {control.size} bound pairs, got {len(bounds)}"
        )

    clearances_before = signed_clearances(control, admissibility)
    objective_before = float(objective(control))

    if bool(np.all(clearances_before >= clearance_margin)):
        return {
            "repaired_control": control.copy(),
            "feasible": True,
            "objective_before": objective_before,
            "objective_after": objective_before,
            "clearance_before_min": float(np.min(clearances_before)),
            "clearance_before_max": float(np.max(clearances_before)),
            "clearance_after_min": float(np.min(clearances_before)),
            "clearance_after_max": float(np.max(clearances_before)),
            "iterations": 0,
            "message": "already feasible at the requested margin; no repair run",
            "was_projected": False,
            "method": method,
        }

    x0, was_projected = _project_infeasible_centers(control, admissibility)
    lower = np.array([lo for lo, _hi in bounds], dtype=np.float64)
    upper = np.array([hi for _lo, hi in bounds], dtype=np.float64)
    x0 = np.clip(x0, lower, upper)

    def constraint_fun(z: np.ndarray) -> np.ndarray:
        return signed_clearances(z, admissibility) - float(clearance_margin)

    result = minimize(
        lambda z: float(objective(z)),
        x0,
        jac=lambda z: np.asarray(gradient(z), dtype=np.float64),
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": constraint_fun}],
        options={"maxiter": int(max_iter), "ftol": 1.0e-12, "disp": False},
    )

    repaired_control = np.asarray(result.x, dtype=np.float64).reshape(-1)
    clearances_after = signed_clearances(repaired_control, admissibility)
    feasible = bool(np.all(clearances_after >= clearance_margin - 1.0e-9))
    objective_after = float(objective(repaired_control))

    return {
        "repaired_control": repaired_control,
        "feasible": feasible,
        "objective_before": objective_before,
        "objective_after": objective_after,
        "clearance_before_min": float(np.min(clearances_before)),
        "clearance_before_max": float(np.max(clearances_before)),
        "clearance_after_min": float(np.min(clearances_after)),
        "clearance_after_max": float(np.max(clearances_after)),
        "iterations": int(getattr(result, "nit", 0)),
        "message": str(result.message),
        "was_projected": bool(was_projected),
        "method": method,
    }
