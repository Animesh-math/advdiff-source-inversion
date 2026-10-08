"""Post-initialization polish methods for the Application1 inverse problem.

Initialization (sensitivity / adaptive / smart / mesh_random / manual) produces
the starting control.  This module owns the SECOND half of that pipeline: which
optimizer polishes that start.

Three methods are exposed:

``lbfgsb``        the historical ``Src/Optimizers/optimizer.py`` L-BFGS-B run.
                  Default, so every existing run is byte-for-byte unchanged.
``gauss_newton``  bound-constrained trust-region Newton driven by the dense
                  Gauss-Newton Hessian from ``hessian_utils`` (``mode=
                  "gauss_newton"``): the residual-through-state term only.
``newton``        the same driver with the EXACT reduced Hessian
                  (``hessian_utils`` ``mode="exact"``), which additionally
                  contracts the base adjoint with ``D^2 S(z)``.

No new curvature code is written here.  ``hessian_utils.hessian_matrix`` is the
sole source of second-order information; this module only restricts it to the
active variables of the current ``--mode`` and hands it to SciPy's
``trust-constr``, the one SciPy method that accepts both a Hessian and bounds.

Solve accounting
----------------
``solve_counter()`` patches the two module-level solver entry points that every
caller reaches through ``gradient_test`` (``run_forward`` and
``rk4_discrete_source_adjoint``) so a polish reports its EXACT forward/adjoint
solve count.  Both the L-BFGS-B and the Newton-type paths are counted the same
way, which is what makes the two comparable.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
from scipy.optimize import minimize


LBFGSB = "lbfgsb"
GAUSS_NEWTON = "gauss_newton"
NEWTON = "newton"
POST_INIT_METHODS = (LBFGSB, GAUSS_NEWTON, NEWTON)
DEFAULT_POST_INIT_METHOD = LBFGSB

#: ``--post-init-method`` -> ``hessian_utils`` Hessian mode.
HESSIAN_MODE_BY_METHOD = {
    GAUSS_NEWTON: "gauss_newton",
    NEWTON: "exact",
}

#: SciPy method used for the Newton-type polish. ``trust-constr`` is the only
#: SciPy minimizer that takes a user Hessian AND bounds, and the bounds here are
#: the same mesh-bounding-box / amplitude-floor box L-BFGS-B is given.
NEWTON_SCIPY_METHOD = "trust-constr"

DEFAULT_NEWTON_GTOL = 1.0e-12
DEFAULT_NEWTON_XTOL = 1.0e-14
DEFAULT_NEWTON_INITIAL_TR_RADIUS = 1.0


#: ``--post-init-scaling``: optional diagonal variable scaling of the L-BFGS-B
#: polish. ``none`` (default) is the historical unscaled run. ``gn-diag`` runs
#: L-BFGS-B in y = D z with D = sqrt(diag H_GN) at the initialized control.
#: ``on-stall`` runs the unscaled polish first and switches to the scaled one
#: from the current iterate only if it has not converged after
#: ``--post-init-stall-iters`` iterations, so runs that converge before then are
#: byte-for-byte unchanged.
SCALING_NONE = "none"
SCALING_GN_DIAG = "gn-diag"
SCALING_ON_STALL = "on-stall"
POST_INIT_SCALINGS = (SCALING_NONE, SCALING_GN_DIAG, SCALING_ON_STALL)
DEFAULT_POST_INIT_SCALING = SCALING_NONE
DEFAULT_STALL_ITERS = 150
#: Cap on max(D) / min(D). A variable the sensors barely see has a tiny
#: Hessian diagonal; flooring D keeps its scaled steps from becoming huge.
DEFAULT_SCALING_MAX_RATIO = 1.0e4


def is_newton_type(method: str) -> bool:
    return method in HESSIAN_MODE_BY_METHOD


def validate_post_init_method(method: str) -> str:
    if method not in POST_INIT_METHODS:
        raise ValueError(
            f"Unknown post-init method {method!r}; expected one of "
            f"{list(POST_INIT_METHODS)}"
        )
    return method


def active_index_map(mode: str, num_sources: int) -> np.ndarray:
    """Full-control indices of the active variables of ``mode``.

    Mirrors ``multi_source_utils.pack_active`` / ``restrict_gradient`` exactly,
    and is what restricts the full 3M x 3M Hessian to the active block.
    """
    num_sources = int(num_sources)
    if mode == "location_only":
        indices = [3 * s + c for s in range(num_sources) for c in (0, 1)]
    elif mode == "intensity_only":
        indices = [3 * s + 2 for s in range(num_sources)]
    elif mode == "location_and_intensity":
        indices = list(range(3 * num_sources))
    else:
        raise ValueError(f"Unknown CONTROL_MODE: {mode}")
    return np.asarray(indices, dtype=int)


@contextmanager
def solve_counter(gt_module):
    """Count forward and adjoint solves issued through ``gradient_test``.

    Every solve on the objective, gradient and Hessian paths goes through
    ``gt.run_forward`` / ``gt.rk4_discrete_source_adjoint`` (``hessian_utils``
    reaches them as ``app1.gt.<name>``), so patching the module attributes for
    the duration of a polish counts all of them and nothing else.
    """
    counts = {"forward_solves": 0, "adjoint_solves": 0}
    original_forward = gt_module.run_forward
    original_adjoint = gt_module.rk4_discrete_source_adjoint

    def counting_forward(*args, **kwargs):
        counts["forward_solves"] += 1
        return original_forward(*args, **kwargs)

    def counting_adjoint(*args, **kwargs):
        counts["adjoint_solves"] += 1
        return original_adjoint(*args, **kwargs)

    gt_module.run_forward = counting_forward
    gt_module.rk4_discrete_source_adjoint = counting_adjoint
    try:
        yield counts
    finally:
        gt_module.run_forward = original_forward
        gt_module.rk4_discrete_source_adjoint = original_adjoint


def build_active_hessian(
    *,
    method: str,
    mode: str,
    num_sources: int,
    mesh_data,
    pde: dict,
    target_state: np.ndarray,
    z_template: np.ndarray,
    fixed_values,
    eta1: float,
    lam: float,
    unpack_active,
    hessian_module,
    record=None,
):
    """Return ``hess(z_active) -> (n_active, n_active)`` dense Hessian.

    The Hessian itself comes from ``hessian_utils.hessian_matrix``; this wrapper
    only lifts the active variables to the full control, calls it, and takes the
    active sub-block.
    """
    hessian_mode = HESSIAN_MODE_BY_METHOD[method]
    active = active_index_map(mode, num_sources)

    def hess(z_active):
        control = unpack_active(
            np.asarray(z_active, dtype=np.float64), z_template, mode, fixed_values
        )
        full = hessian_module.hessian_matrix(
            control,
            mesh_data,
            pde,
            target_state,
            eta1=eta1,
            lam=lam,
            mode=hessian_mode,
        )
        block = np.asarray(full, dtype=np.float64)[np.ix_(active, active)]
        # The reduced Hessian is symmetric in exact arithmetic; the coordinate
        # HVP assembly is not symmetric to the last bit, and trust-constr wants
        # a symmetric operator.
        block = 0.5 * (block + block.T)
        if record is not None:
            eigenvalues = np.linalg.eigvalsh(block)
            record.append(
                {
                    "control": control.tolist(),
                    "hessian_mode": hessian_mode,
                    "eigenvalues": [float(v) for v in eigenvalues],
                    "min_eigenvalue": float(eigenvalues[0]),
                    "max_eigenvalue": float(eigenvalues[-1]),
                    "condition_number": (
                        float(abs(eigenvalues[-1]) / abs(eigenvalues[0]))
                        if eigenvalues[0] != 0.0
                        else None
                    ),
                    "positive_definite": bool(eigenvalues[0] > 0.0),
                }
            )
        return block

    return hess


def run_newton_type_polish(
    *,
    method: str,
    objective,
    gradient,
    hess,
    x0: np.ndarray,
    bounds,
    max_iter: int,
    callback=None,
    gtol: float = DEFAULT_NEWTON_GTOL,
    xtol: float = DEFAULT_NEWTON_XTOL,
    initial_tr_radius: float = DEFAULT_NEWTON_INITIAL_TR_RADIUS,
    verbose: int = 0,
):
    """Run the bound-constrained trust-region Newton polish.

    Returns the SciPy ``OptimizeResult``.  ``callback`` is the evaluator's
    existing single-argument iteration callback; ``trust-constr`` calls back with
    ``(xk, state)`` so it is adapted here rather than in the evaluator.
    """
    validate_post_init_method(method)
    if not is_newton_type(method):
        raise ValueError(
            f"run_newton_type_polish got {method!r}; use the L-BFGS-B path for "
            f"{LBFGSB!r}"
        )

    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)

    scipy_bounds = None
    if bounds is not None:
        lower = np.asarray([b[0] for b in bounds], dtype=np.float64)
        upper = np.asarray([b[1] for b in bounds], dtype=np.float64)
        from scipy.optimize import Bounds

        scipy_bounds = Bounds(lower, upper, keep_feasible=False)

    def scipy_callback(xk, state=None):
        if callback is not None:
            callback(np.asarray(xk, dtype=np.float64))
        return False

    print(
        f"[PostInit] method={method} scipy={NEWTON_SCIPY_METHOD} "
        f"hessian={HESSIAN_MODE_BY_METHOD[method]} max_iter={max_iter} "
        f"gtol={gtol:.1e} xtol={xtol:.1e}"
    )

    return minimize(
        fun=lambda z: float(objective(z)),
        x0=x0,
        method=NEWTON_SCIPY_METHOD,
        jac=lambda z: np.asarray(gradient(z), dtype=np.float64).reshape(-1),
        hess=hess,
        bounds=scipy_bounds,
        callback=scipy_callback,
        options={
            "maxiter": int(max_iter),
            "gtol": float(gtol),
            "xtol": float(xtol),
            "initial_tr_radius": float(initial_tr_radius),
            "verbose": int(verbose),
        },
    )


def gn_diagonal_scaling(hess, z_active, max_ratio: float = DEFAULT_SCALING_MAX_RATIO):
    """Return ``(D, record)`` with ``D = sqrt(diag H_GN(z_active))``.

    ``hess`` is a ``build_active_hessian(method=GAUSS_NEWTON, ...)`` callable.
    Entries below ``max(D) / max_ratio`` are raised to that floor.
    """
    z_active = np.asarray(z_active, dtype=np.float64).reshape(-1)
    block = np.asarray(hess(z_active), dtype=np.float64)
    diagonal = np.diag(block).copy()
    raw = np.sqrt(np.maximum(diagonal, 0.0))
    if not np.all(np.isfinite(raw)) or raw.max() <= 0.0:
        raise ValueError(f"Gauss-Newton diagonal unusable for scaling: {diagonal}")
    floor = raw.max() / float(max_ratio)
    scale = np.maximum(raw, floor)
    eigenvalues = np.linalg.eigvalsh(block)
    scaled_eigenvalues = np.linalg.eigvalsh(block / np.outer(scale, scale))

    def _cond(w):
        return float(abs(w[-1]) / abs(w[0])) if w[0] != 0.0 else None

    record = {
        "control_active": z_active.tolist(),
        "hessian_diagonal": diagonal.tolist(),
        "scale_vector": scale.tolist(),
        "max_ratio": float(max_ratio),
        "n_floored": int(np.count_nonzero(raw < floor)),
        "condition_number_unscaled": _cond(eigenvalues),
        "condition_number_scaled": _cond(scaled_eigenvalues),
    }
    return scale, record


def run_scaled_lbfgsb(
    *,
    optimizer_cls,
    objective,
    gradient,
    x0: np.ndarray,
    scale: np.ndarray,
    bounds,
    max_iter: int,
    tol: float,
    options: dict,
    callback=None,
    mesh_data=None,
    problem_data=None,
):
    """L-BFGS-B in y = scale * z; the result's ``x`` (and ``jac``) are in z.

    ``objective``/``gradient``/``callback`` keep working in the original
    variables, so evaluator caching and the iteration history are unchanged.
    The stopping tolerance ``gtol`` applies to the scaled gradient
    ``grad / scale``.
    """
    scale = np.asarray(scale, dtype=np.float64).reshape(-1)
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    scaled_bounds = None
    if bounds is not None:
        scaled_bounds = [
            (
                None if lo is None else float(lo) * s,
                None if hi is None else float(hi) * s,
            )
            for (lo, hi), s in zip(bounds, scale)
        ]

    def scaled_callback(y):
        if callback is not None:
            callback(np.asarray(y, dtype=np.float64) / scale)

    optimizer = optimizer_cls(
        objective_function=lambda y, *_: objective(np.asarray(y) / scale),
        gradient_function=lambda y, *_: (
            np.asarray(gradient(np.asarray(y) / scale), dtype=np.float64) / scale
        ),
        optimization_method="L-BFGS-B",
        max_iter=max_iter,
        tol=tol,
        bounds=scaled_bounds,
        callback=scaled_callback,
        options=options,
    )
    result = optimizer.optimize(
        initial_guess=x0 * scale, mesh_data=mesh_data, problem_data=problem_data
    )
    result.x = np.asarray(result.x, dtype=np.float64) / scale
    if getattr(result, "jac", None) is not None:
        result.jac = np.asarray(result.jac, dtype=np.float64) * scale
    return result


def run_lbfgsb_polish(
    *,
    optimizer_cls,
    objective,
    gradient,
    x0: np.ndarray,
    bounds,
    max_iter: int,
    tol: float,
    options: dict,
    callback=None,
    mesh_data=None,
    problem_data=None,
    scaling: str = DEFAULT_POST_INIT_SCALING,
    stall_iters: int = DEFAULT_STALL_ITERS,
    max_ratio: float = DEFAULT_SCALING_MAX_RATIO,
    gn_hess=None,
):
    """The L-BFGS-B polish with optional diagonal scaling.

    Returns ``(result, scaling_record)``. ``scaling="none"`` is exactly the
    historical single unscaled ``Optimizer`` run. ``gn_hess`` (an active
    Gauss-Newton Hessian callable) is required for the other two modes.
    """
    if scaling not in POST_INIT_SCALINGS:
        raise ValueError(
            f"Unknown post-init scaling {scaling!r}; expected one of "
            f"{list(POST_INIT_SCALINGS)}"
        )
    common = dict(
        optimizer_cls=optimizer_cls,
        objective=objective,
        gradient=gradient,
        bounds=bounds,
        tol=tol,
        options=options,
        callback=callback,
        mesh_data=mesh_data,
        problem_data=problem_data,
    )
    record = {"scaling": scaling}

    def unscaled(x_start, iterations):
        optimizer = optimizer_cls(
            objective_function=objective,
            gradient_function=gradient,
            optimization_method="L-BFGS-B",
            max_iter=iterations,
            tol=tol,
            bounds=bounds,
            callback=callback,
            options=options,
        )
        return optimizer.optimize(
            initial_guess=x_start, mesh_data=mesh_data, problem_data=problem_data
        )

    if scaling == SCALING_NONE:
        return unscaled(x0, max_iter), record

    if gn_hess is None:
        raise ValueError(f"post-init scaling {scaling!r} needs a Gauss-Newton Hessian")

    x_start = np.asarray(x0, dtype=np.float64).reshape(-1)
    remaining = int(max_iter)
    first = None
    if scaling == SCALING_ON_STALL:
        record["stall_iters"] = int(stall_iters)
        if int(stall_iters) >= int(max_iter):
            record["switched"] = False
            record["note"] = "stall_iters >= max_iter: unscaled run only"
            return unscaled(x0, max_iter), record
        first = unscaled(x_start, int(stall_iters))
        # status 1 = iteration (or evaluation) limit; anything else is a normal
        # stop of the unscaled run and is returned unchanged.
        if int(first.status) != 1:
            record["switched"] = False
            return first, record
        record["switched"] = True
        record["switch_iteration"] = int(first.nit)
        record["unscaled_phase"] = {
            "iterations": int(first.nit),
            "nfev": int(getattr(first, "nfev", 0) or 0),
            "njev": int(getattr(first, "njev", 0) or 0),
            "message": str(first.message),
            "objective": float(first.fun),
        }
        x_start = np.asarray(first.x, dtype=np.float64)
        remaining = int(max_iter) - int(first.nit)
        print(
            f"[PostInit] on-stall: not converged after {first.nit} unscaled "
            f"iterations (J={float(first.fun):.6e}); switching to Gauss-Newton "
            f"diagonal scaling for the remaining {remaining}"
        )

    scale, scale_record = gn_diagonal_scaling(gn_hess, x_start, max_ratio=max_ratio)
    record.update(scale_record)
    print(
        "[PostInit] gn-diag scaling: cond "
        f"{scale_record['condition_number_unscaled']:.3e} -> "
        f"{scale_record['condition_number_scaled']:.3e}  "
        f"D={np.array2string(scale, precision=3)}"
    )
    result = run_scaled_lbfgsb(x0=x_start, scale=scale, max_iter=remaining, **common)
    if first is not None:
        # Report the whole polish: counts span both phases.
        result.nit = int(first.nit) + int(result.nit)
        result.nfev = int(getattr(first, "nfev", 0) or 0) + int(result.nfev)
        result.njev = int(getattr(first, "njev", 0) or 0) + int(
            getattr(result, "njev", 0) or 0
        )
    return result, record


__all__ = [
    "DEFAULT_POST_INIT_METHOD",
    "DEFAULT_POST_INIT_SCALING",
    "DEFAULT_SCALING_MAX_RATIO",
    "DEFAULT_STALL_ITERS",
    "POST_INIT_SCALINGS",
    "SCALING_GN_DIAG",
    "SCALING_NONE",
    "SCALING_ON_STALL",
    "gn_diagonal_scaling",
    "run_lbfgsb_polish",
    "run_scaled_lbfgsb",
    "GAUSS_NEWTON",
    "HESSIAN_MODE_BY_METHOD",
    "LBFGSB",
    "NEWTON",
    "NEWTON_SCIPY_METHOD",
    "POST_INIT_METHODS",
    "active_index_map",
    "build_active_hessian",
    "is_newton_type",
    "run_newton_type_polish",
    "solve_counter",
    "validate_post_init_method",
]
