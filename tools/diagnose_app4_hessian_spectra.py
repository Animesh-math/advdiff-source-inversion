"""Diagnose App4 fake-basin failures via Hessian spectra at fixed controls.

This is a read-only diagnostic. It reuses the verified App4 Hessian/HVP
pipeline (``hessian_utils.hessian_matrix``, exact mode) and the RK4
discrete-adjoint gradient to characterise a small, hand-picked list of
controls (truths, recovered solutions, and known fake-basin failures) for
seed-7 and seed-13.

For each control it reports, at the appropriate ``tfinal`` and Gaussian width
``sigma2``:

- objective J(z),
- full gradient norm and bound-projected gradient norm,
- the dense 6x6 exact Hessian, symmetrised,
- eigenvalues, number of negative eigenvalues, min/max eigenvalue,
- condition number from nonzero |eigenvalues|,
- the eigenvector of the most-negative and smallest-magnitude eigenvalues.

It runs a handful of forward/adjoint solves per control; it is NOT a sweep.
Nothing is modified in the solver; outputs go under
``Applications/Test Application4/output/hessian_diagnostics/``.

Usage (from repo root):
    python3 \
        tools/diagnose_app4_hessian_spectra.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "Applications" / "Test Application4"
sys.path.insert(0, str(APP_DIR))

import hessian_utils  # noqa: E402
import run_optimization as app4  # noqa: E402
from mesh_admissibility import MeshAdmissibility  # noqa: E402
from multi_source_utils import (  # noqa: E402
    MESH_DOMAIN_BOUND_CASE,
    component_names,
)

VELOCITY_FIELD_FILE = (
    APP_DIR / "output" / "navier_stokes" / "app4_navier_stokes_velocity_nodes.npz"
)
MEASUREMENT_BOUNDARIES = ["right", "top", "inner_corner"]
AMPLITUDE_FLOOR = 0.05
AMPLITUDE_CEIL = 2.0
OUT_DIR = APP_DIR / "output" / "hessian_diagnostics"

# Truths.
T7 = [0.64, 0.86, 1.20, 1.20, 0.48, 0.85]          # seed-7 alt truth
T13 = [0.20, 0.88, 0.80, 1.20, 0.48, 1.25]         # seed-13 hard truth

# Diagnostic controls: (label, control, truth, tfinal, sigma2, note).
# All seed-7 tfinal2/tfinal4 runs used sigma schedule ending at target 0.003;
# fakes were "trapped" already at the 0.005 stage, so we evaluate both.
CASES = [
    # ---- seed-7, tfinal = 2.0 ----
    ("s7_truth_tf2_s003", T7, T7, 2.0, 0.003, "seed-7 truth == exact recovery"),
    ("s7_fakeA_tf2_s003",
     [0.6650123164, 0.8721533128, 0.9512961588,
      0.2960063119, 0.7267462434, 1.6768934563],
     T7, 2.0, 0.003, "seed-7 tfinal2 fake (start-1 final, ABNORMAL_LNSRCH)"),
    ("s7_fakeA_tf2_s005",
     [0.6650123164, 0.8721533128, 0.9512961588,
      0.2960063119, 0.7267462434, 1.6768934563],
     T7, 2.0, 0.005, "seed-7 tfinal2 fake evaluated at trap-stage sigma"),
    # ---- seed-7, tfinal = 4.0 ----
    ("s7_truth_tf4_s003", T7, T7, 4.0, 0.003, "seed-7 truth at tfinal=4 baseline"),
    ("s7_fakeB_tf4_s003",
     [0.6874781198, 0.8850292029, 0.8037973945,
      0.3078048113, 0.7404461406, 1.6232185987],
     T7, 4.0, 0.003, "seed-7 tfinal4 fake final control"),
    ("s7_fakeB_tf4_s005",
     [0.6874781198, 0.8850292029, 0.8037973945,
      0.3078048113, 0.7404461406, 1.6232185987],
     T7, 4.0, 0.005, "seed-7 tfinal4 fake evaluated at trap-stage sigma"),
    # ---- seed-13, tfinal = 4.0 ----
    ("s13_truth_tf4_s003", T13, T13, 4.0, 0.003, "seed-13 truth baseline"),
    ("s13_start1_fail_tf4_s003",
     [0.8815838255077459, 0.5499999999964803, 1.035431554389654,
      1.0667425762979899, 0.18316430506872736, 0.06026439894999556],
     T13, 4.0, 0.003, "seed-13 start-1 final (grad~1.12; y1@cutout, A2~floor)"),
]


def build_pde(mesh_data, tfinal: float, sigma2: float) -> dict:
    pde = dict(app4.PDE)
    pde["objective_time_rule"] = app4.OBJECTIVE_TIME_RULE
    pde["tfinal"] = float(tfinal)
    pde["sigma2"] = float(sigma2)
    velocity_nodes, velocity_summary = app4.load_velocity_field_file(
        str(VELOCITY_FIELD_FILE), mesh_data
    )
    pde["velocity_nodes"] = velocity_nodes
    pde["velocity_field"] = velocity_summary
    return pde


def gradient_at(control, mesh_data, pde, target_state):
    state = app4.gt.run_forward(mesh_data, pde, control)
    _greg, _gadj, gradient, _graw = app4.gt.compute_gradient_parts(
        control, state, target_state, mesh_data, pde,
        eta1=app4.ETA1, lam=app4.LAMBDA_REG,
        gradient_time_rule=app4.GRADIENT_TIME_RULE,
    )
    return np.asarray(gradient, dtype=np.float64), state


def objective_at(control, state, target_state, mesh_data, pde):
    return float(app4.gt.compute_objective(
        state, target_state, mesh_data,
        eta1=app4.ETA1, lam=app4.LAMBDA_REG, z=np.asarray(control),
        dt_out=pde["dt"] * pde["jprint"], time_rule=app4.OBJECTIVE_TIME_RULE,
    ))


def projected_gradient(control, gradient, floor, ceil):
    """Bound-projected gradient for a minimisation box [floor/0, .., 2] problem.

    Location bounds are the full mesh bbox (effectively inactive for interior
    centres); amplitude bounds are [floor, ceil]. At a lower-active variable we
    keep only the non-negative part of g; at an upper-active variable only the
    non-positive part.
    """
    lower = np.array([0.0, 0.0, floor, 0.0, 0.0, floor])
    upper = np.array([1.4, 1.0, ceil, 1.4, 1.0, ceil])
    z = np.asarray(control, dtype=np.float64)
    pg = gradient.copy()
    tol = 1.0e-9
    at_low = z <= lower + tol
    at_high = z >= upper - tol
    pg[at_low] = np.minimum(pg[at_low], 0.0)
    pg[at_high] = np.maximum(pg[at_high], 0.0)
    return pg, at_low, at_high


def classify(objective, grad_norm, pgrad_norm, eigenvalues, tol_grad=1.0e-3):
    neg = int(np.sum(eigenvalues < -1.0e-10))
    pos = int(np.sum(eigenvalues > 1.0e-10))
    min_eig = float(eigenvalues.min())
    stationary = pgrad_norm < tol_grad
    if not stationary:
        return "nonstationary (line-search artifact / not a critical point)"
    if neg == 0 and pos == len(eigenvalues):
        kind = "true local minimum" if objective < 1.0e-8 else "fake local minimum"
        return f"{kind} (Hessian positive definite)"
    if neg > 0 and pos > 0:
        return f"saddle / indefinite ({neg} negative curvature directions)"
    if neg == len(eigenvalues):
        return "local maximum (negative definite)"
    return "semidefinite / degenerate"


def diagnose_one(label, control, truth, tfinal, sigma2, note):
    control = np.asarray(control, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    mesh_data = app4.configure_measurement_boundaries(
        app4.build_mesh_data(), MEASUREMENT_BOUNDARIES
    )
    pde = build_pde(mesh_data, tfinal, sigma2)

    target_state = app4.gt.run_forward(mesh_data, pde, truth)
    gradient, state = gradient_at(control, mesh_data, pde, target_state)
    objective = objective_at(control, state, target_state, mesh_data, pde)
    pgrad, at_low, at_high = projected_gradient(
        control, gradient, AMPLITUDE_FLOOR, AMPLITUDE_CEIL
    )

    H = hessian_utils.hessian_matrix(
        control, mesh_data, pde, target_state,
        eta1=app4.ETA1, lam=app4.LAMBDA_REG, mode="exact",
    )
    H_sym = 0.5 * (H + H.T)
    eigvals, eigvecs = np.linalg.eigh(H_sym)
    asym_abs = float(np.linalg.norm(H - H.T, ord="fro"))
    asym_rel = asym_abs / max(1.0, float(np.linalg.norm(H, ord="fro")))

    nonzero = np.abs(eigvals) > 1.0e-12
    cond = None
    if np.any(nonzero):
        a = np.abs(eigvals[nonzero])
        cond = float(a.max() / a.min())

    most_neg_vec = eigvecs[:, 0]
    smallest_idx = int(np.argmin(np.abs(eigvals)))
    smallest_vec = eigvecs[:, smallest_idx]

    verdict = classify(
        objective, float(np.linalg.norm(gradient)),
        float(np.linalg.norm(pgrad)), eigvals,
    )

    result = {
        "label": label,
        "note": note,
        "tfinal": tfinal,
        "sigma2": sigma2,
        "control": control.tolist(),
        "truth": truth.tolist(),
        "objective": objective,
        "gradient": gradient.tolist(),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "projected_gradient_norm": float(np.linalg.norm(pgrad)),
        "active_lower": [bool(x) for x in at_low],
        "active_upper": [bool(x) for x in at_high],
        "hessian": H.tolist(),
        "hessian_asymmetry_abs": asym_abs,
        "hessian_asymmetry_rel": asym_rel,
        "eigenvalues": eigvals.tolist(),
        "num_negative_eigs": int(np.sum(eigvals < -1.0e-10)),
        "num_positive_eigs": int(np.sum(eigvals > 1.0e-10)),
        "min_eigenvalue": float(eigvals.min()),
        "max_eigenvalue": float(eigvals.max()),
        "condition_number_abs": cond,
        "most_negative_eigenvector": most_neg_vec.tolist(),
        "smallest_magnitude_eig": float(eigvals[smallest_idx]),
        "smallest_magnitude_eigenvector": smallest_vec.tolist(),
        "verdict": verdict,
    }

    print(f"\n===== {label} =====")
    print(f"  note: {note}")
    print(f"  tfinal={tfinal}  sigma2={sigma2}")
    print(f"  control = {np.array2string(control, precision=6)}")
    print(f"  J = {objective:.6e}")
    print(f"  |grad| = {np.linalg.norm(gradient):.6e}   "
          f"|proj grad| = {np.linalg.norm(pgrad):.6e}")
    print(f"  active lower={result['active_lower']} upper={result['active_upper']}")
    print(f"  eigenvalues = {np.array2string(eigvals, precision=6)}")
    print(f"  n_neg={result['num_negative_eigs']}  n_pos={result['num_positive_eigs']}"
          f"  min={eigvals.min():.4e}  max={eigvals.max():.4e}  cond={cond}")
    print(f"  Hessian asym rel = {asym_rel:.3e}")
    print(f"  most-negative eigvec = {np.array2string(most_neg_vec, precision=4)}")
    print(f"  VERDICT: {verdict}")
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("components (control layout):", component_names())
    print(f"sensors: {MEASUREMENT_BOUNDARIES}  amplitude_floor={AMPLITUDE_FLOOR}")

    # Sanity: confirm admissibility class loads (not used to gate forward solves).
    _adm = MeshAdmissibility(
        app4.configure_measurement_boundaries(
            app4.build_mesh_data(), MEASUREMENT_BOUNDARIES
        )
    )
    _ = MESH_DOMAIN_BOUND_CASE

    results = []
    for case in CASES:
        results.append(diagnose_one(*case))

    (OUT_DIR / "hessian_diagnostics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    # Compact CSV table.
    lines = [
        "label,tfinal,sigma2,objective,grad_norm,proj_grad_norm,"
        "n_neg,n_pos,min_eig,max_eig,cond,verdict"
    ]
    for r in results:
        lines.append(
            f"{r['label']},{r['tfinal']},{r['sigma2']},"
            f"{r['objective']:.6e},{r['gradient_norm']:.6e},"
            f"{r['projected_gradient_norm']:.6e},"
            f"{r['num_negative_eigs']},{r['num_positive_eigs']},"
            f"{r['min_eigenvalue']:.6e},{r['max_eigenvalue']:.6e},"
            f"{r['condition_number_abs']},\"{r['verdict']}\""
        )
    (OUT_DIR / "hessian_diagnostics_table.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(f"\nWrote:\n  {OUT_DIR / 'hessian_diagnostics.json'}")
    print(f"  {OUT_DIR / 'hessian_diagnostics_table.csv'}")


if __name__ == "__main__":
    main()
