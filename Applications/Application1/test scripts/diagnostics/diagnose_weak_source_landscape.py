"""Is a missed weak source a stopped optimizer, or an unidentifiable source?

READ-ONLY, truth-aware. Given a recovered control and the true control, scans the
weak source's center along the straight path from where the optimizer left it to
where it belongs, with the other center frozen and BOTH amplitudes eliminated by
the exact bounded variable-projection fit (the forward map is linear in
amplitude, so J(a) is exactly quadratic and is fitted from 6 evaluations of the
driver's own objective using 2 forward solves per point).

Reading the result:

* J decreases monotonically toward the truth  -> the optimizer stopped early:
  FINAL LOCAL OPTIMIZATION is the failure point.
* J rises, or is flat within the amplitude floor -> the truth is not downhill
  from where the optimizer sits: an OBSERVABILITY / landscape limit, not a
  solver bug.

Usage:
  cd /path/to/Advection-Diffusion-PDECO
  python3 \
    "Applications/Application1/test scripts/diagnostics/diagnose_weak_source_landscape.py" \
    --case <case-name> [--source-index 1] [--sigma2 400] [--points 9]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import sys as _sys_dc
from pathlib import Path as _Path_dc
_sys_dc.path.insert(0, str(_Path_dc(__file__).resolve().parents[1]))  # _diag_common
import _diag_common as dc
from mesh_admissibility import MeshAdmissibility

ro = dc.ro
gt = dc.gt

AMP_LO, AMP_HI = dc.AMPLITUDE_BOUNDS


def vp_objective(mesh_data, target, centers, amp_lo, amp_hi):
    """Exact bounded VP fit over all amplitudes for fixed centers."""
    unit_states = []
    for index in range(len(centers)):
        z = np.zeros(3 * len(centers))
        for k, c in enumerate(centers):
            z[3 * k] = c[0]
            z[3 * k + 1] = c[1]
            z[3 * k + 2] = 1.0 if k == index else 0.0
        unit_states.append(gt.run_forward(mesh_data, ro.PDE, z))

    def control(a):
        z = np.zeros(3 * len(centers))
        for k, c in enumerate(centers):
            z[3 * k] = c[0]
            z[3 * k + 1] = c[1]
            z[3 * k + 2] = a[k]
        return z

    def objective(a):
        state = sum(a[k] * unit_states[k] for k in range(len(centers)))
        return float(gt.compute_objective(
            state, target, mesh_data, eta1=ro.ETA1, lam=ro.LAMBDA_REG,
            z=control(a), dt_out=ro.PDE["dt"] * ro.PDE["jprint"],
            time_rule=ro.OBJECTIVE_TIME_RULE))

    pts = [(0., 0.), (1., 0.), (0., 1.), (2., 0.), (0., 2.), (1., 1.)]
    design = [[0.5 * a * a, a * b, 0.5 * b * b, a, b, 1.0] for a, b in pts]
    values = [objective(np.array(p)) for p in pts]
    coef, *_ = np.linalg.lstsq(np.array(design), np.array(values), rcond=None)
    grid = np.linspace(amp_lo, amp_hi, 401)
    A, B = np.meshgrid(grid, grid, indexing="ij")
    J = (0.5 * (coef[0] * A ** 2 + 2 * coef[1] * A * B + coef[2] * B ** 2)
         + coef[3] * A + coef[4] * B + coef[5])
    k = np.unravel_index(np.argmin(J), J.shape)
    return float(J[k]), float(A[k]), float(B[k])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--source-index", type=int, default=1,
                        help="0-based index of the source to scan")
    parser.add_argument("--sigma2", type=float, default=400.0)
    parser.add_argument("--tfinal", type=float, default=150.0)
    parser.add_argument("--points", type=int, default=9)
    parser.add_argument("--amplitude-floor", type=float, default=0.1)
    args = parser.parse_args()

    base = dc.APP_DIR / "output" / args.case / "full"
    summary_path = base / "optimization_summary.json"
    if not summary_path.exists():
        starts = sorted(base.glob("start_*/optimization_summary.json"))
        summary_path = starts[0]
    summary = json.loads(summary_path.read_text())
    true_control = np.asarray(summary["true_control"], dtype=np.float64)
    final_control = np.asarray(summary["final_control"], dtype=np.float64)

    mesh_data = ro.configure_measurement_boundaries(
        ro.build_mesh_data(), list(summary["measurement_boundaries"])
    )
    admiss = MeshAdmissibility(mesh_data)
    velocity, vsummary = ro.load_velocity_field_file(
        str(ro.DEFAULT_VELOCITY_FIELD_FILE), mesh_data
    )
    ro.PDE["tfinal"] = float(args.tfinal)
    ro.PDE["sigma2"] = float(args.sigma2)
    ro.PDE["velocity_nodes"] = velocity
    ro.PDE["velocity_field"] = vsummary

    target = gt.run_forward(mesh_data, ro.PDE, true_control)
    true_sources = true_control.reshape(-1, 3)
    final_sources = final_control.reshape(-1, 3)
    index = int(args.source_index)
    other = [k for k in range(final_sources.shape[0]) if k != index]

    start = final_sources[index, :2].copy()
    goal = true_sources[index, :2].copy()
    print(f"case: {args.case}   sigma2={args.sigma2:g}  tfinal={args.tfinal:g}")
    print(f"scanning source {index + 1}: ({start[0]:.2f},{start[1]:.2f}) -> "
          f"({goal[0]:.2f},{goal[1]:.2f})   "
          f"{float(np.hypot(*(goal - start))):.2f} m")
    print(f"other source(s) frozen at "
          + ", ".join(f"({final_sources[k, 0]:.2f},{final_sources[k, 1]:.2f})"
                      for k in other))
    print(f"amplitudes by exact bounded VP fit on "
          f"[{args.amplitude_floor}, {AMP_HI}]\n")

    print(f"{'s':>6} {'x':>9} {'y':>9} {'in':>6} {'J_VP':>14} "
          f"{'a_other':>9} {'a_scan':>9}")
    values = []
    for s in np.linspace(0.0, 1.0, int(args.points)):
        point = start + s * (goal - start)
        inside = admiss.is_point_in_domain(point[0], point[1])
        if not inside:
            print(f"{s:>6.3f} {point[0]:>9.3f} {point[1]:>9.3f} {'OUT':>6}")
            values.append(None)
            continue
        centers = [final_sources[k, :2] for k in other] + [point]
        J, a_other, a_scan = vp_objective(
            mesh_data, target, centers, args.amplitude_floor, AMP_HI
        )
        values.append(J)
        print(f"{s:>6.3f} {point[0]:>9.3f} {point[1]:>9.3f} {'yes':>6} "
              f"{J:>14.6e} {a_other:>9.4f} {a_scan:>9.4f}")

    finite = [v for v in values if v is not None]
    blocked = any(v is None for v in values[1:-1])
    visible = bool(admiss.segment_in_domain(start, goal, spacing=0.5))
    print()
    if len(finite) >= 2:
        print(f"J at the optimizer's point : {finite[0]:.6e}")
        print(f"J at the true position     : {finite[-1]:.6e}")
        print(f"straight path admissible   : {not blocked} "
              f"(visibility check: {visible})")
        lower_at_truth = finite[-1] < finite[0]
        if blocked or not visible:
            # Monotonicity over the admissible SUBSET of a blocked path proves
            # nothing: the optimizer cannot travel through a building, so a lower
            # value on the far side is not reachable by any local step.
            print("VERDICT: the straight path is BLOCKED by an obstacle, so the "
                  "scan says nothing about\n         reachability by local steps."
                  + (
                      "\n         The truth is nonetheless much lower, so the two "
                      "positions are in DIFFERENT\n         streets. Failure point "
                      "= SOURCE SELECTION placing this source in a street\n"
                      "         separated from the true one; the final "
                      "optimization then converged against\n         the "
                      "intervening wall, which is a constrained optimum, not an "
                      "early stop."
                      if lower_at_truth else
                      "\n         The truth is not lower either, so this source is "
                      "weakly identifiable here."
                  ))
        else:
            monotone = all(
                b <= a * (1.0 + 1.0e-9) for a, b in zip(finite, finite[1:])
            )
            if monotone and lower_at_truth:
                print("VERDICT: J decreases monotonically along an admissible path "
                      "to the truth.\n         Failure point = FINAL LOCAL "
                      "OPTIMIZATION (the optimizer stopped early).")
            elif lower_at_truth:
                print("VERDICT: the truth is lower and the path is admissible, but "
                      "J is not monotone --\n         an intervening ridge "
                      "separates the two. Failure point = landscape/basin\n"
                      "         structure, not the line search.")
            else:
                print("VERDICT: the true position is NOT lower than where the "
                      "optimizer stopped.\n         Failure point = OBSERVABILITY: "
                      "this source is not identifiable from the\n         selected "
                      "measurement boundaries at this width, so no initialization "
                      "can fix it.")


if __name__ == "__main__":
    main()
