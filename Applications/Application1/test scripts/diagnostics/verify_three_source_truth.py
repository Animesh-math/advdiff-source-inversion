"""Pre-flight verification of a candidate Application1 M-source truth (read-only).

Run BEFORE launching an expensive recovery so the truth is known to be
physically admissible and observationally non-degenerate.  Reports, for the
control given on the command line (default: the three-source truth chosen for
the first M=3 feasibility test):

  * fluid admissibility of every center on the DC mesh;
  * nearest obstacle/boundary clearance of every center;
  * pairwise center distances, in metres and in Gaussian radii;
  * pairwise UNIT-trace observation coherence
    mu_ij = |<t_i, t_j>| / (||t_i|| ||t_j||) in the exact discrete observation
    norm the objective uses (one forward solve per source, so M solves total);
  * per-source explained fraction of the zero-source objective, as a cheap
    observability check that no source is effectively invisible.

Nothing here is imported by production code and nothing is written except the
optional JSON report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

import sys as _sys_dc
from pathlib import Path as _Path_dc
_sys_dc.path.insert(0, str(_Path_dc(__file__).resolve().parents[1]))  # _diag_common
import _diag_common as D
import run_optimization as ro
import sensitivity_initialization as sinit


# The three-source truth for the first M=3 feasibility test.  Chosen from the
# DC geometry alone (see Applications/Application1/README.md): three different vertical bands
# of the domain, three different x corridors, comparable amplitudes inside
# [0.8, 1.5], and no center in the documented near-unobservable lower-left.
DEFAULT_TRUTH = [
    275.0, 120.0, 1.30,
    200.0, 425.0, 1.20,
    110.0, 660.0, 1.05,
]
DEFAULT_BOUNDARIES = ("left", "right", "bottom")
DEFAULT_SIGMA2 = 1600.0
DEFAULT_TFINAL = 150.0
AMPLITUDE_BOUNDS = (0.1, 2.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", nargs="+", type=float, default=DEFAULT_TRUTH)
    parser.add_argument(
        "--measurement-boundaries", nargs="+", default=list(DEFAULT_BOUNDARIES)
    )
    parser.add_argument("--sigma2", type=float, default=DEFAULT_SIGMA2)
    parser.add_argument("--tfinal", type=float, default=DEFAULT_TFINAL)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    control = np.asarray(args.control, dtype=np.float64)
    if control.size == 0 or control.size % 3 != 0:
        print(f"FAIL: control length {control.size} is not a positive multiple of 3")
        return 1
    num_sources = control.size // 3
    sources = control.reshape(num_sources, 3)

    print("=== Candidate truth ===")
    print(f"num sources M     : {num_sources}")
    print(f"control dimension : {control.size}")
    print(f"control           : {np.array2string(control, precision=6)}")
    print(f"measurement set   : {list(args.measurement_boundaries)}")
    print(f"sigma2 (init width): {args.sigma2:g}  (radius {np.sqrt(args.sigma2):.2f} m)")
    print(f"tfinal            : {args.tfinal:g} s")

    t0 = time.perf_counter()
    mesh_data, pde, admiss = D.build_setup(
        tuple(args.measurement_boundaries), args.sigma2, tfinal=args.tfinal
    )
    ok = True

    print("\n=== Fluid admissibility and wall clearance ===")
    records = []
    for index, (x, y, amplitude) in enumerate(sources, start=1):
        inside = bool(admiss.is_point_in_domain(float(x), float(y)))
        clearance = float(admiss.distance_to_boundary(float(x), float(y)))
        amp_ok = bool(AMPLITUDE_BOUNDS[0] < amplitude < AMPLITUDE_BOUNDS[1])
        ok = ok and inside and amp_ok
        print(
            f"  source {index}: ({x:8.2f}, {y:8.2f})  A={amplitude:.4f}  "
            f"fluid-admissible={inside}  wall clearance={clearance:8.3f} m  "
            f"amplitude strictly inside {AMPLITUDE_BOUNDS}={amp_ok}"
        )
        records.append(
            {
                "source": index,
                "center": [float(x), float(y)],
                "amplitude": float(amplitude),
                "fluid_admissible": inside,
                "wall_clearance_m": clearance,
                "amplitude_strictly_inside_bounds": amp_ok,
            }
        )

    print("\n=== Pairwise center distances ===")
    radius = float(np.sqrt(args.sigma2))
    distances = []
    for i, j in combinations(range(num_sources), 2):
        d = float(np.hypot(*(sources[i, :2] - sources[j, :2])))
        visible = bool(admiss.segment_in_domain(sources[i, :2], sources[j, :2]))
        print(
            f"  s{i + 1}-s{j + 1}: {d:8.2f} m  ({d / radius:6.2f} init radii, "
            f"{d / np.sqrt(400.0):6.2f} target radii)  "
            f"mutually visible through fluid={visible}"
        )
        distances.append(
            {
                "pair": [i + 1, j + 1],
                "distance_m": d,
                "distance_init_radii": d / radius,
                "mutually_visible": visible,
            }
        )

    print("\n=== Unit-trace observation coherence (exact discrete norm) ===")
    print(f"  ({num_sources} forward solves)")
    ctx = sinit._ObsTraceContext(
        mesh_data, ro.ETA1, pde["dt"] * pde["jprint"], ro.OBJECTIVE_TIME_RULE
    )
    traces, norms = [], []
    for index in range(num_sources):
        src = sinit._gaussian_sum_source(
            [(sources[index, 0], sources[index, 1], 1.0)], args.sigma2
        )
        state = D.gt.run_forward(mesh_data, pde, sinit._DUMMY_Z, source_func=src)
        trace = ctx.extract(state)
        traces.append(trace)
        norms.append(ctx.norm2(trace))
    coherences = []
    max_mu = 0.0
    for i, j in combinations(range(num_sources), 2):
        mu = sinit._coherence_from_norms(
            ctx.inner(traces[i], traces[j]), norms[i], norms[j]
        )
        max_mu = max(max_mu, mu)
        print(f"  mu(s{i + 1}, s{j + 1}) = {mu:.6f}")
        coherences.append({"pair": [i + 1, j + 1], "coherence": float(mu)})
    print(f"  max pairwise coherence = {max_mu:.6f}")

    print("\n=== Per-source observability (share of the zero-source objective) ===")
    full_trace = np.zeros_like(traces[0])
    for index in range(num_sources):
        full_trace = full_trace + float(sources[index, 2]) * traces[index]
    j_zero = ctx.tracking_objective(np.zeros_like(full_trace), full_trace)
    for index in range(num_sources):
        without = full_trace - float(sources[index, 2]) * traces[index]
        j_without = ctx.tracking_objective(without, full_trace)
        share = (j_without / j_zero) if j_zero > 0.0 else float("nan")
        print(
            f"  source {index + 1}: removing it leaves residual J = {j_without:.6e} "
            f"({100.0 * share:7.3f}% of J_zero = {j_zero:.6e})"
        )

    print(f"\nwall time: {time.perf_counter() - t0:.1f} s")
    print(f"VERDICT: {'PASS' if ok else 'FAIL'}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {
                    "control": control.tolist(),
                    "num_sources": int(num_sources),
                    "measurement_boundaries": list(args.measurement_boundaries),
                    "sigma2": float(args.sigma2),
                    "tfinal": float(args.tfinal),
                    "sources": records,
                    "pairwise_distances": distances,
                    "pairwise_coherence": coherences,
                    "max_pairwise_coherence": float(max_mu),
                    "zero_source_objective": float(j_zero),
                    "verdict": "PASS" if ok else "FAIL",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"report: {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
