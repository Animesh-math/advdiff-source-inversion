"""Resolution / stability evidence behind the Application2 transport settings.

    PY "test scripts/select_transport_settings.py" [--out DIR] [--candidates K ...] [--tfinal T]

For each candidate tracer diffusivity k and each mesh level (with its own
velocity field) this records

  * cell Peclet numbers |v| h / (2 k) (area-weighted statistics);
  * the spectral radius of M_lumped^{-1} K and the resulting RK4 step limit;
  * a forward solve with ALL THREE bed releases active (covers every case by
    linearity), from which it reports the undershoot min(u) / max(u) on each
    mesh and the coarse-vs-fine difference of the full concentration field at
    several times (fine field P1-interpolated to the coarse nodes, lumped-mass
    L2 norm).

Only full-field forward-model metrics are used: no sensor traces, no inverse
solve, no recovery outcome.  The adopted values live in ``transport_config.py``;
this script documents why and is rerun by the setup pipeline for the record.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import transport_config as tc  # noqa: E402
import transport_model as tm  # noqa: E402

ROUND_STEPS = (0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.0125, 0.01, 0.008, 0.006, 0.005)
SNAPSHOT_TIMES = (60.0, 120.0, 180.0, 240.0, 300.0)


def choose_steps(rho_coarse: float, rho_fine: float):
    limit = tc.RK4_STABILITY_TARGET
    dt_c = next(s for s in ROUND_STEPS if s * rho_coarse <= limit)
    m = int(math.ceil(dt_c * rho_fine / limit - 1.0e-12))
    return dt_c, dt_c / m, m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=tm.DATASET_DIR / "common" / "settings_selection")
    ap.add_argument("--candidates", type=float, nargs="+", default=[tm.nsc.NU, 5.0e-3, 1.0e-2])
    ap.add_argument("--tfinal", type=float, default=tc.TFINAL)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    z_all = []
    for bay in (1, 2, 3):
        r = tc.RELEASES[bay]
        z_all += [r["center"][0], r["center"][1], r["amplitude"]]
    source = tm.gaussian_source(z_all, tc.SIGMA2)
    md = {lv: tm.build_transport_mesh_data(lv) for lv in ("coarse", "fine")}
    P_fine_to_coarse, _ = tm.interpolation_matrix(md["fine"], md["coarse"]["coord"])
    times = [t for t in SNAPSHOT_TIMES if t <= args.tfinal + 1e-9]

    rows = []
    for k in args.candidates:
        t0 = time.perf_counter()
        ops = {lv: tm.SparseTransport(md[lv], tm.build_pde(lv, md[lv], k=k, dt=1.0, tfinal=args.tfinal))
               for lv in ("coarse", "fine")}
        rho = {lv: ops[lv].spectral_radius() for lv in ops}
        dt_c, dt_f, m = choose_steps(rho["coarse"], rho["fine"])
        dts = {"coarse": dt_c, "fine": dt_f}
        fields, extrema = {}, {}
        for lv in ("coarse", "fine"):
            ops[lv].pde["dt"] = dts[lv]
            snap_steps = [int(round(t / dts[lv])) for t in times]
            every = max(1, int(round(1.0 / dts[lv])))          # extrema sampled once per second
            samples, snaps = ops[lv].solve(ops[lv].load_vector(source), sample_every=every,
                                           snapshot_steps=snap_steps)
            fields[lv] = [snaps[s] for s in snap_steps]
            extrema[lv] = {"min": float(samples.min()), "max": float(samples.max()),
                           "undershoot_min_over_max": float(samples.min() / samples.max())}
        mass = ops["coarse"].mass
        diffs = []
        for t, uc, uf in zip(times, fields["coarse"], fields["fine"]):
            ufc = P_fine_to_coarse @ uf
            diffs.append({"t": t,
                          "rel_l2": float(np.sqrt(np.sum(mass * (uc - ufc) ** 2) / np.sum(mass * ufc ** 2))),
                          "max_abs_over_max": float(np.max(np.abs(uc - ufc)) / np.max(np.abs(ufc)))})
        row = {
            "k": float(k),
            "peclet": {lv: tm.cell_peclet(md[lv], k) for lv in md},
            "spectral_radius": rho,
            "dt": dts, "fine_substeps": m,
            "dt_rho": {lv: dts[lv] * rho[lv] for lv in rho},
            "num_steps": {lv: tm.num_steps(dts[lv], args.tfinal) for lv in dts},
            "extrema_all_three_releases": extrema,
            "coarse_vs_fine_field": diffs,
            "seconds": time.perf_counter() - t0,
        }
        rows.append(row)
        print(f"k={k:.4g}: Pe_max coarse/fine {row['peclet']['coarse']['max']:.2f}/{row['peclet']['fine']['max']:.2f}; "
              f"rho {rho['coarse']:.2f}/{rho['fine']:.2f}; dt {dt_c:g}/{dt_f:g} (x{m}); "
              f"undershoot coarse/fine {extrema['coarse']['undershoot_min_over_max']:.2e}/"
              f"{extrema['fine']['undershoot_min_over_max']:.2e}; coarse-fine rel L2 "
              + ", ".join(f"{d['rel_l2']:.3f}@{d['t']:g}s" for d in diffs) + f"  [{row['seconds']:.0f} s]",
              flush=True)

    record = {
        "purpose": "resolution/stability evidence for the transport settings; full-field forward metrics only",
        "test_source": "all three bed releases active (transport_config.RELEASES), sigma2 = %g" % tc.SIGMA2,
        "tfinal": args.tfinal, "rk4_stability_target_dt_rho": tc.RK4_STABILITY_TARGET,
        "adopted": {"k": tc.DIFFUSIVITY_K, "dt": dict(tc.DT), "sigma2": tc.SIGMA2, "tfinal": tc.TFINAL},
        "candidates": rows,
    }
    tm.write_json(args.out / "settings_selection.json", record)
    lines = ["| k [m²/s] | level | Pe max | Pe area-median | area Pe>1 | ρ(M⁻¹K) [1/s] | dt [s] | dt·ρ | steps | min u / max u |"
             " coarse–fine rel. L2 at t = " + ", ".join(f"{t:g}" for t in times) + " s |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for lv in ("coarse", "fine"):
            pe = r["peclet"][lv]
            lines.append(
                f"| {r['k']:.4g} | {lv} | {pe['max']:.2f} | {pe['area_median']:.2f} | {100 * pe['area_fraction_above_1']:.1f} % "
                f"| {r['spectral_radius'][lv]:.1f} | {r['dt'][lv]:g} | {r['dt_rho'][lv]:.2f} | {r['num_steps'][lv]} "
                f"| {r['extrema_all_three_releases'][lv]['undershoot_min_over_max']:.1e} | "
                + (", ".join(f"{d['rel_l2']:.3f}" for d in r["coarse_vs_fine_field"]) if lv == "coarse" else "")
                + " |")
    (args.out / "settings_selection.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}/settings_selection.json and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
