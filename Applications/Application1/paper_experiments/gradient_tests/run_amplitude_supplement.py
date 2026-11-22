"""Supplementary amplitude-block Taylor test (same model/evaluator as the primary test).

Why: at the primary test points (the saved L-BFGS-B starting controls) the
amplitudes come from the OSQP joint amplitude solve, so dJ/dA ~ 1e-9 and the
primary g.q is carried by the center components only.  To exercise the adjoint
AMPLITUDE gradient with a non-negligible directional derivative, this test uses
a second saved nonstationary control from the SAME source run: the iteration-1
iterate in full/optimization_history.csv (physical coordinates), where the
centers have moved and the amplitudes are no longer optimal.

Direction: amplitude-only, q = D r, with r ~ N(0, I) on the M amplitude entries
(numpy default_rng(seed), normalized), centers zero, D = AMPLITUDE_SCALE.  Same
eps sequence, admissibility rule and |g.q| acceptance rule as the primary test.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import run_paper_gradient_tests as P


def run_case(case_id: str, out_dir: Path) -> dict:
    import run_optimization as ro
    from mesh_admissibility import MeshAdmissibility
    from multi_source_utils import AMPLITUDE_BOUNDS, reshape_control

    t0 = time.time()
    rec = P.recover_settings(case_id)
    a = rec["driver_args"]
    run = Path(rec["run_dir"])
    with open(run / "full" / "optimization_history.csv", encoding="utf-8") as fh:
        rows_h = list(csv.DictReader(fh))
    h1 = rows_h[1]
    M = int(rec["num_sources"])
    cols = [f"{c}{i}" for i in range(1, M + 1) for c in ("x", "y", "A")]
    z = np.array([float(h1[c]) for c in cols])
    rec_obj, rec_gn = float(h1["objective"]), float(h1["gradient_norm"])

    ro.PDE["tfinal"] = float(a["tfinal"])
    mesh_data = ro.configure_measurement_boundaries(ro.build_mesh_data(), a["measurement_boundaries"])
    mesh_data = ro.configure_observation(mesh_data, argparse.Namespace(
        observation_mode=a["observation_mode"], sensor_node_ids=a["sensor_node_ids"],
        sensor_coordinates=a["sensor_coordinates"], sensor_weights=a["sensor_weights"],
        sensor_equal_weight=a["sensor_equal_weight"], sensor_snap_tolerance=a["sensor_snap_tolerance"]))
    vn, vs = ro.load_velocity_field_file(a["velocity_field_file"], mesh_data)
    ro.PDE["velocity_nodes"], ro.PDE["velocity_field"] = vn, vs
    adm = MeshAdmissibility(mesh_data)
    assert list(mesh_data["observation"].metadata()["sensor_node_ids"]) == list(
        rec["summary_observation"]["sensor_node_ids"])

    true_state = ro.gt.run_forward(mesh_data, ro.PDE, np.asarray(rec["true_control"]))
    feas = ro.new_feasibility_stats(a["feasibility_mode"])
    _c, _h, objective, gradient, _cb = ro.build_evaluator(
        mesh_data, true_state, a["mode"], a["bound_case"], {}, z.copy(), adm,
        feasibility_mode=a["feasibility_mode"], feasibility_stats=feas)
    j0 = float(objective(z.copy()))
    g = np.asarray(gradient(z.copy()), dtype=np.float64)
    P.log(f"[{case_id}/amp] z(iter1) = {np.array2string(z, precision=8)}")
    P.log(f"[{case_id}/amp] J = {j0:.16e} (history iter1 objective {rec_obj:.16e})")
    P.log(f"[{case_id}/amp] g = {np.array2string(g, precision=8)}")
    amp_mask = np.tile([False, False, True], M)
    g_amp = g[amp_mask]

    amp_lo = float(rec["amplitude_floor"]); amp_hi = float(AMPLITUDE_BOUNDS[1])

    def admissible(c):
        amps = reshape_control(c)[:, 2]
        return adm.validate_sources_in_domain(c) and bool(np.all(amps >= amp_lo) and np.all(amps <= amp_hi))

    rejected, q, seed_used = [], None, None
    thr = P.MIN_DIRECTIONAL_FRACTION * P.AMPLITUDE_SCALE * float(np.linalg.norm(g_amp))
    for seed in range(P.SEED0, P.SEED0 + P.MAX_DRAWS):
        r = np.random.default_rng(seed).normal(size=M)
        r /= np.linalg.norm(r)
        qq = np.zeros_like(z); qq[amp_mask] = P.AMPLITUDE_SCALE * r
        bad = [float(e) for e in P.EPS if not admissible(z + e * qq)]
        gq = float(g @ qq)
        if bad:
            rejected.append({"seed": seed, "reason": "inadmissible", "eps": bad}); continue
        if abs(gq) < thr:
            rejected.append({"seed": seed, "reason": "small |g.q|", "g_dot_q": gq, "threshold": thr}); continue
        q, r_used, seed_used = qq, r, seed
        break
    gq = float(g @ q)
    P.log(f"[{case_id}/amp] seed={seed_used} rejected={len(rejected)} g.q={gq:.6e} ||g_A||={np.linalg.norm(g_amp):.6e}")

    rows = []
    for e in P.EPS:
        je = float(objective((z + e * q).copy()))
        d = je - j0
        rows.append({"eps": float(e), "J_eps": je, "J_eps_minus_J": d, "E0": abs(d),
                     "E1": abs(d - e * gq), "E0_over_J": abs(d) / j0, "E1_over_J": abs(d - e * gq) / j0})
        P.log(f"[{case_id}/amp] eps={e:9.3e} E0={abs(d):.6e} E1={abs(d - e * gq):.6e}")
    for i, row in enumerate(rows):
        if i == 0:
            row["E0_rate"] = row["E1_rate"] = None; continue
        p = rows[i - 1]; ln = np.log(p["eps"] / row["eps"])
        row["E0_rate"] = float(np.log(p["E0"] / row["E0"]) / ln) if row["E0"] > 0 and p["E0"] > 0 else None
        row["E1_rate"] = float(np.log(p["E1"] / row["E1"]) / ln) if row["E1"] > 0 and p["E1"] > 0 else None
    an = P.analyse(rows, j0, gq)
    P.log(f"[{case_id}/amp] analysis: {json.dumps({k: v for k, v in an.items() if not k.endswith('_all')})}")

    res = {"case_id": case_id, "short_label": P.CASES[case_id]["short_label"],
           "purpose": "amplitude-block gradient check (non-zero dJ/dA)",
           "source_run": rec["run_dir"],
           "test_point": "optimization_history.csv iteration 1 (saved L-BFGS-B iterate, physical coords)",
           "history_iter1_objective": rec_obj, "history_iter1_gradient_norm_logged": rec_gn,
           "history_iter1_gradient_norm_note": "logged norm is of the optimizer's active (possibly gn-diag scaled) gradient",
           "z": z.tolist(), "J_z": j0, "J_z_rel_diff_vs_history": abs(j0 - rec_obj) / abs(rec_obj),
           "gradient": g.tolist(), "gradient_amplitude_block": g_amp.tolist(),
           "direction": {"q": q.tolist(), "r_unit_amplitudes": r_used.tolist(), "seed": seed_used,
                         "amplitude_scale": P.AMPLITUDE_SCALE, "centers": "zero",
                         "rejected_draws": rejected},
           "g_dot_q": gq, "eps": P.EPS.tolist(), "rows": rows, "analysis": an,
           "feasibility_stats": feas, "elapsed_s": time.time() - t0}
    d = out_dir / "supplement_amplitude" / case_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    with open(d / "taylor_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return res


def plot(out_dir: Path, cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "mathtext.fontset": "cm"})
    res = [json.loads((out_dir / "supplement_amplitude" / c / "result.json").read_text()) for c in cases]
    fig, axes = plt.subplots(1, len(res), figsize=(5.0 * len(res), 4.6))
    for k, (ax, r) in enumerate(zip(np.atleast_1d(axes), res)):
        eps = np.array(r["eps"])
        n0 = np.array([x["E0_over_J"] for x in r["rows"]]); n1 = np.array([x["E1_over_J"] for x in r["rows"]])
        hs = [ax.loglog(eps, n0, "o-", color="#1f5fa8", ms=5, label=r"$E_0(\varepsilon)/J_h(z)$")[0],
              ax.loglog(eps, n1, "s-", color="#c4452b", ms=5, label=r"$E_1(\varepsilon)/J_h(z)$")[0],
              ax.loglog(eps[-9:], 0.15 * n0[-1] * eps[-9:] / eps[-1], "--", color="0.35", label="slope 1")[0],
              ax.loglog(eps[3:12], 0.08 * n1[3] * (eps[3:12] / eps[3]) ** 2, ":", color="0.1", lw=1.6, label="slope 2")[0]]
        ax.set_xlabel(r"$\varepsilon$"); ax.set_title(r["short_label"] + "\n(amplitude-only $q$, iterate 1)", fontsize=11)
        if k == 0:
            ax.set_ylabel("normalized Taylor remainder")
        ax.grid(True, ls=":", lw=0.6, color="0.75")
    fig.legend(hs, [h.get_label() for h in hs], loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / "figures" / f"supplement_amplitude_taylor_3panel.{ext}", dpi=300, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cases", nargs="+", default=list(P.CASES))
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir).resolve()
    if not args.plot_only:
        for c in args.cases:
            P.log(f"===== supplement {c} start =====")
            r = run_case(c, out)
            P.log(f"===== supplement {c} done in {r['elapsed_s']:.1f}s =====")
    plot(out, args.cases)
    P.log("SUPPLEMENT DONE")


if __name__ == "__main__":
    main()
