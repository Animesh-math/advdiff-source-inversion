"""Readiness checks for the Application2 source-identification datasets.

    PY validate_readiness.py [--dataset-dir DIR] [--cases 12 13 23] [--skip-gn-check]

No sensitivity initialization, centre search or optimization is run.  Checks:

  V1  fine-mesh sampling at the fixed sensor coordinates (P1 interpolation)
  V2  saved datasets reload, are reproducible from the recorded seed, and are
      unchanged by objective evaluations
  V3  coarse forward predictions and the weighted objective are consistent
      (W = I, a non-uniform diagonal W, a full symmetric W); the sparse RK4 used
      for data generation reproduces the shared dense solver
  V4  directional Taylor tests of the coarse discrete-adjoint gradient against
      the SAVED clean and noisy data at feasible non-truth controls
  V5  source admissibility (beds, partitions, bathroom wall, exterior) and
      Gaussian resolution on both meshes; analytic source derivatives
  V6  the eventual initialization / optimizer entry points are wired to the
      coarse mesh, its velocity and the dataset (run_inversion.py --stage wiring)
  D   discretization-mismatch diagnostic: coarse model at each TRUE control
      against the clean fine-mesh data.  This is the only place the truth is
      read; its output stays under validation/ and is not an inversion input.

Writes validation/validation_results.{json,csv}, validation/discretization_mismatch.{json,csv},
validation/taylor_tests.json, validation/wiring/<case>_<data>/wiring_check.json.
Exit status 0 only if every check passed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import transport_config as tc
import transport_model as tm
import run_inversion as ri

geo, mdc = tm.geo, tm.mdc

# Feasible evaluation controls defined from geometry only (side passages, aisle,
# bathroom); V4 verifies each is admissible and far from every true release.
TAYLOR_CONTROLS = {
    "12": [0.5, 3.6, 0.8, 6.6, 1.2, 1.5],
    "13": [3.6, 5.0, 1.4, 8.0, 2.6, 0.5],
    "23": [10.0, 1.0, 0.9, 4.5, 2.9, 1.2],
}
TAYLOR_STEPS = (1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5)
ADMISSIBILITY_POINTS = [
    ("bed_1 interior", (1.5, 4.5), False), ("bed_2 interior", (4.5, 4.5), False),
    ("bed_3 interior", (7.5, 4.5), False), ("partition_1 interior", (3.0, 4.5), False),
    ("partition_2 interior", (6.0, 4.5), False), ("bathroom wall (lower piece)", (9.0, 0.3), False),
    ("bathroom wall (upper piece)", (9.0, 1.7), False), ("exterior above the bathroom annex", (10.0, 4.0), False),
    ("exterior left of the room", (-0.2, 3.0), False), ("exterior above the top wall", (4.5, 6.2), False),
    ("aisle", (4.5, 1.5), True), ("bay 1 left passage", (0.5, 4.5), True), ("bay 2 right passage", (5.5, 4.5), True),
    ("bay 3 head gap above the bed", (7.5, 5.8), True), ("bathroom doorway", (9.0, 1.0), True),
    ("bathroom interior", (10.0, 1.0), True),
]


class Results:
    def __init__(self, log):
        self.rows, self.log = [], log

    def add(self, check_id: str, name: str, passed: bool, detail: str = "", data=None):
        self.rows.append({"id": check_id, "name": name, "passed": bool(passed), "detail": str(detail),
                          "data": data})
        self.log(f"  [{'PASS' if passed else 'FAIL'}] {check_id} {name}: {detail}")


def objective_direct(traces, data, weights, eta1, dt) -> float:
    """J = (eta1 dt / 2) sum_{n>=1} r_n^T W r_n written out independently of the shared code."""
    r = traces[1:] - data[1:]
    wr = r * weights[None, :] if weights.ndim == 1 else r @ weights
    return 0.5 * eta1 * dt * float(np.sum(wr * r))


def check_sampling(out: Path, res: Results):
    sensors = json.loads((out / "common" / "sensors.json").read_text())
    xy = np.asarray(sensors["coordinates"])
    md = {lv: tm.build_transport_mesh_data(lv) for lv in ("coarse", "fine")}
    nodes = np.asarray(sensors["node_ids"])
    res.add("V1a", "sensor coordinates are the coarse-mesh node coordinates",
            np.array_equal(md["coarse"]["coord"][nodes], xy), f"{len(nodes)} sensors")
    res.add("V1b", "no sensor on a clean-air Dirichlet node or outside the fluid",
            np.intersect1d(nodes, md["coarse"]["clean_air_dirichlet_nodes"]).size == 0
            and bool(mdc.point_in_fluid_mesh(md["fine"], xy).all()) and bool(mdc.point_in_fluid_mesh(md["coarse"], xy).all()))
    P, info = tm.interpolation_matrix(md["fine"], xy)
    lin = P @ (md["fine"]["coord"][:, 0] + 2.0 * md["fine"]["coord"][:, 1])
    err = float(np.max(np.abs(lin - (xy[:, 0] + 2.0 * xy[:, 1]))))
    w = np.asarray([i["weights"] for i in info])
    res.add("V1c", "fine-mesh P1 interpolation reproduces a linear field at the sensor coordinates",
            err < 1e-12 and w.min() >= 0.0 and np.allclose(w.sum(axis=1), 1.0), f"max error {err:.1e}")
    recorded = [s["fine_mesh_interpolation"] for s in sensors["sensors"]]
    res.add("V1d", "interpolation cells/weights equal those recorded at generation",
            all(a["cell"] == b["cell"] and np.allclose(a["weights"], b["weights"], atol=1e-14)
                for a, b in zip(info, recorded)))
    worst = 0.0
    for case in tc.CASES:
        cdir = out / tc.case_dir_name(case)
        if not cdir.exists():
            continue
        snap = np.load(cdir / "truth" / "fine_state_snapshots.npz")
        clean = np.load(cdir / "data" / "observations_clean.npz")["observations"]
        for t, state in zip(snap["times"], snap["states"]):
            n = int(round(t / tc.OBSERVATION_INTERVAL))
            worst = max(worst, float(np.max(np.abs(P @ state - clean[n]))))
    res.add("V1e", "saved clean traces equal the interpolated saved fine-mesh fields at the snapshot times",
            worst < 1e-13, f"max abs difference {worst:.1e}")
    return md


def check_datasets(out: Path, cases, res: Results):
    for case in cases:
        cdir = out / tc.case_dir_name(case)
        cfg = json.loads((cdir / "config" / "inversion_config.json").read_text())
        rec = json.loads((cdir / "data" / "noise_record.json").read_text())
        text = (cdir / "config" / "inversion_config.json").read_text()
        res.add("V2a", f"case {case}: inversion configuration holds no truth field",
                cfg["contains_truth"] is False
                and not any(word in text for word in ("true_control", "occupied", "release", "truth/")))
        a = {k: np.load(cdir / "data" / f"observations_{k}.npz") for k in ("clean", "noisy")}
        b = {k: np.load(cdir / "data" / f"observations_{k}.npz") for k in ("clean", "noisy")}
        noise = np.load(cdir / "data" / "noise.npz")
        clean, noisy, eps = a["clean"]["observations"], a["noisy"]["observations"], noise["noise"]
        ok_hash = all(tm.array_sha256(a[k]["observations"]) == cfg["observations"][k]["observations_sha256"]
                      == tm.array_sha256(b[k]["observations"])
                      and mdc.sha256_file(cdir / "data" / f"observations_{k}.npz") == cfg["observations"][k]["file_sha256"]
                      for k in ("clean", "noisy"))
        res.add("V2b", f"case {case}: clean/noisy files reload identically and match the recorded hashes", ok_hash)
        res.add("V2c", f"case {case}: d_noisy = d_clean + noise exactly; n = 0 sample is zero and noise-free",
                np.array_equal(noisy, clean + eps) and not clean[0].any() and not eps[0].any()
                and not a["clean"]["in_objective"][0] and a["clean"]["in_objective"][1:].all())
        s = float(rec["nominal_noise_level"]) * float(np.sqrt(np.mean(clean[1:] ** 2)))
        redraw = np.random.default_rng(int(rec["seed"])).normal(0.0, s, size=clean[1:].shape)
        rel = float(np.linalg.norm(eps[1:]) / np.linalg.norm(clean[1:]))
        res.add("V2d", f"case {case}: noise reproducible from the recorded seed; one common std",
                np.array_equal(redraw, eps[1:]) and np.isclose(s, rec["s_noise_absolute"], rtol=1e-14, atol=0.0)
                and int(rec["seed"]) == int(noise["seed"]) == tc.NOISE_SEEDS[case],
                f"seed {rec['seed']}, s_noise {s:.6e}, realized relative noise norm {rel:.6f}")
        res.add("V2e", f"case {case}: realized relative noise norm consistent with the nominal level",
                abs(rel / rec["nominal_noise_level"] - 1.0) < 0.02 and np.isclose(rel, rec["realized_relative_noise_norm"]),
                f"{rel:.6f} vs nominal {rec['nominal_noise_level']}")
        res.add("V2f", f"case {case}: negative noisy samples are kept (no clipping)",
                int(np.count_nonzero(noisy[1:] < 0.0)) == rec["num_negative_noisy_samples"],
                f"{rec['num_negative_noisy_samples']} negative samples of {noisy[1:].size}")


def check_objective(out: Path, case: str, md_unused, res: Results, log):
    """V3 on one case with the noisy data, at a feasible non-truth control."""
    ds = ri.load_dataset(out, case, "noisy")
    problem = ri.build_problem(ds, log=log)
    app, md, pde = problem["app"], problem["mesh_data"], problem["pde"]
    z = np.asarray(TAYLOR_CONTROLS[case])
    file_hash_before = mdc.sha256_file(ds["path"])
    target_hash_before = tm.array_sha256(problem["target"])
    _h, objective, _g, _c, _s = ri.build_evaluator(problem, z)
    j1 = float(objective(z))
    state = app.gt.run_forward(md, pde, z)
    traces = state[:, problem["observation"].node_ids]
    eta1, dt = problem["eta1"], pde["dt"]
    j_direct = objective_direct(traces, ds["observations"], problem["weights"], eta1, dt)
    res.add("V3a", f"case {case}: evaluator objective == direct (eta1 dt / 2) sum_n>=1 r^T W r, W = I",
            abs(j1 - j_direct) <= 1e-12 * abs(j_direct), f"J = {j1:.12e}, relative difference {abs(j1 - j_direct) / abs(j_direct):.1e}")
    rng = np.random.default_rng(7)
    ns = traces.shape[1]
    w_diag = rng.uniform(0.5, 2.0, size=ns)
    B = rng.normal(size=(ns, ns))
    w_full = B @ B.T / ns + np.eye(ns)
    for label, w in (("non-uniform diagonal W", w_diag), ("full symmetric positive definite W", w_full)):
        md_w = dict(md)
        md_w["observation"] = app.obs_op.NodeObservation(problem["observation"].node_ids, w)
        j_shared = float(app.gt.compute_objective(state, problem["target"], md_w, eta1=eta1, lam=0.0, z=z,
                                                  dt_out=dt, time_rule=problem["time_rule"]))
        j_ref = objective_direct(traces, ds["observations"], w, eta1, dt)
        res.add("V3b", f"case {case}: shared objective supports a {label}",
                abs(j_shared - j_ref) <= 1e-12 * abs(j_ref), f"relative difference {abs(j_shared - j_ref) / abs(j_ref):.1e}")
    j2 = float(ri.build_evaluator(problem, z)[1](z))
    res.add("V3c", f"case {case}: repeated evaluation gives the same objective; data file and data array unchanged",
            j1 == j2 and mdc.sha256_file(ds["path"]) == file_hash_before
            and tm.array_sha256(problem["target"]) == target_hash_before
            and tm.array_sha256(problem["target"][:, problem["observation"].node_ids]) == ds["observations_sha256"],
            f"J = {j2:.12e} twice")
    op = tm.SparseTransport(md, pde)
    sparse_traces, _ = op.solve(op.load_vector(tm.gaussian_source(z, pde["sigma2"])),
                                sampler=tm.interpolation_matrix(md, md["coord"][problem["observation"].node_ids])[0])
    diff = float(np.max(np.abs(sparse_traces - traces)) / np.max(np.abs(traces)))
    res.add("V3d", "sparse RK4 (data-generation stepping) reproduces the shared dense forward solver on the coarse mesh",
            diff < 1e-10, f"max relative trace difference {diff:.1e}")


def taylor_test(out: Path, case: str, kind: str, truth_centers, res: Results, log) -> dict:
    ds = ri.load_dataset(out, case, kind)
    problem = ri.build_problem(ds, log=log)
    z0 = np.asarray(TAYLOR_CONTROLS[case], dtype=np.float64)
    feas = ri.control_feasibility(problem, z0)
    lo, hi = problem["amplitude_bounds"]
    dist_truth = min(float(np.hypot(c["center"][0] - t[0], c["center"][1] - t[1])) for c in feas for t in truth_centers)
    admissible = all(c["in_fluid_mesh"] and lo < c["amplitude"] < hi for c in feas)
    rng = np.random.default_rng(100 + int(case))
    d = rng.normal(size=z0.size)
    d /= np.linalg.norm(d)
    _h, objective, gradient, _c, _s = ri.build_evaluator(problem, z0)
    j0 = float(objective(z0))
    g = np.asarray(gradient(z0), dtype=np.float64)
    gd = float(g @ d)
    rows, prev = [], None
    for h in TAYLOR_STEPS:
        zh = z0 + h * d
        if not problem["admissibility"].validate_sources_in_domain(zh):
            raise RuntimeError(f"Taylor point left the fluid domain at h = {h}")
        jh = float(objective(zh))
        rem = abs(jh - j0 - h * gd)
        rate = None if prev is None else float(np.log(prev[1] / rem) / np.log(prev[0] / h))
        rows.append({"h": h, "objective": jh, "remainder": rem, "rate": rate,
                     "fd_slope_relative_error": abs((jh - j0) / h - gd) / abs(gd)})
        prev = (h, rem)
    rates = [r["rate"] for r in rows if r["rate"] is not None]
    best = max(float(np.mean(rates[i:i + 2])) for i in range(len(rates) - 1))
    best_fd = min(r["fd_slope_relative_error"] for r in rows)
    passed = admissible and dist_truth > 0.5 and best >= 1.9 and best_fd < 1e-4
    res.add("V4", f"case {case} ({kind} data): directional Taylor test of the discrete-adjoint gradient",
            passed, f"best two-step remainder rate {best:.3f}, best FD-slope relative error {best_fd:.1e}, "
                    f"J = {j0:.6e}, g.d = {gd:.6e}, distance to nearest true centre {dist_truth:.2f} m")
    return {"case": case, "data": kind, "control": z0, "direction": d, "objective": j0, "gradient": g,
            "gradient_dot_direction": gd, "steps": rows, "best_two_step_rate": best,
            "best_fd_slope_relative_error": best_fd, "feasibility": feas,
            "distance_to_nearest_true_center_m": dist_truth, "passed": passed}


def all_true_centers(out: Path) -> list:
    """Every true release centre of the dataset (validation only; certifies the test controls are non-truth)."""
    centers = []
    for case in tc.CASES:
        path = out / tc.case_dir_name(case) / "truth" / "truth_record.json"
        if path.exists():
            centers += [s["center_xy"] for s in json.loads(path.read_text())["sources"]]
    return centers


def check_admissibility_and_resolution(out: Path, res: Results):
    app = tm.app1()
    md = {lv: tm.build_transport_mesh_data(lv) for lv in ("coarse", "fine")}
    adm = app.MeshAdmissibility(md["coarse"])
    wrong = [name for name, (x, y), expect in ADMISSIBILITY_POINTS if bool(adm.is_point_in_domain(x, y)) != expect]
    analytic_wrong = [name for name, (x, y), expect in ADMISSIBILITY_POINTS
                      if bool(geo.admissible_source_mask(x, y)) != expect]
    res.add("V5a", "mesh admissibility rejects beds, partitions, the bathroom wall and the exterior; accepts the fluid",
            not wrong and not analytic_wrong, f"{len(ADMISSIBILITY_POINTS)} probe points; wrong: {wrong or 'none'}")
    bad = np.array([1.5, 4.5, 1.0, 4.5, 1.5, 1.0])
    good = np.array([0.5, 4.5, 1.0, 4.5, 1.5, 1.0])
    res.add("V5b", "control-level admissibility (validate_sources_in_domain) flags a centre inside a bed",
            (not adm.validate_sources_in_domain(bad)) and adm.validate_sources_in_domain(good))
    radius = float(np.sqrt(tc.SIGMA2))
    gt = app.gt
    rows = []
    ok = True
    for bay, rel in tc.RELEASES.items():
        c = np.asarray(rel["center"])
        clearance = float(geo.distance_to_boundary(c[0], c[1])[0])
        inside = bool(adm.is_point_in_domain(*c)) and not bool(geo.inside_solid(c[0], c[1]))
        z = np.array([c[0], c[1], rel["amplitude"]])
        row = {"bay": bay, "center": c.tolist(), "amplitude": rel["amplitude"], "in_fluid": inside,
               "clearance_m": clearance, "clearance_radii": clearance / radius,
               "analytic_integral_unbounded": rel["amplitude"] * np.pi * tc.SIGMA2}
        for lv in ("coarse", "fine"):
            coord, tri = md[lv]["coord"], md[lv]["connections"]
            load = gt.assemble_load_vector(len(coord), len(tri), coord, tri, tc.NGAUSS,
                                           tm.gaussian_source(z, tc.SIGMA2))
            near = np.hypot(coord[tri].mean(axis=1)[:, 0] - c[0], coord[tri].mean(axis=1)[:, 1] - c[1]) <= radius
            edges = np.stack([np.linalg.norm(coord[tri[:, i]] - coord[tri[:, (i + 1) % 3]], axis=1) for i in range(3)], 1)
            row[lv] = {"integral": float(load.sum()), "triangles_within_radius": int(near.sum()),
                       "mean_edge_within_radius_m": float(edges[near].mean()),
                       "radius_over_mean_edge": float(radius / edges[near].mean())}
        row["coarse_vs_fine_integral_rel"] = abs(row["coarse"]["integral"] - row["fine"]["integral"]) / row["fine"]["integral"]
        row["fine_vs_analytic_rel"] = abs(row["fine"]["integral"] - row["analytic_integral_unbounded"]) / row["analytic_integral_unbounded"]
        ok &= (inside and row["clearance_radii"] >= tc.MIN_RELEASE_CLEARANCE_RADII
               and row["coarse"]["radius_over_mean_edge"] >= tc.MIN_RADIUS_OVER_COARSE_EDGE and row["coarse_vs_fine_integral_rel"] < 5e-3
               and row["fine_vs_analytic_rel"] < 5e-3)
        rows.append(row)
    res.add("V5c", "release centres in the fluid with >= 2 radii of clearance; radius / local mean edge >= "
                   f"{tc.MIN_RADIUS_OVER_COARSE_EDGE:g} on the coarse mesh; source integral agrees between meshes",
            ok, "; ".join(f"bed {r['bay']}: clearance {r['clearance_radii']:.2f} radii, radius/h coarse "
                          f"{r['coarse']['radius_over_mean_edge']:.1f} fine {r['fine']['radius_over_mean_edge']:.1f}, "
                          f"quadrature integral coarse vs fine {r['coarse_vs_fine_integral_rel']:.1e}" for r in rows),
            data=rows)
    amps = [r["amplitude"] for r in tc.RELEASES.values()]
    res.add("V5d", "release amplitudes strictly inside the amplitude bounds",
            all(tc.AMPLITUDE_BOUNDS[0] < a < tc.AMPLITUDE_BOUNDS[1] for a in amps), f"{amps} in {tc.AMPLITUDE_BOUNDS}")
    # analytic source derivatives (the ones the adjoint gradient contracts) against central differences
    ro = app.ro
    z = np.array([2.3, 5.1, 0.9, 5.6, 4.9, 1.2])
    specs = ro.gt.source_derivatives_from_control(z, r2=tc.SIGMA2)
    worst = 0.0
    for j, spec in enumerate(specs):
        e = np.zeros(6)
        e[j] = 1e-6
        for x, y in ((2.4, 5.0), (5.5, 5.05), (2.2, 5.3)):
            fd = (ro.source_from_control(z + e, tc.SIGMA2)(x, y) - ro.source_from_control(z - e, tc.SIGMA2)(x, y)) / 2e-6
            worst = max(worst, abs(fd - spec["func"](x, y)) / max(1e-12, abs(fd)))
    res.add("V5e", "analytic Gaussian source derivatives (x, y, A) match central differences", worst < 1e-6,
            f"max relative error {worst:.1e}")
    return rows


def check_wiring(out: Path, cases, gn_case, res: Results):
    for case in cases:
        for kind in ("clean", "noisy"):
            wdir = out / "validation" / "wiring" / f"{tc.case_dir_name(case)}_{kind}"
            argv = ["--dataset-dir", str(out), "--case", case, "--data", kind, "--stage", "wiring",
                    "--out-dir", str(wdir), "--initial-control", *[repr(v) for v in TAYLOR_CONTROLS[case]]]
            if gn_case == (case, kind):
                argv.append("--check-gn-scaling")
            ri.main(argv)
            w = json.loads((wdir / "wiring_check.json").read_text())
            cfg = json.loads((out / tc.case_dir_name(case) / "config" / "inversion_config.json").read_text())
            cand = w["initializer"]["candidates"]
            ok = (w["uses_truth"] is False and w["mesh"]["level"] == "coarse"
                  and w["mesh"]["sha256"] == cfg["mesh"]["mesh_sha256"]
                  and w["velocity"]["sha256"] == cfg["velocity"]["velocity_sha256"]
                  and w["inputs"]["observations_sha256"] == cfg["observations"][kind]["observations_sha256"]
                  and w["objective_zero_source_agrees"] and np.isfinite(w["objective"])
                  and np.all(np.isfinite(w["gradient"])) and len(w["gradient"]) == 6
                  and w["initializer"]["amplitude_solver"] == "osqp" and w["optimizer"]["scaling"] == "gn-diag"
                  and cand["num_admissible_candidates"] == cand["num_mesh_nodes"]
                  and all(v > 0 for v in cand["candidates_by_region"].values())
                  and not w["initializer"]["executed"] and not w["optimizer"]["executed"])
            detail = (f"J = {w['objective']:.6e}, |g| = {np.linalg.norm(w['gradient']):.3e}, "
                      f"{cand['num_admissible_candidates']} candidate centres {cand['candidates_by_region']}")
            if "gn_diagonal_at_evaluation_control" in w:
                gnr = w["gn_diagonal_at_evaluation_control"]
                ok &= bool(np.all(np.asarray(gnr["scale_vector"]) > 0.0))
                detail += (f"; GN-diagonal scaling path: cond {gnr['condition_number_unscaled']:.2e} -> "
                           f"{gnr['condition_number_scaled']:.2e} ({gnr['cost']['forward_solves']} forward / "
                           f"{gnr['cost']['adjoint_solves']} adjoint sweeps)")
            res.add("V6", f"case {case} ({kind}): initializer and optimizer entry points wired to the coarse mesh, "
                          f"its velocity and this dataset", ok, detail)


def mismatch_diagnostic(out: Path, cases, res: Results, log) -> list:
    rows = []
    for case in cases:
        cdir = out / tc.case_dir_name(case)
        truth = json.loads((cdir / "truth" / "truth_record.json").read_text())
        z_true = np.asarray(truth["true_control"], dtype=np.float64)
        ds = ri.load_dataset(out, case, "clean")
        problem = ri.build_problem(ds, log=log)
        app = problem["app"]
        state = app.gt.run_forward(problem["mesh_data"], problem["pde"], z_true)
        traces = state[:, problem["observation"].node_ids]
        clean = ds["observations"]
        noisy = np.load(cdir / "data" / "observations_noisy.npz")["observations"]
        eps = np.load(cdir / "data" / "noise.npz")["noise"]
        eta1, dt, w = problem["eta1"], problem["pde"]["dt"], problem["weights"]
        j_mis = objective_direct(traces, clean, w, eta1, dt)
        j_noisy = objective_direct(traces, noisy, w, eta1, dt)
        j_noise = 0.5 * eta1 * dt * float(np.sum(eps[1:] ** 2))
        diff = traces[1:] - clean[1:]
        per_sensor = np.linalg.norm(diff, axis=0) / np.maximum(np.linalg.norm(clean[1:], axis=0), 1e-300)
        np.savez(out / "validation" / f"coarse_traces_at_truth_{tc.case_dir_name(case)}.npz",
                 times=ds["times"], coarse_traces=traces)
        row = {
            "case": case,
            "objective_coarse_at_truth_vs_clean": j_mis,
            "objective_coarse_at_truth_vs_noisy": j_noisy,
            "objective_noise_only": j_noise,
            "objective_zero_source_clean": problem["j_zero"],
            "relative_misfit_norm_clean": float(np.linalg.norm(diff) / np.linalg.norm(clean[1:])),
            "nominal_noise_level": tc.NOISE_LEVEL,
            "mismatch_over_noise_objective_ratio": j_mis / j_noise,
            "per_sensor_relative_misfit": per_sensor,
            "max_abs_trace_difference": float(np.max(np.abs(diff))),
            "state_min": float(state.min()), "state_max": float(state.max()),
        }
        rows.append(row)
        res.add("D", f"case {case}: coarse model at the true control vs clean fine-mesh data (diagnostic, not a pass/fail)",
                np.isfinite(j_mis), f"J = {j_mis:.6e} (relative misfit norm {row['relative_misfit_norm_clean']:.4f}; "
                                    f"noise-only objective {j_noise:.6e}; ratio {row['mismatch_over_noise_objective_ratio']:.2f})")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, default=tm.DATASET_DIR)
    ap.add_argument("--cases", nargs="+", default=list(tc.CASES), choices=list(tc.CASES))
    ap.add_argument("--skip-gn-check", action="store_true")
    args = ap.parse_args()
    out = args.dataset_dir.resolve()
    vdir = out / "validation"
    vdir.mkdir(parents=True, exist_ok=True)
    logf = open(out / "logs" / "validate_readiness.log", "a")

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    res = Results(log)
    taylor, mismatch, resolution = [], [], []
    stages = [
        ("V1 fine-mesh sampling", lambda: check_sampling(out, res)),
        ("V2 datasets", lambda: check_datasets(out, args.cases, res)),
        ("V5 admissibility and Gaussian resolution", lambda: resolution.extend(check_admissibility_and_resolution(out, res))),
        ("V3 objective consistency", lambda: check_objective(out, args.cases[0], None, res, log)),
        ("V6 wiring", lambda: check_wiring(out, args.cases, None if args.skip_gn_check else (args.cases[0], "noisy"), res)),
        ("V4 Taylor tests", lambda: [taylor.append(taylor_test(out, c, k, all_true_centers(out), res, log))
                                     for c in args.cases for k in ("clean", "noisy")]),
        ("D discretization mismatch at the true controls", lambda: mismatch.extend(mismatch_diagnostic(out, args.cases, res, log))),
    ]
    for name, fn in stages:
        log(f"=== {name}")
        t0 = time.perf_counter()
        try:
            fn()
        except Exception:  # noqa: BLE001 - record the failure and keep validating
            res.add("ERR", name, False, traceback.format_exc().strip().splitlines()[-1])
            log(traceback.format_exc())
        log(f"    ({time.perf_counter() - t0:.0f} s)")

    all_passed = all(r["passed"] for r in res.rows)
    tm.write_json(vdir / "validation_results.json", {
        "dataset_version": tc.DATASET_VERSION, "all_passed": all_passed,
        "num_checks": len(res.rows), "num_failed": sum(not r["passed"] for r in res.rows),
        "optimization_or_initialization_run": False, "checks": res.rows})
    with open(vdir / "validation_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "passed", "name", "detail"])
        for r in res.rows:
            w.writerow([r["id"], r["passed"], r["name"], r["detail"]])
    tm.write_json(vdir / "taylor_tests.json", taylor)
    tm.write_json(vdir / "gaussian_resolution.json", resolution)
    tm.write_json(vdir / "discretization_mismatch.json", {
        "note": "coarse-model misfit at the TRUE control against the clean fine-mesh observations; a diagnostic of "
                "the coarse/fine model mismatch, kept out of every inversion input",
        "cases": mismatch})
    if mismatch:
        with open(vdir / "discretization_mismatch.csv", "w", newline="") as f:
            cols = ["case", "objective_coarse_at_truth_vs_clean", "relative_misfit_norm_clean",
                    "objective_coarse_at_truth_vs_noisy", "objective_noise_only",
                    "mismatch_over_noise_objective_ratio", "objective_zero_source_clean", "max_abs_trace_difference"]
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(mismatch)
    log(f"validation finished: {'ALL PASSED' if all_passed else 'FAILURES'} "
        f"({sum(not r['passed'] for r in res.rows)} failed of {len(res.rows)})")
    return 0 if all_passed else 3


if __name__ == "__main__":
    sys.exit(main())
