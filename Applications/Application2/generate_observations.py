"""Generate the Application2 source-identification observation datasets.

    PY generate_observations.py [--out DATASET_DIR] [--cases 12 13 23] [--noise-level 0.01]

Order of operations (each stage writes its record before the next starts):

  1 preflight   geometry / mesh / velocity identity and convergence checks for
                BOTH levels (aborts on any mismatch, e.g. a no-bed or
                bottom-supply field)                  -> common/mesh_velocity_identifiers.json
  2 sensors     fix the sensor layout from geometry + the coarse mesh only,
                BEFORE any concentration field exists -> common/sensors.{json,csv}
  3 stability   RK4 step check per mesh with its own velocity field,
                cell Peclet numbers                   -> common/stability.json
  4 cases       one FINE-mesh forward solve per true configuration, sampled at
                the fixed sensor coordinates by P1 interpolation at the common
                observation times; one noise realization per case
                                                      -> case_XX/{config,truth,data,logs}

Nothing here runs an initialization, a centre search or an optimization.

Truth separation: ``case_XX/config/inversion_config.json`` and
``case_XX/data/observations_*.npz`` hold everything the inversion may read
(observations, sensor locations, source count, admissible bounds, model
settings).  The true control lives only in ``case_XX/truth/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

import transport_config as tc
import transport_model as tm

geo, mdc, nsc = tm.geo, tm.mdc, tm.nsc


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a")

    def __call__(self, msg: str = "") -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


# --------------------------------------------------------------------------- #
# 1 preflight
# --------------------------------------------------------------------------- #
def preflight(md: dict, log) -> dict:
    """Confirm both meshes / velocity fields are the approved v3 ones. Raises on failure."""
    flow_dir = tm.FLOW_DIR
    checks, record = [], {"flow_version": tc.FLOW_VERSION, "flow_dir": str(flow_dir), "levels": {}}

    def check(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": str(detail)})
        log(f"  [{'ok' if ok else 'FAIL'}] {name} {detail}")

    check("flow output folder is the geometry's OUTPUT_VERSION", flow_dir.name == geo.OUTPUT_VERSION, flow_dir.name)
    run_state = (flow_dir / "RUN_STATE").read_text().strip() if (flow_dir / "RUN_STATE").exists() else "(missing)"
    check("flow pipeline state DONE_CONVERGED", run_state.startswith("DONE_CONVERGED"), run_state)
    sums = flow_dir / "env" / "sha256_application2_sources.txt"
    if sums.exists():
        changed = []
        for line in sums.read_text().splitlines():
            digest, rel = line.split(None, 1)
            path = tm.APP2_DIR / rel.strip().lstrip("*")
            if not path.exists() or mdc.sha256_file(path) != digest:
                changed.append(rel.strip())
        check("geometry/mesh/flow sources unchanged since the flow fields were generated", not changed, changed or "")
    geometry_hash = geo.geometry_hash()
    for level in ("coarse", "fine"):
        m = md[level]
        coord, tri = m["coord"], m["connections"]
        info = json.loads(mdc.mesh_info_file(level, flow_dir).read_text())
        meta_path = mdc.velocity_npz_file(level, flow_dir).with_name(
            f"application2_navier_stokes_run_metadata_{level}.json")
        meta = json.loads(meta_path.read_text())
        log(f"level {level}: {len(coord)} nodes, {len(tri)} triangles")
        check(f"{level}: mesh file sha256 == mesh_info == velocity association",
              m["mesh_sha256"] == info["msh_sha256"] == meta["association"]["mesh_sha256"], m["mesh_sha256"][:16])
        check(f"{level}: geometry hash == mesh_info == velocity association",
              geometry_hash == info["geometry_sha256"] == meta["association"]["geometry_sha256"], geometry_hash[:16])
        check(f"{level}: velocity association level", meta["association"]["mesh_level"] == level == meta["level"])
        check(f"{level}: transport node coordinates hash",
              mdc.coords_fingerprint(coord) == meta["association"]["transport_coords_sha256"])
        pic = meta["picard"]
        check(f"{level}: Navier-Stokes converged", meta["status"] == "converged" and pic["converged"]
              and pic["final_relative_update"] < pic["tolerance"],
              f"{pic['iterations']} Picard iterations, final update {pic['final_relative_update']:.2e}")
        par = meta["parameters"]
        check(f"{level}: approved flow parameters (nu, supply speed, Picard tolerance)",
              par["nu_m2_per_s"] == nsc.NU and par["inlet_mean_speed_m_per_s"] == nsc.INLET_MEAN_SPEED
              and par["picard_tol_relative_L2_update"] == nsc.PICARD_TOL,
              f"nu={par['nu_m2_per_s']}, U_mean={par['inlet_mean_speed_m_per_s']}")
        outlets = meta["boundary_conditions"]["outlets"]
        check(f"{level}: natural outlet conditions recorded", "natural" in outlets and "NO outlet pressure" in outlets)
        # --- geometry of THIS mesh: three bed holes, two partitions, top supplies, door + bathroom exhaust
        cx, cy = coord[tri, 0].mean(axis=1), coord[tri, 1].mean(axis=1)
        check(f"{level}: no triangle inside a bed/partition/bathroom wall or outside room+annex",
              not np.any(geo.inside_solid(cx, cy)) and bool(np.all(geo.inside_union(cx, cy))))
        topo = info["topology_and_solids"]
        check(f"{level}: one component, four boundary loops (outer + 3 beds)",
              topo["edge_connected_components"] == 1 and topo["boundary_loops"] == 4)
        names = m["boundary_nodes_by_name"]
        for bed, (x0, x1, y0, y1) in ((b, geo.SOLIDS[b]) for b in geo.BED_NAMES.values()):
            p = coord[names[bed]]
            check(f"{level}: {bed} boundary is the bed rectangle",
                  np.isclose(p[:, 0].min(), x0) and np.isclose(p[:, 0].max(), x1)
                  and np.isclose(p[:, 1].min(), y0) and np.isclose(p[:, 1].max(), y1))
        for part in geo.PARTITIONS:
            check(f"{level}: {part} present", part in names and names[part].size > 0)
        vel = m["velocity_nodes"]
        inlet_ok, inlet_flux_sign = True, []
        for inlet, (a, b) in zip(geo.INLET_NAMES, geo.VENT_X_RANGES):
            p, v = coord[names[inlet]], vel[names[inlet]]
            expected = nsc.inlet_velocity(np.vstack([p[:, 0], p[:, 1]]))
            inlet_ok &= bool(np.allclose(p[:, 1], geo.INLET_Y) and p[:, 0].min() >= a - 1e-9 and p[:, 0].max() <= b + 1e-9
                             and np.allclose(v, expected.T, atol=1e-9) and v[:, 1].min() < -0.1)
            inlet_flux_sign.append(float(v[:, 1].min()))
        check(f"{level}: supplies on the top wall, nodal velocity = downward parabola (peak {nsc.INLET_PEAK_SPEED:.3f})",
              inlet_ok, f"min v_y per supply {inlet_flux_sign}")
        check(f"{level}: outlets at the corridor door (y = 0) and bathroom exhaust (x = 11)",
              np.allclose(coord[names['outlet_door'], 1], 0.0)
              and np.allclose(coord[names['outlet_exhaust'], 0], geo.DOMAIN_X_MAX))
        outflow = {"outlet_door": float(-vel[names["outlet_door"], 1].min()),
                   "outlet_exhaust": float(vel[names["outlet_exhaust"], 0].min())}
        check(f"{level}: no backflow at outlet nodes (min outward normal velocity >= 0)",
              min(outflow.values()) >= -1e-12, outflow)
        wall_nodes = np.unique(np.concatenate([names[w] for w in geo.WALL_NAMES]))
        open_nodes = np.unique(np.concatenate([names[n] for n in (*geo.INLET_NAMES, *geo.OUTLET_NAMES)]))
        wall_only = np.setdiff1d(wall_nodes, open_nodes)
        check(f"{level}: no-slip velocity at wall/bed/partition nodes",
              float(np.abs(vel[wall_only]).max()) < 1e-10, f"max |v| = {np.abs(vel[wall_only]).max():.1e}")
        stale = "upward" in meta["boundary_conditions"].get("inlets", "")
        record["levels"][level] = {
            "mesh_file": m["mesh_file"], "mesh_sha256": m["mesh_sha256"], "geometry_sha256": geometry_hash,
            "n_nodes": int(len(coord)), "n_triangles": int(len(tri)),
            "velocity_file": m["velocity_file"], "velocity_sha256": m["velocity_sha256"],
            "navier_stokes_metadata": str(meta_path),
            "picard_iterations": pic["iterations"], "picard_final_relative_update": pic["final_relative_update"],
            "nu_m2_per_s": par["nu_m2_per_s"], "inlet_mean_speed_m_per_s": par["inlet_mean_speed_m_per_s"],
            "outlet_condition": outlets,
            "flow_git_commit_at_generation": meta["provenance"]["git_commit"],
            "note_metadata_wording": ("the flow metadata string boundary_conditions.inlets says 'parabolic upward "
                                      "profile' (stale wording from the earlier bottom-supply versions); the nodal "
                                      "velocity itself is verified above to be the DOWNWARD top-wall supply profile"
                                      if stale else None),
        }
    record["checks"] = checks
    record["all_passed"] = all(c["passed"] for c in checks)
    if not record["all_passed"]:
        failed = [c["check"] for c in checks if not c["passed"]]
        raise RuntimeError(f"preflight failed: {failed}")
    return record


# --------------------------------------------------------------------------- #
# 3 stability
# --------------------------------------------------------------------------- #
def stability(ops: dict, md: dict, log) -> dict:
    out = {"rk4_real_axis_limit": tc.RK4_REAL_AXIS_LIMIT, "target_dt_rho": tc.RK4_STABILITY_TARGET,
           "observation_interval_s": tc.OBSERVATION_INTERVAL, "fine_substeps_per_observation": tc.FINE_SUBSTEPS,
           "tfinal_s": tc.TFINAL, "levels": {}}
    if abs(tc.DT["coarse"] - tc.FINE_SUBSTEPS * tc.DT["fine"]) > 1e-15:
        raise ValueError("coarse dt must be an integer multiple (FINE_SUBSTEPS) of the fine dt")
    if tc.NUM_STEPS["fine"] != tc.FINE_SUBSTEPS * tc.NUM_STEPS["coarse"]:
        raise ValueError("fine and coarse step counts do not give common observation times")
    for level, op in ops.items():
        rho = op.spectral_radius()
        dt = tc.DT[level]
        rec = {"dt_s": dt, "num_steps": tc.NUM_STEPS[level], "spectral_radius_Minv_K_per_s": rho,
               "dt_times_spectral_radius": dt * rho, "largest_stable_dt_at_target_s": tc.RK4_STABILITY_TARGET / rho,
               "stable": bool(dt * rho <= tc.RK4_STABILITY_TARGET),
               "cell_peclet": tm.cell_peclet(md[level], tc.DIFFUSIVITY_K),
               "velocity_file": md[level]["velocity_file"], "mesh_sha256": md[level]["mesh_sha256"]}
        out["levels"][level] = rec
        log(f"  {level}: rho(M^-1 K) = {rho:.3f} 1/s, dt = {dt} s, dt*rho = {dt * rho:.3f} "
            f"(target <= {tc.RK4_STABILITY_TARGET}); cell Pe max {rec['cell_peclet']['max']:.2f}")
        if not rec["stable"]:
            raise RuntimeError(f"{level}: dt * rho = {dt * rho:.3f} exceeds the RK4 stability target")
    return out


# --------------------------------------------------------------------------- #
# 4 cases
# --------------------------------------------------------------------------- #
def noise_std(clean: np.ndarray, level: float) -> float:
    """s_noise = level * RMS of the clean samples that enter the objective (n >= 1)."""
    used = clean[1:]
    return float(level) * float(np.sqrt(np.mean(used ** 2)))


def generate_case(case: str, out: Path, op_fine, P_fine, sensors: dict, identifiers: dict, noise_level: float,
                  seed: int, md: dict) -> dict:
    cdir = out / tc.case_dir_name(case)
    for sub in ("config", "truth", "data", "logs"):
        (cdir / sub).mkdir(parents=True, exist_ok=True)
    log = Tee(cdir / "logs" / "generate_fine.log")
    z_true = np.asarray(tc.true_control(case), dtype=np.float64)
    bays = tc.CASES[case]
    log(f"case {case}: fine-mesh forward solve, releases near beds {bays}, z* = {z_true.tolist()}")
    centers = z_true.reshape(-1, 3)[:, :2]
    in_fine = mdc.point_in_fluid_mesh(md["fine"], centers)
    in_coarse = mdc.point_in_fluid_mesh(md["coarse"], centers)
    clearance = geo.distance_to_boundary(centers[:, 0], centers[:, 1])
    radius = float(np.sqrt(tc.SIGMA2))
    if not (in_fine.all() and in_coarse.all()):
        raise ValueError(f"case {case}: a release centre is not inside the fluid mesh")
    if clearance.min() < tc.MIN_RELEASE_CLEARANCE_RADII * radius:
        raise ValueError(f"case {case}: release clearance {clearance.min():.3f} m < "
                         f"{tc.MIN_RELEASE_CLEARANCE_RADII} e-folding radii")
    lo, hi = tc.AMPLITUDE_BOUNDS
    if not np.all((z_true[2::3] > lo) & (z_true[2::3] < hi)):
        raise ValueError(f"case {case}: amplitudes must lie strictly inside {tc.AMPLITUDE_BOUNDS}")

    t0 = time.perf_counter()
    load = op_fine.load_vector(tm.gaussian_source(z_true, tc.SIGMA2))
    nsteps = tc.NUM_STEPS["fine"]
    snap_steps = sorted({nsteps // 4, nsteps // 2, 3 * nsteps // 4, nsteps})
    clean, snaps = op_fine.solve(load, sample_every=tc.FINE_SUBSTEPS, sampler=P_fine, snapshot_steps=snap_steps,
                                 progress=(nsteps // 10, lambda s, n: log(f"  step {s}/{n}")))
    seconds = time.perf_counter() - t0
    n_frames, n_sensors = clean.shape
    if n_frames != tc.NUM_OBSERVATION_TIMES + 1:
        raise RuntimeError(f"expected {tc.NUM_OBSERVATION_TIMES + 1} observation frames, got {n_frames}")
    times = np.arange(n_frames) * tc.OBSERVATION_INTERVAL
    in_objective = np.ones(n_frames, dtype=bool)
    in_objective[0] = False                              # objective_time_rule = exclude_first
    log(f"  done in {seconds:.1f} s: {n_frames} frames x {n_sensors} sensors; "
        f"clean range [{clean.min():.3e}, {clean.max():.3e}]")

    # ---- noise: drawn ONCE here, saved, never redrawn
    s_noise = noise_std(clean, noise_level)
    rng = np.random.default_rng(seed)
    noise = np.zeros_like(clean)
    noise[1:] = rng.normal(0.0, s_noise, size=(n_frames - 1, n_sensors))
    noisy = clean + noise                                 # no clipping of negative values
    used_clean, used_noise = clean[1:], noise[1:]
    n_obs = int(used_clean.size)
    noise_record = {
        "case": case, "model": "d_noisy = d_clean + eps, eps_j^n ~ N(0, s_noise^2) i.i.d. over sensors and times",
        "nominal_noise_level": float(noise_level), "seed": int(seed),
        "generator": "numpy.random.default_rng(seed).normal(0, s_noise, size=(N, n_sensors)); rows = times n = 1..N",
        "numpy_version": np.__version__,
        "s_noise_absolute": s_noise,
        "s_noise_definition": "nominal_noise_level * sqrt(mean_{n=1..N, j} d_clean[n, j]^2)",
        "num_observations_in_objective": n_obs, "num_times_in_objective": int(n_frames - 1),
        "num_sensors": int(n_sensors),
        "initial_sample_n0": "excluded from the objective and from the RMS; d_clean[0] = 0 and noise[0] = 0",
        "clean_rms": float(np.sqrt(np.mean(used_clean ** 2))),
        "realized_noise_sample_std": float(np.std(used_noise)),
        "realized_noise_mean": float(np.mean(used_noise)),
        "realized_relative_noise_norm": float(np.linalg.norm(used_noise) / np.linalg.norm(used_clean)),
        "num_negative_noisy_samples": int(np.count_nonzero(noisy[1:] < 0.0)), "negative_values_clipped": False,
        "expected_objective_at_noise_only": 0.5 * tc.ETA1 * tc.OBSERVATION_INTERVAL * tc.SENSOR_WEIGHT * float(
            np.sum(used_noise ** 2)),
        "weights": "W = I for this equal-variance case (configurable in the inversion)",
    }

    common = {"times": times, "in_objective": in_objective,
              "sensor_node_ids": np.asarray(sensors["node_ids"], dtype=int),
              "sensor_xy": np.asarray(sensors["coordinates"], dtype=np.float64)}
    files = {}
    for kind, arr in (("clean", clean), ("noisy", noisy)):
        path = cdir / "data" / f"observations_{kind}.npz"
        np.savez(path, observations=arr, **common)
        files[kind] = {"file": f"data/{path.name}", "file_sha256": mdc.sha256_file(path),
                       "observations_sha256": tm.array_sha256(arr)}
    noise_path = cdir / "data" / "noise.npz"
    np.savez(noise_path, noise=noise, seed=np.int64(seed), s_noise=np.float64(s_noise),
             nominal_noise_level=np.float64(noise_level), **common)
    noise_record["noise_file"] = f"data/{noise_path.name}"
    noise_record["noise_sha256"] = tm.array_sha256(noise)
    tm.write_json(cdir / "data" / "noise_record.json", noise_record)
    with open(cdir / "data" / "observations_clean_noisy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s"] + [f"clean_{s['label']}" for s in sensors["sensors"]]
                   + [f"noisy_{s['label']}" for s in sensors["sensors"]])
        stride = max(1, (n_frames - 1) // 600)             # readable subsample; the NPZ files are authoritative
        for n in range(0, n_frames, stride):
            w.writerow([f"{times[n]:.6f}"] + [f"{v:.10e}" for v in clean[n]] + [f"{v:.10e}" for v in noisy[n]])

    # ---- truth (never read by the inversion)
    np.savez(cdir / "truth" / "fine_state_snapshots.npz",
             times=np.asarray([s * tc.DT["fine"] for s in snap_steps]),
             states=np.stack([snaps[s] for s in snap_steps]), coord=md["fine"]["coord"],
             connections=md["fine"]["connections"])
    truth = {
        "case": case, "WARNING": "TRUTH RECORD - must never be passed to the inversion",
        "occupied_bays": list(bays), "true_control": z_true.tolist(),
        "control_ordering": "[x1, y1, A1, x2, y2, A2]",
        "sources": [{"bay": int(b), "center_xy": [float(c[0]), float(c[1])], "amplitude": float(a),
                     "distance_to_nearest_boundary_m": float(d),
                     "clearance_in_e_folding_radii": float(d / radius),
                     "total_release_rate_per_m_depth": float(a * np.pi * tc.SIGMA2)}
                    for b, c, a, d in zip(bays, centers, z_true[2::3], clearance)],
        "description": "stationary Gaussian tracer releases in the fluid beside the bed heads (not geometrically "
                       "resolved patient mouths); beds are solid obstacles",
        "data_mesh_level": "fine", "fine_dt_s": tc.DT["fine"], "fine_steps": nsteps,
        "fine_solve_seconds": seconds,
        "fine_state_snapshots": "truth/fine_state_snapshots.npz",
        "observation_files": files,
    }
    tm.write_json(cdir / "truth" / "truth_record.json", truth)

    # ---- inversion configuration: NO truth
    bbox = [float(md["coarse"]["coord"][:, 0].min()), float(md["coarse"]["coord"][:, 0].max()),
            float(md["coarse"]["coord"][:, 1].min()), float(md["coarse"]["coord"][:, 1].max())]
    config = {
        "case_label": tc.case_dir_name(case),
        "dataset_version": tc.DATASET_VERSION,
        "contains_truth": False,
        "num_sources": tc.NUM_SOURCES,
        "control_ordering": "[x1, y1, A1, x2, y2, A2]",
        "admissible_set": {
            "amplitude_bounds": list(tc.AMPLITUDE_BOUNDS),
            "center_box_bounds_xmin_xmax_ymin_ymax": bbox,
            "center_region": "any point inside a triangle of the coarse fluid mesh (beds, partitions, the bathroom "
                             "wall and the exterior are not meshed); not restricted to any bay",
        },
        "inversion_mesh_level": tc.INVERSION_MESH_LEVEL,
        "mesh": {k: identifiers["levels"]["coarse"][k] for k in ("mesh_file", "mesh_sha256", "geometry_sha256")},
        "velocity": {k: identifiers["levels"]["coarse"][k] for k in ("velocity_file", "velocity_sha256")},
        "transport": {"diffusivity_k": tc.DIFFUSIVITY_K, "sigma2": tc.SIGMA2, "tfinal": tc.TFINAL,
                      "dt": tc.DT["coarse"], "jprint": 1, "ngauss": tc.NGAUSS,
                      "clean_air_dirichlet_boundaries": list(tc.CLEAN_AIR_DIRICHLET_NAMES)},
        "objective": {"eta1": tc.ETA1, "lambda_reg": tc.LAMBDA_REG, "time_rule": tc.OBJECTIVE_TIME_RULE,
                      "sensor_weights": [tc.SENSOR_WEIGHT] * n_sensors,
                      "sensor_weights_note": "diagonal of W; W = I for the equal-variance noise of this dataset"},
        "sensors": {"node_ids": sensors["node_ids"], "coordinates": sensors["coordinates"],
                    "labels": sensors["labels"], "names": [s["name"] for s in sensors["sensors"]],
                    "layout": sensors["layout"]},
        "observations": {
            "times": "t_n = n * %g s, n = 0 .. %d; n = 0 is not in the objective" % (tc.OBSERVATION_INTERVAL,
                                                                                    n_frames - 1),
            "observation_interval_s": tc.OBSERVATION_INTERVAL, "num_frames": int(n_frames),
            "clean": files["clean"], "noisy": files["noisy"],
            "noisy_nominal_noise_level": float(noise_level),
        },
    }
    tm.write_json(cdir / "config" / "inversion_config.json", config)
    log(f"  s_noise = {s_noise:.6e} (nominal {noise_level:g}), seed {seed}, realized relative noise norm "
        f"{noise_record['realized_relative_noise_norm']:.6f}")
    return {"case": case, "bays": list(bays), "clean_min": float(clean.min()), "clean_max": float(clean.max()),
            "clean_rms": noise_record["clean_rms"], "s_noise": s_noise, "seed": int(seed),
            "realized_relative_noise_norm": noise_record["realized_relative_noise_norm"],
            "num_observations": n_obs, "fine_solve_seconds": seconds, "files": files}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=tm.DATASET_DIR)
    ap.add_argument("--cases", nargs="+", default=list(tc.CASES), choices=list(tc.CASES))
    ap.add_argument("--noise-level", type=float, default=tc.NOISE_LEVEL,
                    help="nominal global-RMS noise level (default %(default)s)")
    args = ap.parse_args()
    out = args.out.resolve()
    existing = [c for c in args.cases if (out / tc.case_dir_name(c) / "data").exists()]
    if existing:
        print(f"refusing: datasets already exist for cases {existing} in {out} (the saved noise must stay fixed)",
              file=sys.stderr)
        return 1
    log = Tee(out / "logs" / "generate_observations.log")
    log(f"dataset {tc.DATASET_VERSION} -> {out}  (python {platform.python_version()}, numpy {np.__version__})")

    log("stage 1: preflight (geometry / mesh / velocity identity)")
    md = {lv: tm.build_transport_mesh_data(lv) for lv in ("coarse", "fine")}
    identifiers = preflight(md, log)
    tm.write_json(out / "common" / "mesh_velocity_identifiers.json", identifiers)

    log("stage 2: fix sensors (geometry + coarse mesh only; no concentration field exists yet)")
    sensors = tm.fix_sensors(md["coarse"])
    P_fine, fine_info = tm.interpolation_matrix(md["fine"], sensors["coordinates"])
    for s, fi in zip(sensors["sensors"], fine_info):
        s["fine_mesh_interpolation"] = fi
    sensors["fine_mesh_sha256"] = md["fine"]["mesh_sha256"]
    sensors["fine_sampling"] = ("P1 finite-element interpolation of the fine solution at the coarse-node "
                                "coordinates (containing fine triangle, barycentric weights)")
    tm.write_json(out / "common" / "sensors.json", sensors)
    with open(out / "common" / "sensors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "name", "region", "wall", "x_m", "y_m", "coarse_node_id", "wall_offset_m",
                    "wall_point_x_m", "wall_point_y_m", "intended_x_m", "intended_y_m", "snap_distance_m",
                    "distance_to_nearest_boundary_m", "description"])
        for s in sensors["sensors"]:
            wp = s["wall_point"] or ["", ""]
            w.writerow([s["label"], s["name"], s["region"], s["wall"] or "", repr(s["xy"][0]), repr(s["xy"][1]),
                        s["coarse_node_id"], f"{s.get('wall_offset_m', float('nan')):.6f}", wp[0], wp[1],
                        s["intended_xy"][0], s["intended_xy"][1], f"{s['snap_distance_m']:.6f}",
                        f"{s['distance_to_boundary_m']:.6f}", s["description"]])
    if tc.PROPOSED_SENSOR_LAYOUTS:
        tm.write_json(out / "common" / "proposed_sensor_layouts_NOT_ACTIVE.json", {
            "note": "recorded for a possible later extension; NOT used by this dataset or by the inversion",
            "layouts": {name: tm.fix_sensors(md["coarse"], name) for name in tc.PROPOSED_SENSOR_LAYOUTS}})
    log(f"  layout {sensors['layout']}: {sensors['num_sensors']} sensors; max snap distance "
        f"{max(s['snap_distance_m'] for s in sensors['sensors']):.4f} m")
    tm.write_json(out / "common" / "transport_config.json", tc.as_dict())

    log("stage 3: operators and RK4 stability (each mesh with its own velocity field)")
    ops = {lv: tm.SparseTransport(md[lv], tm.build_pde(lv, md[lv])) for lv in ("coarse", "fine")}
    stab = stability(ops, md, log)
    tm.write_json(out / "common" / "stability.json", stab)

    log("stage 4: fine-mesh data generation")
    rows = [generate_case(case, out, ops["fine"], P_fine, sensors, identifiers, args.noise_level,
                          tc.NOISE_SEEDS[case], md) for case in args.cases]
    tm.write_json(out / "common" / "generation_summary.json",
                  {"dataset_version": tc.DATASET_VERSION, "noise_level": args.noise_level, "cases": rows})
    with open(out / "common" / "generation_summary.csv", "w", newline="") as f:
        cols = ["case", "bays", "num_observations", "clean_min", "clean_max", "clean_rms", "s_noise", "seed",
                "realized_relative_noise_norm", "fine_solve_seconds"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log("data generation finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
