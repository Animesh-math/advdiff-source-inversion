"""Application2 coarse-mesh source identification from SAVED observations.

    PY run_inversion.py --case 12 --data noisy --stage wiring
    PY run_inversion.py --case 12 --data noisy --stage full      # NOT run during dataset preparation

Reads only ``<dataset>/case_XX/config/inversion_config.json`` and the selected
``data/observations_{clean,noisy}.npz`` (observations, sensor locations, the
prescribed source count, admissible bounds, model settings).  It never opens
``truth/``; recovery errors are computed afterwards by a separate scoring step.

It is the discretize-then-optimize workflow of Application1 with the same code:

  forward / objective / discrete adjoint   Src/Objective/gradient_test.py -> Src/Solvers    (RK4, P1, lumped mass)
  node-sensor objective with weights W     Src/Solvers/observation_operator.NodeObservation
  evaluator (raw feasibility mode)         Application1/run_optimization.build_evaluator
  admissibility                            Application1/mesh_admissibility.MeshAdmissibility on the coarse room mesh
  sensitivity initialization + OSQP        Application1/sensitivity_initialization.build_sensitivity_initial_control
  GN-diagonal scaling + L-BFGS-B           Application1/post_init_optimizers (gn_diagonal_scaling, run_lbfgsb_polish)

Application1 is not modified.  Its driver generates its own data from a true
control, so this file re-implements only the thin orchestration (single stage,
no sigma continuation) around those components and feeds them the saved data:
the data enter as a nodal array that is zero except in the sensor columns,
which is all a node observation ever reads (d = C u_d).

Three Application1 length/objective-scale literals
--------------------------------------------------
``sensitivity_initialization.py`` hard-codes three tolerances chosen for the DC
domain (Gaussian radius 20 m, J(0) ~ 1e5-1e6):

    Nelder-Mead centre refinement   xatol = 0.5 [m],  fatol = 1.0 [objective units]
    branch de-duplication radius    max(2.0 [m], 0.25 * sqrt(sigma2))

At room scale (radius 0.20 m) they would stop the refinement at once and merge
branches up to 2 m apart.  By default this driver therefore loads the
initializer through a shim that substitutes exactly these three literals with
values scaled to the room problem (``--init-refine-xatol``,
``--init-refine-fatol-rel``, ``--init-dedup-floor``); the shim aborts unless it
finds the literals exactly as expected.  ``--init-app1-literal-tolerances`` uses
the unmodified Application1 module instead.

Stages
------
wiring  build everything, evaluate the objective and its adjoint gradient at one
        feasible control, count the initializer's admissible candidates, and
        (``--check-gn-scaling``) evaluate the Gauss-Newton diagonal there.  No
        initialization, no centre search, no optimization.
init    sensitivity initialization only.
full    sensitivity initialization (or ``--initial-control``) + GN-diag L-BFGS-B.

Solve accounting: every forward / adjoint sweep is counted at the shared entry
points and reported separately for initialization, scaling and optimization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np

import transport_config as tc
import transport_model as tm

MODE = "location_and_intensity"
SHIM_NM = 'options={"xatol": 0.5, "fatol": 1.0, "maxfev": '
SHIM_DEDUP = "dedup_radius = max(2.0, 0.25 * float(np.sqrt(sigma2)))"
# Application1 reference scales behind the default shim values.
APP1_GAUSSIAN_RADIUS_M = 20.0
APP1_NM_XATOL_M = 0.5
APP1_DEDUP_FLOOR_M = 2.0
DEFAULT_FATOL_REL = 1.0e-6        # Application1: fatol = 1.0 against J(zero source) ~ 4e5 .. 1e6


# --------------------------------------------------------------------------- #
# Dataset (truth-free)
# --------------------------------------------------------------------------- #
def load_dataset(dataset_dir: Path, case: str, kind: str) -> dict:
    cdir = Path(dataset_dir) / tc.case_dir_name(case)
    cfg_path = cdir / "config" / "inversion_config.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("contains_truth") is not False:
        raise ValueError(f"{cfg_path} is not marked truth-free")
    entry = cfg["observations"][kind]
    path = cdir / entry["file"]
    if tm.mdc.sha256_file(path) != entry["file_sha256"]:
        raise ValueError(f"{path} does not match the sha256 recorded in the inversion configuration")
    with np.load(path, allow_pickle=False) as d:
        obs = np.asarray(d["observations"], dtype=np.float64)
        times = np.asarray(d["times"], dtype=np.float64)
        in_objective = np.asarray(d["in_objective"], dtype=bool)
        node_ids = np.asarray(d["sensor_node_ids"], dtype=int)
        xy = np.asarray(d["sensor_xy"], dtype=np.float64)
    if tm.array_sha256(obs) != entry["observations_sha256"]:
        raise ValueError(f"{path}: observation array hash mismatch")
    return {"case_dir": cdir, "config_path": cfg_path, "config": cfg, "kind": kind, "path": path,
            "observations": obs, "times": times, "in_objective": in_objective, "node_ids": node_ids,
            "sensor_xy": xy, "observations_sha256": entry["observations_sha256"],
            "file_sha256": entry["file_sha256"]}


# --------------------------------------------------------------------------- #
# Problem assembly
# --------------------------------------------------------------------------- #
class SolveCounter:
    """Counts actual PDE sweeps at the shared entry points (cache hits never reach them)."""

    def __init__(self, gt):
        self.counts = {"forward": 0, "adjoint": 0}
        self._gt = gt
        self._fwd, self._adj = gt.run_forward, gt.rk4_discrete_source_adjoint

        def forward(*a, **k):
            self.counts["forward"] += 1
            return self._fwd(*a, **k)

        def adjoint(*a, **k):
            self.counts["adjoint"] += 1
            return self._adj(*a, **k)

        gt.run_forward, gt.rk4_discrete_source_adjoint = forward, adjoint

    def snapshot(self):
        return dict(self.counts), time.perf_counter()

    @staticmethod
    def delta(a, b):
        (ca, ta), (cb, tb) = a, b
        return {"forward_solves": cb["forward"] - ca["forward"], "adjoint_solves": cb["adjoint"] - ca["adjoint"],
                "seconds": tb - ta}


def build_problem(ds: dict, *, sensor_weights=None, eta1=None, log=print) -> dict:
    """Coarse mesh + its velocity + saved data + observation operator, wired into Application1's globals."""
    app = tm.app1()
    ro, gt = app.ro, app.gt
    cfg = ds["config"]
    level = cfg["inversion_mesh_level"]
    if level != tc.INVERSION_MESH_LEVEL:
        raise ValueError(f"inversion mesh level {level!r} != {tc.INVERSION_MESH_LEVEL!r}")
    md = tm.build_transport_mesh_data(level)
    for label, got, want in (("mesh", md["mesh_sha256"], cfg["mesh"]["mesh_sha256"]),
                             ("geometry", md["geometry_sha256"], cfg["mesh"]["geometry_sha256"]),
                             ("velocity", md["velocity_sha256"], cfg["velocity"]["velocity_sha256"])):
        if got != want:
            raise ValueError(f"{label} sha256 {got[:12]} differs from the dataset's {want[:12]}")
    tr = cfg["transport"]
    pde = tm.build_pde(level, md, k=tr["diffusivity_k"], dt=tr["dt"], tfinal=tr["tfinal"], sigma2=tr["sigma2"])
    obs = ds["observations"]
    n_frames = tm.num_steps(pde["dt"], pde["tfinal"]) + 1
    if obs.shape[0] != n_frames:
        raise ValueError(f"dataset has {obs.shape[0]} frames; the coarse solver saves {n_frames}")
    if not np.allclose(ds["times"], np.arange(n_frames) * pde["dt"], rtol=0.0, atol=1e-9):
        raise ValueError("observation times are not n * dt of the coarse solver")
    if ds["in_objective"][0] or not ds["in_objective"][1:].all() or cfg["objective"]["time_rule"] != "exclude_first":
        raise ValueError("dataset objective mask is not 'exclude_first'")
    node_ids = np.asarray(cfg["sensors"]["node_ids"], dtype=int)
    if not np.array_equal(node_ids, ds["node_ids"]):
        raise ValueError("sensor node ids of the configuration and the data file differ")
    if not np.array_equal(md["coord"][node_ids], np.asarray(cfg["sensors"]["coordinates"])):
        raise ValueError("sensor coordinates are not the coarse-mesh node coordinates")
    if np.intersect1d(node_ids, md["clean_air_dirichlet_nodes"]).size:
        raise ValueError("a sensor sits on a clean-air Dirichlet node")

    weights = np.asarray(cfg["objective"]["sensor_weights"] if sensor_weights is None else sensor_weights,
                         dtype=np.float64)
    observation = app.obs_op.NodeObservation(
        node_ids, weights, coords=md["coord"][node_ids],
        description=("diagonal sensor weights W = diag(w)" if weights.ndim == 1 else "symmetric sensor weight matrix W"),
        selection="application2_fixed_sensor_layout")
    md["observation"] = observation

    target = np.zeros((n_frames, md["coord"].shape[0]))
    target[:, node_ids] = obs                      # d = C u_d: only the sensor columns are ever read

    eta1 = float(cfg["objective"]["eta1"] if eta1 is None else eta1)
    lam = float(cfg["objective"]["lambda_reg"])
    ro.PDE.clear()
    ro.PDE.update(pde)
    ro.ETA1, ro.LAMBDA_REG = eta1, lam
    ro.OBJECTIVE_TIME_RULE, ro.GRADIENT_TIME_RULE = cfg["objective"]["time_rule"], tc.GRADIENT_TIME_RULE

    admissibility = app.MeshAdmissibility(md)
    adm = cfg["admissible_set"]
    box = admissibility.bounding_box
    if not np.allclose(box, adm["center_box_bounds_xmin_xmax_ymin_ymax"]):
        raise ValueError("mesh bounding box differs from the dataset's centre box")
    amp = tuple(float(v) for v in adm["amplitude_bounds"])
    m = int(cfg["num_sources"])
    bounds = [b for _ in range(m) for b in ((box[0], box[1]), (box[2], box[3]), amp)]

    r = obs[1:]
    wr = r * weights[None, :] if weights.ndim == 1 else r @ weights
    j_zero = 0.5 * eta1 * pde["dt"] * float(np.sum(wr * r))
    counter = SolveCounter(gt)
    log(f"problem: case {cfg['case_label']} ({ds['kind']} data), coarse mesh {md['mesh_sha256'][:12]}, "
        f"{md['coord'].shape[0]} nodes, velocity {md['velocity_sha256'][:12]}, {node_ids.size} sensors, "
        f"{n_frames - 1} observation times, dt = {pde['dt']} s, k = {pde['k']}, sigma2 = {pde['sigma2']}, "
        f"eta1 = {eta1}, J(zero source) = {j_zero:.10e}")
    return {"app": app, "mesh_data": md, "pde": ro.PDE, "target": target, "observation": observation,
            "weights": weights, "admissibility": admissibility, "bounds": bounds, "amplitude_bounds": amp,
            "num_sources": m, "eta1": eta1, "lam": lam, "time_rule": cfg["objective"]["time_rule"],
            "j_zero": j_zero, "counter": counter, "n_frames": n_frames}


def build_evaluator(problem: dict, z_template: np.ndarray):
    ro = problem["app"].ro
    stats = ro.new_feasibility_stats("raw")
    _cache, history, objective, gradient, callback = ro.build_evaluator(
        problem["mesh_data"], problem["target"], MODE, ro.MESH_DOMAIN_BOUND_CASE, {},
        np.asarray(z_template, dtype=np.float64), problem["admissibility"],
        feasibility_mode="raw", feasibility_stats=stats)
    return history, objective, gradient, callback, stats


def gn_hessian(problem: dict, z_template: np.ndarray):
    app = problem["app"]
    return app.pio.build_active_hessian(
        method=app.pio.GAUSS_NEWTON, mode=MODE, num_sources=problem["num_sources"],
        mesh_data=problem["mesh_data"], pde=problem["pde"], target_state=problem["target"],
        z_template=np.asarray(z_template, dtype=np.float64), fixed_values={},
        eta1=problem["eta1"], lam=problem["lam"], unpack_active=app.ro.unpack_active,
        hessian_module=app.ro._hessian_utils())


# --------------------------------------------------------------------------- #
# Initializer (Application1 module, optionally through the three-literal shim)
# --------------------------------------------------------------------------- #
def load_initializer(args, problem: dict, log=print):
    app = problem["app"]
    path = tm.APP1_DIR / "sensitivity_initialization.py"
    source = path.read_text()
    record = {"application1_file": str(path), "application1_file_sha256": hashlib.sha256(source.encode()).hexdigest()}
    if args.init_app1_literal_tolerances:
        record.update({"shim": False, "nelder_mead_xatol": APP1_NM_XATOL_M, "nelder_mead_fatol": 1.0,
                       "dedup_floor": APP1_DEDUP_FLOOR_M})
        return app.sinit, record
    radius = float(np.sqrt(problem["pde"]["sigma2"]))
    xatol = (APP1_NM_XATOL_M / APP1_GAUSSIAN_RADIUS_M * radius if args.init_refine_xatol is None
             else float(args.init_refine_xatol))
    dedup = (APP1_DEDUP_FLOOR_M / APP1_GAUSSIAN_RADIUS_M * radius if args.init_dedup_floor is None
             else float(args.init_dedup_floor))
    fatol = float(args.init_refine_fatol_rel) * float(problem["j_zero"])
    if source.count(SHIM_NM) != 2 or source.count(SHIM_DEDUP) != 1:
        raise RuntimeError(
            "Application1/sensitivity_initialization.py no longer contains the three tolerance literals this "
            "shim replaces; review the file (or pass --init-app1-literal-tolerances)")
    patched = source.replace(SHIM_NM, 'options={"xatol": _APP2_NM_XATOL, "fatol": _APP2_NM_FATOL, "maxfev": ')
    patched = patched.replace(SHIM_DEDUP,
                              "dedup_radius = max(_APP2_DEDUP_FLOOR, 0.25 * float(np.sqrt(sigma2)))")
    name = "application2_sensitivity_initialization"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module._APP2_NM_XATOL, module._APP2_NM_FATOL, module._APP2_DEDUP_FLOOR = xatol, fatol, dedup
    sys.modules[name] = module
    with warnings.catch_warnings():               # docstring escapes of the Application1 source ("\int")
        warnings.simplefilter("ignore", SyntaxWarning)
        code = compile(patched, f"{path} [Application2 tolerance shim]", "exec")
    exec(code, module.__dict__)
    record.update({
        "shim": True, "nelder_mead_xatol": xatol, "nelder_mead_fatol": fatol, "dedup_floor": dedup,
        "fatol_rule": f"{args.init_refine_fatol_rel:g} * J(zero source)",
        "replaced_literals": [SHIM_NM + "...  (2 occurrences)", SHIM_DEDUP],
        "rationale": "Application1 literals rescaled by the Gaussian-radius ratio "
                     f"{radius:g} m / {APP1_GAUSSIAN_RADIUS_M:g} m and by the objective scale; nothing else differs",
    })
    log(f"initializer shim: Nelder-Mead xatol = {xatol:g} m, fatol = {fatol:.3e}, dedup floor = {dedup:g} m")
    return module, record


def initializer_kwargs(args, problem: dict) -> dict:
    """The Application1 M = 2 production sensitivity preset (run_sensor_layout_comparison.sh)."""
    return dict(
        eta1=problem["eta1"], lam=problem["lam"], objective_time_rule=problem["time_rule"],
        min_separation=args.sensitivity_min_separation, min_value_tol=0.0,
        boundary_buffer=float(args.sensitivity_mask_boundary_buffer), building_buffer=0.0,
        save_plots=False, plot_dir=None, plot_context=None,
        score="predicted_reduction", top_k=int(args.sensitivity_top_k), beam_width=int(args.sensitivity_beam_width),
        refine_centers=True, coherence_cap=float(args.sensitivity_coherence_cap),
        refine_max_fev=int(args.sensitivity_refine_max_fev), refine_pair=True,
        pair_refine_max_fev=int(args.sensitivity_pair_refine_max_fev),
        geometry_aware_branches=True, obstacle_aware_diversity=True,
        reselect_second=True, reselect_threshold=0.25, reselect_max_passes=2,
        amplitude_solver=args.amplitude_solver, true_centers_for_plots=None,
    )


def candidate_summary(sinit, problem: dict, boundary_buffer: float) -> dict:
    """Admissible candidate centres of the initializer (mask only; no search)."""
    md, adm = problem["mesh_data"], problem["admissibility"]
    coord = md["coord"]
    mask = sinit._base_admissibility_mask(
        coord, adm, sinit._outer_boundary_node_coords(md, coord),
        sinit._building_boundary_node_coords(md, coord), float(boundary_buffer), 0.0)
    geo = tm.geo
    x, y = coord[mask, 0], coord[mask, 1]
    regions = {f"bay_{b}": int(np.count_nonzero((x >= geo.BAYS[b]["x"][0]) & (x <= geo.BAYS[b]["x"][1])
                                                 & (y > geo.PARTITION_BOTTOM_Y))) for b in geo.BAYS}
    regions["aisle"] = int(np.count_nonzero((y <= geo.PARTITION_BOTTOM_Y) & (x <= geo.BATH_WALL_X[0])))
    regions["bathroom_and_doorway"] = int(np.count_nonzero(x > geo.BATH_WALL_X[0]))
    return {"candidate_dictionary": "all coarse-mesh nodes passing the fluid-mesh admissibility mask",
            "num_mesh_nodes": int(coord.shape[0]), "num_admissible_candidates": int(mask.sum()),
            "boundary_buffer_m": float(boundary_buffer), "candidates_by_region": regions,
            "restricted_to_occupied_bays": False}


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def control_feasibility(problem: dict, z) -> list:
    adm = problem["admissibility"]
    out = []
    for i, (x, y, a) in enumerate(np.asarray(z, dtype=np.float64).reshape(-1, 3), 1):
        inside = bool(adm.is_point_in_domain(float(x), float(y)))
        out.append({"source": i, "center": [float(x), float(y)], "amplitude": float(a), "in_fluid_mesh": inside,
                    "signed_clearance_m": float(adm.signed_clearance(float(x), float(y)))})
    return out


def base_record(args, ds: dict, problem: dict) -> dict:
    md = problem["mesh_data"]
    return {
        "application": "Application2", "dataset_version": ds["config"]["dataset_version"],
        "case_label": ds["config"]["case_label"], "data_kind": ds["kind"], "stage": args.stage,
        "uses_truth": False,
        "inputs": {"inversion_config": str(ds["config_path"]), "observations_file": str(ds["path"]),
                   "observations_file_sha256": ds["file_sha256"], "observations_sha256": ds["observations_sha256"]},
        "mesh": {"level": md["mesh_level"], "file": md["mesh_file"], "sha256": md["mesh_sha256"],
                 "geometry_sha256": md["geometry_sha256"], "nodes": int(md["coord"].shape[0]),
                 "triangles": int(md["connections"].shape[0])},
        "velocity": {"file": md["velocity_file"], "sha256": md["velocity_sha256"]},
        "pde": {k: v for k, v in problem["pde"].items() if k != "velocity_nodes"},
        "objective": {"eta1": problem["eta1"], "lambda_reg": problem["lam"], "time_rule": problem["time_rule"],
                      "sensor_weights": problem["weights"], "weight_kind": problem["observation"].weight_kind},
        "sensors": {"node_ids": problem["observation"].node_ids, "coordinates": problem["observation"].coords},
        "num_sources": problem["num_sources"], "bounds": problem["bounds"],
        "clean_air_dirichlet": md["dirichlet_alias_note"],
        "objective_zero_source": problem["j_zero"],
        "argv": sys.argv[1:],
    }


def stage_wiring(args, ds, problem, out: Path, log) -> dict:
    app, counter = problem["app"], problem["counter"]
    record = base_record(args, ds, problem)
    z = np.asarray(args.initial_control if args.initial_control is not None else default_wiring_control(problem))
    record["evaluation_control"] = z
    record["evaluation_control_feasibility"] = control_feasibility(problem, z)
    # J(0) through the shared objective code == the direct formula on the saved data
    j_zero_shared = float(app.gt.compute_objective(
        np.zeros_like(problem["target"]), problem["target"], problem["mesh_data"], eta1=problem["eta1"],
        lam=0.0, z=np.zeros_like(z), dt_out=problem["pde"]["dt"], time_rule=problem["time_rule"]))
    record["objective_zero_source_shared_code"] = j_zero_shared
    record["objective_zero_source_agrees"] = bool(abs(j_zero_shared - problem["j_zero"])
                                                  <= 1e-12 * max(1.0, abs(problem["j_zero"])))
    _history, objective, gradient, _callback, _stats = build_evaluator(problem, z)
    a = counter.snapshot()
    j = float(objective(z))
    b = counter.snapshot()
    g = np.asarray(gradient(z), dtype=np.float64)
    c = counter.snapshot()
    record["objective"] = j
    record["gradient"] = g
    record["objective_cost"] = counter.delta(a, b)
    record["gradient_cost"] = counter.delta(b, c)
    log(f"wiring: J = {j:.10e}, |g| = {np.linalg.norm(g):.6e}  (forward {record['objective_cost']['seconds']:.1f} s, "
        f"adjoint gradient {record['gradient_cost']['seconds']:.1f} s)")
    sinit, shim_record = load_initializer(args, problem, log)
    record["initializer"] = {"entry_point": f"{sinit.__name__}.build_sensitivity_initial_control",
                             "tolerances": shim_record,
                             "amplitude_solver": sinit.require_amplitude_solver(args.amplitude_solver),
                             "candidates": candidate_summary(sinit, problem, args.sensitivity_mask_boundary_buffer),
                             "keyword_arguments": {k: v for k, v in initializer_kwargs(args, problem).items()},
                             "executed": False}
    record["optimizer"] = optimizer_settings(args, problem)
    record["optimizer"]["executed"] = False
    if args.check_gn_scaling:
        d = counter.snapshot()
        scale, scale_record = app.pio.gn_diagonal_scaling(gn_hessian(problem, z), z,
                                                          max_ratio=args.post_init_scaling_max_ratio)
        e = counter.snapshot()
        record["gn_diagonal_at_evaluation_control"] = {**scale_record, "cost": counter.delta(d, e),
                                                       "note": "wiring check of the scaling code path only"}
        log(f"wiring: GN diagonal D = {np.array2string(scale, precision=4)}, cond "
            f"{scale_record['condition_number_unscaled']:.3e} -> {scale_record['condition_number_scaled']:.3e}")
    record["total_counted"] = dict(counter.counts)
    tm.write_json(out / "wiring_check.json", record)
    return record


def default_wiring_control(problem: dict) -> np.ndarray:
    """A feasible control built from geometry and bounds only (two aisle points, mid-range amplitudes)."""
    lo, hi = problem["amplitude_bounds"]
    geo = tm.geo
    pts = [(geo.PARTITIONS["partition_1"][0] + 0.05, 1.5), (geo.PARTITIONS["partition_2"][0] + 0.05, 1.5)]
    z = []
    for x, y in pts[: problem["num_sources"]]:
        z += [x, y, 0.5 * (lo + hi)]
    return np.asarray(z)


def optimizer_settings(args, problem: dict) -> dict:
    options = {"ftol": 1.0e-14, "gtol": 1.0e-9, "maxls": 20, "disp": True}
    if args.post_init_lbfgsb_maxcor is not None:
        options["maxcor"] = int(args.post_init_lbfgsb_maxcor)
    return {"entry_point": "post_init_optimizers.run_lbfgsb_polish", "method": "L-BFGS-B",
            "scaling": args.post_init_scaling, "scaling_max_ratio": args.post_init_scaling_max_ratio,
            "max_iter": int(args.max_iter), "tol": 1.0e-8, "options": options, "feasibility_mode": "raw",
            "bounds": problem["bounds"]}


def stage_init(args, ds, problem, out: Path, log) -> dict:
    counter = problem["counter"]
    sinit, shim_record = load_initializer(args, problem, log)
    a = counter.snapshot()
    control, _diagnostics, extras = sinit.build_sensitivity_initial_control(
        problem["mesh_data"], problem["pde"], problem["target"], problem["num_sources"],
        problem["amplitude_bounds"], problem["admissibility"], **initializer_kwargs(args, problem))
    b = counter.snapshot()
    control = np.asarray(control, dtype=np.float64)
    record = {
        "initial_control": control, "feasibility": control_feasibility(problem, control),
        "tolerances": shim_record, "cost": counter.delta(a, b),
        "initializer_reported": {k: extras.get(k) for k in (
            "zero_source_objective", "initial_exact_objective", "n_forward_solves", "n_adjoint_solves",
            "n_candidate_traces", "wall_time_seconds", "source_levels", "n_dedup_removed", "branch_controls",
            "pair_refinement", "second_source_reselection")},
        "keyword_arguments": initializer_kwargs(args, problem),
    }
    tm.write_json(out / "initialization.json", record)
    log(f"initialization: control = {np.array2string(control, precision=8)}  "
        f"({record['cost']['forward_solves']} forward / {record['cost']['adjoint_solves']} adjoint sweeps, "
        f"{record['cost']['seconds']:.0f} s)")
    return record


def stage_full(args, ds, problem, out: Path, log) -> dict:
    app, counter = problem["app"], problem["counter"]
    pio = app.pio
    record = base_record(args, ds, problem)
    if args.initial_control is not None:
        z0 = np.asarray(args.initial_control, dtype=np.float64)
        record["initialization"] = {"mode": "manual", "initial_control": z0,
                                    "cost": {"forward_solves": 0, "adjoint_solves": 0, "seconds": 0.0}}
    else:
        record["initialization"] = {"mode": "sensitivity", **stage_init(args, ds, problem, out, log)}
        z0 = np.asarray(record["initialization"]["initial_control"], dtype=np.float64)
    history, objective, gradient, callback, stats = build_evaluator(problem, z0)
    settings = optimizer_settings(args, problem)
    record["optimizer"] = settings

    scaling_cost = {"forward_solves": 0, "adjoint_solves": 0, "seconds": 0.0}
    original_gn = pio.gn_diagonal_scaling

    def timed_gn(*a, **k):
        s0 = counter.snapshot()
        try:
            return original_gn(*a, **k)
        finally:
            d = counter.delta(s0, counter.snapshot())
            for key in scaling_cost:
                scaling_cost[key] += d[key]

    pio.gn_diagonal_scaling = timed_gn
    block0 = counter.snapshot()
    try:
        j0 = float(objective(z0))
        g0 = np.asarray(gradient(z0), dtype=np.float64)
        history.append({"iteration": 0, "objective": j0, "gradient_norm": float(np.linalg.norm(g0)),
                        "control": z0.copy()})
        hess = gn_hessian(problem, z0) if args.post_init_scaling != pio.SCALING_NONE else None
        result, scaling_record = pio.run_lbfgsb_polish(
            optimizer_cls=app.Optimizer, objective=objective, gradient=gradient, x0=z0,
            bounds=problem["bounds"], max_iter=int(args.max_iter), tol=settings["tol"],
            options=settings["options"], callback=callback, mesh_data=problem["mesh_data"],
            problem_data={"pde": problem["pde"], "mode": MODE}, scaling=args.post_init_scaling,
            stall_iters=pio.DEFAULT_STALL_ITERS, max_ratio=float(args.post_init_scaling_max_ratio), gn_hess=hess)
    finally:
        pio.gn_diagonal_scaling = original_gn
    z = np.asarray(result.x, dtype=np.float64)
    jf = float(objective(z))
    gf = np.asarray(gradient(z), dtype=np.float64)
    block = counter.delta(block0, counter.snapshot())
    record["result"] = {
        "final_control": z, "final_objective": jf, "final_gradient": gf,
        "final_gradient_norm": float(np.linalg.norm(gf)), "initial_objective": j0,
        "iterations": int(result.nit), "nfev": int(getattr(result, "nfev", 0) or 0),
        "njev": int(getattr(result, "njev", 0) or 0), "success": bool(result.success),
        "message": str(result.message), "feasibility": control_feasibility(problem, z),
        "raw_feasibility_stats": stats, "scaling": scaling_record,
    }
    record["solve_accounting"] = {
        "initialization": record["initialization"]["cost"],
        "scaling_gn_diagonal": scaling_cost,
        "optimization_lbfgsb": {k: block[k] - scaling_cost[k] for k in block},
        "total": {"forward_solves": counter.counts["forward"], "adjoint_solves": counter.counts["adjoint"]},
        "definition": "actual gt.run_forward / gt.rk4_discrete_source_adjoint sweeps; optimization includes the "
                      "objective/gradient at the initial control and the final re-evaluation (cache hits cost 0)",
    }
    tm.write_json(out / "inversion_summary.json", record)
    with open(out / "optimization_history.csv", "w", newline="") as f:
        w = csv.writer(f)
        names = [n for i in range(1, problem["num_sources"] + 1) for n in (f"x{i}", f"y{i}", f"A{i}")]
        w.writerow(["iteration", "objective", "gradient_norm"] + names)
        for h in history:
            w.writerow([h["iteration"], repr(h["objective"]), repr(h["gradient_norm"])]
                       + [repr(float(v)) for v in h["control"]])
    log(f"final: J = {jf:.10e}, |g| = {np.linalg.norm(gf):.3e}, control = {np.array2string(z, precision=8)}, "
        f"{result.nit} iterations, {result.message}")
    return record


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, default=tm.DATASET_DIR)
    ap.add_argument("--case", required=True, choices=list(tc.CASES))
    ap.add_argument("--data", required=True, choices=("clean", "noisy"))
    ap.add_argument("--stage", default="wiring", choices=("wiring", "init", "full"))
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <dataset>/case_XX/inversion_<data>[_<tag>]")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--initial-control", type=float, nargs="+", default=None,
                    help="manual start [x1 y1 A1 x2 y2 A2] (skips the sensitivity initialization)")
    ap.add_argument("--sensor-weight", type=float, default=None, help="scalar w: W = w * I (default: dataset's W)")
    ap.add_argument("--sensor-weights-file", type=Path, default=None,
                    help=".npy with a length-Ns diagonal or an (Ns, Ns) symmetric W")
    ap.add_argument("--eta1", type=float, default=None, help="objective scale (default: dataset's)")
    # sensitivity initialization (Application1 M = 2 production preset)
    ap.add_argument("--amplitude-solver", default="osqp", choices=("osqp", "enumeration"))
    ap.add_argument("--sensitivity-top-k", type=int, default=24)
    ap.add_argument("--sensitivity-beam-width", type=int, default=3)
    ap.add_argument("--sensitivity-refine-max-fev", type=int, default=60)
    ap.add_argument("--sensitivity-pair-refine-max-fev", type=int, default=60)
    ap.add_argument("--sensitivity-coherence-cap", type=float, default=0.3)
    ap.add_argument("--sensitivity-min-separation", type=float, default=None)
    ap.add_argument("--sensitivity-mask-boundary-buffer", type=float, default=0.0,
                    help="exclude candidate centres closer than this [m] to any wall/bed/partition node")
    ap.add_argument("--init-refine-xatol", type=float, default=None,
                    help="Nelder-Mead xatol [m] (default 0.025 * sqrt(sigma2), Application1's ratio)")
    ap.add_argument("--init-refine-fatol-rel", type=float, default=DEFAULT_FATOL_REL,
                    help="Nelder-Mead fatol as a fraction of J(zero source)")
    ap.add_argument("--init-dedup-floor", type=float, default=None,
                    help="branch de-duplication floor [m] (default 0.1 * sqrt(sigma2), Application1's ratio)")
    ap.add_argument("--init-app1-literal-tolerances", action="store_true",
                    help="use Application1's hard-coded 0.5 m / 1.0 / 2.0 m unchanged (no shim)")
    # local optimization (Application1 GN-diagonal configuration)
    ap.add_argument("--post-init-scaling", default="gn-diag", choices=("none", "gn-diag", "on-stall"))
    ap.add_argument("--post-init-scaling-max-ratio", type=float, default=1.0e4)
    ap.add_argument("--post-init-lbfgsb-maxcor", type=int, default=30)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--check-gn-scaling", action="store_true",
                    help="wiring stage: also evaluate the Gauss-Newton diagonal at the evaluation control")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    ds = load_dataset(args.dataset_dir, args.case, args.data)
    out = args.out_dir or (ds["case_dir"] / ("inversion_" + args.data + (f"_{args.tag}" if args.tag else "")))
    if args.stage != "wiring" and out.exists() and any(out.iterdir()):
        print(f"refusing: {out} already holds results (use --tag)", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / f"{args.stage}.log", "a")

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    weights = None
    if args.sensor_weights_file is not None:
        weights = np.load(args.sensor_weights_file)
    elif args.sensor_weight is not None:
        weights = np.full(len(ds["config"]["sensors"]["node_ids"]), float(args.sensor_weight))
    problem = build_problem(ds, sensor_weights=weights, eta1=args.eta1, log=log)
    if args.initial_control is not None and len(args.initial_control) != 3 * problem["num_sources"]:
        raise ValueError(f"--initial-control needs {3 * problem['num_sources']} values")
    {"wiring": stage_wiring, "init": stage_init, "full": stage_full}[args.stage](args, ds, problem, out, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
