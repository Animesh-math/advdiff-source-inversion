"""Optimization driver for Test Application2 with multi-source Gaussian control."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import meshio

import gradient_test as gt
from multi_source_utils import (
    NUM_SOURCES,
    active_bounds_for_mode,
    build_initial_control,
    build_reference_control,
    component_names,
    mode_fixed_values,
    pack_active,
    restrict_gradient,
    reshape_control,
    unpack_active,
)


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
OPTIMIZER_DIR = PROJECT_ROOT / "Src" / "Optimizers"

if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_OPT = _load_module_from_file("optimizer_mod", OPTIMIZER_DIR / "optimizer.py")
Optimizer = _OPT.Optimizer


@dataclass
class EvalCache:
    z_active: np.ndarray | None = None
    z_full: np.ndarray | None = None
    state_out: np.ndarray | None = None
    j_tracking: float | None = None
    j_reg: float | None = None
    j_total: float | None = None
    g_full: np.ndarray | None = None


def ensure_visuals_dir() -> dict[str, Path]:
    base_dir = APP_DIR / "visuals"
    paths = {
        "base": base_dir,
        "optimized_state": base_dir / "optimized_state",
        "reference_state": base_dir / "reference_state",
        "initial_state": base_dir / "initial_state",
        "source_paths": base_dir / "source_paths",
        "traces": base_dir / "traces",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_pvd_collection(pvd_path: Path, entries: list[tuple[float, str]]) -> None:
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>',
    ]
    for t, rel_file in entries:
        lines.append(f'    <DataSet timestep="{t:.12g}" group="" part="0" file="{rel_file}"/>')
    lines.extend(['  </Collection>', '</VTKFile>'])
    pvd_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mesh_points_and_cells(mesh_data):
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)
    connections = np.asarray(mesh_data["connections"], dtype=int)
    if coord.ndim != 2 or coord.shape[1] not in (2, 3):
        raise ValueError(f"Expected mesh_data['coord'] with shape (n, 2) or (n, 3), got {coord.shape}")
    if connections.ndim != 2 or connections.shape[1] != 3:
        raise ValueError(f"Expected triangular mesh_data['connections'] with shape (m, 3), got {connections.shape}")

    points = coord if coord.shape[1] == 3 else np.column_stack([coord, np.zeros(coord.shape[0])])
    return points, [("triangle", connections)]


def _state_frames(state_out: np.ndarray, num_points: int) -> np.ndarray:
    state = np.asarray(state_out, dtype=np.float64)
    if state.ndim == 1:
        if state.size != num_points:
            raise ValueError(f"1D state has {state.size} values, expected {num_points} mesh points")
        return state.reshape(1, num_points)
    if state.ndim != 2:
        raise ValueError(f"Expected state array with 1 or 2 dimensions, got shape {state.shape}")
    if state.shape[1] == num_points:
        return state
    if state.shape[0] == num_points:
        return state.T
    raise ValueError(
        "Could not align state array with mesh points: "
        f"state shape {state.shape}, num_points {num_points}"
    )


def export_state_sequence_to_vtu(
    state_out: np.ndarray,
    mesh_data,
    out_dir: Path,
    prefix: str,
    dt_out: float,
    max_frames: int = 150,
) -> Path:
    points, cells = _mesh_points_and_cells(mesh_data)
    frames = _state_frames(state_out, points.shape[0])

    for stale in out_dir.glob(f"{prefix}_*.vtu"):
        stale.unlink()

    if max_frames <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames}")

    num_frames = frames.shape[0]
    if num_frames <= max_frames:
        export_indices = np.arange(num_frames, dtype=int)
    else:
        export_indices = np.unique(
            np.linspace(0, num_frames - 1, max_frames, dtype=int)
        )

    pvd_entries: list[tuple[float, str]] = []
    for out_index, iframe in enumerate(export_indices):
        values = frames[iframe]
        vtu_name = f"{prefix}_{out_index:06d}.vtu"
        vtu_path = out_dir / vtu_name
        mesh = meshio.Mesh(points=points, cells=cells, point_data={"u": values})
        meshio.write(vtu_path, mesh)
        pvd_entries.append((float(iframe * dt_out), vtu_name))

    pvd_path = out_dir / f"{prefix}.pvd"
    write_pvd_collection(pvd_path, pvd_entries)
    print(
        f"{prefix}: exported {len(export_indices)} of {frames.shape[0]} frames "
        f"from state shape {np.asarray(state_out).shape}"
    )
    return pvd_path


def write_control_history(history: list[dict], traces_dir: Path) -> tuple[Path, Path]:
    names = component_names()
    csv_path = traces_dir / "control_history.csv"
    npy_path = traces_dir / "control_history.npy"
    header = ["iter", "J", *names]

    rows = []
    for item in history:
        z_full = np.asarray(item["z_full"], dtype=np.float64)
        rows.append([float(item["iter"]), float(item["J"]), *z_full.tolist()])

    rows_array = np.asarray(rows, dtype=np.float64)
    np.savetxt(
        csv_path,
        rows_array,
        delimiter=",",
        header=",".join(header),
        comments="",
    )
    np.save(npy_path, rows_array)
    return csv_path, npy_path


def write_source_paths(history: list[dict], source_dir: Path) -> list[Path]:
    for stale in source_dir.glob("source_*_path.vtu"):
        stale.unlink()

    csv_path = source_dir / "source_paths.csv"
    lines = ["iter,source_id,x,y,A"]

    by_source = {i: [] for i in range(NUM_SOURCES)}
    for item in history:
        iteration = int(item["iter"])
        Z = reshape_control(np.asarray(item["z_full"], dtype=np.float64))
        for i, (x, y, amp) in enumerate(Z):
            lines.append(f"{iteration},{i + 1},{x:.16e},{y:.16e},{amp:.16e}")
            by_source[i].append((iteration, float(x), float(y), float(amp)))

    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    path_files = [csv_path]
    for i, values in by_source.items():
        if not values:
            continue
        points = np.asarray([[x, y, 0.0] for _, x, y, _ in values], dtype=np.float64)
        if len(values) >= 2:
            line_cells = np.column_stack(
                [
                    np.arange(len(values) - 1, dtype=int),
                    np.arange(1, len(values), dtype=int),
                ]
            )
            cells = [("line", line_cells)]
        else:
            cells = [("vertex", np.array([[0]], dtype=int))]
        mesh = meshio.Mesh(
            points=points,
            cells=cells,
            point_data={
                "iteration": np.asarray([it for it, _, _, _ in values], dtype=np.float64),
                "A": np.asarray([amp for _, _, _, amp in values], dtype=np.float64),
            },
        )
        vtu_path = source_dir / f"source_{i + 1}_path.vtu"
        meshio.write(vtu_path, mesh)
        path_files.append(vtu_path)

    return path_files


def write_final_sources(
    traces_dir: Path,
    z_opt_full: np.ndarray,
    z0_full: np.ndarray,
    z_ref_full: np.ndarray,
    j_opt: float,
    nit: int,
) -> Path:
    Zopt = reshape_control(z_opt_full)
    data = {
        "optimized_control": np.asarray(z_opt_full, dtype=np.float64).tolist(),
        "optimized_sources": [
            {"source_id": i + 1, "x": float(x), "y": float(y), "A": float(amp)}
            for i, (x, y, amp) in enumerate(Zopt)
        ],
        "initial_control": np.asarray(z0_full, dtype=np.float64).tolist(),
        "reference_control": np.asarray(z_ref_full, dtype=np.float64).tolist(),
        "final_objective": float(j_opt),
        "num_iterations": int(nit),
    }
    json_path = traces_dir / "final_sources.json"
    json_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return json_path


def export_visualization_outputs(
    mesh_data,
    pde,
    z_opt_full: np.ndarray,
    z_ref_full: np.ndarray,
    z0_full: np.ndarray,
    optimized_state_out: np.ndarray,
    reference_state_out: np.ndarray,
    initial_state_out: np.ndarray,
    control_history: list[dict],
    j_opt: float,
    nit: int,
    max_paraview_frames: int = 150,
) -> list[Path]:
    paths = ensure_visuals_dir()
    dt_out = pde["dt"] * pde["jprint"]

    exported = [
        export_state_sequence_to_vtu(
            optimized_state_out,
            mesh_data,
            paths["optimized_state"],
            "optimized_state",
            dt_out,
            max_frames=max_paraview_frames,
        ),
        export_state_sequence_to_vtu(
            reference_state_out,
            mesh_data,
            paths["reference_state"],
            "reference_state",
            dt_out,
            max_frames=max_paraview_frames,
        ),
        export_state_sequence_to_vtu(
            initial_state_out,
            mesh_data,
            paths["initial_state"],
            "initial_state",
            dt_out,
            max_frames=max_paraview_frames,
        ),
    ]

    control_csv, control_npy = write_control_history(control_history, paths["traces"])
    exported.extend([control_csv, control_npy])
    exported.extend(write_source_paths(control_history, paths["source_paths"]))
    exported.append(
        write_final_sources(
            paths["traces"],
            z_opt_full=z_opt_full,
            z0_full=z0_full,
            z_ref_full=z_ref_full,
            j_opt=j_opt,
            nit=nit,
        )
    )

    print("\nVisualization files written to:")
    for path in exported:
        print(f"  {path}")
    return exported


def choose_control_mode() -> str:
    print("Choose control mode:")
    print("  1. location_only")
    print("  2. intensity_only")
    print("  3. location_and_intensity")
    choice = input("Enter 1, 2, or 3: ").strip()

    mapping = {
        "1": "location_only",
        "2": "intensity_only",
        "3": "location_and_intensity",
        "": "location_and_intensity",
    }
    if choice in mapping:
        mode = mapping[choice]
    else:
        print("Invalid input. Defaulting to location_and_intensity.")
        mode = "location_and_intensity"

    print(f"Selected CONTROL_MODE: {mode}")
    return mode


def run_optimization(
    mesh_data,
    pde,
    mode: str,
    z_ref_full: np.ndarray,
    z0_full: np.ndarray,
    fixed_values,
    eta1: float,
    lambda_reg: float,
    objective_time_rule: str,
    gradient_time_rule: str,
):
    z_ref_full = np.asarray(z_ref_full, dtype=np.float64)
    z0_full = np.asarray(z0_full, dtype=np.float64)
    dt_out = pde["dt"] * pde["jprint"]
    ud_out = gt.run_forward(mesh_data, pde, z_ref_full)
    z0_active = pack_active(z0_full, mode)
    z_ref_active = pack_active(z_ref_full, mode)
    bounds = active_bounds_for_mode(mode)

    cache = EvalCache()
    iter_counter = 0
    control_history: list[dict] = []

    problem_data = {
        "pde": pde,
        "mode": mode,
        "z_ref_full": z_ref_full,
        "z0_full": z0_full,
        "fixed_values": fixed_values,
        "eta1": float(eta1),
        "lambda_reg": float(lambda_reg),
        "objective_time_rule": objective_time_rule,
        "gradient_time_rule": gradient_time_rule,
        "dt_out": dt_out,
        "ud_out": ud_out,
    }

    def same_active(z_active: np.ndarray) -> bool:
        if cache.z_active is None:
            return False
        return np.array_equal(np.asarray(z_active, dtype=np.float64), cache.z_active)

    def ensure_eval(z_active: np.ndarray) -> None:
        nonlocal cache
        za = np.asarray(z_active, dtype=np.float64)
        if same_active(za):
            return

        z_full = unpack_active(za, z0_full, mode, fixed_values)
        state_out = gt.run_forward(mesh_data, pde, z_full)
        j_track, j_reg, j_total = gt.compute_objective_parts(
            state_out,
            ud_out,
            mesh_data,
            eta1=eta1,
            lam=lambda_reg,
            z=z_full,
            dt_out=dt_out,
            time_rule=objective_time_rule,
        )

        cache = EvalCache(
            z_active=za.copy(),
            z_full=z_full.copy(),
            state_out=state_out,
            j_tracking=float(j_track),
            j_reg=float(j_reg),
            j_total=float(j_total),
            g_full=None,
        )

    def ensure_grad(z_active: np.ndarray) -> None:
        ensure_eval(z_active)
        if cache.g_full is not None:
            return

        _, _, g_total, _ = gt.compute_gradient_parts(
            cache.z_full,
            cache.state_out,
            ud_out,
            mesh_data,
            pde,
            eta1=eta1,
            lam=lambda_reg,
            gradient_time_rule=gradient_time_rule,
        )
        cache.g_full = np.asarray(g_total, dtype=np.float64)

    def objective_active(z_active: np.ndarray, _mesh_data, _problem_data) -> float:
        ensure_eval(z_active)
        return float(cache.j_total)

    def gradient_active(z_active: np.ndarray, _mesh_data, _problem_data) -> np.ndarray:
        ensure_grad(z_active)
        return restrict_gradient(cache.g_full, mode)

    def iteration_callback(z_active: np.ndarray) -> None:
        nonlocal iter_counter
        iter_counter += 1
        j = objective_active(z_active, mesh_data, problem_data)
        g = gradient_active(z_active, mesh_data, problem_data)
        z_full = unpack_active(z_active, z0_full, mode, fixed_values)
        control_history.append(
            {
                "iter": iter_counter,
                "J": float(j),
                "z_full": z_full.copy(),
            }
        )
        d_full = float(np.linalg.norm(z_full - z_ref_full))
        d_active = float(np.linalg.norm(np.asarray(z_active, dtype=np.float64) - z_ref_active))
        z_full_str = np.array2string(z_full, precision=10, suppress_small=False)
        print(
            f"iter={iter_counter:03d} "
            f"J={j:.10e} "
            f"||g_active||={np.linalg.norm(g):.6e} "
            f"||z-zref||={d_full:.6e} "
            f"||za-zaref||={d_active:.6e}"
        )
        print(f"  z_full={z_full_str}")

    print("\n=== Setup ===")
    print(f"NUM_SOURCES: {NUM_SOURCES}")
    print(f"CONTROL_MODE: {mode}")
    print(f"z_ref_full: {np.array2string(z_ref_full, precision=16)}")
    print(f"z0_full:    {np.array2string(z0_full, precision=16)}")
    print(f"z0_active:  {np.array2string(z0_active, precision=16)}")
    print(f"z_ref_active: {np.array2string(z_ref_active, precision=16)}")
    print(f"active_bounds: {bounds}")
    print(f"pde: {pde}")
    print(f"eta1: {eta1:.6e}")
    print(f"lambda_reg: {lambda_reg:.6e}")
    print(f"objective_time_rule: {objective_time_rule}")
    print(f"gradient_time_rule:  {gradient_time_rule}")

    j_ref = gt.compute_objective(
        gt.run_forward(mesh_data, pde, z_ref_full),
        ud_out,
        mesh_data,
        eta1=eta1,
        lam=lambda_reg,
        z=z_ref_full,
        dt_out=dt_out,
        time_rule=objective_time_rule,
    )
    j0 = objective_active(z0_active, mesh_data, problem_data)
    g0_active = gradient_active(z0_active, mesh_data, problem_data)
    g0_full = cache.g_full.copy()
    control_history.append(
        {
            "iter": 0,
            "J": float(j0),
            "z_full": z0_full.copy(),
        }
    )

    print(f"J(z_ref_full): {j_ref:.16e}")
    print(f"J(z0_full):    {j0:.16e}")
    print(f"grad(z0_full): {np.array2string(g0_full, precision=16)}")
    print(f"||grad_active(z0)||: {np.linalg.norm(g0_active):.16e}")

    rng = np.random.default_rng(19)
    d_active = rng.normal(size=z0_active.size)
    d_active /= np.linalg.norm(d_active)
    h = 1.0e-6
    jh = objective_active(z0_active + h * d_active, mesh_data, problem_data)
    fd_slope = (jh - j0) / h
    adj_slope = float(np.dot(g0_active, d_active))
    print("\n=== Pre-Optimization Directional Check ===")
    print(f"h: {h:.1e}")
    print(f"direction_active: {np.array2string(d_active, precision=16)}")
    print(f"FD slope:  {fd_slope:.16e}")
    print(f"Adj slope: {adj_slope:.16e}")
    print(f"difference:{(fd_slope - adj_slope):.16e}")

    cache = EvalCache()
    optimizer = Optimizer(
        objective_function=objective_active,
        gradient_function=gradient_active,
        optimization_method="L-BFGS-B",
        max_iter=150,
        tol=1.0e-8,
        bounds=bounds,
        callback=iteration_callback,
        options={
            "ftol": 1.0e-14,
            "gtol": 1.0e-10,
            "maxls": 20,
            "disp": True,
        },
    )
    res = optimizer.optimize(
        initial_guess=z0_active,
        mesh_data=mesh_data,
        problem_data=problem_data,
    )

    z_opt_active = np.asarray(res.x, dtype=np.float64)
    z_opt_full = unpack_active(z_opt_active, z0_full, mode, fixed_values)
    j_opt = objective_active(z_opt_active, mesh_data, problem_data)
    g_opt_active = gradient_active(z_opt_active, mesh_data, problem_data)

    print("\n=== Final Summary ===")
    print(f"success: {res.success}")
    print(f"message: {res.message}")
    print(f"nit: {res.nit}")
    print(f"nfev: {res.nfev}")
    print(f"njev: {res.njev}")
    print(f"z_active_final: {np.array2string(z_opt_active, precision=16)}")
    print(f"z_full_final:   {np.array2string(z_opt_full, precision=16)}")
    print(f"z_ref_full:     {np.array2string(z_ref_full, precision=16)}")
    print(f"z0_full:        {np.array2string(z0_full, precision=16)}")
    print(f"J_initial:      {j0:.16e}")
    print(f"J_final:        {j_opt:.16e}")
    print(f"objective_reduction_factor J0/Jf: {j0 / max(j_opt, 1.0e-30):.16e}")
    print(f"grad_active_final: {np.array2string(g_opt_active, precision=16)}")
    print(f"||grad_active_final||: {np.linalg.norm(g_opt_active):.16e}")
    if hasattr(res, "jac") and res.jac is not None:
        print(f"reported_result_jac: {np.array2string(np.asarray(res.jac), precision=16)}")
    print(f"distance_full_to_ref: {np.linalg.norm(z_opt_full - z_ref_full):.16e}")
    print(f"distance_active_to_ref: {np.linalg.norm(z_opt_active - z_ref_active):.16e}")

    if not control_history or not np.array_equal(control_history[-1]["z_full"], z_opt_full):
        control_history.append(
            {
                "iter": int(getattr(res, "nit", iter_counter)),
                "J": float(j_opt),
                "z_full": z_opt_full.copy(),
            }
        )

    export_visualization_outputs(
        mesh_data=mesh_data,
        pde=pde,
        z_opt_full=z_opt_full,
        z_ref_full=z_ref_full,
        z0_full=z0_full,
        optimized_state_out=cache.state_out,
        reference_state_out=ud_out,
        initial_state_out=gt.run_forward(mesh_data, pde, z0_full),
        control_history=control_history,
        j_opt=j_opt,
        nit=int(getattr(res, "nit", iter_counter)),
        max_paraview_frames=150,
    )
    return res


if __name__ == "__main__":
    CONTROL_MODE = choose_control_mode()

    pde = {
        "lx": 1.0,
        "ly": 0.4,
        "k": 0.1,
        "vx": 1.0,
        "vy": 0.0,
        "dt": 1.0e-4,
        "tfinal": 1.5,
        "jprint": 1,
        "ngauss": 3,
        "sigma2": 0.005,
    }

    eta1 = 1.0e5
    lambda_reg = 1.0e-4
    objective_time_rule = "exclude_first"
    gradient_time_rule = "trapezoid"

    # z_ref_full = build_reference_control()
    # z0_full = build_initial_control()

    z_ref_full = np.array([
        0.20, 0.10, 1.40,
        0.80, 0.30, 1.22
        ], dtype=np.float64)

    z0_full = np.array([
        0.75, 0.05, 0.60,
        0.36, 0.21, 0.39
        ], dtype=np.float64)

    mesh_data = gt.build_mesh_data()

    fixed_values = mode_fixed_values(CONTROL_MODE, z_ref_full=z_ref_full)
    run_optimization(
        mesh_data=mesh_data,
        pde=pde,
        mode=CONTROL_MODE,
        z_ref_full=z_ref_full,
        z0_full=z0_full,
        fixed_values=fixed_values,
        eta1=eta1,
        lambda_reg=lambda_reg,
        objective_time_rule=objective_time_rule,
        gradient_time_rule=gradient_time_rule,
    )
