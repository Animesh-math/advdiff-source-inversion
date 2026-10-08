"""Optimization driver for the rectangle-with-circular-hole geometry."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
import re
import sys

import meshio
import numpy as np

from mesh_data_config import (
    RECTANGLE_HEIGHT,
    RECTANGLE_WIDTH,
    build_mesh_data,
)


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
APP2_DIR = PROJECT_ROOT / "Applications" / "Test Application2"
OPTIMIZER_FILE = PROJECT_ROOT / "Src" / "Optimizers" / "optimizer.py"
OUTPUT_ROOT = APP_DIR / "output"

if str(APP2_DIR) not in sys.path:
    sys.path.append(str(APP2_DIR))

import gradient_test as gt
from mesh_admissibility import MeshAdmissibility
from multi_source_utils import (
    AMPLITUDE_BOUNDS,
    INITIAL_CONTROL,
    MESH_DOMAIN_BOUND_CASE,
    MODE_OUTPUT_NAMES,
    NUM_SOURCES,
    TRUE_CONTROL,
    active_bounds_for_mode,
    bound_case_names,
    component_names,
    control_dim,
    mesh_domain_active_bounds_for_mode,
    mode_fixed_values,
    pack_active,
    restrict_gradient,
    reshape_control,
    source_from_control,
    unpack_active,
    validate_control_within_bounds,
    validate_sources_outside_obstacle,
)


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Optimizer = _load_module_from_file("optimizer_mod_app3", OPTIMIZER_FILE).Optimizer


PDE = {
    "lx": RECTANGLE_WIDTH,
    "ly": RECTANGLE_HEIGHT,
    "k": 0.02,
    "vx": 1.0,
    "vy": 0.0,
    "dt": 1.0e-3,
    "tfinal": 1.0,
    "jprint": 1,
    "ngauss": 3,
    "sigma2": 0.003,
}

ETA1 = 1.0e2
LAMBDA_REG = 0.0
OBJECTIVE_TIME_RULE = "exclude_first"
GRADIENT_TIME_RULE = "trapezoid"
INVALID_CONTROL_PENALTY = 1.0e6


@dataclass
class EvalCache:
    z_active: np.ndarray | None = None
    z_full: np.ndarray | None = None
    state_out: np.ndarray | None = None
    j_total: float | None = None
    g_full: np.ndarray | None = None


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
    if choice not in mapping:
        print("Invalid input. Defaulting to location_and_intensity.")
    mode = mapping.get(choice, "location_and_intensity")
    print(f"Selected CONTROL_MODE: {mode}")
    return mode


def validate_case_name(case_name: str | None) -> str | None:
    if case_name is None:
        return None
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", case_name):
        raise ValueError(
            "case-name must start with an alphanumeric character and contain "
            "only letters, numbers, underscores, and hyphens"
        )
    return case_name


def mode_output_dir(mode: str, case_name: str | None) -> Path:
    base = OUTPUT_ROOT
    if case_name is not None:
        base = base / case_name
    return base / MODE_OUTPUT_NAMES[mode]


def ensure_output_dirs(
    mode: str, case_name: str | None, run_name: str | None = None
) -> dict[str, Path]:
    base = mode_output_dir(mode, case_name)
    if run_name is not None:
        base = base / run_name
    paths = {
        "base": base,
        "true_state": base / "true_state",
        "initial_state": base / "initial_state",
        "optimized_state": base / "optimized_state",
        "source_paths": base / "source_paths",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_pvd_collection(path: Path, entries: list[tuple[float, str]]) -> None:
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for time, filename in entries:
        lines.append(
            f'    <DataSet timestep="{time:.12g}" group="" part="0" '
            f'file="{filename}"/>'
        )
    lines.extend(["  </Collection>", "</VTKFile>"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_state_sequence(
    state_out: np.ndarray,
    mesh_data,
    out_dir: Path,
    prefix: str,
    max_frames: int,
) -> Path:
    if max_frames <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames}")

    points = np.asarray(mesh_data["coord"], dtype=np.float64)
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(points.shape[0])])
    cells = [("triangle", np.asarray(mesh_data["connections"], dtype=int))]
    frames = np.asarray(state_out, dtype=np.float64)

    for stale in out_dir.glob(f"{prefix}_*.vtu"):
        stale.unlink()

    if len(frames) <= max_frames:
        frame_indices = np.arange(len(frames), dtype=int)
    else:
        frame_indices = np.unique(
            np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        )

    entries = []
    dt_out = PDE["dt"] * PDE["jprint"]
    for output_index, frame_index in enumerate(frame_indices):
        filename = f"{prefix}_{output_index:06d}.vtu"
        meshio.write(
            out_dir / filename,
            meshio.Mesh(
                points=points,
                cells=cells,
                point_data={"u": frames[frame_index]},
            ),
        )
        entries.append((float(frame_index * dt_out), filename))

    pvd_path = out_dir / f"{prefix}.pvd"
    write_pvd_collection(pvd_path, entries)
    print(
        f"{prefix}: exported {len(frame_indices)} of {len(frames)} frames "
        f"from state shape {frames.shape}"
    )
    return pvd_path


def export_source_fields(
    mesh_data, controls: dict[str, np.ndarray], output_dir: Path
) -> Path:
    points = np.asarray(mesh_data["coord"], dtype=np.float64)
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(points.shape[0])])
    cells = [("triangle", np.asarray(mesh_data["connections"], dtype=int))]

    point_data = {}
    for name, control in controls.items():
        source = source_from_control(control, r2=PDE["sigma2"])
        point_data[name] = np.asarray(
            [source(x, y) for x, y in points[:, :2]], dtype=np.float64
        )

    path = output_dir / "source_fields.vtu"
    meshio.write(path, meshio.Mesh(points=points, cells=cells, point_data=point_data))
    return path


def write_source_paths(history: list[dict], output_dir: Path) -> list[Path]:
    for stale in output_dir.glob("source_*_path.vtu"):
        stale.unlink()

    csv_path = output_dir / "source_paths.csv"
    lines = ["iter,source_id,x,y,A"]
    by_source = {source_id: [] for source_id in range(NUM_SOURCES)}

    for item in history:
        iteration = int(item["iteration"])
        for source_id, (x, y, amplitude) in enumerate(
            reshape_control(item["control"])
        ):
            lines.append(
                f"{iteration},{source_id + 1},{x:.16e},{y:.16e},"
                f"{amplitude:.16e}"
            )
            by_source[source_id].append(
                (iteration, float(x), float(y), float(amplitude))
            )
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    paths = [csv_path]
    for source_id, values in by_source.items():
        points = np.asarray(
            [[x, y, 0.0] for _, x, y, _ in values], dtype=np.float64
        )
        if len(points) >= 2:
            cells = [
                (
                    "line",
                    np.column_stack(
                        [
                            np.arange(len(points) - 1, dtype=int),
                            np.arange(1, len(points), dtype=int),
                        ]
                    ),
                )
            ]
        else:
            cells = [("vertex", np.array([[0]], dtype=int))]
        path = output_dir / f"source_{source_id + 1}_path.vtu"
        meshio.write(
            path,
            meshio.Mesh(
                points=points,
                cells=cells,
                point_data={
                    "iteration": np.asarray(
                        [value[0] for value in values], dtype=np.float64
                    ),
                    "A": np.asarray(
                        [value[3] for value in values], dtype=np.float64
                    ),
                },
            ),
        )
        paths.append(path)
    return paths


def print_mesh_summary(mesh_data) -> None:
    boundary_nodes = mesh_data["boundary_nodes_by_name"]
    boundary_edges = mesh_data["boundary_edges_by_name"]
    print("=== Mesh Summary ===")
    print(f"coordinates: {len(mesh_data['coord'])}")
    print(f"elements: {len(mesh_data['connections'])}")
    print(f"boundaries: {sorted(boundary_edges)}")
    print(f"obstacle nodes: {len(boundary_nodes['obstacle'])}")
    print(f"obstacle edges: {len(boundary_edges['obstacle'])}")
    print("obstacle treatment: natural no-flux boundary")


def validate_control_for_region(
    control: np.ndarray,
    label: str,
    bound_case: str,
    admissibility: MeshAdmissibility,
) -> None:
    """Validate a control using mesh geometry or a legacy source box."""
    if bound_case != MESH_DOMAIN_BOUND_CASE:
        validate_sources_outside_obstacle(control, label)
        validate_control_within_bounds(control, bound_case, label)
        return

    control = np.asarray(control, dtype=np.float64)
    if not np.all(np.isfinite(control)):
        raise ValueError(f"{label} control must contain only finite values")
    for source_id, (x, y, amplitude) in enumerate(
        reshape_control(control), start=1
    ):
        if not admissibility.is_point_in_domain(float(x), float(y)):
            raise ValueError(
                f"{label} source {source_id} center ({x}, {y}) lies outside "
                "the fluid mesh domain"
            )
        if not (AMPLITUDE_BOUNDS[0] <= amplitude <= AMPLITUDE_BOUNDS[1]):
            raise ValueError(
                f"{label} source {source_id} amplitude {amplitude} is outside "
                f"the allowed bounds {AMPLITUDE_BOUNDS}"
            )


def build_evaluator(
    mesh_data,
    target_state,
    mode: str,
    bound_case: str,
    fixed_values,
    z0_full: np.ndarray,
    admissibility: MeshAdmissibility,
):
    cache = EvalCache()
    history = []

    def ensure_state(z_active: np.ndarray) -> None:
        nonlocal cache
        z_active = np.asarray(z_active, dtype=np.float64)
        if cache.z_active is not None and np.array_equal(
            z_active, cache.z_active
        ):
            return

        control = unpack_active(z_active, z0_full, mode, fixed_values)
        if (
            bound_case == MESH_DOMAIN_BOUND_CASE
            and not admissibility.validate_sources_in_domain(control)
        ):
            cache = EvalCache(
                z_active=z_active.copy(),
                z_full=control.copy(),
                state_out=None,
                j_total=INVALID_CONTROL_PENALTY,
                g_full=np.zeros_like(control),
            )
            return
        if bound_case != MESH_DOMAIN_BOUND_CASE:
            validate_control_for_region(
                control, "evaluated", bound_case, admissibility
            )
        state = gt.run_forward(mesh_data, PDE, control)
        objective = gt.compute_objective(
            state,
            target_state,
            mesh_data,
            eta1=ETA1,
            lam=LAMBDA_REG,
            z=control,
            dt_out=PDE["dt"] * PDE["jprint"],
            time_rule=OBJECTIVE_TIME_RULE,
        )
        cache = EvalCache(
            z_active=z_active.copy(),
            z_full=control.copy(),
            state_out=state,
            j_total=float(objective),
            g_full=None,
        )

    def objective(z_active, _mesh_data=None, _problem_data=None):
        ensure_state(z_active)
        return float(cache.j_total)

    def gradient(z_active, _mesh_data=None, _problem_data=None):
        ensure_state(z_active)
        if cache.g_full is None:
            _, _, gradient_full, _ = gt.compute_gradient_parts(
                cache.z_full,
                cache.state_out,
                target_state,
                mesh_data,
                PDE,
                eta1=ETA1,
                lam=LAMBDA_REG,
                gradient_time_rule=GRADIENT_TIME_RULE,
            )
            cache.g_full = np.asarray(gradient_full, dtype=np.float64)
        return restrict_gradient(cache.g_full, mode)

    def callback(z_active):
        value = objective(z_active)
        grad = gradient(z_active)
        control = unpack_active(z_active, z0_full, mode, fixed_values)
        if bound_case != MESH_DOMAIN_BOUND_CASE:
            validate_control_for_region(
                control, "iteration", bound_case, admissibility
            )
        iteration = int(history[-1]["iteration"]) + 1 if history else 1
        history.append(
            {
                "iteration": iteration,
                "objective": float(value),
                "gradient_norm": float(np.linalg.norm(grad)),
                "control": control.copy(),
            }
        )
        print(
            f"iter={iteration:03d} J={value:.10e} "
            f"||g_active||={np.linalg.norm(grad):.6e}"
        )
        print(f"  control={np.array2string(control, precision=10)}")

    return cache, history, objective, gradient, callback


def write_run_outputs(
    result,
    mode: str,
    bound_case: str,
    case_name: str | None,
    output_dir: Path,
    true_control: np.ndarray,
    input_initial_control: np.ndarray,
    initial_control: np.ndarray,
    final_control: np.ndarray,
    initial_objective: float,
    final_objective: float,
    final_gradient: np.ndarray,
    history: list[dict],
    active_bounds,
    max_paraview_frames: int,
    init_mode: str,
    start_index: int | None,
) -> tuple[Path, Path]:
    summary = {
        "optimization_mode": mode,
        "case_name": case_name,
        "bound_case": bound_case,
        "source_region": bound_case,
        "initialization_mode": init_mode,
        "start_index": start_index,
        "active_bounds": [list(bounds) for bounds in active_bounds],
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "input_initial_control": input_initial_control.tolist(),
        "initial_control": initial_control.tolist(),
        "true_control": true_control.tolist(),
        "final_control": final_control.tolist(),
        "initial_objective": float(initial_objective),
        "final_objective": float(final_objective),
        "final_gradient_norm": float(np.linalg.norm(final_gradient)),
        "max_paraview_frames": int(max_paraview_frames),
        "pde": PDE,
        "obstacle_boundary_condition": "natural no-flux",
        "invalid_control_penalty": (
            INVALID_CONTROL_PENALTY
            if bound_case == MESH_DOMAIN_BOUND_CASE
            else None
        ),
        "obstacle_constraint": (
            "fluid-mesh point location with a finite invalid-iterate penalty"
            if bound_case == MESH_DOMAIN_BOUND_CASE
            else "legacy source boxes plus explicit circular-obstacle validation"
        ),
        "inverse_problem_scope": (
            "controlled recovery with full mesh bounding-box bounds"
            if bound_case == MESH_DOMAIN_BOUND_CASE
            else "legacy controlled recovery under source-region boxes"
        ),
    }
    summary_path = output_dir / "optimization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    history_path = output_dir / "optimization_history.csv"
    rows = [
        [
            item["iteration"],
            item["objective"],
            item["gradient_norm"],
            *item["control"],
        ]
        for item in history
    ]
    np.savetxt(
        history_path,
        np.asarray(rows, dtype=np.float64),
        delimiter=",",
        header="iteration,objective,gradient_norm," + ",".join(component_names()),
        comments="",
    )
    return summary_path, history_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=tuple(MODE_OUTPUT_NAMES),
        help="Optimization mode. Omit to use the App2-style interactive prompt.",
    )
    parser.add_argument(
        "--bound-case",
        choices=bound_case_names(),
        default=MESH_DOMAIN_BOUND_CASE,
        help=(
            "Source-region behavior. Defaults to mesh_domain, which uses the "
            "full mesh bounding box plus mesh admissibility. Other choices "
            "are deprecated legacy source-label boxes."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("smoke", "gradient", "short", "full"),
        default="full",
        help="Stop after the selected verification stage.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=50,
        help="Maximum iterations for the full optimization.",
    )
    parser.add_argument(
        "--max-paraview-frames",
        type=int,
        default=150,
        help="Maximum frames exported for each ParaView state sequence.",
    )
    parser.add_argument(
        "--true-control",
        nargs=control_dim(),
        type=float,
        metavar=("X1", "Y1", "A1", "X2", "Y2", "A2"),
        help=(
            "Reference control as six values: x1 y1 A1 x2 y2 A2. "
            "Defaults to App3 TRUE_CONTROL."
        ),
    )
    parser.add_argument(
        "--initial-control",
        nargs=control_dim(),
        type=float,
        metavar=("X1", "Y1", "A1", "X2", "Y2", "A2"),
        help=(
            "Initial control as six values: x1 y1 A1 x2 y2 A2. "
            "Defaults to App3 INITIAL_CONTROL."
        ),
    )
    parser.add_argument(
        "--init-mode",
        choices=("manual", "mesh_random"),
        default="manual",
        help="Initialization strategy. Defaults to manual.",
    )
    parser.add_argument(
        "--num-starts",
        type=int,
        default=1,
        help="Number of mesh-random starts. Defaults to 1.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Optional seed for reproducible mesh-random initialization.",
    )
    parser.add_argument(
        "--min-source-separation",
        type=float,
        help="Optional minimum distance between sampled source centers.",
    )
    parser.add_argument(
        "--case-name",
        help=(
            "Optional run name. Named outputs are written under "
            "output/<case-name>/<mode-folder>/."
        ),
    )
    return parser.parse_args()


def run_single_optimization(
    args,
    mode: str,
    case_name: str | None,
    mesh_data,
    admissibility: MeshAdmissibility,
    true_control: np.ndarray,
    true_state: np.ndarray,
    initial_template: np.ndarray,
    active_bounds,
    init_mode: str,
    start_index: int | None = None,
    run_name: str | None = None,
):
    """Run one optimization while preserving the established output workflow."""
    fixed_values = mode_fixed_values(mode, z_ref_full=initial_template)
    z0_active = pack_active(initial_template, mode)
    initial_control = unpack_active(
        z0_active, initial_template, mode, fixed_values
    )
    validate_control_for_region(
        initial_control, "mode-specific initial", args.bound_case, admissibility
    )
    output_paths = ensure_output_dirs(mode, case_name, run_name=run_name)

    print("\n=== Setup ===")
    print(f"optimization mode: {mode}")
    print(f"initialization mode: {init_mode}")
    if start_index is not None:
        print(f"start index: {start_index}")
    print(f"case name: {case_name or '(default output)'}")
    print(f"source region: {args.bound_case}")
    print(f"point locator: {admissibility.locator_name}")
    print(f"active bounds: {active_bounds}")
    if args.bound_case == MESH_DOMAIN_BOUND_CASE:
        print(
            "inverse problem scope: controlled recovery with full mesh "
            "bounding-box bounds and mesh-domain admissibility"
        )
    else:
        print("inverse problem scope: deprecated legacy source-region boxes")
    print(f"output folder: {output_paths['base']}")
    print(f"max_paraview_frames: {args.max_paraview_frames}")
    print(f"true control: {np.array2string(true_control, precision=16)}")
    print(f"initial control: {np.array2string(initial_control, precision=16)}")

    print("\n=== Forward Smoke Tests ===")
    initial_state = gt.run_forward(mesh_data, PDE, initial_control)
    print(f"true forward state shape: {true_state.shape}")
    print(f"initial forward state shape: {initial_state.shape}")
    true_pvd = export_state_sequence(
        true_state,
        mesh_data,
        output_paths["true_state"],
        "true_state",
        args.max_paraview_frames,
    )
    initial_pvd = export_state_sequence(
        initial_state,
        mesh_data,
        output_paths["initial_state"],
        "initial_state",
        args.max_paraview_frames,
    )
    if args.stage == "smoke":
        return None

    _cache, history, objective, gradient, callback = build_evaluator(
        mesh_data,
        true_state,
        mode,
        args.bound_case,
        fixed_values,
        initial_template,
        admissibility,
    )

    print("\n=== Objective and Gradient Smoke Test ===")
    initial_objective = objective(z0_active)
    initial_gradient = gradient(z0_active)
    print(f"initial objective: {initial_objective:.16e}")
    print(f"initial gradient: {np.array2string(initial_gradient, precision=16)}")
    print(f"initial gradient norm: {np.linalg.norm(initial_gradient):.16e}")
    if initial_gradient.shape != z0_active.shape:
        raise ValueError(
            f"Gradient shape {initial_gradient.shape} does not match "
            f"active control shape {z0_active.shape}"
        )
    if args.stage == "gradient":
        return None

    history.append(
        {
            "iteration": 0,
            "objective": float(initial_objective),
            "gradient_norm": float(np.linalg.norm(initial_gradient)),
            "control": initial_control.copy(),
        }
    )

    max_iter = 2 if args.stage == "short" else args.max_iter
    optimizer = Optimizer(
        objective_function=objective,
        gradient_function=gradient,
        optimization_method="L-BFGS-B",
        max_iter=max_iter,
        tol=1.0e-8,
        bounds=active_bounds,
        callback=callback,
        options={
            "ftol": 1.0e-14,
            "gtol": 1.0e-9,
            "maxls": 20,
            "disp": True,
        },
    )
    result = optimizer.optimize(
        initial_guess=z0_active,
        mesh_data=mesh_data,
        problem_data={"pde": PDE, "mode": mode},
    )

    z_opt_active = np.asarray(result.x, dtype=np.float64)
    final_control = unpack_active(
        z_opt_active, initial_template, mode, fixed_values
    )
    validate_control_for_region(
        final_control, "final optimized", args.bound_case, admissibility
    )
    final_objective = objective(z_opt_active)
    final_gradient = gradient(z_opt_active)
    final_state = gt.run_forward(mesh_data, PDE, final_control)

    if not np.array_equal(history[-1]["control"], final_control):
        history.append(
            {
                "iteration": int(result.nit),
                "objective": float(final_objective),
                "gradient_norm": float(np.linalg.norm(final_gradient)),
                "control": final_control.copy(),
            }
        )

    optimized_pvd = export_state_sequence(
        final_state,
        mesh_data,
        output_paths["optimized_state"],
        "optimized_state",
        args.max_paraview_frames,
    )
    source_fields = export_source_fields(
        mesh_data,
        {
            "true_source": true_control,
            "initial_source": initial_control,
            "optimized_source": final_control,
        },
        output_paths["base"],
    )
    source_path_files = write_source_paths(history, output_paths["source_paths"])
    summary_path, history_path = write_run_outputs(
        result,
        mode,
        args.bound_case,
        case_name,
        output_paths["base"],
        true_control,
        initial_template,
        initial_control,
        final_control,
        initial_objective,
        final_objective,
        final_gradient,
        history,
        active_bounds,
        args.max_paraview_frames,
        init_mode,
        start_index,
    )

    print("\n=== Final Summary ===")
    print(f"optimization mode: {mode}")
    print(f"source region: {args.bound_case}")
    print(f"success: {result.success}")
    print(f"message: {result.message}")
    print(f"iterations: {result.nit}")
    print(f"initial objective: {initial_objective:.16e}")
    print(f"final objective: {final_objective:.16e}")
    print(f"final gradient norm: {np.linalg.norm(final_gradient):.16e}")
    print(f"final control: {np.array2string(final_control, precision=16)}")
    print(f"true control:  {np.array2string(true_control, precision=16)}")
    print(f"output folder: {output_paths['base']}")
    print("ParaView files to open:")
    for path in (true_pvd, initial_pvd, optimized_pvd, source_fields):
        print(f"  {path}")
    print("Additional outputs:")
    for path in (summary_path, history_path, *source_path_files):
        print(f"  {path}")

    return {
        "start_index": start_index,
        "sampled_initial_control": initial_control.tolist(),
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "initial_objective": float(initial_objective),
        "final_objective": float(final_objective),
        "final_gradient_norm": float(np.linalg.norm(final_gradient)),
        "final_control": final_control.tolist(),
        "output_folder": str(output_paths["base"]),
    }


def write_multistart_summary(
    mode: str,
    case_name: str | None,
    args,
    records: list[dict],
    best_start_index: int,
) -> tuple[Path, Path]:
    output_dir = mode_output_dir(mode, case_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "optimization_mode": mode,
        "initialization_mode": "mesh_random",
        "num_starts": args.num_starts,
        "random_seed": args.random_seed,
        "min_source_separation": args.min_source_separation,
        "best_start_index": best_start_index,
        "starts": records,
    }
    json_path = output_dir / "multistart_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    csv_path = output_dir / "multistart_summary.csv"
    fieldnames = [
        "start_index",
        "sampled_initial_control",
        "success",
        "iterations",
        "initial_objective",
        "final_objective",
        "final_gradient_norm",
        "final_control",
        "output_folder",
        "message",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["sampled_initial_control"] = json.dumps(
                row["sampled_initial_control"]
            )
            row["final_control"] = json.dumps(row["final_control"])
            writer.writerow(row)
    return json_path, csv_path


def main() -> None:
    args = parse_args()
    mode = args.mode if args.mode is not None else choose_control_mode()
    case_name = validate_case_name(args.case_name)
    if args.num_starts <= 0:
        raise ValueError("--num-starts must be positive")
    if args.initial_control is not None and args.num_starts > 1:
        raise ValueError(
            "--initial-control is authoritative and cannot be combined with "
            "--num-starts greater than 1"
        )
    if args.init_mode == "manual" and args.num_starts != 1:
        raise ValueError("--num-starts greater than 1 requires --init-mode mesh_random")
    if (
        args.initial_control is None
        and args.init_mode == "mesh_random"
        and args.bound_case != MESH_DOMAIN_BOUND_CASE
    ):
        raise ValueError("--init-mode mesh_random requires --bound-case mesh_domain")
    if args.num_starts > 1 and args.stage in ("smoke", "gradient"):
        raise ValueError("Mesh-random multistart requires --stage short or full")

    true_control = np.asarray(
        TRUE_CONTROL if args.true_control is None else args.true_control,
        dtype=np.float64,
    )
    mesh_data = build_mesh_data()
    admissibility = MeshAdmissibility(mesh_data)
    validate_control_for_region(
        true_control, "true/reference", args.bound_case, admissibility
    )

    if args.bound_case == MESH_DOMAIN_BOUND_CASE:
        active_bounds = mesh_domain_active_bounds_for_mode(
            mode, admissibility.bounding_box
        )
    else:
        active_bounds = active_bounds_for_mode(mode, bound_case=args.bound_case)

    if args.initial_control is not None:
        if args.init_mode == "mesh_random":
            print(
                "warning: --initial-control is authoritative; ignoring "
                "--init-mode mesh_random and using the supplied control"
            )
        init_mode = "manual"
        initial_controls = [np.asarray(args.initial_control, dtype=np.float64)]
    elif args.init_mode == "manual":
        init_mode = "manual"
        initial_controls = [INITIAL_CONTROL.copy()]
    else:
        init_mode = "mesh_random"
        rng = np.random.default_rng(args.random_seed)
        initial_controls = [
            admissibility.sample_control_in_domain(
                NUM_SOURCES,
                AMPLITUDE_BOUNDS,
                rng=rng,
                min_separation=args.min_source_separation,
            )
            for _start_index in range(args.num_starts)
        ]
        for start_index, control in enumerate(initial_controls, start=1):
            print(
                f"sampled initial control start {start_index:03d}: "
                f"{np.array2string(control, precision=16)}"
            )

    for start_index, initial_control in enumerate(initial_controls, start=1):
        validate_control_for_region(
            initial_control,
            f"initial start {start_index}",
            args.bound_case,
            admissibility,
        )

    print_mesh_summary(mesh_data)
    true_state = gt.run_forward(mesh_data, PDE, true_control)
    records = []
    is_multistart = len(initial_controls) > 1
    for start_index, initial_control in enumerate(initial_controls, start=1):
        run_name = f"start_{start_index:03d}" if is_multistart else None
        try:
            record = run_single_optimization(
                args,
                mode,
                case_name,
                mesh_data,
                admissibility,
                true_control,
                true_state,
                initial_control,
                active_bounds,
                init_mode,
                start_index=start_index if is_multistart else None,
                run_name=run_name,
            )
        except Exception as exc:
            if not is_multistart:
                raise
            record = {
                "start_index": start_index,
                "sampled_initial_control": initial_control.tolist(),
                "success": False,
                "message": f"{type(exc).__name__}: {exc}",
                "iterations": 0,
                "initial_objective": None,
                "final_objective": None,
                "final_gradient_norm": None,
                "final_control": None,
                "output_folder": str(
                    mode_output_dir(mode, case_name) / f"start_{start_index:03d}"
                ),
            }
            print(f"start {start_index:03d} failed: {record['message']}")
        if record is not None:
            records.append(record)

    if not is_multistart:
        return

    candidates = [
        record
        for record in records
        if record["final_objective"] is not None
        and np.isfinite(record["final_objective"])
    ]
    if not candidates:
        best_start_index = 0
    else:
        best_record = min(candidates, key=lambda item: item["final_objective"])
        best_start_index = int(best_record["start_index"])
    json_path, csv_path = write_multistart_summary(
        mode, case_name, args, records, best_start_index
    )
    print("\n=== Multistart Summary ===")
    print(f"starts completed: {len(records)}")
    print(f"best start index: {best_start_index}")
    print(f"summary JSON: {json_path}")
    print(f"summary CSV: {csv_path}")
    if not candidates:
        raise RuntimeError("All multistart optimization runs failed")


if __name__ == "__main__":
    main()
