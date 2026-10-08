"""Optimization driver for the App4 L-shaped city/building geometry."""

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
    OUTER_HEIGHT,
    OUTER_WIDTH,
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


Optimizer = _load_module_from_file("optimizer_mod_app4", OPTIMIZER_FILE).Optimizer


PDE = {
    "lx": OUTER_WIDTH,
    "ly": OUTER_HEIGHT,
    "k": 0.02,
    "vx": 1.0,
    "vy": 0.5,
    "dt": 1.0e-3,
    "tfinal": 2.0,
    "jprint": 1,
    "ngauss": 3,
    "sigma2": 0.003,
}

ETA1 = 1.0e2
LAMBDA_REG = 0.0
OBJECTIVE_TIME_RULE = "exclude_first"
GRADIENT_TIME_RULE = "trapezoid"
INVALID_CONTROL_PENALTY = 1.0e6
MEASUREMENT_UNION_BOUNDARY = "__selected_measurement_boundaries__"

# Smart initialization defaults. These are truth-free: they only encode a
# "balanced, non-degenerate" prior on the sampled starts. The moderate amplitude
# band deliberately excludes the near-dead (A ~ floor) and near-saturated
# (A ~ upper bound) draws that the App4 diagnostics identified as the dominant
# random-start failure modes (dead-source collapse and A=2 saturation). The
# default separation keeps two candidate source centers meaningfully apart on
# the L-shaped domain. None of these values use the true control.
SMART_DEFAULT_AMPLITUDE_LOW = 0.3
SMART_DEFAULT_AMPLITUDE_HIGH = 1.3
SMART_DEFAULT_MIN_SEPARATION = 0.25

# Adaptive multistart defaults. Adaptive runs starts sequentially and uses the
# already-paid-for objective/gradient/status of completed starts to decide the
# next start: perturb a promising candidate (exploitation) or draw a fresh smart
# start (exploration). None of these use the true control; classification relies
# only on the target-sigma objective, gradient norm, optimizer status, and
# amplitude pathology checks.
ADAPTIVE_DEFAULT_INITIAL_RANDOM_STARTS = 2
ADAPTIVE_DEFAULT_MAX_STARTS = 6
ADAPTIVE_DEFAULT_SUCCESS_TOL = 1.0e-8
ADAPTIVE_DEFAULT_PROMISING_FACTOR = 0.75
ADAPTIVE_DEFAULT_PERTURB_RADIUS = 0.08
ADAPTIVE_DEFAULT_AMPLITUDE_PERTURB = 0.15
ADAPTIVE_DEFAULT_EXPLORATION_PROBABILITY = 0.35
# Internal (non-CLI) tolerances for classification. The good/bad objective gap
# in App4 recovery is many orders of magnitude, so these are deliberately loose.
ADAPTIVE_EXCELLENT_GRAD_TOL = 1.0e-2
ADAPTIVE_LARGE_GRAD_TOL = 1.0


def apply_amplitude_floor(active_bounds, mode: str, floor, num_sources: int = NUM_SOURCES):
    """Return active_bounds with the amplitude lower bound raised to ``floor``.

    Only the Gaussian amplitude variables are affected; x/y coordinate bounds
    are left unchanged. The amplitude slot ordering mirrors the bounds builders
    in multi_source_utils (x, y, A per source). If ``floor`` is None the bounds
    are returned unchanged.
    """
    if floor is None:
        return active_bounds
    floor = float(floor)
    original = [tuple(bound) for bound in active_bounds]
    floored = []
    index = 0
    for _source_id in range(int(num_sources)):
        if mode in ("location_only", "location_and_intensity"):
            floored.append(original[index])
            floored.append(original[index + 1])
            index += 2
        if mode in ("intensity_only", "location_and_intensity"):
            _lo, hi = original[index]
            floored.append((floor, hi))
            index += 1
    return floored


def resolve_smart_sampling_config(args, amplitude_bounds):
    """Resolve the truth-free smart-initialization sampling parameters.

    Smart initialization reuses the existing admissible mesh-domain sampler
    (``MeshAdmissibility.sample_control_in_domain``) but draws amplitudes from a
    moderate, balanced interval instead of the full ``[floor, upper]`` range,
    and enforces a minimum source separation. This avoids the near-dead and
    near-saturated random draws that dominate the documented App4 failure modes.

    Returns ``(sampling_amplitude_bounds, min_separation, config)`` where
    ``config`` is a JSON-serializable record for logging/summaries. Uses only
    geometry and amplitude bounds; it never consults the true control.
    """
    effective_floor = float(amplitude_bounds[0])
    upper = float(AMPLITUDE_BOUNDS[1])

    low = float(args.smart_amplitude_low)
    high = float(args.smart_amplitude_high)
    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError("--smart-amplitude-low/high must be finite")
    if low >= high:
        raise ValueError(
            "--smart-amplitude-low must be strictly less than --smart-amplitude-high"
        )
    if high > upper:
        raise ValueError(
            f"--smart-amplitude-high must be <= the amplitude upper bound {upper}"
        )
    if low < effective_floor:
        print(
            f"warning: --smart-amplitude-low {low} is below the effective "
            f"amplitude floor {effective_floor}; raising it to the floor"
        )
        low = effective_floor
    if low >= high:
        raise ValueError(
            "smart amplitude interval collapsed after applying the amplitude "
            "floor; increase --smart-amplitude-high or lower --amplitude-floor"
        )

    if args.min_source_separation is not None:
        min_separation = float(args.min_source_separation)
        separation_source = "min_source_separation"
    else:
        min_separation = float(SMART_DEFAULT_MIN_SEPARATION)
        separation_source = "smart_default"

    config = {
        "smart_amplitude_low": float(low),
        "smart_amplitude_high": float(high),
        "min_source_separation": float(min_separation),
        "min_separation_source": separation_source,
        "effective_amplitude_floor": effective_floor,
        "uses_true_control": False,
    }
    return (float(low), float(high)), float(min_separation), config


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


def resolve_input_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, PROJECT_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Velocity field file not found: {path_text}. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def load_velocity_field_file(path_text: str, mesh_data, tolerance: float = 1.0e-10):
    path = resolve_input_path(path_text)
    mesh_coords = np.asarray(mesh_data["coord"], dtype=np.float64)
    with np.load(path) as data:
        keys = set(data.files)
        required = {"coords", "velocity_nodes"}
        missing = required - keys
        if missing:
            raise ValueError(
                f"Velocity field file {path} is missing required key(s): "
                f"{sorted(missing)}"
            )
        coords = np.asarray(data["coords"], dtype=np.float64)
        velocity_nodes = np.asarray(data["velocity_nodes"], dtype=np.float64)

    num_nodes = mesh_coords.shape[0]
    if velocity_nodes.shape != (num_nodes, 2):
        raise ValueError(
            "velocity_nodes must have shape "
            f"({num_nodes}, 2) for the current App4 mesh, got "
            f"{velocity_nodes.shape} from {path}"
        )
    if not np.all(np.isfinite(velocity_nodes)):
        bad = np.where(~np.isfinite(velocity_nodes).all(axis=1))[0]
        raise ValueError(
            f"velocity_nodes contains non-finite values at node indices {bad[:20]}"
        )
    if coords.ndim != 2 or coords.shape[0] != num_nodes or coords.shape[1] < 2:
        raise ValueError(
            "coords must have shape (num_nodes, 2) or (num_nodes, 3) with "
            f"num_nodes={num_nodes}; got {coords.shape} from {path}"
        )

    coord_diff = np.max(np.abs(coords[:, :2] - mesh_coords[:, :2]))
    if coord_diff > tolerance:
        raise ValueError(
            "Velocity field coordinates do not match the current App4 mesh "
            "node order. Do not silently reorder nodes; regenerate the "
            f"velocity field for this mesh. max |coord difference| = "
            f"{coord_diff:.16e}, tolerance = {tolerance:.1e}, file = {path}"
        )

    magnitudes = np.linalg.norm(velocity_nodes, axis=1)
    summary = {
        "path": str(path),
        "coords_shape": list(coords.shape),
        "velocity_nodes_shape": list(velocity_nodes.shape),
        "velocity_magnitude_min": float(np.min(magnitudes)),
        "velocity_magnitude_max": float(np.max(magnitudes)),
        "coordinate_validation_passed": True,
        "coordinate_validation_tolerance": float(tolerance),
        "max_coordinate_difference": float(coord_diff),
    }
    print("\n=== Velocity Field ===")
    print(f"velocity field path: {summary['path']}")
    print(f"coords shape: {coords.shape}")
    print(f"velocity_nodes shape: {velocity_nodes.shape}")
    print(
        "velocity magnitude min/max: "
        f"{summary['velocity_magnitude_min']:.16e} / "
        f"{summary['velocity_magnitude_max']:.16e}"
    )
    print("coordinate validation passed: yes")
    return velocity_nodes, summary


def serializable_pde_config(pde: dict) -> dict:
    serializable = {}
    for key, value in pde.items():
        if key == "velocity_nodes":
            continue
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, np.generic):
            serializable[key] = value.item()
        else:
            serializable[key] = value
    if "velocity_nodes" in pde:
        serializable["velocity_mode"] = "nodal"
    else:
        serializable["velocity_mode"] = "constant"
    return serializable


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
    selected = mesh_data.get("selected_measurement_boundaries")
    if selected is None:
        selected = (mesh_data.get("measurement_boundary", "right"),)
    print("=== Mesh Summary ===")
    print(f"coordinates: {len(mesh_data['coord'])}")
    print(f"elements: {len(mesh_data['connections'])}")
    print(
        "boundaries: "
        f"{sorted(name for name in boundary_edges if not name.startswith('__'))}"
    )
    print(f"measurement boundaries: {list(selected)}")
    building_names = sorted(
        name for name in boundary_edges if name.startswith("building_")
    )
    for name in building_names:
        print(f"{name} nodes: {len(boundary_nodes[name])}")
        print(f"{name} edges: {len(boundary_edges[name])}")
    print("building boundary treatment: natural no-flux boundary")


def _normalize_default_measurement_boundaries(mesh_data) -> tuple[str, ...]:
    default = mesh_data.get("measurement_boundary", "right")
    if isinstance(default, str):
        names = (default,)
    else:
        try:
            names = tuple(default)
        except TypeError as exc:
            raise TypeError(
                "mesh_data['measurement_boundary'] must be a string or iterable "
                "of boundary names"
            ) from exc
    if not names:
        raise ValueError("At least one measurement boundary is required")
    return tuple(str(name) for name in names)


def _unique_edges(edge_blocks) -> np.ndarray:
    if not edge_blocks:
        return np.zeros((0, 2), dtype=int)
    edges = np.vstack([np.asarray(block, dtype=int) for block in edge_blocks])
    if edges.size == 0:
        return np.zeros((0, 2), dtype=int)
    return np.unique(edges, axis=0)


def configure_measurement_boundaries(mesh_data, requested_boundaries):
    """Return mesh_data with selected measurement boundaries wired in."""
    boundary_edges = mesh_data["boundary_edges_by_name"]
    boundary_nodes = mesh_data["boundary_nodes_by_name"]
    available = tuple(sorted(name for name in boundary_edges if not name.startswith("__")))
    if requested_boundaries:
        names = tuple(str(name) for name in requested_boundaries)
    else:
        names = _normalize_default_measurement_boundaries(mesh_data)

    # Preserve order while avoiding duplicated boundary integration.
    names = tuple(dict.fromkeys(names))
    missing = [name for name in names if name not in boundary_edges]
    if missing:
        raise ValueError(
            "Unknown measurement boundary name(s): "
            f"{', '.join(missing)}. Available boundaries: {', '.join(available)}"
        )

    building_names = [name for name in names if name.startswith("building_")]
    if building_names:
        print(
            "warning: selected measurement boundaries include internal building "
            "walls, which are usually not intended observation boundaries: "
            f"{building_names}"
        )

    configured = dict(mesh_data)
    configured_edges = dict(boundary_edges)
    configured_nodes = dict(boundary_nodes)
    if len(names) == 1:
        configured["measurement_boundary"] = names[0]
    else:
        configured_edges[MEASUREMENT_UNION_BOUNDARY] = _unique_edges(
            [boundary_edges[name] for name in names]
        )
        configured_nodes[MEASUREMENT_UNION_BOUNDARY] = np.unique(
            np.concatenate(
                [np.asarray(boundary_nodes[name], dtype=int) for name in names]
            )
        )
        configured["measurement_boundary"] = MEASUREMENT_UNION_BOUNDARY
    configured["boundary_edges_by_name"] = configured_edges
    configured["boundary_nodes_by_name"] = configured_nodes
    configured["selected_measurement_boundaries"] = names
    return configured


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
    measurement_boundaries: tuple[str, ...],
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
    amplitude_floor=None,
    sigma2_schedule=None,
    target_sigma2=None,
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
        "amplitude_floor": (
            float(amplitude_floor) if amplitude_floor is not None else None
        ),
        "sigma2_continuation": (
            [float(s) for s in sigma2_schedule]
            if sigma2_schedule is not None and len(sigma2_schedule) > 1
            else None
        ),
        "target_sigma2": (
            float(target_sigma2) if target_sigma2 is not None else None
        ),
        "max_paraview_frames": int(max_paraview_frames),
        "pde": serializable_pde_config(PDE),
        "measurement_boundary": (
            measurement_boundaries[0]
            if len(measurement_boundaries) == 1
            else None
        ),
        "measurement_boundaries": list(measurement_boundaries),
        "building_boundary_condition": "natural no-flux",
        "invalid_control_penalty": (
            INVALID_CONTROL_PENALTY
            if bound_case == MESH_DOMAIN_BOUND_CASE
            else None
        ),
        "geometry_constraint": (
            "App4 fluid-mesh point location with analytic building/cutout "
            "exclusion and a finite invalid-iterate penalty"
            if bound_case == MESH_DOMAIN_BOUND_CASE
            else "legacy source boxes plus explicit App4 geometry validation"
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
        "--tfinal",
        type=float,
        help="Override PDE['tfinal'] for diagnostic experiments.",
    )
    parser.add_argument(
        "--amplitude-floor",
        type=float,
        help=(
            "Positive lower bound for Gaussian amplitude variables. If omitted, "
            "the existing amplitude lower bound (0.0) is used. Applies to the "
            "L-BFGS-B amplitude bounds and mesh_random amplitude sampling; does "
            "not alter x/y coordinate bounds or the true control."
        ),
    )
    parser.add_argument(
        "--sigma2-continuation",
        nargs="+",
        type=float,
        metavar="SIGMA2",
        help=(
            "Optional Gaussian-width continuation schedule, for example "
            "--sigma2-continuation 0.02 0.01 0.005 0.003. Each stage solves the "
            "inverse problem at that sigma2 (with matching synthetic target "
            "data) and warm-starts the next stage from the previous optimum. "
            "The last stage is the target problem. If omitted, a single "
            "optimization at PDE['sigma2'] is run (existing behavior)."
        ),
    )
    parser.add_argument(
        "--measurement-boundaries",
        nargs="+",
        metavar="BOUNDARY",
        help=(
            "One or more measurement boundaries used for the boundary misfit, "
            "for example: --measurement-boundaries right top. If omitted, "
            "uses mesh_data['measurement_boundary']."
        ),
    )
    parser.add_argument(
        "--true-control",
        nargs=control_dim(),
        type=float,
        metavar=("X1", "Y1", "A1", "X2", "Y2", "A2"),
        help=(
            "Reference control as six values: x1 y1 A1 x2 y2 A2. "
            "Defaults to App4 TRUE_CONTROL."
        ),
    )
    parser.add_argument(
        "--initial-control",
        nargs=control_dim(),
        type=float,
        metavar=("X1", "Y1", "A1", "X2", "Y2", "A2"),
        help=(
            "Initial control as six values: x1 y1 A1 x2 y2 A2. "
            "Defaults to App4 INITIAL_CONTROL."
        ),
    )
    parser.add_argument(
        "--init-mode",
        choices=("manual", "mesh_random", "smart", "adaptive"),
        default="manual",
        help=(
            "Initialization strategy. Defaults to manual. 'mesh_random' samples "
            "admissible starts with uniform amplitudes in the full amplitude "
            "range. 'smart' samples the same admissible mesh-domain centers but "
            "draws balanced amplitudes from a moderate interval and enforces a "
            "minimum source separation, avoiding near-dead and near-saturated "
            "starts. 'adaptive' runs starts sequentially and uses the outcomes "
            "of completed starts to perturb promising candidates or draw fresh "
            "smart starts (see --adaptive-* options). No mode uses the true "
            "control for initialization or selection."
        ),
    )
    parser.add_argument(
        "--smart-amplitude-low",
        type=float,
        default=SMART_DEFAULT_AMPLITUDE_LOW,
        help=(
            "Lower end of the balanced amplitude sampling interval for "
            "--init-mode smart. Raised to the effective amplitude floor if "
            f"lower. Defaults to {SMART_DEFAULT_AMPLITUDE_LOW}."
        ),
    )
    parser.add_argument(
        "--smart-amplitude-high",
        type=float,
        default=SMART_DEFAULT_AMPLITUDE_HIGH,
        help=(
            "Upper end of the balanced amplitude sampling interval for "
            "--init-mode smart. Must not exceed the amplitude upper bound. "
            f"Defaults to {SMART_DEFAULT_AMPLITUDE_HIGH}."
        ),
    )
    parser.add_argument(
        "--adaptive-initial-random-starts",
        type=int,
        default=ADAPTIVE_DEFAULT_INITIAL_RANDOM_STARTS,
        help=(
            "Adaptive mode: number of initial exploration starts drawn with the "
            f"smart sampler. Defaults to {ADAPTIVE_DEFAULT_INITIAL_RANDOM_STARTS}."
        ),
    )
    parser.add_argument(
        "--adaptive-max-starts",
        type=int,
        default=ADAPTIVE_DEFAULT_MAX_STARTS,
        help=(
            "Adaptive mode: maximum number of sequential starts (the loop may "
            "stop earlier with --adaptive-stop-on-success). Defaults to "
            f"{ADAPTIVE_DEFAULT_MAX_STARTS}."
        ),
    )
    parser.add_argument(
        "--adaptive-success-tol",
        type=float,
        default=ADAPTIVE_DEFAULT_SUCCESS_TOL,
        help=(
            "Adaptive mode: a start is 'excellent' if its final target-sigma "
            "objective is at or below this tolerance (with a small gradient "
            f"norm). Defaults to {ADAPTIVE_DEFAULT_SUCCESS_TOL}."
        ),
    )
    parser.add_argument(
        "--adaptive-promising-factor",
        type=float,
        default=ADAPTIVE_DEFAULT_PROMISING_FACTOR,
        help=(
            "Adaptive mode: a non-pathological start is 'promising' if its final "
            "objective is at most this factor times the median completed "
            f"objective. Defaults to {ADAPTIVE_DEFAULT_PROMISING_FACTOR}."
        ),
    )
    parser.add_argument(
        "--adaptive-perturb-radius",
        type=float,
        default=ADAPTIVE_DEFAULT_PERTURB_RADIUS,
        help=(
            "Adaptive mode: maximum source-center perturbation radius when "
            "spawning a start from a promising candidate. Defaults to "
            f"{ADAPTIVE_DEFAULT_PERTURB_RADIUS}."
        ),
    )
    parser.add_argument(
        "--adaptive-amplitude-perturb",
        type=float,
        default=ADAPTIVE_DEFAULT_AMPLITUDE_PERTURB,
        help=(
            "Adaptive mode: maximum absolute amplitude perturbation when "
            "spawning from a promising candidate. Defaults to "
            f"{ADAPTIVE_DEFAULT_AMPLITUDE_PERTURB}."
        ),
    )
    parser.add_argument(
        "--adaptive-exploration-probability",
        type=float,
        default=ADAPTIVE_DEFAULT_EXPLORATION_PROBABILITY,
        help=(
            "Adaptive mode: probability of drawing a fresh smart start instead "
            "of perturbing a promising candidate, keeping exploration/"
            f"exploitation balance. Defaults to "
            f"{ADAPTIVE_DEFAULT_EXPLORATION_PROBABILITY}."
        ),
    )
    parser.add_argument(
        "--adaptive-stop-on-success",
        action="store_true",
        help=(
            "Adaptive mode: stop as soon as an 'excellent' start is found "
            "instead of running to --adaptive-max-starts."
        ),
    )
    parser.add_argument(
        "--num-starts",
        type=int,
        default=1,
        help=(
            "Number of mesh_random/smart starts. Defaults to 1. Ignored for "
            "--init-mode adaptive, which uses --adaptive-max-starts."
        ),
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
    parser.add_argument(
        "--velocity-field-file",
        help=(
            "Optional NPZ file with coords and velocity_nodes in the current "
            "App4 mesh node order. If omitted, the existing constant velocity "
            "vx/vy behavior is preserved."
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
    target_sigma2: float,
    sigma2_stages: list[float],
    amplitude_floor,
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
    print(
        "measurement boundaries: "
        f"{list(mesh_data.get('selected_measurement_boundaries', (mesh_data.get('measurement_boundary', 'right'),)))}"
    )
    if "velocity_nodes" in PDE:
        velocity_summary = PDE.get("velocity_field", {})
        print("velocity mode: nodal velocity field")
        print(f"velocity field file: {velocity_summary.get('path', '(unknown)')}")
    else:
        print(f"velocity mode: constant vx={PDE['vx']}, vy={PDE['vy']}")
    print(f"tfinal: {PDE['tfinal']}")
    print(
        "amplitude floor: "
        + (f"{amplitude_floor}" if amplitude_floor is not None else "0.0 (default)")
    )
    if len(sigma2_stages) > 1:
        print(f"sigma2 continuation schedule: {sigma2_stages}")
        print(f"sigma2 target (final stage): {target_sigma2}")
    else:
        print(f"sigma2 (no continuation): {sigma2_stages[0]}")
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

    # Gaussian-width (sigma2) continuation. Without --sigma2-continuation this
    # is a single stage at the target sigma2 and reproduces prior behavior.
    # Approach A: each stage regenerates its own synthetic target data at the
    # stage kernel width, warm-starting the next stage from the previous
    # optimum. The final stage is the target problem.
    stage_records = []
    current_control = initial_control.copy()
    last_result = None
    for stage_index, stage_sigma2 in enumerate(sigma2_stages, start=1):
        PDE["sigma2"] = float(stage_sigma2)
        if float(stage_sigma2) == float(target_sigma2):
            stage_true_state = true_state
        else:
            stage_true_state = gt.run_forward(mesh_data, PDE, true_control)

        stage_fixed_values = mode_fixed_values(mode, z_ref_full=current_control)
        stage_z0_active = pack_active(current_control, mode)
        (
            _stage_cache,
            stage_history,
            stage_objective,
            stage_gradient,
            stage_callback,
        ) = build_evaluator(
            mesh_data,
            stage_true_state,
            mode,
            args.bound_case,
            stage_fixed_values,
            current_control,
            admissibility,
        )

        stage_initial_objective = float(stage_objective(stage_z0_active))
        stage_optimizer = Optimizer(
            objective_function=stage_objective,
            gradient_function=stage_gradient,
            optimization_method="L-BFGS-B",
            max_iter=max_iter,
            tol=1.0e-8,
            bounds=active_bounds,
            callback=stage_callback,
            options={
                "ftol": 1.0e-14,
                "gtol": 1.0e-9,
                "maxls": 20,
                "disp": True,
            },
        )
        stage_result = stage_optimizer.optimize(
            initial_guess=stage_z0_active,
            mesh_data=mesh_data,
            problem_data={"pde": PDE, "mode": mode},
        )

        stage_z_opt = np.asarray(stage_result.x, dtype=np.float64)
        stage_final_control = unpack_active(
            stage_z_opt, current_control, mode, stage_fixed_values
        )
        stage_final_objective = float(stage_objective(stage_z_opt))
        stage_final_gradient = stage_gradient(stage_z_opt)

        # Append this stage's trajectory to the combined history.
        for entry in stage_history:
            next_iteration = history[-1]["iteration"] + 1 if history else 0
            history.append(
                {
                    "iteration": next_iteration,
                    "objective": float(entry["objective"]),
                    "gradient_norm": float(entry["gradient_norm"]),
                    "control": np.asarray(entry["control"], dtype=np.float64).copy(),
                }
            )

        label = f"start {start_index:03d} " if start_index is not None else ""
        print(
            f"[sigma-continuation] {label}stage {stage_index}/{len(sigma2_stages)} "
            f"sigma2={stage_sigma2:.6g} J0={stage_initial_objective:.10e} "
            f"Jf={stage_final_objective:.10e} status={stage_result.message}"
        )
        print(
            "  stage final control="
            f"{np.array2string(stage_final_control, precision=10)}"
        )

        stage_records.append(
            {
                "stage_index": stage_index,
                "sigma2": float(stage_sigma2),
                "stage_initial_control": current_control.tolist(),
                "stage_initial_objective": stage_initial_objective,
                "stage_final_objective": stage_final_objective,
                "stage_final_gradient_norm": float(
                    np.linalg.norm(stage_final_gradient)
                ),
                "stage_final_control": stage_final_control.tolist(),
                "iterations": int(stage_result.nit),
                "success": bool(stage_result.success),
                "message": str(stage_result.message),
            }
        )

        current_control = stage_final_control.copy()
        last_result = stage_result

    # Restore the target kernel width and evaluate the final quantities on the
    # target problem so the summary is an apples-to-apples measure.
    PDE["sigma2"] = float(target_sigma2)
    result = last_result
    final_control = current_control.copy()
    z_opt_active = pack_active(final_control, mode)
    validate_control_for_region(
        final_control, "final optimized", args.bound_case, admissibility
    )
    final_objective = objective(z_opt_active)
    final_gradient = gradient(z_opt_active)
    final_state = gt.run_forward(mesh_data, PDE, final_control)

    if not np.array_equal(history[-1]["control"], final_control):
        history.append(
            {
                "iteration": history[-1]["iteration"] + 1 if history else 0,
                "objective": float(final_objective),
                "gradient_norm": float(np.linalg.norm(final_gradient)),
                "control": final_control.copy(),
            }
        )

    if len(sigma2_stages) > 1:
        continuation_summary_path = (
            output_paths["base"] / "sigma_continuation_summary.json"
        )
        continuation_summary_path.write_text(
            json.dumps(
                {
                    "start_index": start_index,
                    "target_sigma2": float(target_sigma2),
                    "sigma2_schedule": [float(s) for s in sigma2_stages],
                    "amplitude_floor": (
                        float(amplitude_floor)
                        if amplitude_floor is not None
                        else None
                    ),
                    "target_initial_objective": float(initial_objective),
                    "target_final_objective": float(final_objective),
                    "target_final_control": final_control.tolist(),
                    "stages": stage_records,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"sigma continuation summary: {continuation_summary_path}")

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
        tuple(
            mesh_data.get(
                "selected_measurement_boundaries",
                (mesh_data.get("measurement_boundary", "right"),),
            )
        ),
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
        amplitude_floor=amplitude_floor,
        sigma2_schedule=sigma2_stages,
        target_sigma2=target_sigma2,
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
    init_mode: str = "mesh_random",
    smart_config: dict | None = None,
    adaptive_config: dict | None = None,
) -> tuple[Path, Path]:
    output_dir = mode_output_dir(mode, case_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "optimization_mode": mode,
        "initialization_mode": init_mode,
        "num_starts": args.num_starts,
        "random_seed": args.random_seed,
        "min_source_separation": args.min_source_separation,
        "smart_initialization": smart_config,
        "adaptive_multistart": adaptive_config,
        "amplitude_floor": (
            float(args.amplitude_floor)
            if args.amplitude_floor is not None
            else None
        ),
        "sigma2_continuation": (
            [float(s) for s in args.sigma2_continuation]
            if args.sigma2_continuation
            else None
        ),
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
        # Adaptive-mode provenance columns; blank for non-adaptive runs.
        "classification",
        "initialization_reason",
        "parent_start_index",
        "perturbation_radius",
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


def _sample_disk_offset(radius: float, rng) -> np.ndarray:
    """Sample a uniformly distributed offset inside a disk of the given radius."""
    r = float(radius) * np.sqrt(float(rng.random()))
    theta = 2.0 * np.pi * float(rng.random())
    return np.array([r * np.cos(theta), r * np.sin(theta)], dtype=np.float64)


def perturb_control_for_adaptive(
    parent_control,
    admissibility: MeshAdmissibility,
    rng,
    perturb_radius: float,
    amplitude_perturb: float,
    amplitude_bounds,
    min_separation,
    num_sources: int = NUM_SOURCES,
    max_attempts: int = 200,
):
    """Perturb a promising control while preserving admissibility and bounds.

    Source centers are jittered within ``perturb_radius`` and amplitudes within
    ``amplitude_perturb`` (clipped to ``amplitude_bounds``). Candidates whose
    centers leave the fluid mesh, or violate ``min_separation`` when set, are
    rejected and re-sampled. Returns a new admissible control, or ``None`` if no
    admissible perturbation was found. This uses only geometry and bounds; it
    never consults the true control.
    """
    parent = np.asarray(parent_control, dtype=np.float64).reshape(-1)
    lo = float(amplitude_bounds[0])
    hi = float(amplitude_bounds[1])
    separation = 0.0 if min_separation is None else float(min_separation)
    for _attempt in range(int(max_attempts)):
        candidate = parent.copy()
        centers = []
        admissible = True
        for source_id in range(int(num_sources)):
            base_center = parent[3 * source_id : 3 * source_id + 2]
            new_center = base_center + _sample_disk_offset(perturb_radius, rng)
            if not admissibility.is_point_in_domain(
                float(new_center[0]), float(new_center[1])
            ):
                admissible = False
                break
            centers.append(new_center)
            candidate[3 * source_id : 3 * source_id + 2] = new_center
            new_amp = parent[3 * source_id + 2] + rng.uniform(
                -amplitude_perturb, amplitude_perturb
            )
            candidate[3 * source_id + 2] = float(np.clip(new_amp, lo, hi))
        if not admissible:
            continue
        if separation > 0.0:
            separated = all(
                np.linalg.norm(centers[i] - centers[j]) >= separation
                for i in range(len(centers))
                for j in range(i + 1, len(centers))
            )
            if not separated:
                continue
        return candidate
    return None


def _adaptive_amplitudes(control, num_sources: int = NUM_SOURCES):
    values = np.asarray(control, dtype=np.float64).reshape(-1)
    return [float(values[3 * i + 2]) for i in range(int(num_sources))]


def adaptive_is_pathological(
    record: dict, amplitude_bounds, success_tol: float, large_grad_tol: float
) -> bool:
    """Flag obviously broken candidates without consulting the true control.

    Covers the documented App4 random-start failure modes: dead / near-floor
    sources, amplitude saturation at the upper bound, abnormal line search with
    a poor objective, and large residual gradient with a poor objective.
    """
    obj = record.get("final_objective")
    gnorm = record.get("final_gradient_norm")
    control = record.get("final_control")
    message = str(record.get("message", "")).upper()
    if control is None or obj is None or not np.isfinite(obj):
        return True
    lo = float(amplitude_bounds[0])
    hi = float(AMPLITUDE_BOUNDS[1])
    span = max(hi - lo, 1.0e-12)
    amplitudes = _adaptive_amplitudes(control)
    if any(amp <= lo + 0.05 * span for amp in amplitudes):
        return True
    if any(amp >= hi - 0.02 * span for amp in amplitudes):
        return True
    if "ABNORMAL" in message and obj > success_tol:
        return True
    if gnorm is not None and gnorm > large_grad_tol and obj > success_tol:
        return True
    return False


def classify_adaptive_start(
    record: dict, prior_finite_objectives, args, amplitude_bounds
) -> str:
    """Classify a completed start as 'excellent', 'promising', or 'bad'.

    Uses only the final target-sigma objective, gradient norm, optimizer status,
    and amplitude pathology checks. Never uses the true control.
    """
    obj = record.get("final_objective")
    gnorm = record.get("final_gradient_norm")
    if obj is None or not np.isfinite(obj):
        return "bad"
    if obj <= args.adaptive_success_tol and (
        gnorm is None or gnorm <= ADAPTIVE_EXCELLENT_GRAD_TOL
    ):
        return "excellent"
    if adaptive_is_pathological(
        record, amplitude_bounds, args.adaptive_success_tol, ADAPTIVE_LARGE_GRAD_TOL
    ):
        return "bad"
    finite = [
        float(value)
        for value in prior_finite_objectives
        if value is not None and np.isfinite(value)
    ]
    reference = finite + [float(obj)]
    best_obj = min(reference)
    median_obj = float(np.median(reference))
    # The current incumbent best (including the very first start) is always a
    # promising exploitation target; additionally, anything substantially better
    # than the median is promising.
    if float(obj) <= best_obj:
        return "promising"
    if obj <= args.adaptive_promising_factor * median_obj:
        return "promising"
    return "bad"


def run_adaptive_multistart(
    args,
    mode: str,
    case_name: str | None,
    mesh_data,
    admissibility: MeshAdmissibility,
    true_control: np.ndarray,
    true_state: np.ndarray,
    active_bounds,
    target_sigma2: float,
    sigma2_stages: list[float],
    sampling_amplitude_bounds,
    sampling_min_separation,
    amplitude_bounds,
    smart_config: dict,
    rng,
) -> None:
    """Sequential adaptive multistart driven by completed-start outcomes.

    The first ``--adaptive-initial-random-starts`` starts are smart exploration
    draws. Thereafter each start either perturbs the best promising candidate
    (exploitation) or draws a fresh smart start (exploration), governed by
    ``--adaptive-exploration-probability``. Bad candidates are never perturbed.
    Optionally stops early when an excellent candidate is found. Adaptivity uses
    only already-computed objective/gradient/status data, never the truth.
    """
    init_mode = "adaptive"
    max_starts = int(args.adaptive_max_starts)
    initial_random = int(args.adaptive_initial_random_starts)
    exploration_probability = float(args.adaptive_exploration_probability)

    adaptive_config = {
        "initial_random_starts": initial_random,
        "max_starts": max_starts,
        "success_tol": float(args.adaptive_success_tol),
        "promising_factor": float(args.adaptive_promising_factor),
        "perturb_radius": float(args.adaptive_perturb_radius),
        "amplitude_perturb": float(args.adaptive_amplitude_perturb),
        "exploration_probability": exploration_probability,
        "stop_on_success": bool(args.adaptive_stop_on_success),
        "smart_sampler": smart_config,
        "uses_true_control": False,
    }

    print("\n=== Adaptive Multistart ===")
    print(f"initial random (smart) starts: {initial_random}")
    print(f"max starts: {max_starts}")
    print(f"success tolerance: {args.adaptive_success_tol}")
    print(f"promising factor (of median objective): {args.adaptive_promising_factor}")
    print(f"perturbation radius: {args.adaptive_perturb_radius}")
    print(f"amplitude perturbation: {args.adaptive_amplitude_perturb}")
    print(
        "exploration probability (fresh smart draw): "
        f"{exploration_probability}"
    )
    print(f"stop on success: {bool(args.adaptive_stop_on_success)}")
    print(
        "smart sampler amplitude interval: "
        f"[{sampling_amplitude_bounds[0]}, {sampling_amplitude_bounds[1]}], "
        f"min separation: {sampling_min_separation}"
    )
    print("adaptive multistart uses the true control: no")

    print_mesh_summary(mesh_data)

    def sample_smart_control():
        return admissibility.sample_control_in_domain(
            NUM_SOURCES,
            sampling_amplitude_bounds,
            rng=rng,
            min_separation=sampling_min_separation,
        )

    archive: list[dict] = []
    records: list[dict] = []

    start_index = 0
    while start_index < max_starts:
        start_index += 1
        parent_start_index = None
        perturbation_radius = None

        if start_index <= initial_random:
            initial_control = sample_smart_control()
            initialization_reason = "initial_random_smart"
            print(
                f"start {start_index:03d} generated as an initial smart "
                "exploration draw"
            )
        else:
            promising = [
                item for item in archive if item["classification"] == "promising"
            ]
            draw = float(rng.random())
            spawn = bool(promising) and (draw >= exploration_probability)
            if spawn:
                parent = min(promising, key=lambda item: item["final_objective"])
                candidate = perturb_control_for_adaptive(
                    parent["final_control"],
                    admissibility,
                    rng,
                    args.adaptive_perturb_radius,
                    args.adaptive_amplitude_perturb,
                    amplitude_bounds,
                    sampling_min_separation,
                )
                if candidate is None:
                    initial_control = sample_smart_control()
                    initialization_reason = "fresh_perturb_failed"
                    print(
                        f"start {start_index:03d} generated fresh because a valid "
                        f"perturbation of promising start "
                        f"{int(parent['start_index']):03d} could not be found"
                    )
                else:
                    initial_control = candidate
                    parent_start_index = int(parent["start_index"])
                    perturbation_radius = float(args.adaptive_perturb_radius)
                    initialization_reason = "perturb_promising"
                    print(
                        f"start {start_index:03d} generated by perturbing "
                        f"promising start {parent_start_index:03d} "
                        f"(radius {perturbation_radius})"
                    )
            else:
                initial_control = sample_smart_control()
                if promising:
                    initialization_reason = "fresh_exploration"
                    print(
                        f"start {start_index:03d} generated fresh (exploration "
                        f"draw {draw:.3f} < p={exploration_probability})"
                    )
                else:
                    initialization_reason = "fresh_no_promising"
                    print(
                        f"start {start_index:03d} generated fresh because no "
                        "promising candidate is available yet"
                    )

        initial_control = np.asarray(initial_control, dtype=np.float64)
        print(
            f"adaptive start {start_index:03d} initial control "
            f"({initialization_reason}): "
            f"{np.array2string(initial_control, precision=16)}"
        )

        try:
            validate_control_for_region(
                initial_control,
                f"adaptive start {start_index}",
                args.bound_case,
                admissibility,
            )
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
                target_sigma2,
                sigma2_stages,
                args.amplitude_floor,
                start_index=start_index,
                run_name=f"start_{start_index:03d}",
            )
        except Exception as exc:
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

        prior_finite = [item["final_objective"] for item in archive]
        classification = classify_adaptive_start(
            record, prior_finite, args, amplitude_bounds
        )
        record["classification"] = classification
        record["initialization_reason"] = initialization_reason
        record["parent_start_index"] = parent_start_index
        record["perturbation_radius"] = perturbation_radius
        records.append(record)
        archive.append(
            {
                "start_index": start_index,
                "classification": classification,
                "final_objective": record["final_objective"],
                "final_control": record["final_control"],
            }
        )

        objective_text = (
            f"{record['final_objective']:.6e}"
            if record["final_objective"] is not None
            and np.isfinite(record["final_objective"])
            else "n/a"
        )
        print(
            f"start {start_index:03d} classified {classification} "
            f"(J={objective_text}, reason={initialization_reason})"
        )

        if classification == "excellent" and args.adaptive_stop_on_success:
            print(
                f"stopping early after start {start_index:03d}: an excellent "
                "candidate was found and --adaptive-stop-on-success is set"
            )
            break

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
        mode,
        case_name,
        args,
        records,
        best_start_index,
        init_mode=init_mode,
        smart_config=smart_config,
        adaptive_config=adaptive_config,
    )

    counts = {
        label: sum(1 for record in records if record["classification"] == label)
        for label in ("excellent", "promising", "bad")
    }
    print("\n=== Adaptive Multistart Summary ===")
    print(f"starts completed: {len(records)}")
    print(f"classification counts: {counts}")
    print(f"best start index: {best_start_index}")
    print(f"summary JSON: {json_path}")
    print(f"summary CSV: {csv_path}")
    if not candidates:
        raise RuntimeError("All adaptive multistart optimization runs failed")


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
        raise ValueError(
            "--num-starts greater than 1 requires --init-mode mesh_random or smart"
        )
    if (
        args.initial_control is None
        and args.init_mode in ("mesh_random", "smart")
        and args.bound_case != MESH_DOMAIN_BOUND_CASE
    ):
        raise ValueError(
            f"--init-mode {args.init_mode} requires --bound-case mesh_domain"
        )
    if args.num_starts > 1 and args.stage in ("smoke", "gradient"):
        raise ValueError("Mesh-random multistart requires --stage short or full")
    if args.init_mode == "adaptive":
        if args.initial_control is not None:
            raise ValueError(
                "--init-mode adaptive generates its own starts and cannot be "
                "combined with --initial-control"
            )
        if args.bound_case != MESH_DOMAIN_BOUND_CASE:
            raise ValueError("--init-mode adaptive requires --bound-case mesh_domain")
        if args.stage in ("smoke", "gradient"):
            raise ValueError("--init-mode adaptive requires --stage short or full")
        if args.num_starts != 1:
            print(
                "warning: --num-starts is ignored for --init-mode adaptive; the "
                "start count is controlled by --adaptive-max-starts"
            )
        if args.adaptive_initial_random_starts < 1:
            raise ValueError("--adaptive-initial-random-starts must be >= 1")
        if args.adaptive_max_starts < args.adaptive_initial_random_starts:
            raise ValueError(
                "--adaptive-max-starts must be >= --adaptive-initial-random-starts"
            )
        if not 0.0 <= args.adaptive_exploration_probability <= 1.0:
            raise ValueError(
                "--adaptive-exploration-probability must lie in [0, 1]"
            )
        if not (np.isfinite(args.adaptive_success_tol) and args.adaptive_success_tol > 0.0):
            raise ValueError("--adaptive-success-tol must be finite and > 0")
        if not (
            np.isfinite(args.adaptive_promising_factor)
            and args.adaptive_promising_factor > 0.0
        ):
            raise ValueError("--adaptive-promising-factor must be finite and > 0")
        if not (
            np.isfinite(args.adaptive_perturb_radius)
            and args.adaptive_perturb_radius > 0.0
        ):
            raise ValueError("--adaptive-perturb-radius must be finite and > 0")
        if not (
            np.isfinite(args.adaptive_amplitude_perturb)
            and args.adaptive_amplitude_perturb >= 0.0
        ):
            raise ValueError("--adaptive-amplitude-perturb must be finite and >= 0")
    if args.tfinal is not None:
        PDE["tfinal"] = args.tfinal

    if args.amplitude_floor is not None:
        if not np.isfinite(args.amplitude_floor):
            raise ValueError("--amplitude-floor must be finite")
        if not (AMPLITUDE_BOUNDS[0] <= args.amplitude_floor < AMPLITUDE_BOUNDS[1]):
            raise ValueError(
                f"--amplitude-floor must lie in [{AMPLITUDE_BOUNDS[0]}, "
                f"{AMPLITUDE_BOUNDS[1]})"
            )
    if args.sigma2_continuation is not None:
        if len(args.sigma2_continuation) == 0:
            raise ValueError("--sigma2-continuation requires at least one value")
        if any(s <= 0.0 or not np.isfinite(s) for s in args.sigma2_continuation):
            raise ValueError("--sigma2-continuation values must be finite and > 0")

    # The target problem is the current PDE sigma2; the continuation schedule
    # (if any) must end at this value. Captured before any stage mutates PDE.
    target_sigma2 = float(PDE["sigma2"])
    sigma2_stages = (
        [float(s) for s in args.sigma2_continuation]
        if args.sigma2_continuation
        else [target_sigma2]
    )
    if float(sigma2_stages[-1]) != target_sigma2:
        print(
            "warning: --sigma2-continuation final stage "
            f"{sigma2_stages[-1]} does not equal the target sigma2 "
            f"{target_sigma2}; final objective is still reported at the target."
        )

    # Effective amplitude bounds for sampling; the L-BFGS-B amplitude lower
    # bound is floored separately below. x/y bounds are never affected.
    amplitude_bounds = (
        AMPLITUDE_BOUNDS
        if args.amplitude_floor is None
        else (float(args.amplitude_floor), AMPLITUDE_BOUNDS[1])
    )

    true_control = np.asarray(
        TRUE_CONTROL if args.true_control is None else args.true_control,
        dtype=np.float64,
    )
    mesh_data = configure_measurement_boundaries(
        build_mesh_data(), args.measurement_boundaries
    )
    if args.velocity_field_file is not None:
        velocity_nodes, velocity_summary = load_velocity_field_file(
            args.velocity_field_file, mesh_data
        )
        PDE["velocity_nodes"] = velocity_nodes
        PDE["velocity_field"] = velocity_summary
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
    active_bounds = apply_amplitude_floor(active_bounds, mode, args.amplitude_floor)

    if args.init_mode == "adaptive":
        sampling_amplitude_bounds, sampling_min_separation, smart_config = (
            resolve_smart_sampling_config(args, amplitude_bounds)
        )
        rng = np.random.default_rng(args.random_seed)
        true_state = gt.run_forward(mesh_data, PDE, true_control)
        run_adaptive_multistart(
            args,
            mode,
            case_name,
            mesh_data,
            admissibility,
            true_control,
            true_state,
            active_bounds,
            target_sigma2,
            sigma2_stages,
            sampling_amplitude_bounds,
            sampling_min_separation,
            amplitude_bounds,
            smart_config,
            rng,
        )
        return

    smart_config = None
    if args.initial_control is not None:
        if args.init_mode in ("mesh_random", "smart"):
            print(
                "warning: --initial-control is authoritative; ignoring "
                f"--init-mode {args.init_mode} and using the supplied control"
            )
        init_mode = "manual"
        initial_controls = [np.asarray(args.initial_control, dtype=np.float64)]
    elif args.init_mode == "manual":
        init_mode = "manual"
        initial_controls = [INITIAL_CONTROL.copy()]
    else:
        # mesh_random and smart share the admissible mesh-domain sampler; smart
        # only narrows the amplitude interval and enforces a separation default.
        init_mode = args.init_mode
        if init_mode == "smart":
            sampling_amplitude_bounds, sampling_min_separation, smart_config = (
                resolve_smart_sampling_config(args, amplitude_bounds)
            )
            print("\n=== Smart Initialization ===")
            print(
                "amplitude sampling interval: "
                f"[{sampling_amplitude_bounds[0]}, {sampling_amplitude_bounds[1]}] "
                "(balanced; excludes near-dead and near-saturated draws)"
            )
            print(
                "minimum source separation: "
                f"{sampling_min_separation} (source: {smart_config['min_separation_source']})"
            )
            print("smart initialization uses the true control: no")
        else:
            sampling_amplitude_bounds = amplitude_bounds
            sampling_min_separation = args.min_source_separation
        rng = np.random.default_rng(args.random_seed)
        initial_controls = [
            admissibility.sample_control_in_domain(
                NUM_SOURCES,
                sampling_amplitude_bounds,
                rng=rng,
                min_separation=sampling_min_separation,
            )
            for _start_index in range(args.num_starts)
        ]
        for start_index, control in enumerate(initial_controls, start=1):
            print(
                f"sampled {init_mode} initial control start {start_index:03d}: "
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
                target_sigma2,
                sigma2_stages,
                args.amplitude_floor,
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
        mode,
        case_name,
        args,
        records,
        best_start_index,
        init_mode=init_mode,
        smart_config=smart_config,
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
