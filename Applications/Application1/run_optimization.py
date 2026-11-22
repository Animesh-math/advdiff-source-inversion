"""Optimization driver for Application1 (DC Navy Yard advection-diffusion).

Ported from App4's ``run_optimization.py`` and adapted to the DC Navy
Yard mesh (coordinates in projected meters) and the Application1 Navier-Stokes
nodal velocity field.  The inverse problem, discrete-adjoint gradient, L-BFGS-B
optimizer, sigma-continuation, amplitude floor, and manual / mesh_random / smart
/ adaptive initialization behaviors are preserved from App4; only
the geometry-dependent numeric defaults (coordinate bounds, Gaussian width,
source separation, PDE parameters) and the velocity-field source differ.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from dataclasses import dataclass
from itertools import permutations
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
OBJECTIVE_DIR = PROJECT_ROOT / "Src" / "Objective"  # shared gradient_test module
OPTIMIZER_FILE = PROJECT_ROOT / "Src" / "Optimizers" / "optimizer.py"
OUTPUT_ROOT = APP_DIR / "output"

if str(OBJECTIVE_DIR) not in sys.path:
    sys.path.append(str(OBJECTIVE_DIR))

import gradient_test as gt
import post_init_optimizers as pio
import source_count_inference as sci
import source_insertion_sensitivity as sis
import sensitivity_initialization as sinit
from mesh_admissibility import MeshAdmissibility
from final_feasibility import (
    DEFAULT_FINAL_FEASIBILITY_MARGIN,
    DEFAULT_FINAL_FEASIBILITY_MAX_ITER,
    DEFAULT_FINAL_FEASIBILITY_METHOD,
    FINAL_FEASIBILITY_METHODS,
    final_feasibility_polish,
    signed_clearances,
)
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
    infer_num_sources,
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


_HESSIAN_UTILS = None


def _hessian_utils():
    """Import ``hessian_utils`` lazily.

    ``hessian_utils`` imports this module (it reads ETA1/LAMBDA_REG and the
    ``gt`` handle from it), so importing it at module scope would be circular.
    Only the Newton-type post-init methods need it.
    """
    global _HESSIAN_UTILS
    if _HESSIAN_UTILS is None:
        import hessian_utils as _hu

        _HESSIAN_UTILS = _hu
    return _HESSIAN_UTILS


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Optimizer = _load_module_from_file("optimizer_mod_app1", OPTIMIZER_FILE).Optimizer
obs_op = _load_module_from_file(
    "observation_operator_app1",
    PROJECT_ROOT / "Src" / "Solvers" / "observation_operator.py",
)


# Default Application1 Navier-Stokes nodal velocity field. Auto-loaded when
# --velocity-field-file is omitted and --constant-velocity is not set, so the
# DC transport uses the real geometry-aware wind field by default.
DEFAULT_VELOCITY_FIELD_FILE = (
    APP_DIR
    / "output"
    / "navier_stokes"
    / "application1_navier_stokes_velocity_nodes.npz"
)


# PDE configuration in PHYSICAL DC UNITS (meters, seconds, m^2/s). The DC fluid
# domain spans ~302 m (x) by ~700 m (y); mean triangle edge is ~8 m. These
# values are the deliberate physical re-scaling of the App4
# normalized configuration (see the port notes / README):
#   * k = 10 m^2/s  -> urban effective/eddy diffusivity; Peclet U*L/k ~ 70,
#                      matching App4's ~70 with U~2.4 m/s and L~300 m.
#   * sigma2 = 400 m^2 -> Gaussian e-folding radius 20 m (~2.5 mean elements),
#                      i.e. well resolved by the DC mesh.
#   * dt = 0.1 s    -> explicit RK4 stable at the smallest (~1.6 m) element for
#                      k = 10 (real-axis limit) and the ~10 m/s peak velocity.
#   * tfinal = 60 s -> long enough for a source plume to advect to the outflow
#                      (east/south) observation boundaries at U ~ 2.4 m/s.
# lx/ly are ignored by the shared forward solver (geometry is the mesh); they
# are kept only for API parity with the App4 PDE signature.
# vx/vy are the CONSTANT-velocity fallback (the ambient wind) used only when no
# nodal velocity field is loaded; real runs use the Navier-Stokes field.
PDE = {
    "lx": OUTER_WIDTH,
    "ly": OUTER_HEIGHT,
    "k": 10.0,
    "vx": 2.07,
    "vy": -0.36,
    "dt": 0.1,
    "tfinal": 150.0,
    "jprint": 1,
    "ngauss": 3,
    "sigma2": 400.0,
}

ETA1 = 1.0e2
LAMBDA_REG = 0.0
OBJECTIVE_TIME_RULE = "exclude_first"
GRADIENT_TIME_RULE = "trapezoid"
# Invalid mesh-domain iterates (a source center inside a building hole or
# outside the fluid mesh) are intercepted before any forward/adjoint solve. A
# flat low penalty with an exactly zero gradient is unsafe: at broad continuation
# stages the legitimate objective can be ~1e8, so a constant 1e6 looked like a
# dramatic improvement, and the zero gradient created a fake stationary point at
# which L-BFGS-B declared convergence (and then warm-started later stages from
# the invalid control). Instead, an invalid iterate now receives:
#   (1) a large, objective-scale-aware baseline guaranteed to exceed any valid
#       objective seen so far (with a fixed floor for the first evaluation), so
#       it can never look better than a legitimate objective; plus
#   (2) a smooth squared-distance term whose gradient restores the offending
#       centers toward the fluid domain, so there is no fake zero-gradient
#       optimum.
# The distance term is C1 (both value and gradient vanish as a center re-enters
# the domain), so valid evaluations are completely unaffected.
INVALID_CONTROL_PENALTY_FLOOR = 1.0e12
INVALID_CONTROL_PENALTY_OBJECTIVE_FACTOR = 1.0e3
INVALID_CONTROL_DISTANCE_WEIGHT = 1.0e6
# How an evaluated control whose source center left the fluid mesh is handled.
#   "penalty" : historical behavior. The objective jumps to
#               max(1.0e12, 1.0e3 * largest valid objective) plus a quadratic
#               distance term. That jump is DISCONTINUOUS at the fluid boundary,
#               so an L-BFGS-B line search cannot bracket it and aborts
#               (ABNORMAL_TERMINATION_IN_LNSRCH) when an iterate sits within a
#               line-search step of a wall.
#   "project" : the offending centers are projected onto the fluid domain and
#               the objective/gradient are evaluated THERE. The reduced
#               objective becomes continuous across the wall and the gradient is
#               the correct chain-rule derivative of J(P(z)): the component
#               normal to the wall is removed, leaving a feasible descent
#               direction ALONG the boundary.
FEASIBILITY_MODES = ("penalty", "project", "raw")
DEFAULT_FEASIBILITY_MODE = "penalty"
MEASUREMENT_UNION_BOUNDARY = "__selected_measurement_boundaries__"

# Observation operator selection. "boundary" is the historical (and default)
# boundary-integral misfit over the selected measurement boundaries and is left
# bit-for-bit unchanged: no observation operator object is attached to mesh_data
# and every solver takes exactly the code path it took before. "nodes" observes
# the FE state at discrete mesh nodes through
# Src/Solvers/observation_operator.NodeObservation.
OBSERVATION_MODES = ("boundary", "nodes")
DEFAULT_OBSERVATION_MODE = "boundary"
# Point-sensor spatial weighting. "equal" is the documented default convention
# for ordinary interior sensors; "boundary_quadrature" is the regression /
# equivalence configuration that reproduces the boundary objective exactly.
SENSOR_WEIGHT_MODES = ("equal", "boundary_quadrature")
DEFAULT_SENSOR_WEIGHT_MODE = "equal"
DEFAULT_SENSOR_EQUAL_WEIGHT = 1.0
# Coordinate snapping tolerance in METERS. The DC mesh has a ~7.5 m median edge,
# so any fluid-interior request resolves within a few metres; 20 m rejects a
# coordinate that is not actually on the mesh instead of snapping it across the
# domain.
DEFAULT_SENSOR_SNAP_TOLERANCE = 20.0

# Smart initialization defaults. These are truth-free: they only encode a
# "balanced, non-degenerate" prior on the sampled starts. The moderate amplitude
# band deliberately excludes the near-dead (A ~ floor) and near-saturated
# (A ~ upper bound) draws that the App4 diagnostics identified as the dominant
# random-start failure modes (dead-source collapse and A=2 saturation). The
# default separation keeps two candidate source centers meaningfully apart on
# the DC street domain. None of these values use the true control.
# Amplitudes are dimensionless (unchanged from App4); the separation is in
# METERS for the DC domain (App4's 0.25 normalized ~ 18% of width -> ~40 m here).
SMART_DEFAULT_AMPLITUDE_LOW = 0.3
SMART_DEFAULT_AMPLITUDE_HIGH = 1.3
SMART_DEFAULT_MIN_SEPARATION = 40.0

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
# Perturbation radius is in METERS (App4's 0.08 normalized ~ 6% of width ->
# ~18 m here; 15 m keeps jitters within a couple of street corridors). The
# amplitude perturbation stays dimensionless.
ADAPTIVE_DEFAULT_PERTURB_RADIUS = 15.0
ADAPTIVE_DEFAULT_AMPLITUDE_PERTURB = 0.15
ADAPTIVE_DEFAULT_EXPLORATION_PROBABILITY = 0.35
# Internal (non-CLI) tolerances for classification. The good/bad objective gap
# in App4 recovery is many orders of magnitude, so these are deliberately loose.
ADAPTIVE_EXCELLENT_GRAD_TOL = 1.0e-2
ADAPTIVE_LARGE_GRAD_TOL = 1.0

# Dead-source sensitivity reinsertion (adaptive mode; OFF by default via CLI).
# A source counts as "dead" when its amplitude sits at/just above the floor; the
# amplitude-at-floor test is decisive, a bound-pinned center is only recorded as
# a secondary signal. Reinsertion uses a fixed moderate amplitude and reuses the
# existing source-insertion sensitivity field to pick the new center. None of
# these constants change source bounds, the amplitude floor, minimum separation,
# or optimizer tolerances.
DEAD_SOURCE_AMPLITUDE_MARGIN = 0.05     # dead if amp <= floor + this margin
REINSERTION_AMPLITUDE = 0.8             # fixed moderate reinsertion amplitude
ADAPTIVE_DEFAULT_MAX_REINSERTIONS = 2   # cap for the whole adaptive run
BASIN_ROUND_M = 1.0                     # survivor-center rounding for basin dedup
BOUND_PIN_TOL_M = 1.0                   # center "pinned" within this of a bound

# Greedy sensitivity-initialization default (see --init-mode sensitivity). An
# insertion is accepted only when the minimum masked insertion sensitivity is
# below -SENSITIVITY_DEFAULT_MIN_VALUE_TOL (a nonnegative-amplitude descent). The
# default requires a strictly negative sensitivity.
SENSITIVITY_DEFAULT_MIN_VALUE_TOL = 0.0


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
            f"({num_nodes}, 2) for the current Application1 mesh, got "
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
            "Velocity field coordinates do not match the current Application1 "
            "mesh node order. Do not silently reorder nodes; regenerate the "
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


def export_sensor_points(mesh_data, output_dir: Path) -> Path | None:
    """Write the node-sensor set as a ParaView point cloud (``sensors.vtu``).

    One vertex cell per sensor so ParaView renders it directly (Points /
    Glyph).  Returns ``None`` in the default boundary observation mode, which
    has no discrete sensor set.  The JSON summary already records the same ids,
    coordinates and weights; this file only makes them visible.
    """
    observation = mesh_data.get("observation")
    if observation is None:
        return None

    coord = np.asarray(mesh_data["coord"], dtype=np.float64)
    node_ids = np.asarray(observation.node_ids, dtype=int)
    xy = coord[node_ids, :2]
    points = np.column_stack([xy, np.zeros(node_ids.size, dtype=np.float64)])
    # One vertex cell per point; meshio point_data is indexed by point.
    cells = [("vertex", np.arange(node_ids.size, dtype=int).reshape(-1, 1))]

    if observation.weight_kind == obs_op.DIAGONAL_WEIGHTS:
        weight = np.asarray(observation.weights, dtype=np.float64)
        row_sum_weight = weight.copy()
    else:
        weights = observation.weights
        diagonal = (
            weights.diagonal()
            if hasattr(weights, "diagonal")
            else np.diag(weights)
        )
        weight = np.asarray(diagonal, dtype=np.float64).reshape(-1)
        row_sum_weight = np.asarray(
            weights.sum(axis=1), dtype=np.float64
        ).reshape(-1)

    point_data = {
        "sensor_id": np.arange(node_ids.size, dtype=np.float64),
        "node_id": node_ids.astype(np.float64),
        "weight": weight,
        "row_sum_weight": row_sum_weight,
        "x": xy[:, 0].copy(),
        "y": xy[:, 1].copy(),
    }

    path = output_dir / "sensors.vtu"
    meshio.write(path, meshio.Mesh(points=points, cells=cells, point_data=point_data))
    print(
        f"sensors: exported {node_ids.size} node sensor(s) "
        f"({observation.weight_kind} weights) to {path}"
    )
    return path


def write_source_paths(history: list[dict], output_dir: Path) -> list[Path]:
    for stale in output_dir.glob("source_*_path.vtu"):
        stale.unlink()

    csv_path = output_dir / "source_paths.csv"
    lines = ["iter,source_id,x,y,A"]
    # Source count comes from the recorded controls, so this writes one path per
    # source for any M.
    num_sources = (
        infer_num_sources(history[0]["control"]) if history else NUM_SOURCES
    )
    by_source = {source_id: [] for source_id in range(num_sources)}

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


def _normalize_boundary_name(name: str, available) -> str:
    """Map user-facing aliases to canonical DC boundary names.

    Accepts zero-padded building aliases (``building_03`` -> ``building_3``) and
    the cardinal outer-boundary names.  Unknown names are returned unchanged so
    the caller can raise a clear error listing the available boundaries.
    """
    name = str(name)
    if name in available:
        return name
    match = re.fullmatch(r"building_0*(\d+)", name)
    if match:
        canonical = f"building_{int(match.group(1))}"
        if canonical in available:
            return canonical
    return name


def configure_measurement_boundaries(mesh_data, requested_boundaries):
    """Return mesh_data with selected measurement boundaries wired in."""
    boundary_edges = mesh_data["boundary_edges_by_name"]
    boundary_nodes = mesh_data["boundary_nodes_by_name"]
    available = tuple(sorted(name for name in boundary_edges if not name.startswith("__")))
    if requested_boundaries:
        names = tuple(
            _normalize_boundary_name(name, available) for name in requested_boundaries
        )
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


def parse_sensor_node_ids(tokens) -> list[int]:
    """Parse ``--sensor-node-ids`` tokens (comma and/or space separated)."""
    values: list[int] = []
    for token in tokens:
        for piece in str(token).replace(";", ",").split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                values.append(int(piece))
            except ValueError as exc:
                raise ValueError(
                    f"--sensor-node-ids expects integers; got {piece!r}"
                ) from exc
    if not values:
        raise ValueError("--sensor-node-ids was supplied but contains no ids")
    return values


def parse_sensor_coordinates(text: str) -> np.ndarray:
    """Parse ``--sensor-coordinates`` as ``"x1,y1;x2,y2;..."``."""
    pairs = []
    for chunk in str(text).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p for p in chunk.replace(",", " ").split() if p]
        if len(parts) != 2:
            raise ValueError(
                "--sensor-coordinates expects \"x1,y1;x2,y2;...\"; got the "
                f"entry {chunk!r}"
            )
        try:
            pairs.append((float(parts[0]), float(parts[1])))
        except ValueError as exc:
            raise ValueError(
                f"--sensor-coordinates entry {chunk!r} is not a numeric pair"
            ) from exc
    if not pairs:
        raise ValueError("--sensor-coordinates was supplied but contains no points")
    return np.asarray(pairs, dtype=np.float64)


def measurement_observation_edges(mesh_data) -> np.ndarray:
    """Edges of the currently selected measurement boundary/boundaries."""
    return np.asarray(
        mesh_data["boundary_edges_by_name"][
            mesh_data.get("measurement_boundary", "right")
        ],
        dtype=int,
    )


def configure_observation(mesh_data, args):
    """Attach the requested observation operator to ``mesh_data``.

    ``--observation-mode boundary`` (the default) attaches nothing, so the
    objective, the RK4 discrete adjoint, and the sensitivity initializer keep
    their existing boundary-integral code paths untouched.

    ``--observation-mode nodes`` builds a ``NodeObservation`` with an explicit
    weight representation and stores it as ``mesh_data["observation"]``.
    """
    observation_mode = getattr(args, "observation_mode", DEFAULT_OBSERVATION_MODE)
    sensor_ids_arg = getattr(args, "sensor_node_ids", None)
    sensor_coords_arg = getattr(args, "sensor_coordinates", None)
    weight_mode = getattr(args, "sensor_weights", DEFAULT_SENSOR_WEIGHT_MODE)

    if observation_mode == "boundary":
        for name, value in (
            ("--sensor-node-ids", sensor_ids_arg),
            ("--sensor-coordinates", sensor_coords_arg),
        ):
            if value:
                raise ValueError(
                    f"{name} requires --observation-mode nodes; the default "
                    "boundary observation takes no sensor arguments"
                )
        if weight_mode != DEFAULT_SENSOR_WEIGHT_MODE:
            raise ValueError(
                "--sensor-weights requires --observation-mode nodes"
            )
        return mesh_data

    coord = np.asarray(mesh_data["coord"], dtype=np.float64)
    npoin = coord.shape[0]
    selected = mesh_data.get(
        "selected_measurement_boundaries",
        (mesh_data.get("measurement_boundary", "right"),),
    )
    snap_records = None

    if sensor_ids_arg:
        if sensor_coords_arg:
            print(
                "note: both --sensor-node-ids and --sensor-coordinates were "
                "supplied; node ids take precedence and the coordinates are "
                "ignored"
            )
        node_ids = obs_op.resolve_node_ids(
            parse_sensor_node_ids(sensor_ids_arg), npoin
        )
        selection = "node_ids"
    elif sensor_coords_arg:
        requested = parse_sensor_coordinates(sensor_coords_arg)
        node_ids, snap_records = obs_op.snap_coordinates_to_nodes(
            requested, coord, float(getattr(args, "sensor_snap_tolerance"))
        )
        resolved = obs_op.resolve_node_ids(node_ids, npoin)
        if resolved.size != node_ids.size:
            print(
                "note: distinct requested coordinates snapped to the same mesh "
                "node; the duplicate sensors were removed"
            )
        node_ids = resolved
        selection = "coordinates_snapped"
    elif weight_mode == "boundary_quadrature":
        # Boundary-equivalent node observation test configuration: every node of
        # the selected measurement boundaries.
        node_ids = np.unique(measurement_observation_edges(mesh_data).reshape(-1))
        selection = "measurement_boundary_nodes"
    else:
        raise ValueError(
            "--observation-mode nodes requires --sensor-node-ids or "
            "--sensor-coordinates (or --sensor-weights boundary_quadrature to "
            "use every node of the selected measurement boundaries)"
        )

    if weight_mode == "boundary_quadrature":
        edges = measurement_observation_edges(mesh_data)
        boundary_nodes, weight_matrix = obs_op.boundary_quadrature_weights(
            coord, edges, *gt.interval_quadrature_0_to_1(3)
        )
        if not np.array_equal(np.sort(node_ids), boundary_nodes):
            raise ValueError(
                "--sensor-weights boundary_quadrature reproduces the discrete "
                "boundary objective only when the sensor set is exactly the "
                f"node set of {list(selected)} ({boundary_nodes.size} nodes); "
                f"the requested selection has {node_ids.size} node(s) and does "
                "not match. Use --sensor-weights equal for an arbitrary sensor "
                "subset."
            )
        # Reorder to the assembled matrix's node order.
        node_ids = boundary_nodes
        weights = weight_matrix
        description = (
            "exact P1 boundary observation matrix (consistent edge mass "
            "matrix, 3-point Gauss rule) of "
            f"{'+'.join(selected)}; W is symmetric with off-diagonal node "
            "coupling and is NOT a diagonal nodal weighting"
        )
    else:
        equal_weight = float(getattr(args, "sensor_equal_weight"))
        if not (np.isfinite(equal_weight) and equal_weight > 0.0):
            raise ValueError("--sensor-equal-weight must be finite and > 0")
        weights = np.full(node_ids.size, equal_weight, dtype=np.float64)
        description = (
            f"equal diagonal sensor weights w = {equal_weight:g} "
            "(||r||_W^2 = w * sum_s r_s^2)"
        )

    extra = {"weight_mode": weight_mode}
    if weight_mode == "boundary_quadrature":
        extra["boundary_equivalent_of"] = list(selected)
    if snap_records is not None:
        extra["coordinate_snapping"] = snap_records
        extra["snap_tolerance"] = float(getattr(args, "sensor_snap_tolerance"))

    observation = obs_op.NodeObservation(
        node_ids,
        weights,
        coords=coord[node_ids, :2],
        description=description,
        selection=selection,
        extra=extra,
    )

    configured = dict(mesh_data)
    configured["observation"] = observation
    return configured


def print_observation_summary(mesh_data, max_listed: int = 40) -> None:
    """Print the resolved observation operator (and persist-ready metadata)."""
    observation = mesh_data.get("observation")
    selected = list(
        mesh_data.get(
            "selected_measurement_boundaries",
            (mesh_data.get("measurement_boundary", "right"),),
        )
    )
    print("\n=== Observation Operator ===")
    if observation is None:
        edges = measurement_observation_edges(mesh_data)
        print("observation mode: boundary (default, unchanged)")
        print(f"measurement boundaries: {selected}")
        print(f"boundary edges integrated: {len(edges)}")
        print(
            "spatial form: int_Gamma_obs |u - u_d|^2 ds with the 3-point "
            "interval Gauss rule on each P1 edge"
        )
        return

    coords = observation.coords
    print("observation mode: nodes (discrete mesh-node sensors)")
    print(f"sensor selection: {observation.selection}")
    print(f"resolved sensor count: {observation.num_sensors}")
    print(f"sensor weighting: {observation.description}")
    print(f"weight representation: {observation.weight_kind}")
    if observation.extra.get("weight_mode") == "boundary_quadrature":
        print(f"boundary-equivalent of: {selected}")
    for record in observation.extra.get("coordinate_snapping", []) or []:
        print(
            f"  requested ({record['requested'][0]:.4f}, "
            f"{record['requested'][1]:.4f}) -> node {record['node_id']} at "
            f"({record['node_coordinate'][0]:.4f}, "
            f"{record['node_coordinate'][1]:.4f}), snap distance "
            f"{record['snap_distance']:.4f} m"
        )
    listed = observation.num_sensors <= max_listed
    print(
        "sensor node ids: "
        + (
            ", ".join(str(int(i)) for i in observation.node_ids)
            if listed
            else f"{observation.num_sensors} ids (persisted in the run summary)"
        )
    )
    if listed and coords is not None:
        for node, (x, y) in zip(observation.node_ids, coords):
            print(f"  node {int(node):5d} at ({x:.4f}, {y:.4f})")


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


def new_feasibility_stats(mode: str = DEFAULT_FEASIBILITY_MODE) -> dict:
    """Fresh projection/penalty accounting record for one evaluator."""
    return {
        "feasibility_mode": str(mode),
        "invalid_evaluations": 0,
        "projected_evaluations": 0,
        "penalized_evaluations": 0,
        "projected_source_events": 0,
        "max_projection_distance": 0.0,
        "last_projection_distance": None,
        # "raw" mode only: evaluations / accepted iterates whose Gaussian center
        # sits inside a building hole or otherwise outside the fluid mesh.
        "raw_infeasible_evaluations": 0,
        "raw_infeasible_iterates": 0,
        "raw_max_outside_distance": 0.0,
        "raw_infeasible_iterate_log": [],
    }


def project_control_centers(control: np.ndarray, admissibility: MeshAdmissibility):
    """Project every inadmissible source center onto the fluid mesh.

    Returns ``(projected_control, events)``; ``events`` is one record per moved
    source with the outward unit normal used for the gradient chain rule.
    """
    projected = np.asarray(control, dtype=np.float64).copy()
    events = []
    for source_id, (x, y, _amplitude) in enumerate(reshape_control(projected)):
        point, was_projected, distance = admissibility.project_into_domain(
            float(x), float(y)
        )
        if not was_projected:
            continue
        offset = 3 * source_id
        original = np.array([float(x), float(y)], dtype=np.float64)
        outward = original - point
        norm = float(np.hypot(outward[0], outward[1]))
        projected[offset] = float(point[0])
        projected[offset + 1] = float(point[1])
        events.append(
            {
                "source_id": source_id,
                "original": original.tolist(),
                "projected": [float(point[0]), float(point[1])],
                "distance": float(distance),
                "normal": (outward / norm).tolist() if norm > 0.0 else None,
            }
        )
    return projected, events


def build_evaluator(
    mesh_data,
    target_state,
    mode: str,
    bound_case: str,
    fixed_values,
    z0_full: np.ndarray,
    admissibility: MeshAdmissibility,
    *,
    feasibility_mode: str = DEFAULT_FEASIBILITY_MODE,
    feasibility_stats: dict | None = None,
):
    if feasibility_mode not in FEASIBILITY_MODES:
        raise ValueError(
            f"Unknown feasibility mode {feasibility_mode!r}; "
            f"expected one of {list(FEASIBILITY_MODES)}"
        )
    cache = EvalCache()
    projection_events: list[dict] = []
    history = []
    # Running scale for the invalid-iterate baseline: the largest valid objective
    # observed by this evaluator. Guarantees an invalid iterate is never cheaper
    # than a legitimate objective, at any continuation stage / tfinal.
    max_valid_objective = 0.0

    def invalid_control_penalty(control: np.ndarray):
        """Objective-scale-aware penalty and restoring gradient for an invalid
        control (a source center outside the fluid mesh domain)."""
        grad_full = np.zeros_like(control)
        distance_sq_sum = 0.0
        for source_id, (x, y, _amplitude) in enumerate(reshape_control(control)):
            dist_sq, (dx, dy) = admissibility.nearest_domain_distance_sq(
                float(x), float(y)
            )
            distance_sq_sum += dist_sq
            grad_full[3 * source_id] = INVALID_CONTROL_DISTANCE_WEIGHT * dx
            grad_full[3 * source_id + 1] = INVALID_CONTROL_DISTANCE_WEIGHT * dy
        baseline = max(
            INVALID_CONTROL_PENALTY_FLOOR,
            INVALID_CONTROL_PENALTY_OBJECTIVE_FACTOR * max_valid_objective,
        )
        j_total = baseline + INVALID_CONTROL_DISTANCE_WEIGHT * distance_sq_sum
        return float(j_total), grad_full

    def ensure_state(z_active: np.ndarray) -> None:
        nonlocal cache, max_valid_objective, projection_events
        z_active = np.asarray(z_active, dtype=np.float64)
        if cache.z_active is not None and np.array_equal(
            z_active, cache.z_active
        ):
            return

        control = unpack_active(z_active, z0_full, mode, fixed_values)
        projection_events = []
        if (
            bound_case == MESH_DOMAIN_BOUND_CASE
            and not admissibility.validate_sources_in_domain(control)
        ):
            if feasibility_stats is not None:
                feasibility_stats["invalid_evaluations"] += 1
            if feasibility_mode == "raw":
                # Evaluate the TRUE objective at the RAW center. The Gaussian and
                # its center derivatives are evaluated at the raw center; the
                # forward/adjoint solves still run only on the fluid-domain mesh,
                # so a center inside a building simply contributes the tail of its
                # Gaussian to the fluid. No projection, no penalty, and (because
                # projection_events stays empty) no wall-normal component is
                # removed from the gradient.
                if feasibility_stats is not None:
                    feasibility_stats["raw_infeasible_evaluations"] += 1
                    worst = 0.0
                    for _sid, (_x, _y, _a) in enumerate(reshape_control(control)):
                        dist_sq, _g = admissibility.nearest_domain_distance_sq(
                            float(_x), float(_y)
                        )
                        worst = max(worst, float(np.sqrt(max(dist_sq, 0.0))))
                    feasibility_stats["raw_max_outside_distance"] = max(
                        float(feasibility_stats["raw_max_outside_distance"]), worst
                    )
            elif feasibility_mode == "penalty":
                if feasibility_stats is not None:
                    feasibility_stats["penalized_evaluations"] += 1
                j_invalid, g_invalid = invalid_control_penalty(control)
                cache = EvalCache(
                    z_active=z_active.copy(),
                    z_full=control.copy(),
                    state_out=None,
                    j_total=j_invalid,
                    g_full=g_invalid,
                )
                return
            else:
                # "project": evaluate the true objective at the projected control.
                control, projection_events = project_control_centers(
                    control, admissibility
                )
                if feasibility_stats is not None:
                    distances = [event["distance"] for event in projection_events]
                    feasibility_stats["projected_evaluations"] += 1
                    feasibility_stats["projected_source_events"] += len(
                        projection_events
                    )
                    if distances:
                        feasibility_stats["last_projection_distance"] = float(
                            max(distances)
                        )
                        feasibility_stats["max_projection_distance"] = float(
                            max(
                                feasibility_stats["max_projection_distance"],
                                max(distances),
                            )
                        )
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
        if np.isfinite(objective):
            max_valid_objective = max(max_valid_objective, float(objective))
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
            gradient_full = np.asarray(gradient_full, dtype=np.float64)
            # Chain rule for the projected evaluation J(P(z)): the projection is
            # locally the identity along the wall and constant across it, so the
            # component of the position gradient normal to the wall is removed.
            # Without this the reported slope would disagree with the objective
            # the line search measures.
            for event in projection_events:
                normal = event.get("normal")
                if normal is None:
                    continue
                offset = 3 * int(event["source_id"])
                n = np.asarray(normal, dtype=np.float64)
                g_xy = gradient_full[offset:offset + 2]
                gradient_full[offset:offset + 2] = g_xy - float(g_xy @ n) * n
            cache.g_full = gradient_full
        return restrict_gradient(cache.g_full, mode)

    def callback(z_active):
        value = objective(z_active)
        grad = gradient(z_active)
        # cache.z_full is the control that was actually evaluated, i.e. the
        # projected one when this iterate was corrected in "project" mode.
        control = np.asarray(cache.z_full, dtype=np.float64).copy()
        if bound_case != MESH_DOMAIN_BOUND_CASE:
            validate_control_for_region(
                control, "iteration", bound_case, admissibility
            )
        iteration = int(history[-1]["iteration"]) + 1 if history else 1
        if (
            feasibility_mode == "raw"
            and bound_case == MESH_DOMAIN_BOUND_CASE
            and not admissibility.validate_sources_in_domain(control)
        ):
            offenders = []
            for source_id, (sx, sy, samp) in enumerate(reshape_control(control), 1):
                if admissibility.is_point_in_domain(float(sx), float(sy)):
                    continue
                dist_sq, _g = admissibility.nearest_domain_distance_sq(
                    float(sx), float(sy)
                )
                offenders.append(
                    {
                        "source_id": source_id,
                        "center": [float(sx), float(sy)],
                        "amplitude": float(samp),
                        "distance_outside_fluid": float(np.sqrt(max(dist_sq, 0.0))),
                    }
                )
            if feasibility_stats is not None:
                feasibility_stats["raw_infeasible_iterates"] += 1
                feasibility_stats["raw_infeasible_iterate_log"].append(
                    {
                        "iteration": iteration,
                        "objective": float(value),
                        "sources": offenders,
                    }
                )
            print(
                "  [raw feasibility] iterate has "
                f"{len(offenders)} source center(s) OUTSIDE the fluid mesh: "
                + "; ".join(
                    f"s{o['source_id']} ({o['center'][0]:.4f},{o['center'][1]:.4f}) "
                    f"A={o['amplitude']:.4f} {o['distance_outside_fluid']:.4f} m outside"
                    for o in offenders
                )
            )
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


def build_source_count_polish_fn(
    args,
    mode: str,
    mesh_data,
    true_state: np.ndarray,
    admissibility: MeshAdmissibility,
    feasibility_mode: str,
    method: str,
    max_iter: int,
):
    """Per-level polish callback for ``source_count_inference``.

    Reuses the driver's own ``build_evaluator`` (same PDE, observation operator,
    feasibility mode and geometry handling) and the ``--post-init-method``
    plumbing, so an inference polish is the same optimizer a production run
    uses -- only the amplitude LOWER bound differs, and that is the point: the
    caller passes 0 so the m-source model stays nested inside the (m+1)-source
    model.  Returns ``(polished_control, objective, info)``.
    """

    def polish(control, amplitude_lower):
        control = np.asarray(control, dtype=np.float64).reshape(-1)
        num_sources = infer_num_sources(control)
        bounds = mesh_domain_active_bounds_for_mode(
            mode, admissibility.bounding_box, num_sources=num_sources
        )
        # Amplitude lower bound is the caller's (0 during inference), NOT the
        # production --amplitude-floor.
        bounds = apply_amplitude_floor(
            bounds, mode, float(amplitude_lower), num_sources=num_sources
        )
        fixed_values = mode_fixed_values(mode, z_ref_full=control)
        feasibility_stats = new_feasibility_stats(feasibility_mode)
        _cache, _history, objective, gradient, _callback = build_evaluator(
            mesh_data,
            true_state,
            mode,
            args.bound_case,
            fixed_values,
            control,
            admissibility,
            feasibility_mode=feasibility_mode,
            feasibility_stats=feasibility_stats,
        )
        z0 = pack_active(control, mode)
        lower = np.asarray([b[0] for b in bounds], dtype=np.float64)
        upper = np.asarray([b[1] for b in bounds], dtype=np.float64)
        z0 = np.clip(z0, lower, upper)

        with pio.solve_counter(gt) as counts:
            initial_objective = float(objective(z0))
            if method == pio.LBFGSB:
                result = Optimizer(
                    objective_function=objective,
                    gradient_function=gradient,
                    optimization_method="L-BFGS-B",
                    max_iter=int(max_iter),
                    tol=1.0e-8,
                    bounds=bounds,
                    callback=None,
                    options={
                        "ftol": 1.0e-14,
                        "gtol": 1.0e-9,
                        "maxls": 20,
                        "disp": False,
                    },
                ).optimize(
                    initial_guess=z0,
                    mesh_data=mesh_data,
                    problem_data={"pde": PDE, "mode": mode},
                )
            else:
                hess = pio.build_active_hessian(
                    method=method,
                    mode=mode,
                    num_sources=num_sources,
                    mesh_data=mesh_data,
                    pde=PDE,
                    target_state=true_state,
                    z_template=control,
                    fixed_values=fixed_values,
                    eta1=ETA1,
                    lam=LAMBDA_REG,
                    unpack_active=unpack_active,
                    hessian_module=_hessian_utils(),
                )
                result = pio.run_newton_type_polish(
                    method=method,
                    objective=objective,
                    gradient=gradient,
                    hess=hess,
                    x0=z0,
                    bounds=bounds,
                    max_iter=int(max_iter),
                )
            z_opt = np.asarray(result.x, dtype=np.float64)
            polished = unpack_active(z_opt, control, mode, fixed_values)
            final_objective = float(objective(z_opt))

        info = {
            "method": method,
            "iterations": int(getattr(result, "nit", -1)),
            "success": bool(result.success),
            "message": str(result.message),
            "initial_objective": initial_objective,
            "final_objective": final_objective,
            "amplitude_lower_bound": float(amplitude_lower),
            "n_forward_solves": int(counts["forward_solves"]),
            "n_adjoint_solves": int(counts["adjoint_solves"]),
            "max_iter": int(max_iter),
        }
        return polished, final_objective, info

    return polish


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
    feasibility=None,
    observation=None,
    post_init=None,
    final_feasibility=None,
) -> tuple[Path, Path]:
    # `true_control` is a synthetic-benchmark scoring input ONLY: in a real
    # application the truth is unknown, so it must never decide the reported
    # source count, control dimension, headers, or run validity. Every
    # dimension/label below follows the REPORTED (final) control instead.
    final_num_sources = infer_num_sources(final_control)
    initial_num_sources = infer_num_sources(initial_control)
    true_num_sources = infer_num_sources(true_control)  # scoring only
    num_sources = final_num_sources

    recovery_matched, recovery_match_error = safe_match_sources_by_permutation(
        true_control, final_control
    )
    initial_recovery_matched, initial_recovery_match_error = (
        safe_match_sources_by_permutation(true_control, initial_control)
    )

    summary = {
        "optimization_mode": mode,
        "num_sources": int(num_sources),
        "final_num_sources": int(final_num_sources),
        "initial_num_sources": int(initial_num_sources),
        "true_num_sources": int(true_num_sources),
        "inferred_matches_true_count": bool(final_num_sources == true_num_sources),
        "control_dim": int(control_dim(num_sources)),
        "case_name": case_name,
        "bound_case": bound_case,
        "source_region": bound_case,
        "initialization_mode": init_mode,
        # Which optimizer polished the initialized control. Initialization is
        # unchanged by this choice; "lbfgsb" is the historical default.
        "post_init_method": (
            pio.DEFAULT_POST_INIT_METHOD
            if post_init is None
            else post_init.get("post_init_method", pio.DEFAULT_POST_INIT_METHOD)
        ),
        "post_init": post_init,
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
        # Source labels are unidentifiable, so recovery errors are reported
        # after the optimal label permutation (exhaustive over M!). Undefined
        # (None, with an explanatory error string) when the reported control's
        # source count differs from the synthetic truth's -- see
        # safe_match_sources_by_permutation.
        "recovery_permutation_matched": recovery_matched,
        "recovery_permutation_match_error": recovery_match_error,
        "initial_recovery_permutation_matched": initial_recovery_matched,
        "initial_recovery_permutation_match_error": initial_recovery_match_error,
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
        # Full node-sensor provenance (ids, coordinates, weights, snapping) so
        # every node-mode run is reproducible from its summary alone.
        "observation": (
            {"observation_mode": "boundary"}
            if observation is None
            else observation.metadata()
        ),
        "building_boundary_condition": "natural no-flux",
        "feasibility": (dict(feasibility) if feasibility is not None else None),
        "final_feasibility_polish": final_feasibility,
        "invalid_control_penalty": (
            {
                "baseline_floor": INVALID_CONTROL_PENALTY_FLOOR,
                "objective_scale_factor": INVALID_CONTROL_PENALTY_OBJECTIVE_FACTOR,
                "distance_weight": INVALID_CONTROL_DISTANCE_WEIGHT,
                "scheme": (
                    "max(floor, factor * largest_valid_objective) + "
                    "distance_weight * sum_i dist_sq(center_i, fluid_domain)"
                ),
            }
            if bound_case == MESH_DOMAIN_BOUND_CASE
            else None
        ),
        "geometry_constraint": (
            "DC fluid-mesh point location (21 building holes excluded exactly by "
            "the mesh) with a large objective-scale-aware invalid-iterate penalty "
            "plus a smooth squared-distance restoring gradient"
            if bound_case == MESH_DOMAIN_BOUND_CASE
            else "legacy DC bounding-box source boxes plus explicit geometry "
            "validation"
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
        header="iteration,objective,gradient_norm,"
        + ",".join(component_names(num_sources)),
        comments="",
    )
    return summary_path, history_path


def match_sources_by_permutation(true_control, candidate_control) -> dict:
    """Permutation-invariant per-source recovery errors for arbitrary M.

    Source labels are not identifiable -- the objective is symmetric under
    relabelling -- so errors are reported after the label permutation that
    minimizes the total position error.  Exhaustive over M! assignments (2 for
    M=2, 6 for M=3, 24 for M=4), which is exact and cheap at the source counts
    Application1 runs.  ``permutation[i]`` is the candidate source matched to
    true source ``i``.
    """
    true_sources = reshape_control(true_control)
    candidate_sources = reshape_control(candidate_control)
    num_sources = int(true_sources.shape[0])
    if candidate_sources.shape[0] != num_sources:
        raise ValueError(
            f"Cannot match a {candidate_sources.shape[0]}-source control "
            f"against a {num_sources}-source truth"
        )

    best_perm, best_cost = None, np.inf
    for perm in permutations(range(num_sources)):
        cost = sum(
            float(np.hypot(*(candidate_sources[perm[i], :2] - true_sources[i, :2])))
            for i in range(num_sources)
        )
        if cost < best_cost:
            best_perm, best_cost = perm, cost
    assert best_perm is not None

    position_errors = [
        float(np.hypot(*(candidate_sources[best_perm[i], :2] - true_sources[i, :2])))
        for i in range(num_sources)
    ]
    amplitude_errors = [
        float(abs(candidate_sources[best_perm[i], 2] - true_sources[i, 2]))
        for i in range(num_sources)
    ]
    return {
        "num_sources": num_sources,
        "permutation": [int(p) for p in best_perm],
        "labels_permuted": tuple(best_perm) != tuple(range(num_sources)),
        "position_errors_m": position_errors,
        "amplitude_errors": amplitude_errors,
        "max_position_error_m": float(max(position_errors)),
        "max_amplitude_error": float(max(amplitude_errors)),
        "total_position_error_m": float(best_cost),
    }


def safe_match_sources_by_permutation(true_control, candidate_control):
    """Best-effort ``match_sources_by_permutation`` for SCORING/REPORTING only.

    ``match_sources_by_permutation`` correctly raises when the two controls
    describe different numbers of sources -- a permutation match is
    mathematically undefined there. But that is a real, expected situation
    once ``--infer-num-sources`` can return an M different from
    ``--true-control``'s M (real applications don't know the truth at all),
    so every REPORTING call site must not crash the run over it. This wraps
    the call and returns ``(None, reason)`` instead of raising when the
    counts disagree; ``match_sources_by_permutation`` itself is left exactly
    as strict as before for any caller that needs the hard guarantee.

    Returns ``(match_dict_or_None, error_message_or_None)``.
    """
    true_m = infer_num_sources(true_control)
    candidate_m = infer_num_sources(candidate_control)
    if true_m != candidate_m:
        return None, (
            f"Permutation recovery is undefined because true M={true_m} "
            f"but candidate M={candidate_m}."
        )
    return match_sources_by_permutation(true_control, candidate_control), None


def validate_control_length(values, label: str, num_sources: int) -> np.ndarray:
    """Return ``values`` as a float array of exactly ``3 * num_sources`` entries.

    Controls are never silently truncated or padded: a wrong length is an error
    that names the expected length and the ordering.
    """
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    expected = control_dim(num_sources)
    if array.size != expected:
        raise ValueError(
            f"{label} must have exactly {expected} entries for "
            f"--num-sources {int(num_sources)} "
            f"([x1, y1, A1, ..., x{int(num_sources)}, y{int(num_sources)}, "
            f"A{int(num_sources)}]); got {array.size}"
        )
    return array


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-sources",
        type=int,
        default=NUM_SOURCES,
        metavar="M",
        help=(
            "Number of Gaussian sources M (positive integer, default "
            f"{NUM_SOURCES}). The control has length 3*M with ordering "
            "[x1, y1, A1, ..., xM, yM, AM]. Any supplied --true-control / "
            "--initial-control must have exactly 3*M entries. Values other "
            "than the default require an explicit --true-control (Application1 "
            "only ships two-source reference/initial defaults)."
        ),
    )
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
        "--post-init-method",
        choices=tuple(pio.POST_INIT_METHODS),
        default=pio.DEFAULT_POST_INIT_METHOD,
        help=(
            "Optimizer that POLISHES the initialized control. Initialization "
            "(--init-mode / --initial-control) is unaffected and still produces "
            "the starting control exactly as before. 'lbfgsb' (default) is the "
            "historical L-BFGS-B run and leaves existing runs unchanged. "
            "'gauss_newton' and 'newton' run a bound-constrained trust-region "
            "Newton driven by the dense reduced Hessian from hessian_utils: "
            "'gauss_newton' uses the residual-through-state Gauss-Newton "
            "Hessian, 'newton' uses the exact Hessian (which additionally "
            "contracts the base adjoint with D^2 S(z)). Both cost one tangent "
            "forward solve plus one incremental adjoint solve per control "
            "component per Hessian, so they pay more per iteration and are "
            "intended for polishing an already-good initialization."
        ),
    )
    parser.add_argument(
        "--post-init-scaling",
        choices=tuple(pio.POST_INIT_SCALINGS),
        default=pio.DEFAULT_POST_INIT_SCALING,
        help=(
            "Diagonal variable scaling for the L-BFGS-B polish (--post-init-method "
            "lbfgsb only). 'none' (default) is the historical unscaled run. "
            "'gn-diag' computes the Gauss-Newton Hessian at the initialized "
            "control (3M+1 forward + 3M adjoint solves) and runs L-BFGS-B in "
            "y = D z, D = sqrt(diag H_GN); positions and amplitudes have "
            "curvatures orders of magnitude apart, which unscaled L-BFGS-B "
            "learns slowly. 'on-stall' runs the unscaled polish and switches to "
            "gn-diag scaling from the current iterate only if it has not "
            "converged after --post-init-stall-iters iterations; runs that "
            "converge before then are unchanged. With scaling, gtol applies to "
            "the scaled gradient."
        ),
    )
    parser.add_argument(
        "--post-init-stall-iters",
        type=int,
        default=pio.DEFAULT_STALL_ITERS,
        help=(
            "Unscaled L-BFGS-B iterations before --post-init-scaling on-stall "
            f"switches to scaling (default {pio.DEFAULT_STALL_ITERS}); counts "
            "toward --max-iter."
        ),
    )
    parser.add_argument(
        "--post-init-scaling-max-ratio",
        type=float,
        default=pio.DEFAULT_SCALING_MAX_RATIO,
        help=(
            "Cap on max(D)/min(D) for --post-init-scaling (default "
            f"{pio.DEFAULT_SCALING_MAX_RATIO:g}): entries of D below "
            "max(D)/ratio are raised to it, so weakly observed variables do not "
            "get huge scaled steps."
        ),
    )
    parser.add_argument(
        "--post-init-lbfgsb-maxcor",
        type=int,
        default=None,
        help=(
            "L-BFGS-B memory (maxcor) for the polish; default: not passed, i.e. "
            "SciPy's 10, as in all earlier runs."
        ),
    )
    parser.add_argument(
        "--infer-num-sources",
        action="store_true",
        help=(
            "Infer the number of sources M instead of taking it from "
            "--num-sources. Drives the existing sequential adjoint-sensitivity "
            "machinery one source at a time: for each m it polishes the "
            "m-source model, scores the best next-source birth on the residual "
            "(exact bounded 1-D predicted reduction), inserts and polishes the "
            "candidate, and accepts only when the ACTUAL decrease clears the "
            "tolerance AND no source collapses. Candidate models are polished "
            "with amplitude lower bound 0 so the m-source model stays NESTED in "
            "the (m+1)-source model -- the production --amplitude-floor is never "
            "used as evidence of a source. --num-sources is ignored."
        ),
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=sci.DEFAULT_MAX_SOURCES,
        help=(
            "Upper limit on the inferred source count for --infer-num-sources "
            f"(default {sci.DEFAULT_MAX_SOURCES})."
        ),
    )
    parser.add_argument(
        "--source-count-discrepancy-tol",
        type=float,
        default=sci.DEFAULT_DISCREPANCY_TOL,
        help=(
            "Absolute objective value at/below which the residual counts as "
            "explained and inference stops (the discrepancy principle / "
            "numerical floor). Default 0.0 disables this stop, leaving the "
            "birth test to terminate the sequence."
        ),
    )
    parser.add_argument(
        "--source-count-rel-reduction-tol",
        type=float,
        default=sci.DEFAULT_REL_REDUCTION_TOL,
        help=(
            "A new source is accepted only when the objective decrease exceeds "
            "this fraction of the current objective (default "
            f"{sci.DEFAULT_REL_REDUCTION_TOL:g}). Required because the models "
            "are nested: J_(m+1) <= J_m always, so a decrease alone is not "
            "evidence."
        ),
    )
    parser.add_argument(
        "--source-count-abs-reduction-tol",
        type=float,
        default=sci.DEFAULT_ABS_REDUCTION_TOL,
        help=(
            "Absolute floor on the accepted objective decrease; the effective "
            "threshold is max(abs, rel * J_m). Default "
            f"{sci.DEFAULT_ABS_REDUCTION_TOL:g}."
        ),
    )
    parser.add_argument(
        "--source-count-amp-tol",
        type=float,
        default=sci.DEFAULT_AMP_TOL,
        help=(
            "A candidate model whose smallest polished amplitude is <= this is "
            "rejected as degenerate (it collapsed back to fewer effective "
            f"sources). Default {sci.DEFAULT_AMP_TOL:g}. This is the test that "
            "the zero amplitude lower bound makes meaningful."
        ),
    )
    parser.add_argument(
        "--source-count-top-k",
        type=int,
        default=sci.DEFAULT_TOP_K,
        help=(
            "Basin-diverse screen size for each birth test (default "
            f"{sci.DEFAULT_TOP_K}). Each screened candidate costs one cached "
            "unit forward solve."
        ),
    )
    parser.add_argument(
        "--source-count-polish-method",
        choices=tuple(pio.POST_INIT_METHODS),
        default=pio.DEFAULT_POST_INIT_METHOD,
        help=(
            "Optimizer used for the per-level polish during inference, from the "
            "same set as --post-init-method (default "
            f"{pio.DEFAULT_POST_INIT_METHOD})."
        ),
    )
    parser.add_argument(
        "--source-count-polish-max-iter",
        type=int,
        default=60,
        help="Iteration cap for each per-level inference polish (default 60).",
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
            "The last stage is the target problem. A start is aborted rather "
            "than warm-started if a stage optimum leaves the fluid mesh domain. "
            "If omitted, a single optimization at PDE['sigma2'] is run "
            "(existing behavior)."
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
        "--observation-mode",
        choices=OBSERVATION_MODES,
        default=DEFAULT_OBSERVATION_MODE,
        help=(
            "Observation operator. 'boundary' (default) is the historical "
            "boundary-integral misfit over --measurement-boundaries and is "
            "completely unchanged. 'nodes' observes the FE state at discrete "
            "mesh nodes: C_s U^n = (U^n_j1, ..., U^n_jNs). Select the sensors "
            "with --sensor-node-ids or --sensor-coordinates, or use "
            "--sensor-weights boundary_quadrature to build the exact "
            "node-equivalent of the selected measurement boundaries."
        ),
    )
    parser.add_argument(
        "--sensor-node-ids",
        nargs="+",
        metavar="IDS",
        help=(
            "Solver mesh-node ids of the point sensors, comma and/or space "
            "separated, for example --sensor-node-ids 41,82,164,233. Requires "
            "--observation-mode nodes. Takes precedence over "
            "--sensor-coordinates."
        ),
    )
    parser.add_argument(
        "--sensor-coordinates",
        metavar="COORDS",
        help=(
            "Sensor locations as \"x1,y1;x2,y2;...\" in local DC meters, each "
            "snapped to the nearest fluid mesh node (requested coordinate, "
            "resolved node id, mesh coordinate and snapping distance are all "
            "reported). Requires --observation-mode nodes."
        ),
    )
    parser.add_argument(
        "--sensor-snap-tolerance",
        type=float,
        default=DEFAULT_SENSOR_SNAP_TOLERANCE,
        metavar="METERS",
        help=(
            "Maximum admissible --sensor-coordinates snapping distance in "
            f"meters (default {DEFAULT_SENSOR_SNAP_TOLERANCE}). A requested "
            "coordinate further than this from every mesh node is rejected "
            "rather than silently snapped across the domain."
        ),
    )
    parser.add_argument(
        "--sensor-weights",
        choices=SENSOR_WEIGHT_MODES,
        default=DEFAULT_SENSOR_WEIGHT_MODE,
        help=(
            "Spatial weighting W_s of the node-sensor residual. 'equal' "
            "(default) uses equal positive weights, i.e. the plain Euclidean "
            "sensor sum sum_s w |r_s|^2 with w = --sensor-equal-weight. "
            "'boundary_quadrature' uses the exact P1 boundary observation "
            "matrix of --measurement-boundaries (a symmetric matrix, NOT a "
            "diagonal weighting) so the node objective reproduces the boundary "
            "objective; with no explicit sensor selection it also defaults the "
            "sensor set to every node of those boundaries."
        ),
    )
    parser.add_argument(
        "--sensor-equal-weight",
        type=float,
        default=DEFAULT_SENSOR_EQUAL_WEIGHT,
        metavar="W",
        help=(
            "Positive weight used by --sensor-weights equal (default "
            f"{DEFAULT_SENSOR_EQUAL_WEIGHT})."
        ),
    )
    parser.add_argument(
        "--true-control",
        nargs="+",
        type=float,
        metavar="V",
        help=(
            "Reference control as 3*M values: x1 y1 A1 ... xM yM AM (meters, "
            "meters, amplitude), where M is --num-sources. Defaults to "
            "Application1 TRUE_CONTROL (two sources)."
        ),
    )
    parser.add_argument(
        "--initial-control",
        nargs="+",
        type=float,
        metavar="V",
        help=(
            "Initial control as 3*M values: x1 y1 A1 ... xM yM AM (meters, "
            "meters, amplitude), where M is --num-sources. Defaults to "
            "Application1 INITIAL_CONTROL (two sources). Authoritative: when "
            "supplied it overrides every generated initialization."
        ),
    )
    parser.add_argument(
        "--init-mode",
        choices=("manual", "mesh_random", "smart", "adaptive", "sensitivity"),
        default="manual",
        help=(
            "Initialization strategy. Defaults to manual. 'mesh_random' samples "
            "admissible starts with uniform amplitudes in the full amplitude "
            "range. 'smart' samples the same admissible mesh-domain centers but "
            "draws balanced amplitudes from a moderate interval and enforces a "
            "minimum source separation, avoiding near-dead and near-saturated "
            "starts. 'adaptive' runs starts sequentially and uses the outcomes "
            "of completed starts to perturb promising candidates or draw fresh "
            "smart starts (see --adaptive-* options). 'sensitivity' greedily "
            "places the requested number of Gaussian sources by following the "
            "adjoint source-insertion sensitivity field (see --sensitivity-* "
            "options), then hands the assembled control to the full optimizer. "
            "No mode uses the true control for initialization or selection."
        ),
    )
    parser.add_argument(
        "--sensitivity-score",
        choices=("raw_g", "predicted_reduction"),
        default="predicted_reduction",
        help=(
            "Sensitivity init placement score. 'predicted_reduction' (default) "
            "screens the top-K basin-diverse G minima and selects by the EXACT "
            "bounded one-dimensional objective reduction "
            "dJ(c) = a* G(c) + 0.5 a*^2 kappa(c) (one unit forward solve per "
            "screened candidate). 'raw_g' is the original argmin G(c) rule; it "
            "is retained for comparison but is biased toward hyper-observable "
            "near-boundary candidates and demonstrably misplaces sources on "
            "Application1 (see Applications/Application1/README.md)."
        ),
    )
    parser.add_argument(
        "--sensitivity-top-k",
        type=int,
        default=24,
        help=(
            "Sensitivity init (predicted_reduction): number of basin-diverse "
            "candidates screened per greedy/beam step (one unit forward solve "
            "each, cached across beam branches/levels). Must exceed the basin "
            "count of the dominant plume for weakly observed sources to enter "
            "the screen (the DC left+right+bottom case needs ~19). Defaults "
            "to 24."
        ),
    )
    parser.add_argument(
        "--sensitivity-beam-width",
        type=int,
        default=1,
        help=(
            "Sensitivity init (predicted_reduction): beam width for multi-"
            "source placement. B > 1 retains B first-source hypotheses, expands "
            "each with its own residual/adjoint recomputation, jointly refits "
            "all amplitudes, and returns the complete control with the smallest "
            "exact objective. Defaults to 1 (plain greedy)."
        ),
    )
    parser.add_argument(
        "--sensitivity-min-value-tol",
        type=float,
        default=SENSITIVITY_DEFAULT_MIN_VALUE_TOL,
        help=(
            "Sensitivity init: a greedy insertion is accepted only when the "
            "minimum masked insertion sensitivity G(c*) < -tol (meaningful "
            f"descent for a nonnegative amplitude). Defaults to "
            f"{SENSITIVITY_DEFAULT_MIN_VALUE_TOL}."
        ),
    )
    parser.add_argument(
        "--sensitivity-min-separation",
        type=float,
        default=None,
        help=(
            "Sensitivity init: minimum separation (m) between greedily placed "
            "source centers. Defaults to --min-source-separation when set, "
            "otherwise disabled."
        ),
    )
    parser.add_argument(
        "--sensitivity-refine-centers",
        action="store_true",
        help=(
            "Sensitivity init (predicted_reduction, beam): after the retained "
            "first-source beam branches are chosen, locally refine each branch's "
            "source center by VARIABLE PROJECTION (only the spatial coordinates "
            "are optimized; the amplitude is eliminated by the exact bounded fit) "
            "BEFORE recomputing that branch's residual sensitivity field for the "
            "second source. Branches that refine to the same basin are "
            "deduplicated. Reuses the existing forward solver and box-QP "
            "amplitude fit; introduces no approximate physics. OFF by default "
            "(default reproduces current behavior)."
        ),
    )
    parser.add_argument(
        "--sensitivity-refine-max-fev",
        type=int,
        default=50,
        help=(
            "Sensitivity init: maximum forward evaluations of the Nelder-Mead "
            "variable-projected center refinement, per beam branch. Only used "
            "with --sensitivity-refine-centers. Defaults to 50."
        ),
    )
    parser.add_argument(
        "--sensitivity-coherence-cap",
        type=float,
        default=None,
        help=(
            "Sensitivity init (predicted_reduction, beam): optional dimensionless "
            "cap in [0, 1] on the observation-space coherence "
            "mu = <t1,t2> / (||t1|| ||t2||) between an already-placed source "
            "response t1 and a screened second-source candidate response t2 "
            "(same weighted inner product as the objective). Candidates with "
            "mu above the cap are rejected. This is an ADDITIONAL screen layered "
            "on top of the existing minimum-distance logic, not a replacement. "
            "Defaults to None (disabled; reproduces current behavior)."
        ),
    )
    parser.add_argument(
        "--feasibility-mode",
        choices=list(FEASIBILITY_MODES),
        default=DEFAULT_FEASIBILITY_MODE,
        help=(
            "How an evaluated control whose source center left the fluid mesh is "
            "handled (--bound-case mesh_domain only). 'penalty' (default) keeps "
            "the historical objective-scale-aware penalty, which is DISCONTINUOUS "
            "at the fluid boundary and can abort the L-BFGS-B line search when an "
            "iterate sits within a step of a wall. 'project' projects the "
            "offending centers onto the fluid domain and evaluates the true "
            "objective and the chain-rule gradient there, which is continuous "
            "across the wall. 'raw' is a DIAGNOSTIC mode: source centers are "
            "allowed to enter building holes during intermediate iterations, the "
            "true objective and the FULL adjoint gradient are evaluated at the raw "
            "center (no projection, no penalty, no wall-normal component removed), "
            "and the forward/adjoint solves still run only on the fluid-domain "
            "mesh, so a center inside a building contributes only its Gaussian "
            "tail. The rectangular outer bounds and the amplitude bounds still "
            "apply. In 'raw' mode a continuation stage may end on an inadmissible "
            "center; that is reported, not aborted, and every accepted iterate "
            "with a center outside the fluid mesh is logged. 'penalty' and "
            "'project' do not change which controls are admissible."
        ),
    )
    parser.add_argument(
        "--final-feasibility-polish",
        action="store_true",
        help=(
            "After normal optimization (and any raw-mode polishing), run a final "
            "short constrained repair/certification pass on the accepted control: "
            "certify every source center is inside the fluid mesh with at least "
            "--final-feasibility-margin clearance, or repair it with a short "
            "constrained solve if not. Does not replace the main optimizer and "
            "does not change source-count inference. OFF by default -- existing "
            "behavior (including 'raw' mode's inadmissible final centers) is "
            "unchanged unless this flag is given."
        ),
    )
    parser.add_argument(
        "--final-feasibility-margin",
        type=float,
        default=DEFAULT_FINAL_FEASIBILITY_MARGIN,
        help=(
            "Required signed clearance (meters) from every source center to the "
            "fluid boundary for the final feasibility polish to certify a control "
            "as feasible without repair. Only used with "
            "--final-feasibility-polish."
        ),
    )
    parser.add_argument(
        "--final-feasibility-max-iter",
        type=int,
        default=DEFAULT_FINAL_FEASIBILITY_MAX_ITER,
        help=(
            "Iteration cap for the final feasibility repair solve. This is a "
            "short certification/repair pass, not the main optimizer. Only used "
            "with --final-feasibility-polish."
        ),
    )
    parser.add_argument(
        "--final-feasibility-method",
        choices=list(FINAL_FEASIBILITY_METHODS),
        default=DEFAULT_FINAL_FEASIBILITY_METHOD,
        help=(
            "SciPy constrained method used for the final feasibility repair "
            "solve. Only used with --final-feasibility-polish."
        ),
    )
    parser.add_argument(
        "--sensitivity-refine-pair",
        action="store_true",
        help=(
            "Sensitivity init (predicted_reduction, beam): after ALL sources are "
            "placed, jointly refine every center of each retained branch by "
            "variable projection -- a single Nelder-Mead search over "
            "(x1, y1, x2, y2) with both amplitudes eliminated by the existing "
            "two-source bounded box-QP at every trial. This is joint refinement, "
            "not alternating single-source refinement. OFF by default."
        ),
    )
    parser.add_argument(
        "--sensitivity-pair-refine-max-fev",
        type=int,
        default=0,
        help=(
            "Sensitivity init: evaluation budget for --sensitivity-refine-pair. "
            "One pair evaluation costs one uncached forward solve PER SOURCE, so "
            "the forward-solve cost is this number times the source count. "
            "Defaults to 0, meaning reuse --sensitivity-refine-max-fev."
        ),
    )
    parser.add_argument(
        "--sensitivity-reselect-second",
        action="store_true",
        help=(
            "Sensitivity init (predicted_reduction, beam): after joint pair "
            "refinement, re-select the LAST source against a rebuilt residual. "
            "The last source was originally screened against the residual of the "
            "placement as it stood before refinement; if refinement then moved "
            "the leading source(s) materially, that residual is stale. This "
            "stage recomputes the leading component's forward solution, the "
            "residual, its adjoint and the insertion-sensitivity field, "
            "rescreens the last source, refits both amplitudes by variable "
            "projection, re-refines both centers jointly, and keeps whichever "
            "control has the smaller exact objective. OFF by default."
        ),
    )
    parser.add_argument(
        "--sensitivity-reselect-threshold",
        type=float,
        default=0.25,
        help=(
            "Sensitivity init: staleness trigger for "
            "--sensitivity-reselect-second, as a fraction of one Gaussian radius "
            "sqrt(sigma2) of leading-source movement, or the same fraction as a "
            "relative fitted-amplitude change. Defaults to 0.25."
        ),
    )
    parser.add_argument(
        "--sensitivity-reselect-max-passes",
        type=int,
        default=2,
        help=(
            "Sensitivity init: maximum re-selection passes per branch for "
            "--sensitivity-reselect-second. A pass also stops early once the "
            "selection is stable or the objective gain is immaterial. "
            "Defaults to 2."
        ),
    )
    parser.add_argument(
        "--sensitivity-reselect-at-target-width",
        action="store_true",
        help=(
            "Sensitivity init: run --sensitivity-reselect-second at the TARGET "
            "sigma2 instead of the first continuation width. Two candidate "
            "streets closer than one Gaussian radius produce nearly identical "
            "observation traces, so at a broad initialization width the screen "
            "cannot tell them apart; at the target width it can. Every retained "
            "branch is re-evaluated at the target width so the comparison and "
            "the final ranking stay on one scale. Only meaningful with "
            "--sigma2-continuation. OFF by default."
        ),
    )
    parser.add_argument(
        "--sensitivity-reselect-refine-max-fev",
        type=int,
        default=0,
        help=(
            "Sensitivity init: evaluation budget for the joint center "
            "refinement inside each re-selection pass. Defaults to 0, meaning "
            "reuse the pair-refinement budget."
        ),
    )
    parser.add_argument(
        "--sensitivity-geometry-aware-branches",
        action="store_true",
        help=(
            "Sensitivity init (predicted_reduction, beam): make beam-branch "
            "handling obstacle-aware. Two branches are the same basin only if "
            "their centers are close AND mutually visible through the fluid "
            "domain, so branches separated by a building are no longer "
            "deduplicated; and each branch's variable-projected refinement is "
            "confined to the region visible from its own starting center, so a "
            "branch cannot migrate into another street and collapse the beam's "
            "geometric diversity. OFF by default."
        ),
    )
    parser.add_argument(
        "--sensitivity-obstacle-aware-diversity",
        action="store_true",
        help=(
            "Sensitivity init (predicted_reduction): make the top-K DIVERSITY "
            "screen obstacle-aware. By default a candidate within the diversity "
            "radius of an already-retained candidate is suppressed on Euclidean "
            "distance alone, which discards a candidate that is close but "
            "separated by a building -- a different street, not a duplicate "
            "basin. With this flag a close candidate is suppressed only when the "
            "straight segment joining the two also stays inside the admissible "
            "fluid domain; leaving through a building wall OR the outer domain "
            "boundary counts as separated and both candidates survive. Can only "
            "retain more candidates, never fewer. OFF by default."
        ),
    )
    parser.add_argument(
        "--sensitivity-branch-multistart",
        action="store_true",
        help=(
            "Sensitivity init (predicted_reduction, beam): run the full "
            "continuation/optimization from EVERY retained beam branch instead of "
            "only the single best-objective branch, and select the run with the "
            "smallest exact final objective. The number of starts is the number "
            "of distinct retained branches (at most --sensitivity-beam-width). "
            "OFF by default."
        ),
    )
    parser.add_argument(
        "--amplitude-solver",
        choices=tuple(sinit.AMPLITUDE_SOLVERS),
        default=sinit.DEFAULT_AMPLITUDE_SOLVER,
        help=(
            "Solver for the JOINT amplitude fit of the sensitivity "
            "initialization: for fixed source centers, "
            "min_{A_min<=a<=A_max} 0.5 a^T H a - a^T h in the exact weighted "
            "observation/time norm, with H = Gram(unit responses) + lambda I. "
            "'osqp' (default) hands that same QP to OSQP (P = H symmetrized, "
            "q = -h, A = I, l/u the amplitude bounds) and has no source-count "
            "cap, so it covers the intended 8-10 source problems; it raises on "
            "a bad termination status or a bound violation and never falls back. "
            "'enumeration' is the original exact 3^n active-set enumeration: "
            f"dependency-free but exponential, and refused above n = "
            f"{sinit._ENUMERATION_MAX_SOURCES}. The selection reaches the "
            "initialization, the variable-projected center-refinement trials "
            "and second-source re-selection alike; it does NOT change the "
            "closed-form single-source ranking amplitude a* = -G(c)/kappa(c)."
        ),
    )
    parser.add_argument(
        "--sensitivity-save-plots",
        action="store_true",
        help=(
            "Sensitivity init: save per-step PNGs distinguishing the integrated "
            "adjoint field S(x) from the insertion sensitivity field G(c). OFF "
            "by default."
        ),
    )
    parser.add_argument(
        "--sensitivity-mask-boundary-buffer",
        type=float,
        default=0.0,
        help=(
            "Sensitivity init: optional buffer (m) that excludes candidate "
            "centers within this distance of the outer domain boundary. Defaults "
            "to 0.0 (disabled; no extra physical assumption)."
        ),
    )
    parser.add_argument(
        "--sensitivity-mask-building-buffer",
        type=float,
        default=0.0,
        help=(
            "Sensitivity init: optional buffer (m) that excludes candidate "
            "centers within this distance of any building wall (building "
            "interiors are already excluded by mesh admissibility). Defaults to "
            "0.0 (disabled)."
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
        "--adaptive-reinsert-dead-sources",
        action="store_true",
        help=(
            "Adaptive mode: when a completed start ends in partial recovery "
            "(one source recovered, another collapsed to the amplitude floor), "
            "spawn a start that revives the dead source using the source-"
            "insertion sensitivity field (fixed reinsertion amplitude "
            f"{REINSERTION_AMPLITUDE}), then run the ordinary full sigma-"
            "continuation. OFF by default; enabling it does not change any "
            "other adaptive behavior."
        ),
    )
    parser.add_argument(
        "--adaptive-max-reinsertions",
        type=int,
        default=ADAPTIVE_DEFAULT_MAX_REINSERTIONS,
        help=(
            "Adaptive mode: maximum dead-source sensitivity reinsertions for the "
            f"whole run. Defaults to {ADAPTIVE_DEFAULT_MAX_REINSERTIONS}. Only "
            "used with --adaptive-reinsert-dead-sources."
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
            "NPZ file with coords and velocity_nodes in the current Application1 "
            "mesh node order. If omitted, the Application1 Navier-Stokes velocity "
            "field is auto-loaded from output/navier_stokes/ (unless "
            "--constant-velocity is set)."
        ),
    )
    parser.add_argument(
        "--constant-velocity",
        action="store_true",
        help=(
            "Use the constant fallback velocity (PDE vx/vy = ambient wind) "
            "instead of the Application1 Navier-Stokes nodal velocity field."
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
    post_init_method = pio.validate_post_init_method(
        getattr(args, "post_init_method", pio.DEFAULT_POST_INIT_METHOD)
    )
    post_init_scaling = getattr(args, "post_init_scaling", pio.DEFAULT_POST_INIT_SCALING)
    post_init_stall_iters = int(
        getattr(args, "post_init_stall_iters", pio.DEFAULT_STALL_ITERS)
    )
    post_init_scaling_max_ratio = float(
        getattr(args, "post_init_scaling_max_ratio", pio.DEFAULT_SCALING_MAX_RATIO)
    )
    post_init_maxcor = getattr(args, "post_init_lbfgsb_maxcor", None)
    if post_init_scaling not in pio.POST_INIT_SCALINGS:
        raise ValueError(f"Unknown --post-init-scaling {post_init_scaling!r}")
    if post_init_method != pio.LBFGSB and (
        post_init_scaling != pio.SCALING_NONE or post_init_maxcor is not None
    ):
        raise ValueError(
            "--post-init-scaling / --post-init-lbfgsb-maxcor apply only to "
            "--post-init-method lbfgsb"
        )
    if post_init_stall_iters < 1:
        raise ValueError("--post-init-stall-iters must be >= 1")
    if post_init_scaling_max_ratio <= 1.0:
        raise ValueError("--post-init-scaling-max-ratio must be > 1")
    if post_init_maxcor is not None and int(post_init_maxcor) < 1:
        raise ValueError("--post-init-lbfgsb-maxcor must be >= 1")
    num_sources = int(infer_num_sources(true_control))
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
    print(f"post-initialization method: {post_init_method}")
    if post_init_method == pio.LBFGSB:
        print(
            f"post-init scaling: {post_init_scaling}"
            + (
                f" (stall after {post_init_stall_iters} iterations)"
                if post_init_scaling == pio.SCALING_ON_STALL
                else ""
            )
            + (f"  maxcor: {post_init_maxcor}" if post_init_maxcor is not None else "")
        )
    if start_index is not None:
        print(f"start index: {start_index}")
    print(f"case name: {case_name or '(default output)'}")
    print(f"source region: {args.bound_case}")
    print(
        "measurement boundaries: "
        f"{list(mesh_data.get('selected_measurement_boundaries', (mesh_data.get('measurement_boundary', 'right'),)))}"
    )
    _observation = mesh_data.get("observation")
    print(
        "observation operator: "
        + (
            "boundary integral"
            if _observation is None
            else f"{_observation.num_sensors} mesh-node sensors "
            f"({_observation.weight_kind} weights)"
        )
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

    feasibility_mode = getattr(args, "feasibility_mode", DEFAULT_FEASIBILITY_MODE)
    feasibility_stats = new_feasibility_stats(feasibility_mode)
    print(f"invalid-control feasibility mode: {feasibility_mode}")

    _cache, history, objective, gradient, callback = build_evaluator(
        mesh_data,
        true_state,
        mode,
        args.bound_case,
        fixed_values,
        initial_template,
        admissibility,
        feasibility_mode=feasibility_mode,
        feasibility_stats=feasibility_stats,
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
    post_init_records: list[dict] = []
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
            feasibility_mode=feasibility_mode,
            feasibility_stats=feasibility_stats,
        )

        stage_projected_before = int(feasibility_stats["projected_evaluations"])
        stage_penalized_before = int(feasibility_stats["penalized_evaluations"])
        stage_hessian_records: list[dict] = []
        # The solve counter only wraps gt.run_forward / gt.rk4_discrete_source_adjoint
        # so both polish methods report exactly comparable solve budgets. It does
        # not touch the numerics of either path.
        with pio.solve_counter(gt) as stage_solve_counts:
            stage_initial_objective = float(stage_objective(stage_z0_active))
            if post_init_method == pio.LBFGSB:
                lbfgsb_options = {
                    "ftol": 1.0e-14,
                    "gtol": 1.0e-9,
                    "maxls": 20,
                    "disp": True,
                }
                if post_init_maxcor is not None:
                    lbfgsb_options["maxcor"] = int(post_init_maxcor)
                stage_gn_hess = None
                if post_init_scaling != pio.SCALING_NONE:
                    stage_gn_hess = pio.build_active_hessian(
                        method=pio.GAUSS_NEWTON,
                        mode=mode,
                        num_sources=num_sources,
                        mesh_data=mesh_data,
                        pde=PDE,
                        target_state=stage_true_state,
                        z_template=current_control,
                        fixed_values=stage_fixed_values,
                        eta1=ETA1,
                        lam=LAMBDA_REG,
                        unpack_active=unpack_active,
                        hessian_module=_hessian_utils(),
                    )
                stage_result, stage_scaling_record = pio.run_lbfgsb_polish(
                    optimizer_cls=Optimizer,
                    objective=stage_objective,
                    gradient=stage_gradient,
                    x0=stage_z0_active,
                    bounds=active_bounds,
                    max_iter=max_iter,
                    tol=1.0e-8,
                    options=lbfgsb_options,
                    callback=stage_callback,
                    mesh_data=mesh_data,
                    problem_data={"pde": PDE, "mode": mode},
                    scaling=post_init_scaling,
                    stall_iters=post_init_stall_iters,
                    max_ratio=post_init_scaling_max_ratio,
                    gn_hess=stage_gn_hess,
                )
            else:
                stage_hess = pio.build_active_hessian(
                    method=post_init_method,
                    mode=mode,
                    num_sources=num_sources,
                    mesh_data=mesh_data,
                    pde=PDE,
                    target_state=stage_true_state,
                    z_template=current_control,
                    fixed_values=stage_fixed_values,
                    eta1=ETA1,
                    lam=LAMBDA_REG,
                    unpack_active=unpack_active,
                    hessian_module=_hessian_utils(),
                    record=stage_hessian_records,
                )
                stage_result = pio.run_newton_type_polish(
                    method=post_init_method,
                    objective=stage_objective,
                    gradient=stage_gradient,
                    hess=stage_hess,
                    x0=stage_z0_active,
                    bounds=active_bounds,
                    max_iter=max_iter,
                    callback=stage_callback,
                )
        stage_solve_counts = dict(stage_solve_counts)

        stage_z_opt = np.asarray(stage_result.x, dtype=np.float64)
        stage_final_control = unpack_active(
            stage_z_opt, current_control, mode, stage_fixed_values
        )
        stage_final_objective = float(stage_objective(stage_z_opt))
        stage_final_gradient = stage_gradient(stage_z_opt)

        # In "project" mode L-BFGS-B may stop at a z whose center lies just
        # outside the mesh but whose PROJECTION is the point that was actually
        # evaluated. Carry the projected control forward so later stages and the
        # final summary describe an admissible source configuration.
        stage_final_projection = None
        if (
            feasibility_mode == "project"
            and args.bound_case == MESH_DOMAIN_BOUND_CASE
            and not admissibility.validate_sources_in_domain(stage_final_control)
        ):
            projected_control, projection_events = project_control_centers(
                stage_final_control, admissibility
            )
            stage_final_projection = [
                {
                    "source_id": int(event["source_id"]),
                    "original": event["original"],
                    "projected": event["projected"],
                    "distance": float(event["distance"]),
                }
                for event in projection_events
            ]
            print(
                f"  stage {stage_index}: optimizer stopped just outside the fluid "
                f"mesh; projected {len(projection_events)} source center(s) back "
                "onto the domain "
                + ", ".join(
                    f"s{int(event['source_id']) + 1} moved {event['distance']:.4g} m"
                    for event in projection_events
                )
            )
            stage_final_control = projected_control
            stage_z_opt = pack_active(stage_final_control, mode)
            stage_final_objective = float(stage_objective(stage_z_opt))
            stage_final_gradient = stage_gradient(stage_z_opt)

        # Abort this start if the stage optimum leaves the fluid mesh domain
        # instead of warm-starting later stages from an invalid control. The
        # objective-scale-aware restoring penalty makes this outcome unlikely,
        # but if a stage still terminates on an inadmissible center (e.g. a
        # line-search stall) propagating it would corrupt every later stage and
        # only surface at the final validation. Raising here keeps the failure
        # local: adaptive/smart multistart records it as a failed start and
        # continues with the next start.
        if (
            args.bound_case == MESH_DOMAIN_BOUND_CASE
            and feasibility_mode != "raw"
            and not admissibility.validate_sources_in_domain(stage_final_control)
        ):
            raise ValueError(
                f"sigma-continuation aborted at stage {stage_index}/"
                f"{len(sigma2_stages)} (sigma2={stage_sigma2:.6g}): optimized "
                f"control {np.array2string(stage_final_control, precision=6)} has "
                "a source center outside the fluid mesh domain; refusing to "
                "warm-start later stages from an invalid control"
            )

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
        print(
            "  stage feasibility: "
            f"projected evals="
            f"{feasibility_stats['projected_evaluations'] - stage_projected_before}  "
            f"penalized evals="
            f"{feasibility_stats['penalized_evaluations'] - stage_penalized_before}"
        )

        stage_post_init = {
            "stage_index": stage_index,
            "post_init_method": post_init_method,
            "iterations": int(stage_result.nit),
            "success": bool(stage_result.success),
            "message": str(stage_result.message),
            "n_objective_evaluations": int(getattr(stage_result, "nfev", 0) or 0),
            "n_gradient_evaluations": int(getattr(stage_result, "njev", 0) or 0),
            "n_hessian_evaluations": int(getattr(stage_result, "nhev", 0) or 0),
            "n_forward_solves": int(stage_solve_counts["forward_solves"]),
            "n_adjoint_solves": int(stage_solve_counts["adjoint_solves"]),
            "stage_initial_objective": float(stage_initial_objective),
            "stage_final_objective": float(stage_final_objective),
        }
        if post_init_method == pio.LBFGSB:
            stage_post_init["lbfgsb_maxcor"] = (
                10 if post_init_maxcor is None else int(post_init_maxcor)
            )
            stage_post_init["scaling"] = stage_scaling_record
        else:
            stage_post_init.update(
                {
                    "scipy_method": pio.NEWTON_SCIPY_METHOD,
                    "hessian_mode": pio.HESSIAN_MODE_BY_METHOD[post_init_method],
                    "hessian_evaluations": stage_hessian_records,
                    "curvature_chain_rule_note": (
                        "the reduced Hessian is assembled at the evaluated "
                        "control and does NOT include the projection/penalty "
                        "chain rule, so feasibility modes other than 'raw' can "
                        "see inconsistent curvature on invalid iterates"
                        if feasibility_mode != "raw"
                        else "feasibility mode 'raw': objective, gradient and "
                        "Hessian are all the true reduced quantities"
                    ),
                }
            )
        post_init_records.append(stage_post_init)
        print(
            f"  stage post-init ({post_init_method}): "
            f"iterations={stage_post_init['iterations']}  "
            f"forward solves={stage_post_init['n_forward_solves']}  "
            f"adjoint solves={stage_post_init['n_adjoint_solves']}  "
            f"nfev/njev/nhev={stage_post_init['n_objective_evaluations']}/"
            f"{stage_post_init['n_gradient_evaluations']}/"
            f"{stage_post_init['n_hessian_evaluations']}"
        )

        stage_records.append(
            {
                "stage_index": stage_index,
                "post_init": stage_post_init,
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
                "projected_evaluations": int(
                    feasibility_stats["projected_evaluations"] - stage_projected_before
                ),
                "penalized_evaluations": int(
                    feasibility_stats["penalized_evaluations"] - stage_penalized_before
                ),
                "final_control_projected": stage_final_projection,
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
    if feasibility_mode == "raw":
        # The whole point of "raw" is to let centers leave the fluid mesh, so the
        # final control may legitimately be inadmissible. Report it rather than
        # aborting; the summary records the exact distance for every source.
        if not admissibility.validate_sources_in_domain(final_control):
            for _sid, (_x, _y, _a) in enumerate(reshape_control(final_control), 1):
                if admissibility.is_point_in_domain(float(_x), float(_y)):
                    continue
                _d2, _g = admissibility.nearest_domain_distance_sq(float(_x), float(_y))
                print(
                    f"  [raw feasibility] FINAL source {_sid} "
                    f"({_x:.6f},{_y:.6f}) is OUTSIDE the fluid mesh by "
                    f"{float(np.sqrt(max(_d2, 0.0))):.6f} m"
                )
    else:
        validate_control_for_region(
            final_control, "final optimized", args.bound_case, admissibility
        )

    final_feasibility_record = None
    if getattr(args, "final_feasibility_polish", False):
        if args.bound_case == MESH_DOMAIN_BOUND_CASE:
            print(
                "\n=== Final Feasibility Polish ===\n"
                f"margin={args.final_feasibility_margin} m  "
                f"max_iter={args.final_feasibility_max_iter}  "
                f"method={args.final_feasibility_method}"
            )
            # The repair certifies/moves SOURCE CENTERS irrespective of which
            # components this run's --mode treats as fixed, so it always uses
            # full location_and_intensity bounds/evaluator here -- not the
            # run's own (possibly location- or intensity-restricted)
            # active_bounds/objective/gradient.
            polish_full_bounds = mesh_domain_active_bounds_for_mode(
                "location_and_intensity",
                admissibility.bounding_box,
                num_sources=num_sources,
            )
            polish_full_bounds = apply_amplitude_floor(
                polish_full_bounds,
                "location_and_intensity",
                amplitude_floor,
                num_sources=num_sources,
            )
            # feasibility_mode="raw" so the repair's own objective/gradient stay
            # well-defined at every intermediate SLSQP iterate, including ones
            # outside the fluid mesh; admissibility is enforced only through
            # the polish's explicit signed-clearance constraints.
            _, _, polish_objective, polish_gradient, _ = build_evaluator(
                mesh_data,
                true_state,
                "location_and_intensity",
                args.bound_case,
                {},
                final_control,
                admissibility,
                feasibility_mode="raw",
                feasibility_stats=None,
            )
            final_feasibility_record = final_feasibility_polish(
                final_control,
                admissibility,
                polish_full_bounds,
                polish_objective,
                polish_gradient,
                clearance_margin=args.final_feasibility_margin,
                max_iter=args.final_feasibility_max_iter,
                method=args.final_feasibility_method,
            )
            print(
                "  before: "
                f"J={final_feasibility_record['objective_before']:.10e}  "
                "clearance "
                f"[{final_feasibility_record['clearance_before_min']:.4f}, "
                f"{final_feasibility_record['clearance_before_max']:.4f}] m"
            )
            print(
                "  after:  "
                f"J={final_feasibility_record['objective_after']:.10e}  "
                "clearance "
                f"[{final_feasibility_record['clearance_after_min']:.4f}, "
                f"{final_feasibility_record['clearance_after_max']:.4f}] m  "
                f"feasible={final_feasibility_record['feasible']}  "
                f"projected_start={final_feasibility_record['was_projected']}  "
                f"iterations={final_feasibility_record['iterations']}  "
                f"message={final_feasibility_record['message']}"
            )
            final_control = np.asarray(
                final_feasibility_record["repaired_control"], dtype=np.float64
            )
            z_opt_active = pack_active(final_control, mode)
            if not final_feasibility_record["feasible"]:
                print(
                    "  WARNING: final feasibility polish did NOT certify the "
                    "repaired control within the iteration budget; consider "
                    "raising --final-feasibility-max-iter or relaxing "
                    "--final-feasibility-margin"
                )
        else:
            print(
                "\nfinal feasibility polish requested but --bound-case is not "
                f"{MESH_DOMAIN_BOUND_CASE!r}; skipping (fluid-mesh clearance is "
                "only defined for the mesh-domain bound case)"
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
    sensor_points = export_sensor_points(mesh_data, output_paths["base"])

    # Aggregate the per-stage post-init records into one summary block. Without
    # sigma2 continuation there is exactly one stage, which is the common case.
    post_init_summary = {
        "post_init_method": post_init_method,
        "n_stages": len(post_init_records),
        "total_iterations": int(
            sum(r["iterations"] for r in post_init_records)
        ),
        "total_forward_solves": int(
            sum(r["n_forward_solves"] for r in post_init_records)
        ),
        "total_adjoint_solves": int(
            sum(r["n_adjoint_solves"] for r in post_init_records)
        ),
        "total_objective_evaluations": int(
            sum(r["n_objective_evaluations"] for r in post_init_records)
        ),
        "total_gradient_evaluations": int(
            sum(r["n_gradient_evaluations"] for r in post_init_records)
        ),
        "total_hessian_evaluations": int(
            sum(r["n_hessian_evaluations"] for r in post_init_records)
        ),
        "stages": post_init_records,
    }
    if post_init_method == pio.LBFGSB:
        post_init_summary["scaling"] = post_init_scaling
        post_init_summary["stall_iters"] = post_init_stall_iters
        post_init_summary["scaling_max_ratio"] = post_init_scaling_max_ratio
        post_init_summary["lbfgsb_maxcor"] = (
            10 if post_init_maxcor is None else int(post_init_maxcor)
        )
    if post_init_method != pio.LBFGSB:
        post_init_summary["scipy_method"] = pio.NEWTON_SCIPY_METHOD
        post_init_summary["hessian_mode"] = pio.HESSIAN_MODE_BY_METHOD[
            post_init_method
        ]
        # The Newton-type path has its own curvature diagnostics; keep them in a
        # dedicated file so optimization_summary.json stays readable.
        post_init_path = output_paths["base"] / "post_init_summary.json"
        post_init_path.write_text(
            json.dumps(
                {
                    **post_init_summary,
                    "initial_control": initial_control.tolist(),
                    "final_control": final_control.tolist(),
                    "initial_objective": float(initial_objective),
                    "final_objective": float(final_objective),
                    "final_gradient_norm": float(np.linalg.norm(final_gradient)),
                    "recovery_permutation_matched": (
                        safe_match_sources_by_permutation(
                            true_control, final_control
                        )[0]
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"post-init summary: {post_init_path}")

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
        feasibility=feasibility_stats,
        observation=mesh_data.get("observation"),
        post_init=post_init_summary,
        final_feasibility=(
            {
                key: (
                    value.tolist()
                    if isinstance(value, np.ndarray)
                    else value
                )
                for key, value in final_feasibility_record.items()
            }
            if final_feasibility_record is not None
            else None
        ),
    )

    print("\n=== Final Summary ===")
    print(f"optimization mode: {mode}")
    print(f"post-initialization method: {post_init_method}")
    print(
        "post-init cost: "
        f"{post_init_summary['total_iterations']} iteration(s), "
        f"{post_init_summary['total_forward_solves']} forward solve(s), "
        f"{post_init_summary['total_adjoint_solves']} adjoint solve(s)"
    )
    print(f"source region: {args.bound_case}")
    print(f"success: {result.success}")
    print(f"message: {result.message}")
    print(f"iterations: {result.nit}")
    print(f"initial objective: {initial_objective:.16e}")
    print(f"final objective: {final_objective:.16e}")
    print(f"final gradient norm: {np.linalg.norm(final_gradient):.16e}")
    print(f"final control: {np.array2string(final_control, precision=16)}")
    print(f"true control:  {np.array2string(true_control, precision=16)}")
    match, match_error = safe_match_sources_by_permutation(true_control, final_control)
    if match is None:
        print(f"per-source recovery: UNDEFINED -- {match_error}")
    else:
        print(
            "per-source recovery (optimal label permutation, "
            f"labels permuted = {match['labels_permuted']}):"
        )
        for true_index in range(match["num_sources"]):
            final_index = match["permutation"][true_index]
            print(
                f"  true s{true_index + 1} <- final s{final_index + 1}: "
                f"position error {match['position_errors_m'][true_index]:.6f} m  "
                f"amplitude error {match['amplitude_errors'][true_index]:.6e}"
            )
        print(
            f"max position error: {match['max_position_error_m']:.6f} m   "
            f"max amplitude error: {match['max_amplitude_error']:.6e}"
        )
    for source_id, (sx, sy, _sa) in enumerate(reshape_control(final_control), 1):
        inside = admissibility.is_point_in_domain(float(sx), float(sy))
        clearance = admissibility.distance_to_boundary(float(sx), float(sy))
        print(
            f"  terminal source {source_id}: fluid-admissible={inside}  "
            f"wall clearance={clearance:.4f} m"
        )
    print(
        f"feasibility ({feasibility_stats['feasibility_mode']}): "
        f"invalid evals={feasibility_stats['invalid_evaluations']}  "
        f"projected={feasibility_stats['projected_evaluations']}  "
        f"penalized={feasibility_stats['penalized_evaluations']}  "
        f"max projection distance="
        f"{feasibility_stats['max_projection_distance']:.6g} m"
    )
    if final_feasibility_record is not None:
        print(
            "final feasibility polish: "
            f"feasible={final_feasibility_record['feasible']}  "
            f"clearance margin={args.final_feasibility_margin} m  "
            "min clearance "
            f"{final_feasibility_record['clearance_before_min']:.4f} -> "
            f"{final_feasibility_record['clearance_after_min']:.4f} m  "
            f"J {final_feasibility_record['objective_before']:.6e} -> "
            f"{final_feasibility_record['objective_after']:.6e}"
        )
    print(f"output folder: {output_paths['base']}")
    print("ParaView files to open:")
    for path in (true_pvd, initial_pvd, optimized_pvd, source_fields, sensor_points):
        if path is not None:
            print(f"  {path}")
    print("Additional outputs:")
    for path in (summary_path, history_path, *source_path_files):
        print(f"  {path}")

    return {
        "start_index": start_index,
        "sampled_initial_control": initial_control.tolist(),
        "post_init_method": post_init_method,
        "post_init": post_init_summary,
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "initial_objective": float(initial_objective),
        "final_objective": float(final_objective),
        "final_gradient_norm": float(np.linalg.norm(final_gradient)),
        "final_control": final_control.tolist(),
        "output_folder": str(output_paths["base"]),
        "feasibility": dict(feasibility_stats),
        "final_feasibility_polish": (
            {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in final_feasibility_record.items()
            }
            if final_feasibility_record is not None
            else None
        ),
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
    num_starts_override: int | None = None,
    branch_controls: list[dict] | None = None,
) -> tuple[Path, Path]:
    output_dir = mode_output_dir(mode, case_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "optimization_mode": mode,
        "initialization_mode": init_mode,
        "sensitivity_branches": branch_controls,
        "num_starts": (
            int(num_starts_override)
            if num_starts_override is not None
            else args.num_starts
        ),
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
        # extrasaction="ignore" lets records carry extra keys (e.g. the nested
        # "reinsertion" provenance, which lives in the JSON summary) without
        # breaking the fixed CSV column set.
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
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


def _basin_key(survivor_centers, round_m: float = BASIN_ROUND_M):
    """Permutation-symmetric key for a partial-recovery basin.

    Two partial recoveries are the "same basin" iff their surviving source
    centers match up to permutation (rounded to ``round_m`` metres). Only the
    survivors define the basin; the parked dead source is what we replace.
    """
    rounded = tuple(
        sorted(
            (round(float(c[0]) / round_m) * round_m, round(float(c[1]) / round_m) * round_m)
            for c in survivor_centers
        )
    )
    return rounded


def detect_partial_recovery(
    final_control,
    amplitude_bounds,
    admissibility: MeshAdmissibility | None = None,
    num_sources: int = NUM_SOURCES,
    dead_margin: float = DEAD_SOURCE_AMPLITUDE_MARGIN,
):
    """Detect a partial recovery: some sources active, at least one collapsed.

    Pure function. A source is "dead" when its amplitude is at/just above the
    amplitude floor (``amp <= floor + dead_margin``) -- this is the decisive
    signal. Whether the dead center is pinned near a spatial (mesh bounding-box)
    bound is recorded as a secondary diagnostic only. Returns ``None`` unless
    there is at least one dead source AND at least one surviving source (so the
    survivor carries real information to keep); otherwise a dict describing the
    dead/survivor split and a permutation-symmetric ``basin_key``. Never consults
    the true control.
    """
    if final_control is None:
        return None
    control = np.asarray(final_control, dtype=np.float64).reshape(-1)
    if control.size != 3 * int(num_sources) or not np.all(np.isfinite(control)):
        return None
    floor = float(amplitude_bounds[0])
    dead_threshold = floor + float(dead_margin)

    dead_indices, survivor_indices = [], []
    for source_id in range(int(num_sources)):
        amp = float(control[3 * source_id + 2])
        if amp <= dead_threshold:
            dead_indices.append(source_id)
        else:
            survivor_indices.append(source_id)

    # Partial recovery requires >=1 dead AND >=1 survivor.
    if not dead_indices or not survivor_indices:
        return None

    bbox = admissibility.bounding_box if admissibility is not None else None

    def _pinned(center):
        if bbox is None:
            return False
        xmin, xmax, ymin, ymax = bbox
        x, y = float(center[0]), float(center[1])
        return (
            abs(x - xmin) <= BOUND_PIN_TOL_M or abs(x - xmax) <= BOUND_PIN_TOL_M
            or abs(y - ymin) <= BOUND_PIN_TOL_M or abs(y - ymax) <= BOUND_PIN_TOL_M
        )

    survivor_centers = [control[3 * s : 3 * s + 2].copy() for s in survivor_indices]
    dead_centers = [control[3 * d : 3 * d + 2].copy() for d in dead_indices]
    return {
        "dead_indices": dead_indices,
        "survivor_indices": survivor_indices,
        "dead_amplitudes": [float(control[3 * d + 2]) for d in dead_indices],
        "dead_centers": [c.tolist() for c in dead_centers],
        "dead_center_bound_pinned": [bool(_pinned(c)) for c in dead_centers],
        "survivor_centers": [c.tolist() for c in survivor_centers],
        "survivor_control": [
            control[3 * s : 3 * s + 3].tolist() for s in survivor_indices
        ],
        "basin_key": _basin_key(survivor_centers),
    }


def select_reinsertion_candidate(
    coord,
    sensitivity,
    survivor_centers,
    boundary_node_set,
    used_locations,
    min_separation,
):
    """Pick argmin S(q) over admissible mesh vertices (pure; no PDE solve).

    Admissible = interior mesh vertex (not an outer-boundary node), at least
    ``min_separation`` from every surviving source, and at least
    ``min_separation`` from every previously used reinsertion location. Requires
    a genuinely negative sensitivity (descent for a positive amplitude); returns
    ``(index, s_value)`` or ``None`` if no admissible negative candidate exists.
    """
    coord = np.asarray(coord, dtype=np.float64)[:, :2]
    sensitivity = np.asarray(sensitivity, dtype=np.float64)
    sep = float(min_separation) if min_separation is not None else 0.0

    best_index, best_value = None, np.inf
    for i in range(coord.shape[0]):
        if i in boundary_node_set:
            continue
        s = float(sensitivity[i])
        if not np.isfinite(s) or s >= 0.0:
            continue
        if s >= best_value:  # cheap early-out before the distance checks
            continue
        p = coord[i]
        if any(np.hypot(p[0] - c[0], p[1] - c[1]) < sep for c in survivor_centers):
            continue
        if any(np.hypot(p[0] - u[0], p[1] - u[1]) < sep for u in used_locations):
            continue
        best_index, best_value = i, s
    if best_index is None:
        return None
    return best_index, best_value


def build_reinsertion_control(
    partial_control,
    detection,
    mesh_data,
    pde,
    true_state,
    admissibility: MeshAdmissibility,
    min_separation,
    used_locations,
    amplitude_bounds,
    num_sources: int = NUM_SOURCES,
):
    """Build a sensitivity-enriched restart that revives one dead source.

    Reuses ``source_insertion_sensitivity``: form ``z_bar`` from the partially
    recovered control with the dead source's amplitude set to 0, do ONE forward +
    ONE adjoint at the current (target) sigma2, evaluate S(q) over all mesh
    vertices, and select the most negative admissible candidate (interior,
    >=min_separation from every survivor, not near a used location). The revived
    source takes a fixed moderate amplitude ``REINSERTION_AMPLITUDE`` (clamped to
    the current amplitude bounds; never the floor). Both sources stay free for the
    subsequent optimization -- only the initial control changes. Returns a
    provenance dict, or ``None`` if there is no genuinely negative candidate.
    """
    control = np.asarray(partial_control, dtype=np.float64).reshape(-1).copy()
    dead_idx = int(detection["dead_indices"][0])  # NUM_SOURCES=2 -> one dead source

    # z_bar: dead source amplitude exactly 0 (center then irrelevant to S).
    z_bar = control.copy()
    z_bar[3 * dead_idx + 2] = 0.0

    # ONE forward + ONE adjoint (target-sigma observation = true_state).
    _state, source_adjoint, inv_mass, W = sis.compute_adjoint_weight_field(
        mesh_data, pde, z_bar, true_state, ETA1, OBJECTIVE_TIME_RULE
    )
    qx, qy, ag = sis.precompute_quadrature_weights(mesh_data, pde, W)
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)[:, :2]
    sensitivity = sis.sensitivity_at_candidates(coord, qx, qy, ag, pde["sigma2"])

    # Outer-boundary nodes are excluded so we never re-seed onto a sensor/inflow
    # wall (matches the sensitivity-enrichment driver's admissibility).
    boundary_node_set = set()
    for name in ("left", "right", "top", "bottom"):
        nodes = mesh_data["boundary_nodes_by_name"].get(name)
        if nodes is not None:
            boundary_node_set.update(int(v) for v in np.asarray(nodes).reshape(-1))

    survivor_centers = [np.asarray(c, dtype=np.float64) for c in detection["survivor_centers"]]
    used = [np.asarray(u, dtype=np.float64) for u in used_locations]
    selection = select_reinsertion_candidate(
        coord, sensitivity, survivor_centers, boundary_node_set, used, min_separation
    )
    if selection is None:
        return None
    q_index, s_value = selection
    q_new = coord[q_index].copy()

    lo, hi = float(amplitude_bounds[0]), float(amplitude_bounds[1])
    a_new = float(np.clip(REINSERTION_AMPLITUDE, lo, hi))

    # Enriched restart: survivors keep their slots; the dead slot is re-seeded.
    z_restart = control.copy()
    z_restart[3 * dead_idx : 3 * dead_idx + 3] = [q_new[0], q_new[1], a_new]

    return {
        "control": z_restart,
        "dead_source_index": dead_idx,
        "survivor_control": detection["survivor_control"],
        "reinsertion_location": [float(q_new[0]), float(q_new[1])],
        "reinsertion_location_vertex_index": int(q_index),
        "sensitivity_value": float(s_value),
        "reinsertion_amplitude": a_new,
        "basin_key": list(detection["basin_key"]),
    }


def _select_reinsertion_parent(
    records, amplitude_bounds, admissibility, consumed_basins
):
    """Pick the lowest-objective unconsumed partial-recovery start, or ``None``."""
    best = None
    for rec in records:
        obj = rec.get("final_objective")
        if obj is None or not np.isfinite(obj):
            continue
        detection = detect_partial_recovery(
            rec.get("final_control"), amplitude_bounds, admissibility
        )
        if detection is None:
            continue
        if detection["basin_key"] in consumed_basins:
            continue
        if best is None or obj < best[0]:
            best = (float(obj), rec, detection)
    return best


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
        "reinsert_dead_sources": bool(args.adaptive_reinsert_dead_sources),
        "max_reinsertions": int(args.adaptive_max_reinsertions),
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
        "dead-source sensitivity reinsertion: "
        f"{bool(args.adaptive_reinsert_dead_sources)} "
        f"(max {int(args.adaptive_max_reinsertions)}, amplitude "
        f"{REINSERTION_AMPLITUDE})"
    )
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

    # Dead-source sensitivity reinsertion state (feature OFF unless the CLI flag
    # is set). ``consumed_basins`` prevents reinserting from the same partial-
    # recovery basin twice (permutation-symmetric key); ``used_locations`` blocks
    # near-duplicate insertion candidates; the budget caps total reinsertions.
    reinsert_enabled = bool(args.adaptive_reinsert_dead_sources)
    max_reinsertions = int(args.adaptive_max_reinsertions)
    reinsertions_used = 0
    consumed_basins: list = []
    used_insertion_locations: list = []

    start_index = 0
    while start_index < max_starts:
        start_index += 1
        parent_start_index = None
        perturbation_radius = None
        reinsertion_provenance = None

        if start_index <= initial_random:
            initial_control = sample_smart_control()
            initialization_reason = "initial_random_smart"
            print(
                f"start {start_index:03d} generated as an initial smart "
                "exploration draw"
            )
        elif (
            reinsert_enabled
            and reinsertions_used < max_reinsertions
            and _select_reinsertion_parent(
                records, amplitude_bounds, admissibility, consumed_basins
            )
            is not None
        ):
            # PRIORITY 2: revive a dead source from an unconsumed partial-recovery
            # basin via the sensitivity field, before any perturb/fresh logic.
            parent_obj, parent_rec, detection = _select_reinsertion_parent(
                records, amplitude_bounds, admissibility, consumed_basins
            )
            parent_idx = int(parent_rec["start_index"])
            print(
                f"partial recovery detected at start {parent_idx:03d} "
                f"(J={parent_obj:.6e}): dead source index "
                f"{detection['dead_indices']} (amp"
                f"={[round(a, 4) for a in detection['dead_amplitudes']]}, "
                f"bound_pinned={detection['dead_center_bound_pinned']}); "
                f"survivor control {detection['survivor_control']}"
            )
            # Consume this basin now (whether or not a candidate is found) so the
            # same partial recovery is never reinserted from again -> no loops.
            consumed_basins.append(detection["basin_key"])
            built = build_reinsertion_control(
                parent_rec["final_control"],
                detection,
                mesh_data,
                PDE,
                true_state,
                admissibility,
                sampling_min_separation,
                used_insertion_locations,
                amplitude_bounds,
            )
            if built is None:
                initial_control = sample_smart_control()
                initialization_reason = "fresh_reinsertion_no_candidate"
                print(
                    f"start {start_index:03d}: no genuinely negative admissible "
                    "reinsertion candidate; falling back to a fresh smart draw"
                )
            else:
                reinsertions_used += 1
                initial_control = built["control"]
                parent_start_index = parent_idx
                initialization_reason = "dead_source_reinsertion"
                built["parent_start_index"] = parent_idx
                built["attempt_number"] = reinsertions_used
                reinsertion_provenance = built
                used_insertion_locations.append(np.asarray(built["reinsertion_location"]))
                print(
                    f"start {start_index:03d} generated by dead-source "
                    f"reinsertion (attempt {reinsertions_used}/{max_reinsertions}) "
                    f"from partial-recovery start {parent_idx:03d}: "
                    f"reinsert dead source {built['dead_source_index']} at "
                    f"({built['reinsertion_location'][0]:.4f}, "
                    f"{built['reinsertion_location'][1]:.4f}), "
                    f"S={built['sensitivity_value']:.6e}, "
                    f"amplitude={built['reinsertion_amplitude']}"
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
        if reinsertion_provenance is not None:
            # Provenance for the dead-source sensitivity reinsertion (JSON-safe:
            # the reinsertion control array is not stored, only scalars/lists).
            record["reinsertion"] = {
                "parent_start_index": reinsertion_provenance["parent_start_index"],
                "attempt_number": reinsertion_provenance["attempt_number"],
                "dead_source_index": reinsertion_provenance["dead_source_index"],
                "survivor_control": reinsertion_provenance["survivor_control"],
                "reinsertion_location": reinsertion_provenance["reinsertion_location"],
                "sensitivity_value": reinsertion_provenance["sensitivity_value"],
                "reinsertion_amplitude": reinsertion_provenance["reinsertion_amplitude"],
            }
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
    if not isinstance(args.num_sources, int) or args.num_sources < 1:
        raise ValueError(
            f"--num-sources must be a positive integer; got {args.num_sources!r}"
        )
    num_sources = int(args.num_sources)
    # Controls are validated against 3*M here, before any mesh work, and are
    # never truncated or padded.
    if args.true_control is not None and args.infer_num_sources:
        # Under inference the source count comes from the data, not from
        # --num-sources, so --true-control only has to be a whole number of
        # sources. It is used for reporting recovery errors, never for inference.
        args.true_control = np.asarray(args.true_control, dtype=np.float64)
        infer_num_sources(args.true_control)
    elif args.true_control is not None:
        args.true_control = validate_control_length(
            args.true_control, "--true-control", num_sources
        )
    elif args.infer_num_sources:
        pass
    elif num_sources != NUM_SOURCES:
        raise ValueError(
            f"--num-sources {num_sources} requires an explicit --true-control "
            f"of {control_dim(num_sources)} values; Application1 only defines a "
            f"{NUM_SOURCES}-source reference control"
        )
    if args.initial_control is not None:
        args.initial_control = validate_control_length(
            args.initial_control, "--initial-control", num_sources
        )
    elif args.init_mode == "manual" and num_sources != NUM_SOURCES:
        raise ValueError(
            f"--num-sources {num_sources} with --init-mode manual requires an "
            f"explicit --initial-control of {control_dim(num_sources)} values; "
            f"Application1 only defines a {NUM_SOURCES}-source initial control"
        )
    if num_sources != NUM_SOURCES and args.init_mode == "adaptive":
        raise ValueError(
            "--init-mode adaptive still assumes exactly two sources (its "
            "dead-source reinsertion selects a single dead source); use "
            "--init-mode sensitivity, smart, mesh_random or manual for "
            "--num-sources other than 2"
        )
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
        if args.adaptive_max_reinsertions < 0:
            raise ValueError("--adaptive-max-reinsertions must be >= 0")
    if args.init_mode == "sensitivity":
        # Greedy sensitivity init needs the mesh-domain admissibility mask/bounds,
        # unless the user supplies an authoritative --initial-control (which wins).
        if args.initial_control is None and args.bound_case != MESH_DOMAIN_BOUND_CASE:
            raise ValueError(
                "--init-mode sensitivity requires --bound-case mesh_domain"
            )
        if args.num_starts != 1:
            raise ValueError(
                "--init-mode sensitivity places a single deterministic greedy "
                "start; use --num-starts 1"
            )
        if not (
            np.isfinite(args.sensitivity_min_value_tol)
            and args.sensitivity_min_value_tol >= 0.0
        ):
            raise ValueError("--sensitivity-min-value-tol must be finite and >= 0")
        if args.sensitivity_top_k < 1:
            raise ValueError("--sensitivity-top-k must be >= 1")
        if args.sensitivity_beam_width < 1:
            raise ValueError("--sensitivity-beam-width must be >= 1")
        if args.sensitivity_beam_width > 1 and args.sensitivity_score != "predicted_reduction":
            raise ValueError(
                "--sensitivity-beam-width > 1 requires "
                "--sensitivity-score predicted_reduction"
            )
        for _name, _val in (
            ("--sensitivity-min-separation", args.sensitivity_min_separation),
            ("--sensitivity-mask-boundary-buffer", args.sensitivity_mask_boundary_buffer),
            ("--sensitivity-mask-building-buffer", args.sensitivity_mask_building_buffer),
        ):
            if _val is not None and (not np.isfinite(_val) or _val < 0.0):
                raise ValueError(f"{_name} must be finite and >= 0")
    if args.infer_num_sources:
        if args.initial_control is not None:
            raise ValueError(
                "--infer-num-sources determines both the source count and its "
                "own starting control; it cannot be combined with "
                "--initial-control"
            )
        if args.init_mode == "adaptive":
            raise ValueError(
                "--infer-num-sources cannot be combined with --init-mode adaptive"
            )
        if args.num_starts != 1:
            raise ValueError(
                "--infer-num-sources runs a single deterministic sequential "
                "inference; use --num-starts 1"
            )
        if args.bound_case != MESH_DOMAIN_BOUND_CASE:
            raise ValueError("--infer-num-sources requires --bound-case mesh_domain")
        if args.stage in ("smoke", "gradient"):
            raise ValueError("--infer-num-sources requires --stage short or full")
        if args.max_sources < 1:
            raise ValueError("--max-sources must be >= 1")
        if args.source_count_top_k < 1:
            raise ValueError("--source-count-top-k must be >= 1")
        if args.source_count_polish_max_iter < 1:
            raise ValueError("--source-count-polish-max-iter must be >= 1")
        for _name, _value in (
            ("--source-count-discrepancy-tol", args.source_count_discrepancy_tol),
            ("--source-count-rel-reduction-tol", args.source_count_rel_reduction_tol),
            ("--source-count-abs-reduction-tol", args.source_count_abs_reduction_tol),
            ("--source-count-amp-tol", args.source_count_amp_tol),
        ):
            if not (np.isfinite(_value) and _value >= 0.0):
                raise ValueError(f"{_name} must be finite and >= 0")

    # Fail at startup (before any forward/adjoint solve) if the selected joint
    # amplitude solver cannot run here. Validated unconditionally so a typo or a
    # missing OSQP install is reported even when --init-mode is not sensitivity.
    args.amplitude_solver = sinit.require_amplitude_solver(args.amplitude_solver)

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
    # Observation operator resolution. The default (boundary) attaches nothing
    # and every downstream solver keeps its existing code path.
    mesh_data = configure_observation(mesh_data, args)
    print_observation_summary(mesh_data)

    # Velocity source resolution:
    #   --constant-velocity            -> constant PDE vx/vy (ambient wind)
    #   --velocity-field-file PATH     -> that nodal field
    #   (default)                      -> auto-load Application1 NS field if present
    if args.constant_velocity:
        if args.velocity_field_file is not None:
            raise ValueError(
                "--constant-velocity cannot be combined with --velocity-field-file"
            )
        velocity_field_file = None
    elif args.velocity_field_file is not None:
        velocity_field_file = args.velocity_field_file
    elif DEFAULT_VELOCITY_FIELD_FILE.exists():
        velocity_field_file = str(DEFAULT_VELOCITY_FIELD_FILE)
        print(
            "no --velocity-field-file given; auto-loading the Application1 "
            "Navier-Stokes velocity field"
        )
    else:
        velocity_field_file = None
        print(
            "warning: no velocity field file found at "
            f"{DEFAULT_VELOCITY_FIELD_FILE}; falling back to constant velocity "
            f"vx={PDE['vx']}, vy={PDE['vy']}"
        )

    if velocity_field_file is not None:
        velocity_nodes, velocity_summary = load_velocity_field_file(
            velocity_field_file, mesh_data
        )
        PDE["velocity_nodes"] = velocity_nodes
        PDE["velocity_field"] = velocity_summary

    admissibility = MeshAdmissibility(mesh_data)
    print("\n=== Geometry / Coordinate Scale (DC Navy Yard, meters) ===")
    xmin, xmax, ymin, ymax = admissibility.bounding_box
    print(f"fluid-domain bounding box x: [{xmin:.4f}, {xmax:.4f}] ({xmax - xmin:.2f} m)")
    print(f"fluid-domain bounding box y: [{ymin:.4f}, {ymax:.4f}] ({ymax - ymin:.2f} m)")
    print(f"mesh nodes: {len(mesh_data['coord'])}, triangles: {len(mesh_data['connections'])}")
    print(f"Gaussian sigma2 (target): {PDE['sigma2']} m^2 (e-folding {np.sqrt(PDE['sigma2']):.2f} m)")
    print(f"diffusivity k: {PDE['k']} m^2/s, dt: {PDE['dt']} s, tfinal: {PDE['tfinal']} s")
    print(f"amplitude bounds: {AMPLITUDE_BOUNDS}")
    print(f"smart min source separation default: {SMART_DEFAULT_MIN_SEPARATION} m")
    print(f"adaptive perturbation radius default: {ADAPTIVE_DEFAULT_PERTURB_RADIUS} m")

    print("\n=== Problem Size And Initialization ===")
    print(f"number of sources M: {num_sources}")
    print(f"control dimension 3*M: {control_dim(num_sources)}")
    print(
        "control ordering: "
        + ", ".join(component_names(num_sources))
    )
    print(
        "initialization mode: "
        + (
            f"manual (--initial-control supplied; overrides --init-mode "
            f"{args.init_mode})"
            if args.initial_control is not None
            else str(args.init_mode)
        )
    )
    if args.init_mode == "sensitivity" and args.initial_control is None:
        print(f"sensitivity score: {args.sensitivity_score}")
        print(f"beam width: {args.sensitivity_beam_width}")
        print(f"sensitivity screen size (top-K): {args.sensitivity_top_k}")
        print(f"coherence cap: {args.sensitivity_coherence_cap}")
    else:
        print(f"beam width: n/a ({args.init_mode})")
        print(f"sensitivity screen size (top-K): n/a ({args.init_mode})")
    print(
        "continuation schedule (sigma2): "
        + (
            " -> ".join(f"{s:g}" for s in sigma2_stages)
            if len(sigma2_stages) > 1
            else f"{sigma2_stages[0]:g} (no continuation)"
        )
    )
    print(f"target sigma2: {target_sigma2:g}")

    validate_control_for_region(
        true_control, "true/reference", args.bound_case, admissibility
    )

    # Adaptive source-count inference. Runs BEFORE the bounds/initialization
    # branches because it determines num_sources; it then hands its inferred
    # control to the ordinary pipeline as the starting control.
    source_count_summary = None
    inferred_control = None
    if args.infer_num_sources:
        true_state = gt.run_forward(mesh_data, PDE, true_control)
        inference_min_separation = (
            args.sensitivity_min_separation
            if args.sensitivity_min_separation is not None
            else args.min_source_separation
        )
        polish_fn = build_source_count_polish_fn(
            args,
            mode,
            mesh_data,
            true_state,
            admissibility,
            getattr(args, "feasibility_mode", DEFAULT_FEASIBILITY_MODE),
            args.source_count_polish_method,
            args.source_count_polish_max_iter,
        )
        print(
            f"\nsource-count inference: --num-sources {num_sources} is IGNORED; "
            "M is inferred from the data"
        )
        source_count_summary = sci.infer_source_count(
            mesh_data,
            PDE,
            true_state,
            admissibility,
            amplitude_upper=float(AMPLITUDE_BOUNDS[1]),
            eta1=ETA1,
            lam=LAMBDA_REG,
            objective_time_rule=OBJECTIVE_TIME_RULE,
            polish_fn=polish_fn,
            max_sources=int(args.max_sources),
            discrepancy_tol=float(args.source_count_discrepancy_tol),
            rel_reduction_tol=float(args.source_count_rel_reduction_tol),
            abs_reduction_tol=float(args.source_count_abs_reduction_tol),
            amp_tol=float(args.source_count_amp_tol),
            top_k=int(args.source_count_top_k),
            min_separation=inference_min_separation,
            boundary_buffer=float(args.sensitivity_mask_boundary_buffer or 0.0),
            building_buffer=float(args.sensitivity_mask_building_buffer or 0.0),
            obstacle_aware_diversity=bool(
                args.sensitivity_obstacle_aware_diversity
            ),
            min_value_tol=float(args.sensitivity_min_value_tol),
            production_amplitude_floor=args.amplitude_floor,
        )
        inferred_m = int(source_count_summary["inferred_num_sources"])
        if inferred_m < 1:
            raise RuntimeError(
                "source-count inference accepted no sources "
                f"(stop reason: {source_count_summary['stop_reason']}); nothing "
                "to optimize"
            )
        inferred_control = np.asarray(
            source_count_summary["final_control"], dtype=np.float64
        )

        # Permutation-matched recovery is SCORING ONLY -- true_control is
        # unknown in a real application -- and is undefined when the inferred
        # and true source counts differ; report that rather than crash.
        true_m = int(infer_num_sources(true_control))
        recovery_matched, recovery_match_error = safe_match_sources_by_permutation(
            true_control, inferred_control
        )
        source_count_summary["recovery_permutation_matched"] = recovery_matched
        source_count_summary["recovery_permutation_match_error"] = (
            recovery_match_error
        )
        source_count_summary["recovery_note"] = (
            f"inferred M = true M = {inferred_m}; errors are reported after "
            "the optimal label permutation"
            if recovery_matched is not None
            else (
                f"inferred M = {inferred_m} but the supplied --true-control has "
                f"M = {true_m}; a permutation match is undefined for unequal "
                "source counts, so no per-source errors are reported"
            )
        )
        source_count_summary["true_control"] = true_control.tolist()
        source_count_summary["true_num_sources"] = true_m
        source_count_summary["inferred_matches_true_count"] = bool(
            inferred_m == true_m
        )
        source_count_summary["observation"] = (
            {"observation_mode": "boundary"}
            if mesh_data.get("observation") is None
            else mesh_data["observation"].metadata()
        )
        source_count_summary["measurement_boundaries"] = list(
            mesh_data.get(
                "selected_measurement_boundaries",
                (mesh_data.get("measurement_boundary", "right"),),
            )
        )
        source_count_summary["pde"] = serializable_pde_config(PDE)
        source_count_summary["polish"] = {
            "method": args.source_count_polish_method,
            "max_iter": int(args.source_count_polish_max_iter),
            "feasibility_mode": getattr(
                args, "feasibility_mode", DEFAULT_FEASIBILITY_MODE
            ),
        }

        print(f"inferred number of sources: {inferred_m}")
        print(
            "inferred control (raw, before final feasibility handling): "
            f"{np.array2string(inferred_control, precision=16)}"
        )

        # Final feasibility handoff. The inference can hand back a control
        # with a center outside the fluid mesh (raw-mode-style behavior).
        # Historically the very next step -- validate_control_for_region --
        # would raise and crash the run AFTER
        # source_count_inference_summary.json was already written below. Now:
        # certify/repair the inferred control first (only if
        # --final-feasibility-polish was requested), record what happened,
        # and if it is still infeasible, write the summary and SKIP final
        # optimization gracefully instead of crashing. true_control plays no
        # role in any of this.
        inferred_control_before_final_feasibility = inferred_control.copy()
        sc_final_feasibility_record = None
        if getattr(args, "final_feasibility_polish", False) and (
            args.bound_case == MESH_DOMAIN_BOUND_CASE
        ):
            sc_polish_bounds = mesh_domain_active_bounds_for_mode(
                "location_and_intensity",
                admissibility.bounding_box,
                num_sources=inferred_m,
            )
            sc_polish_bounds = apply_amplitude_floor(
                sc_polish_bounds,
                "location_and_intensity",
                args.amplitude_floor,
                num_sources=inferred_m,
            )
            _, _, sc_polish_objective, sc_polish_gradient, _ = build_evaluator(
                mesh_data,
                true_state,
                "location_and_intensity",
                args.bound_case,
                {},
                inferred_control,
                admissibility,
                feasibility_mode="raw",
                feasibility_stats=None,
            )
            sc_final_feasibility_record = final_feasibility_polish(
                inferred_control,
                admissibility,
                sc_polish_bounds,
                sc_polish_objective,
                sc_polish_gradient,
                clearance_margin=args.final_feasibility_margin,
                max_iter=args.final_feasibility_max_iter,
                method=args.final_feasibility_method,
            )
            inferred_control = np.asarray(
                sc_final_feasibility_record["repaired_control"], dtype=np.float64
            )
            print(
                "final feasibility polish on inferred control: "
                f"feasible={sc_final_feasibility_record['feasible']}  "
                "clearance "
                f"[{sc_final_feasibility_record['clearance_before_min']:.4f}, "
                f"{sc_final_feasibility_record['clearance_before_max']:.4f}] -> "
                f"[{sc_final_feasibility_record['clearance_after_min']:.4f}, "
                f"{sc_final_feasibility_record['clearance_after_max']:.4f}] m"
            )

        source_count_summary["final_feasibility_polish"] = (
            {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in sc_final_feasibility_record.items()
            }
            if sc_final_feasibility_record is not None
            else None
        )
        source_count_summary["inferred_control_before_final_feasibility"] = (
            inferred_control_before_final_feasibility.tolist()
        )
        source_count_summary["inferred_control_after_final_feasibility"] = (
            inferred_control.tolist()
        )

        inferred_control_validation_error = None
        try:
            validate_control_for_region(
                inferred_control, "inferred", args.bound_case, admissibility
            )
        except ValueError as exc:
            inferred_control_validation_error = str(exc)

        inferred_control_feasible = inferred_control_validation_error is None
        source_count_summary["inferred_control_feasible"] = (
            inferred_control_feasible
        )
        source_count_summary["inferred_control_validation_error"] = (
            inferred_control_validation_error
        )

        skipped_final_optimization_reason = None
        if inferred_control_validation_error is not None:
            skipped_final_optimization_reason = (
                "inferred control failed feasibility validation "
                + (
                    "even after --final-feasibility-polish"
                    if sc_final_feasibility_record is not None
                    else "(--final-feasibility-polish was not requested)"
                )
                + f": {inferred_control_validation_error}"
            )
        source_count_summary["skipped_final_optimization_reason"] = (
            skipped_final_optimization_reason
        )

        inference_dir = mode_output_dir(mode, case_name)
        inference_dir.mkdir(parents=True, exist_ok=True)
        inference_path = inference_dir / "source_count_inference_summary.json"
        inference_path.write_text(
            json.dumps(source_count_summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"source-count inference summary: {inference_path}")

        if skipped_final_optimization_reason is not None:
            print(
                "\nWARNING: skipping final optimization -- "
                f"{skipped_final_optimization_reason}"
            )
            print(
                "source-count inference results are recorded in "
                f"{inference_path}; no optimization_summary.json will be "
                "written for this run"
            )
            return

        num_sources = inferred_m
        print(f"inferred number of sources: {num_sources}")
        print(
            "inferred control (final, handed to the ordinary pipeline): "
            f"{np.array2string(inferred_control, precision=16)}"
        )

    if args.bound_case == MESH_DOMAIN_BOUND_CASE:
        active_bounds = mesh_domain_active_bounds_for_mode(
            mode, admissibility.bounding_box, num_sources=num_sources
        )
    else:
        active_bounds = active_bounds_for_mode(
            mode, num_sources=num_sources, bound_case=args.bound_case
        )
    active_bounds = apply_amplitude_floor(
        active_bounds, mode, args.amplitude_floor, num_sources=num_sources
    )

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
    sensitivity_branch_records = None
    if not args.infer_num_sources:
        true_state = None
    if args.infer_num_sources:
        # The inference already produced the starting control (and already
        # polished it at each accepted level); the ordinary pipeline now runs
        # from it under the PRODUCTION amplitude bounds.
        init_mode = "source_count_inference"
        initial_controls = [inferred_control]
    elif args.initial_control is not None:
        if args.init_mode in ("mesh_random", "smart", "sensitivity"):
            print(
                "warning: --initial-control is authoritative; ignoring "
                f"--init-mode {args.init_mode} and using the supplied control"
            )
        init_mode = "manual"
        initial_controls = [np.asarray(args.initial_control, dtype=np.float64)]
    elif args.init_mode == "manual":
        init_mode = "manual"
        initial_controls = [INITIAL_CONTROL.copy()]
    elif args.init_mode == "sensitivity":
        init_mode = "sensitivity"
        # The observation data u_d is the forward state of the true control,
        # exactly as the optimizer generates it; the greedy placement uses it
        # only as data, never for candidate selection.
        true_state = gt.run_forward(mesh_data, PDE, true_control)
        # Width consistency: with sigma2 continuation the first optimization
        # stage solves the sigma2_stages[0] problem (Approach A regenerates its
        # data at that width), so the initialization must be built for THAT
        # problem, not the target width. Without continuation this is the
        # target width and reproduces prior behavior.
        init_sigma2 = float(sigma2_stages[0])
        pde_init = dict(PDE)
        pde_init["sigma2"] = init_sigma2
        if init_sigma2 == target_sigma2:
            init_observation_state = true_state
        else:
            print(
                f"sensitivity init runs at the first continuation width "
                f"sigma2={init_sigma2:g} (target {target_sigma2:g}); "
                "regenerating the init observation data at that width"
            )
            init_observation_state = gt.run_forward(mesh_data, pde_init, true_control)
        sensitivity_min_separation = (
            args.sensitivity_min_separation
            if args.sensitivity_min_separation is not None
            else args.min_source_separation
        )
        plot_dir = (
            mode_output_dir(mode, case_name) / "sensitivity_init"
            if args.sensitivity_save_plots
            else None
        )
        plot_context = "+".join(
            mesh_data.get(
                "selected_measurement_boundaries",
                (mesh_data.get("measurement_boundary", "right"),),
            )
        )
        if mesh_data.get("observation") is not None:
            plot_context += (
                f" ({mesh_data['observation'].num_sensors} node sensors)"
            )
        plot_context += f", sigma2={init_sigma2:g}"
        sensitivity_control, _sensitivity_diagnostics, sensitivity_extras = (
            sinit.build_sensitivity_initial_control(
                mesh_data,
                pde_init,
                init_observation_state,
                num_sources,
                amplitude_bounds,
                admissibility,
                eta1=ETA1,
                lam=LAMBDA_REG,
                objective_time_rule=OBJECTIVE_TIME_RULE,
                min_separation=sensitivity_min_separation,
                min_value_tol=args.sensitivity_min_value_tol,
                boundary_buffer=args.sensitivity_mask_boundary_buffer,
                building_buffer=args.sensitivity_mask_building_buffer,
                save_plots=args.sensitivity_save_plots,
                plot_dir=plot_dir,
                plot_context=plot_context,
                score=args.sensitivity_score,
                top_k=args.sensitivity_top_k,
                beam_width=args.sensitivity_beam_width,
                refine_centers=args.sensitivity_refine_centers,
                coherence_cap=args.sensitivity_coherence_cap,
                refine_max_fev=args.sensitivity_refine_max_fev,
                refine_pair=args.sensitivity_refine_pair,
                pair_refine_max_fev=args.sensitivity_pair_refine_max_fev,
                geometry_aware_branches=args.sensitivity_geometry_aware_branches,
                obstacle_aware_diversity=(
                    args.sensitivity_obstacle_aware_diversity
                ),
                reselect_second=args.sensitivity_reselect_second,
                reselect_threshold=args.sensitivity_reselect_threshold,
                reselect_max_passes=args.sensitivity_reselect_max_passes,
                reselect_refine_max_fev=args.sensitivity_reselect_refine_max_fev,
                amplitude_solver=args.amplitude_solver,
                reselect_sigma2=(
                    float(target_sigma2)
                    if args.sensitivity_reselect_at_target_width
                    else None
                ),
                reselect_observation_state=(
                    true_state
                    if args.sensitivity_reselect_at_target_width
                    else None
                ),
                true_centers_for_plots=(
                    reshape_control(true_control)[:, :2]
                    if args.sensitivity_save_plots
                    else None
                ),
            )
        )
        print(
            "sensitivity greedy initial control: "
            f"{np.array2string(sensitivity_control, precision=16)}"
        )
        branch_records = sensitivity_extras.get("branch_controls") or []
        sensitivity_branch_records = branch_records
        # Persist the initialization so a run can be reported without parsing
        # stdout: solve budget, retained branches, pair refinement, and any
        # second-source re-selection passes.
        init_summary_dir = mode_output_dir(mode, case_name)
        init_summary_dir.mkdir(parents=True, exist_ok=True)
        init_summary_path = init_summary_dir / "sensitivity_init_summary.json"
        init_summary_path.write_text(
            json.dumps(
                {
                    "num_sources": int(num_sources),
                    "control_dim": int(control_dim(num_sources)),
                    "score": args.sensitivity_score,
                    "top_k": args.sensitivity_top_k,
                    "beam_width": args.sensitivity_beam_width,
                    "sigma2_init": init_sigma2,
                    "options": {
                        "refine_centers": bool(args.sensitivity_refine_centers),
                        "refine_max_fev": int(args.sensitivity_refine_max_fev),
                        "refine_pair": bool(args.sensitivity_refine_pair),
                        "pair_refine_max_fev": int(
                            args.sensitivity_pair_refine_max_fev
                        ),
                        "geometry_aware_branches": bool(
                            args.sensitivity_geometry_aware_branches
                        ),
                        "obstacle_aware_diversity": bool(
                            args.sensitivity_obstacle_aware_diversity
                        ),
                        "reselect_second": bool(args.sensitivity_reselect_second),
                        "reselect_threshold": float(
                            args.sensitivity_reselect_threshold
                        ),
                        "reselect_max_passes": int(
                            args.sensitivity_reselect_max_passes
                        ),
                        "reselect_at_target_width": bool(
                            args.sensitivity_reselect_at_target_width
                        ),
                        "coherence_cap": args.sensitivity_coherence_cap,
                        "branch_multistart": bool(
                            args.sensitivity_branch_multistart
                        ),
                        "amplitude_solver": str(args.amplitude_solver),
                    },
                    "zero_source_objective": sensitivity_extras.get(
                        "zero_source_objective"
                    ),
                    "initial_exact_objective": sensitivity_extras.get(
                        "initial_exact_objective"
                    ),
                    "objective_width_sigma2": sensitivity_extras.get(
                        "objective_width_sigma2"
                    ),
                    "n_forward_solves": sensitivity_extras.get("n_forward_solves"),
                    "n_adjoint_solves": sensitivity_extras.get("n_adjoint_solves"),
                    "n_candidate_traces": sensitivity_extras.get(
                        "n_candidate_traces"
                    ),
                    "wall_time_seconds": sensitivity_extras.get(
                        "wall_time_seconds"
                    ),
                    "source_levels": sensitivity_extras.get("source_levels"),
                    "n_dedup_removed": sensitivity_extras.get("n_dedup_removed"),
                    "diversity": sensitivity_extras.get("diversity"),
                    "selected_control": sensitivity_control.tolist(),
                    "branch_controls": branch_records,
                    "pair_refinement": sensitivity_extras.get("pair_refinement"),
                    "second_source_reselection": sensitivity_extras.get(
                        "second_source_reselection"
                    ),
                    "plot_paths": sensitivity_extras.get("plot_paths"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"sensitivity init summary: {init_summary_path}")
        if args.sensitivity_branch_multistart and len(branch_records) > 1:
            initial_controls = [
                np.asarray(record["control"], dtype=np.float64)
                for record in branch_records
            ]
            print(
                f"branch multistart: running the full optimization from all "
                f"{len(initial_controls)} retained geometric branch(es)"
            )
            for record in branch_records:
                centers = ", ".join(
                    f"({c[0]:.2f},{c[1]:.2f})" for c in record["centers"]
                )
                print(
                    f"  branch {record['branch_index']}: [{centers}] "
                    f"J_exact={record['objective']:.6e}  "
                    "wall distances "
                    + ", ".join(f"{d:.3f} m" for d in record["wall_distances_m"])
                )
        else:
            if args.sensitivity_branch_multistart:
                print(
                    "branch multistart requested but only one distinct branch was "
                    "retained; running the single best branch"
                )
            initial_controls = [sensitivity_control]
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
                num_sources,
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
    if true_state is None:
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
        num_starts_override=len(initial_controls),
        branch_controls=(
            sensitivity_branch_records if sensitivity_branch_records else None
        ),
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
