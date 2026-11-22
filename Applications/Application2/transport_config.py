"""Tracer-transport and source-identification settings for Application2.

SINGLE SOURCE OF TRUTH for the hospital-room source-identification experiment:
tracer transport parameters, the Gaussian release model, the three two-source
cases, the fixed sensor layout, the objective convention, the amplitude bounds
and the observation-noise settings.  Geometry stays in ``room_geometry_config``
and the airflow in ``navier_stokes_config``; nothing here changes either.

Three different physical parameters -- do not confuse them
----------------------------------------------------------
``navier_stokes_config.NU``  effective kinematic VISCOSITY of the airflow [m^2/s]
                             (momentum equation only; fixed with the flow field).
``DIFFUSIVITY_K``            effective TRACER DIFFUSIVITY k [m^2/s] of the
                             advection-diffusion equation below.
``SIGMA2``                   squared e-folding radius rho [m^2] of the Gaussian
                             SOURCE (a length scale of the release, not a
                             transport coefficient).

Transport model (same discretization as Application1; Src/Solvers)
------------------------------------------------------------------
    du/dt + v . grad(u) - k Laplace(u) = S(x; z)   in the fluid domain, 0 < t <= T,
    u(x, 0) = 0,
    u = 0                      on the three supply vents (clean supply air),
    k grad(u) . n = 0          on every wall, partition, bed and bathroom-wall face
                               (impermeable; v = 0 there, so the total flux is zero),
    k grad(u) . n = 0          on the corridor door and the bathroom exhaust
                               (purely advective outflow; the flow fields have no
                               backflow on either outlet).

v is the steady Navier-Stokes field of the SAME mesh (P1 nodal values).  The
advection term is used in non-conservative form, so the natural condition of the
weak form is the zero DIFFUSIVE flux stated above.  P1 triangles, lumped mass,
classical explicit RK4 (Src/Solvers/adv_diff_forward.py).

Source model (Application1 convention, NOT renormalized)
--------------------------------------------------------
    S(x; z) = sum_i A_i exp( -|x - c_i|^2 / rho ),     z = [x1, y1, A1, x2, y2, A2],

rho = SIGMA2 = 2 sigma_std^2, so sqrt(rho) is the e-folding radius.  A_i is the
PEAK volumetric release rate at the centre [concentration unit / s]; the
Gaussian is not divided by its integral.  The total release rate of one source
per metre of out-of-plane depth is  A_i * pi * rho  [concentration unit * m^2/s].
Concentration is in arbitrary tracer units (the problem is linear in A).
These are stationary tracer releases near the bed heads, not geometrically
resolved patient mouths.

How the numerical settings were selected
----------------------------------------
From resolution and stability checks only (``test scripts/select_transport_settings.py``,
recorded in ``<dataset>/common/settings_selection/``); no recovery outcome was
used and none existed when they were fixed.  See ``SETTINGS_RATIONALE``.
"""

from __future__ import annotations

import math

import room_geometry_config as geo

# --------------------------------------------------------------------------- #
# Versions
# --------------------------------------------------------------------------- #
DATASET_VERSION = "source_id_v2_wall8_20261116"
# Earlier dataset versions (preserved on disk, NOT to be used for the paper inversions):
#   source_id_v1_20261116  generated with the superseded 12-sensor "aisle_passage_12" layout
#                          (free-standing points in the bed-side passages and the aisle).
SUPERSEDED_DATASETS = {"source_id_v1_20261116": "aisle_passage_12"}
FLOW_VERSION = geo.OUTPUT_VERSION            # velocity fields / meshes this dataset is tied to
DATA_MESH_LEVEL = "fine"                     # synthetic observations
INVERSION_MESH_LEVEL = "coarse"              # the entire eventual inversion

# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #
DIFFUSIVITY_K = 1.0e-2          # effective tracer diffusivity k [m^2/s]
SIGMA2 = 0.04                   # rho [m^2]; e-folding radius sqrt(rho) = 0.20 m
TFINAL = 300.0                  # observation horizon T [s]
NGAUSS = 3                      # triangle quadrature (as Application1)

# Explicit RK4 steps.  The discrete adjoint of the shared solver needs every
# coarse step saved (jprint = 1), so the coarse step IS the observation interval.
# The fine mesh uses an integer sub-step and is sampled every FINE_SUBSTEPS steps,
# i.e. at exactly the same times t_n = n * OBSERVATION_INTERVAL.
DT = {"coarse": 0.015, "fine": 0.0025}
OBSERVATION_INTERVAL = DT["coarse"]
FINE_SUBSTEPS = 6
NUM_STEPS = {lv: int(math.floor((TFINAL + 0.5 * dt) / dt)) for lv, dt in DT.items()}
NUM_OBSERVATION_TIMES = NUM_STEPS["coarse"]          # n = 1 .. N (n = 0 is the zero initial state)

# RK4 absolute stability: |R(dt * lambda)| <= 1 on the negative real axis for
# dt * lambda <= 2.785.  The pipeline computes the spectral radius of
# M_lumped^{-1} K (with the clean-air rows) for each mesh and its own velocity
# field and aborts unless dt * rho <= RK4_STABILITY_TARGET (the margin used in
# the Application1 mesh-refinement study).
RK4_REAL_AXIS_LIMIT = 2.785
RK4_STABILITY_TARGET = 2.5

CLEAN_AIR_DIRICHLET_NAMES = tuple(geo.INLET_NAMES)   # u = 0 on the supplies

# --------------------------------------------------------------------------- #
# Objective (Application1 node-sensor convention)
# --------------------------------------------------------------------------- #
#   J_h(z) = (eta1 * dt_obs / 2) * sum_{n=1}^{N} || C U^n(z) - d^n ||_W^2 + (lambda/2) |z|^2
# "exclude_first": the n = 0 sample (identically zero initial state) is not in the sum.
ETA1 = 1.0
LAMBDA_REG = 0.0
OBJECTIVE_TIME_RULE = "exclude_first"
GRADIENT_TIME_RULE = "trapezoid"            # legacy argument of the shared gradient code (unused there)
SENSOR_WEIGHT = 1.0                          # W = SENSOR_WEIGHT * I (equal-variance noise); configurable per run

# --------------------------------------------------------------------------- #
# Admissible set handed to the inversion
# --------------------------------------------------------------------------- #
NUM_SOURCES = 2
AMPLITUDE_BOUNDS = (0.1, 2.0)                # [concentration unit / s]; 0.1 = Application1's amplitude floor
# Source centres: anywhere in the fluid mesh (beds, partitions, the bathroom wall
# and the exterior are holes/notches, never inside a triangle).  The L-BFGS-B box
# is the mesh bounding box; admissibility is the mesh point-location test.

# --------------------------------------------------------------------------- #
# Releases: ONE fixed location and amplitude per bed, reused in every case
# --------------------------------------------------------------------------- #
# Location rule (geometry only): beside the head of the bed, on the centreline of
# the passage on the +x side of the bed, RELEASE_HEAD_OFFSET below the head end.
RELEASE_HEAD_OFFSET = 0.40                   # [m] below the bed-head end (y = 5.6)


def _release_center(bay: int):
    bed = geo.BEDS[bay]
    x = 0.5 * (bed["x"][1] + geo.BAYS[bay]["x"][1])
    return (round(x, 10), round(bed["head_y"] - RELEASE_HEAD_OFFSET, 10))


RELEASE_AMPLITUDE = {1: 1.0, 2: 0.7, 3: 1.3}  # distinct, strictly inside AMPLITUDE_BOUNDS
RELEASES = {bay: {"center": _release_center(bay), "amplitude": RELEASE_AMPLITUDE[bay]} for bay in (1, 2, 3)}
MIN_RELEASE_CLEARANCE_RADII = 2.0            # required wall clearance in e-folding radii
MIN_RADIUS_OVER_COARSE_EDGE = 1.0            # e-folding radius / mean local coarse edge (readiness check floor)

CASES = {"12": (1, 2), "13": (1, 3), "23": (2, 3)}


def case_dir_name(case: str) -> str:
    return f"case_{case}"


def true_control(case: str):
    """[x1, y1, A1, x2, y2, A2] of a case (TRUTH: never passed to the inversion)."""
    z = []
    for bay in CASES[case]:
        r = RELEASES[bay]
        z += [r["center"][0], r["center"][1], r["amplitude"]]
    return z


# --------------------------------------------------------------------------- #
# Sensors
# --------------------------------------------------------------------------- #
# DECISION (user, 2026-11-16): use the eight-sensor WALL-MOUNTED layout "wall_8"
# first, for data generation and for the optimization, identical in all three
# source-pair cases.  It is defined from room geometry only and must not be
# adjusted using the true sources or any recovery outcome.
#
#   S1  left exterior wall of Bay 1, beside the upper portion of the bed
#   S2  Bay 1 face of the first partition, beside the lower portion of the bed
#   S3  Bay 2 face of the first partition, upper position
#   S4  Bay 2 face of the second partition, lower position
#   S5  Bay 3 face of the second partition, upper position
#   S6  right exterior wall of Bay 3, lower position
#   S7  bottom wall of the shared aisle, immediately to the right of the corridor doorway
#   S8  bathroom's upper wall, near the exhaust side
#
# Geometric realization (plan view; "upper" = toward the bed head / top wall):
#   upper / lower position = 3/4 / 1/4 of the bed length above the bed foot (y = 5.075 / 4.025);
#   S7 is SENSOR_DOOR_OFFSET to the right of the door jamb; S8 is SENSOR_EXHAUST_OFFSET
#   from the exhaust wall.  Each wall point is moved SENSOR_WALL_OFFSET into the fluid
#   along the wall normal and snapped to the nearest coarse-mesh node that is NOT a
#   boundary node, so every sampling point is slightly inside the adjacent fluid.
#   The snapped coordinates (and their actual wall offsets) are recorded in
#   <dataset>/common/sensors.{json,csv} and are the sampling points on BOTH meshes
#   (P1 interpolation on the fine mesh).
#
# PROPOSED, NOT ACTIVE: "wall_10" = wall_8 plus two sensors (S9, S10) on the same
# bottom aisle wall as S7, evenly spaced along the remaining wall between S7 and
# the bathroom wall.  Recorded for a possible later extension; do not activate
# without an explicit decision (it needs a new DATASET_VERSION).
#
# SUPERSEDED: "aisle_passage_12" (free-standing points in both side passages of
# each bed and a 3 x 2 aisle grid); judged unrealistic.  Dataset
# source_id_v1_20261116 was generated with it and is preserved, not relabelled.
ACTIVE_SENSOR_LAYOUT = "wall_8"
PROPOSED_SENSOR_LAYOUTS = ("wall_10",)
SUPERSEDED_SENSOR_LAYOUTS = ("aisle_passage_12",)
SENSOR_WALL_OFFSET = 0.10        # [m] intended distance of the sampling point from its wall
SENSOR_DOOR_OFFSET = 0.30        # [m] S7: distance to the right of the door jamb x = 1.6
SENSOR_EXHAUST_OFFSET = 0.50     # [m] S8: distance from the exhaust wall x = 11
_BED_Y0, _BED_Y1 = geo.BED_Y_RANGE
SENSOR_UPPER_Y = round(_BED_Y0 + 0.75 * (_BED_Y1 - _BED_Y0), 10)
SENSOR_LOWER_Y = round(_BED_Y0 + 0.25 * (_BED_Y1 - _BED_Y0), 10)
_BED_MID_Y = 0.5 * (_BED_Y0 + _BED_Y1)
AISLE_SENSOR_Y = (0.75, 2.25)


def _wall_sensor(label, name, description, wall, wall_point, normal, region):
    x = round(wall_point[0] + SENSOR_WALL_OFFSET * normal[0], 10)
    y = round(wall_point[1] + SENSOR_WALL_OFFSET * normal[1], 10)
    return {"label": label, "name": name, "description": description, "wall": wall,
            "wall_point": [float(wall_point[0]), float(wall_point[1])], "inward_normal": list(normal),
            "intended_xy": [x, y], "region": region}


def _wall_8():
    p1, p2 = geo.PARTITIONS["partition_1"], geo.PARTITIONS["partition_2"]
    s7_x = round(geo.DOOR_X_RANGE[1] + SENSOR_DOOR_OFFSET, 10)
    s8_x = round(geo.ANNEX_X[1] - SENSOR_EXHAUST_OFFSET, 10)
    return [
        _wall_sensor("S1", "bay1_left_wall_upper", "left exterior wall of Bay 1, beside the upper portion of the bed",
                     "wall_left", (geo.ROOM_X_MIN, SENSOR_UPPER_Y), (1.0, 0.0), "bay_1"),
        _wall_sensor("S2", "bay1_partition1_lower", "Bay 1 face of the first partition, beside the lower portion of the bed",
                     "partition_1", (p1[0], SENSOR_LOWER_Y), (-1.0, 0.0), "bay_1"),
        _wall_sensor("S3", "bay2_partition1_upper", "Bay 2 face of the first partition, upper position",
                     "partition_1", (p1[1], SENSOR_UPPER_Y), (1.0, 0.0), "bay_2"),
        _wall_sensor("S4", "bay2_partition2_lower", "Bay 2 face of the second partition, lower position",
                     "partition_2", (p2[0], SENSOR_LOWER_Y), (-1.0, 0.0), "bay_2"),
        _wall_sensor("S5", "bay3_partition2_upper", "Bay 3 face of the second partition, upper position",
                     "partition_2", (p2[1], SENSOR_UPPER_Y), (1.0, 0.0), "bay_3"),
        _wall_sensor("S6", "bay3_right_wall_lower", "right exterior wall of Bay 3, lower position",
                     "wall_right", (geo.ROOM_X_MAX, SENSOR_LOWER_Y), (-1.0, 0.0), "bay_3"),
        _wall_sensor("S7", "aisle_bottom_wall_right_of_door",
                     "bottom wall of the shared aisle, immediately to the right of the corridor doorway",
                     "wall_bottom", (s7_x, geo.ROOM_Y_MIN), (0.0, 1.0), "aisle"),
        _wall_sensor("S8", "bathroom_top_wall_near_exhaust", "bathroom's upper wall, near the exhaust side",
                     "bath_top", (s8_x, geo.ANNEX_Y[1]), (0.0, -1.0), "bathroom"),
    ]


def _wall_10():
    base = _wall_8()
    x7, x_end = base[6]["wall_point"][0], geo.BATH_WALL_X[0]
    step = (x_end - x7) / 3.0
    extra = [_wall_sensor(f"S{9 + i}", f"aisle_bottom_wall_extension_{i + 1}",
                          "bottom wall of the shared aisle, evenly spaced between S7 and the bathroom wall",
                          "wall_bottom", (round(x7 + (i + 1) * step, 10), geo.ROOM_Y_MIN), (0.0, 1.0), "aisle")
             for i in range(2)]
    return base + extra


def _aisle_passage_12():
    out = []
    for bay in (1, 2, 3):
        bed, span = geo.BEDS[bay], geo.BAYS[bay]["x"]
        out.append((f"bay{bay}_left_passage", 0.5 * (span[0] + bed["x"][0]), _BED_MID_Y, f"bay_{bay}"))
        out.append((f"bay{bay}_right_passage", 0.5 * (bed["x"][1] + span[1]), _BED_MID_Y, f"bay_{bay}"))
    for row, y in zip(("lower", "upper"), AISLE_SENSOR_Y):
        for bay, cx in zip((1, 2, 3), geo.BED_CENTERS_X):
            out.append((f"aisle_{row}_under_bay{bay}", cx, y, "aisle"))
    return [{"label": f"s{i}", "name": n, "description": n, "wall": None, "wall_point": None, "inward_normal": None,
             "intended_xy": [float(x), float(y)], "region": r} for i, (n, x, y, r) in enumerate(out)]


SENSOR_LAYOUTS = {"wall_8": _wall_8, "wall_10": _wall_10, "aisle_passage_12": _aisle_passage_12}
SENSOR_LAYOUT_RULES = {
    "wall_8": "eight wall-mounted sensors (user decision 2026-11-16): geometry only; each wall point moved "
              f"{SENSOR_WALL_OFFSET:g} m into the fluid and snapped to the nearest coarse-mesh node that is not a "
              "boundary node; identical for all cases; not adjusted using true sources, signals or recovery outcomes.",
    "wall_10": "PROPOSED, NOT ACTIVE: wall_8 plus two sensors evenly spaced on the bottom aisle wall between S7 and "
               "the bathroom wall.",
    "aisle_passage_12": "SUPERSEDED: free-standing points in both side passages of each bed and a 3 x 2 aisle grid.",
}


def sensor_targets(layout: str | None = None):
    """Intended sensor records of a layout (default: the ACTIVE one); geometry only."""
    return SENSOR_LAYOUTS[ACTIVE_SENSOR_LAYOUT if layout is None else layout]()


SENSOR_SNAP_TOLERANCE = 0.15                 # [m]; coarse bulk element size is 0.20 m

# --------------------------------------------------------------------------- #
# Observation noise (first noisy experiment)
# --------------------------------------------------------------------------- #
#   s_noise = NOISE_LEVEL * sqrt( mean_{n=1..N, j} |d_clean_j^n|^2 ),  eps ~ N(0, s_noise^2) i.i.d.
# One realization per case, drawn once with numpy.random.default_rng(seed) and saved.
NOISE_LEVEL = 0.01
NOISE_SEEDS = {"12": 2026111612, "13": 2026111613, "23": 2026111623}

SETTINGS_RATIONALE = {
    "diffusivity_k": (
        "k = 1.0e-2 m^2/s is the smallest of the candidates {nu_eff = 2.3244e-3, 5e-3, 1e-2} for which the "
        "cell Peclet number |v| h / (2 k) is <= 1 in every triangle of the COARSE (inversion) mesh (max 0.97), "
        "the classical condition for a non-oscillatory unstabilized Galerkin P1 discretization; the shared "
        "solver has no stabilization.  Measured with all three releases active: k = nu_eff gives cell Pe up to "
        "4.2 on the coarse mesh (46 % of the area above 1) and a small negative undershoot (min u / max u = "
        "-6.6e-4); k = 1e-2 gives a non-negative coarse solution and the smallest coarse-vs-fine field "
        "difference (1.6-1.9 % relative L2, against 2.4-3.0 % for k = nu_eff).  k is an effective (eddy) "
        "diffusivity, a modelling parameter like nu_eff but a different one (nu_eff / k = 0.23); it is far "
        "above molecular diffusion."
    ),
    "sigma2": (
        "sqrt(rho) = 0.20 m is a compromise fixed by the 0.95-1.0 m bed-side passages: the passage centreline "
        "is 2.4 e-folding radii from the nearest solid (Gaussian value 0.4 % there), and the radius spans "
        "1.3 mean element edges on the coarse mesh and 2.5 on the fine mesh at the release locations "
        "(Application1's default was 2.5 on its mesh; its width study went down to about 0.9).  The 3-point "
        "quadrature of the source integral agrees between the two meshes to 3e-4 and the coarse and fine "
        "concentration fields differ by 1.6-1.9 % (settings-selection record).  A wider Gaussian would be "
        "better resolved on the coarse mesh but would overlap the bed and the partition."
    ),
    "tfinal": "T = 300 s, about one nominal air-change time A_fluid / Q_supply = 50.98 / 0.18 = 283 s.",
    "dt": (
        "largest round steps with dt * rho(M^-1 K) <= 2.5 on each mesh with its own velocity field and an "
        "integer fine/coarse ratio, so that both meshes share the observation times."
    ),
    "amplitudes": "arbitrary distinct values within +-30 % of 1, strictly inside the bounds [0.1, 2.0].",
}


def as_dict() -> dict:
    return {
        "dataset_version": DATASET_VERSION,
        "sensor_layout": ACTIVE_SENSOR_LAYOUT,
        "sensor_layout_rule": SENSOR_LAYOUT_RULES[ACTIVE_SENSOR_LAYOUT],
        "proposed_inactive_sensor_layouts": {k: SENSOR_LAYOUT_RULES[k] for k in PROPOSED_SENSOR_LAYOUTS},
        "superseded_datasets": dict(SUPERSEDED_DATASETS),
        "flow_version": FLOW_VERSION,
        "data_mesh_level": DATA_MESH_LEVEL,
        "inversion_mesh_level": INVERSION_MESH_LEVEL,
        "pde": "du/dt + v.grad(u) - k Laplace(u) = S(x; z); u(0) = 0",
        "boundary_conditions": {
            "supplies (inlet_1..3)": "u = 0 (clean supply air)",
            "walls, partitions, beds, bathroom wall": "k grad(u).n = 0 (impermeable, no-slip velocity)",
            "outlets (outlet_door, outlet_exhaust)": "k grad(u).n = 0 (advective outflow)",
        },
        "diffusivity_k_m2_per_s": DIFFUSIVITY_K,
        "flow_viscosity_note": "the airflow's effective viscosity is navier_stokes_config.NU; it is a different "
                               "parameter and is not used by the transport solve",
        "source_formula": "S(x; z) = sum_i A_i exp(-|x - c_i|^2 / rho); z = [x1, y1, A1, x2, y2, A2]",
        "amplitude_convention": "A_i = peak volumetric release rate at the centre [concentration unit / s]; "
                                "the Gaussian is NOT normalized; total rate per metre depth = A_i * pi * rho",
        "sigma2_rho_m2": SIGMA2,
        "e_folding_radius_m": math.sqrt(SIGMA2),
        "gaussian_std_m": math.sqrt(0.5 * SIGMA2),
        "tfinal_s": TFINAL,
        "dt_s": dict(DT),
        "num_steps": dict(NUM_STEPS),
        "observation_interval_s": OBSERVATION_INTERVAL,
        "fine_substeps_per_observation": FINE_SUBSTEPS,
        "num_observation_times_in_objective": NUM_OBSERVATION_TIMES,
        "time_integrator": "classical explicit RK4, lumped mass, P1 triangles (Src/Solvers/adv_diff_forward.py)",
        "rk4_stability_target_dt_rho": RK4_STABILITY_TARGET,
        "objective": "J = (eta1 * dt_obs / 2) * sum_{n=1}^{N} ||C U^n - d^n||_W^2 + (lambda / 2) |z|^2",
        "eta1": ETA1,
        "lambda_reg": LAMBDA_REG,
        "objective_time_rule": OBJECTIVE_TIME_RULE,
        "sensor_weight_W": f"{SENSOR_WEIGHT} * I",
        "num_sources": NUM_SOURCES,
        "amplitude_bounds": list(AMPLITUDE_BOUNDS),
        "noise_level_nominal": NOISE_LEVEL,
        "noise_seeds": dict(NOISE_SEEDS),
        "settings_rationale": dict(SETTINGS_RATIONALE),
    }
