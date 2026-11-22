"""Navier-Stokes configuration for Application2 (ventilated treatment room).

SINGLE SOURCE OF TRUTH for the supply-air speed, the viscosity and the solver
controls.  Both mesh levels (coarse, fine) are solved with exactly these values;
edit here and rerun to revise them.

The room experiment uses a steady, two-dimensional airflow with a spatially
constant effective viscosity. It provides an idealized setting for evaluating
source identification rather than a validated prediction of hospital
ventilation.

PDE (incompressible, steady, KINEMATIC form, vector-Laplacian viscosity):

    -nu * Laplace(v) + (v . grad) v + grad(p) = 0,     div(v) = 0,

with p the kinematic pressure (p_physical / rho) [m^2/s^2].

Boundary conditions (geometry: room_geometry_config.py)
-------------------------------------------------------
  * supplies (top wall y = 6, tags 21-23): Dirichlet, DOWNWARD parabolic profile;
  * all walls, partitions, beds and the bathroom wall (tags 41-47, 51-52, 61-63,
    71-72): no slip v = 0;
  * outlets, corridor door (31, y = 0) and bathroom exhaust (32, x = 11): NATURAL
    outflow (nu grad v - p I) n = 0.  No outlet pressure value, no outlet
    velocity, no pressure pinning or mean constraint: the flow split between the
    two outlets is part of the solution.  These passive natural outlets do not
    model mechanical suction or a negative-pressure room.

The ventilation arrangement is inspired by Fig. 24 / Sec. 7.1 of Lohner et al.,
arXiv:2006.13792 (adapted to three beds; the bathroom ceiling exhaust becomes a
2D boundary outlet).  The constant effective viscosity is a modelling choice of
this idealized experiment; it is not derived from or justified by that paper or
by the Chen-Xu zero-equation model.

Parameter record (2026-11-16)
-----------------------------
The steady problem depends only on Re_slot = U_mean * b / nu (b = 0.60 m vent
width) for this fixed geometry and inlet-profile shape.  Trials (preserved in
output/trials_20261116_picard_viscosity/):

  U_mean  nu        Re_slot  Picard (coarse / fine)        steady solution
  0.30    5.0e-3    36       NOT converged / NOT converged  exists (Newton), linearly UNSTABLE:
                                                            leading eigenvalue -1.20e-3 / -1.14e-3 1/s
  0.30    7.5e-3    24       56 / 56 it                     stable (leading eigenvalue +1.58e-2 1/s)
  0.30    1.0e-2    18       41 / -  it                     -
  0.10    2.3244e-3 25.8     66 / 67 it                     stable (leading eigenvalue +3.95e-3 1/s)

The adopted values are the last row (decided 2026-11-16, user-approved).
These trials used the earlier v1 geometry (no beds, bottom supply / top
extraction; output/v1_nobeds_bottomsupply_20261116/trials_20261116_picard_viscosity/).
The values are retained unchanged for the current geometry (user instruction);
its stability has not been re-assessed.
"""

from __future__ import annotations

import room_geometry_config as _geo

# --------------------------------------------------------------------------- #
# Supply air
# --------------------------------------------------------------------------- #
# INLET_MEAN_SPEED is the MEAN (bulk) normal speed over each 0.60 m supply
# segment [m/s] on the top wall.  The profile is parabolic, DOWNWARD, and vanishes
# at both ends:
#     v_y(x) = -INLET_PEAK_SPEED * 4 s (1 - s),   s = (x - x_a) / (x_b - x_a),   v_x = 0,
# so the PEAK (centreline) speed is 1.5 * INLET_MEAN_SPEED.  Prescribed flux per
# inlet (2D, per metre of out-of-plane extent): Q_in = INLET_MEAN_SPEED * 0.60 m.
#
# Approximate air-change rate, ONLY under these out-of-plane assumptions: the
# plan-view field is the height average of the room flow; each opening spans
# the full room height; supply and extract are uniform over that height; all
# air enters through the three inlets.  Then the room height cancels and
#     ACH = 3 * U_mean * 0.60 m * 3600 s/h / A_fluid,  A_fluid = 50.98 m^2 (room + bathroom
# minus partitions, beds and bathroom wall), i.e. ~12.7 1/h for U_mean = 0.10 m/s
# (11.2 1/h if referred to the gross 58 m^2 floor area).  Real supply and
# extract terminals do not span the room height, so this is an interpretation
# of the 2D model, not a property of a real room.
INLET_MEAN_SPEED = 0.10
INLET_PROFILE = "parabolic"
INLET_PEAK_SPEED = 1.5 * INLET_MEAN_SPEED


def inlet_velocity(x, peak: float = INLET_PEAK_SPEED):
    """Supply velocity at points x (shape (>=2, n)): downward parabola on each top-wall vent."""
    import numpy as np

    vals = np.zeros((2, x.shape[1]))
    for a, b in _geo.VENT_X_RANGES:
        s = (x[0] - a) / (b - a)
        m = (s >= 0.0) & (s <= 1.0) & (np.abs(x[1] - _geo.INLET_Y) < 1e-9)
        vals[1, m] = -peak * 4.0 * s[m] * (1.0 - s[m])
    return vals

# --------------------------------------------------------------------------- #
# Viscosity
# --------------------------------------------------------------------------- #
# NU is a spatially constant EFFECTIVE kinematic viscosity [m^2/s] -- a modelling
# parameter of the idealized experiment.  It is NOT the molecular viscosity of
# air (nu_air ~ 1.5e-5 m^2/s at 20-22 C, 1 atm; mu_air ~ 1.8e-5 Pa s), which at
# these speeds would give Re_slot ~ 4e3 (turbulent, unsteady).  It is also not a
# tracer diffusivity.  With NU = 2.32e-3:
#     Re_slot = U_mean * b / NU      = 0.10 * 0.60 / 2.3244e-3 = 25.8   (b = vent width)
#     Re_room = U_mean * H / NU      = 0.10 * 6.0  / 2.3244e-3 = 258    (H = room depth in plan)
# In the v1 (no-bed, bottom-supply) geometry the steady solution at this Re was linearly stable
# on both meshes and lost stability near Re_slot ~ 35; not re-assessed for the current layout.
NU = 2.3244e-3
NU_KIND = ("spatially constant effective kinematic viscosity (modelling parameter of an idealized 2D "
           "experiment); not molecular air viscosity, not a tracer diffusivity")
NU_MOLECULAR_AIR = 1.5e-5

# Air density [kg/m^3]: informational only (kinematic formulation).
RHO = 1.2

# --------------------------------------------------------------------------- #
# Nonlinear solver (Picard fixed point, as in Application1)
# --------------------------------------------------------------------------- #
# Iteration 1 is a Stokes solve (frozen advecting velocity w = 0); thereafter w
# is the previous iterate.  Converged when ||u_k - u_{k-1}||_L2 / ||u_k||_L2
# < PICARD_TOL.  Identical for both mesh levels.
MAX_PICARD_ITER = 100
PICARD_TOL = 1.0e-8

MODEL_STATEMENT = ("The room experiment uses a steady, two-dimensional airflow with a spatially constant effective "
                   "viscosity. It provides an idealized setting for evaluating source identification rather than a "
                   "validated prediction of hospital ventilation.")
LAYOUT_ATTRIBUTION = ("Ventilation arrangement inspired by Fig. 24 and Sec. 7.1 of Lohner et al., arXiv:2006.13792; "
                      "this three-bed room is an adaptation, not a reproduction, and the paper's bathroom ceiling "
                      "exhaust is represented by a 2D boundary outlet. Not clinically validated.")


def reynolds_numbers(vent_width: float = 0.60, room_height: float = 6.0) -> dict:
    return {
        "Re_slot_mean_speed_vent_width": INLET_MEAN_SPEED * vent_width / NU,
        "Re_room_mean_speed_room_depth": INLET_MEAN_SPEED * room_height / NU,
        "Re_slot_if_molecular_air": INLET_MEAN_SPEED * vent_width / NU_MOLECULAR_AIR,
        "characteristic_lengths_m": {"slot": vent_width, "room": room_height},
    }


def approx_air_changes_per_hour(fluid_area: float | None = None, vent_width: float = 0.60,
                                n_inlets: int = 3) -> float:
    """ACH under the full-height-opening, height-averaged assumptions stated above."""
    if fluid_area is None:
        fluid_area = _geo.fluid_area()
    return n_inlets * INLET_MEAN_SPEED * vent_width * 3600.0 / fluid_area


def as_dict() -> dict:
    return {
        "inlet_mean_speed_m_per_s": INLET_MEAN_SPEED,
        "inlet_peak_speed_m_per_s": INLET_PEAK_SPEED,
        "inlet_speed_convention": "INLET_MEAN_SPEED is the mean over each inlet; peak = 1.5 * mean",
        "inlet_profile": INLET_PROFILE + ": v_y = -peak * 4 s (1 - s) (downward), v_x = 0, s in [0,1] along each "
                                         "top-wall supply",
        "nu_m2_per_s": NU,
        "nu_kind": NU_KIND,
        "rho_kg_per_m3_informational": RHO,
        "max_picard_iter": MAX_PICARD_ITER,
        "picard_tol_relative_L2_update": PICARD_TOL,
        "reynolds_numbers": reynolds_numbers(),
        "approx_ach_full_height_openings": approx_air_changes_per_hour(),
        "approx_ach_area_basis_m2": _geo.fluid_area(),
        "approx_ach_gross_room_area": approx_air_changes_per_hour(
            _geo.gross_area()),
        "approx_ach_assumptions": ("plan-view field = height average; each opening spans the full room height; "
                                   "uniform supply/extract over the height; all air enters via the three inlets"),
        "model_statement": MODEL_STATEMENT,
        "layout_attribution": LAYOUT_ATTRIBUTION,
        "outlet_condition": "natural (nu grad v - p I) n = 0 on outlet_door and outlet_exhaust; split not prescribed",
    }
