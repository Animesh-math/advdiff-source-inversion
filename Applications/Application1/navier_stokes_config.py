"""Navier-Stokes configuration for Application1 (DC Navy Yard).

This module is the SINGLE SOURCE OF TRUTH for the ambient wind and the solver
parameters.  To run a different wind, edit ``WIND_VELOCITY`` below and rerun the
solve command -- nothing else needs to change.  Everything downstream (inflow /
outflow classification, inflow Dirichlet data, the Picard initial guess, and the
run metadata) is derived from this vector.
"""

from __future__ import annotations

import math


# --------------------------------------------------------------------------- #
# Ambient wind  --  the only value you normally edit
# --------------------------------------------------------------------------- #
# Cartesian components in m/s, with  positive x = east,  positive y = north.
#
# Default: a measured wind of ~4.1 knots FROM 280 degrees true, converted to the
# Cartesian direction the air is moving TOWARD:
#     WIND_VELOCITY = (2.07, -0.36) m/s   (mostly eastward, slightly southward)
WIND_VELOCITY = (2.07, -0.36)


# --------------------------------------------------------------------------- #
# Physical parameters
# --------------------------------------------------------------------------- #
# Effective (eddy) KINEMATIC viscosity [m^2/s].
#
# This is deliberately NOT the molecular kinematic viscosity of air
# (~1.5e-5 m^2/s).  On this ~300 m x 700 m urban domain with a ~2 m/s wind, the
# molecular value gives a Reynolds number of ~4e7, which the unstabilised
# Taylor-Hood / Picard solver inherited from Test Application4 cannot resolve on
# this coarse mesh.  NU here is an urban effective/eddy viscosity chosen for a
# stable, reproducible, laminar first solve (domain Reynolds number ~O(1e2)).
# Increase NU for more robustness, decrease it for a sharper (less diffusive)
# field once stability is confirmed.
NU = 5.0

# Air density [kg/m^3].  Informational only: the solver uses the incompressible
# KINEMATIC form  (v.grad)v - nu*Laplace(v) + grad(p) = 0, so the computed
# pressure is kinematic (p/rho) and density does not enter the equations.
RHO = 1.225


# --------------------------------------------------------------------------- #
# Boundary classification
# --------------------------------------------------------------------------- #
# An outer boundary with outward unit normal n is classified from
# flux = WIND_VELOCITY . n :
#     flux < -FLUX_TOLERANCE  -> inflow   (Dirichlet u = WIND_VELOCITY)
#     flux >  FLUX_TOLERANCE  -> outflow  (pressure-outlet p = 0, do-nothing)
#     |flux| <= FLUX_TOLERANCE -> tangential/slip (natural, do-nothing)
# The tolerance is in m/s; it only needs to reject floating-point noise.
FLUX_TOLERANCE = 1.0e-9


# --------------------------------------------------------------------------- #
# Nonlinear solver controls (Picard fixed-point, as in Test Application4)
# --------------------------------------------------------------------------- #
MAX_PICARD_ITER = 50
PICARD_TOL = 1.0e-8   # relative L2 velocity-update tolerance


def wind_speed() -> float:
    """Magnitude of the configured wind vector [m/s]."""
    return math.hypot(WIND_VELOCITY[0], WIND_VELOCITY[1])
