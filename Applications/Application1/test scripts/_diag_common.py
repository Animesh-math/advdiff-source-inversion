"""Shared setup for the sensitivity-initialization diagnostics (read-only).

Reproduces exactly the Application1 configuration that ``run_optimization.py``
builds for ``--init-mode sensitivity`` so every diagnostic below runs against
the same mesh, velocity field, observation set, discrete norms and PDE dict.
Nothing here is imported by production code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parents[1]
OBJECTIVE_DIR = PROJECT_ROOT / "Src" / "Objective"
for p in (str(OBJECTIVE_DIR), str(APP_DIR)):  # APP_DIR last -> ends up first on sys.path
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

import run_optimization as ro  # noqa: E402
import gradient_test as gt  # noqa: E402
import sensitivity_initialization as sinit  # noqa: E402
from mesh_admissibility import MeshAdmissibility  # noqa: E402


TRUE_CONTROL = np.array([250.0, 130.0, 1.60, 15.0, 580.0, 0.98], dtype=np.float64)
AMPLITUDE_BOUNDS = (0.1, 2.0)
TFINAL = 150.0
SIGMA2_STAGES = (1600.0, 800.0, 400.0)

BOUNDARY_SETS = {
    "lrb": ("left", "right", "bottom"),
    "lrbt": ("left", "right", "bottom", "top"),
}


def build_setup(boundaries, sigma2, tfinal=TFINAL):
    """mesh_data + pde dict identical to the production sensitivity-init path."""
    mesh_data = ro.configure_measurement_boundaries(
        ro.build_mesh_data(), list(boundaries)
    )
    pde = dict(ro.PDE)
    pde["tfinal"] = float(tfinal)
    pde["sigma2"] = float(sigma2)
    vel_nodes, vel_summary = ro.load_velocity_field_file(
        str(ro.DEFAULT_VELOCITY_FIELD_FILE), mesh_data
    )
    pde["velocity_nodes"] = vel_nodes
    pde["velocity_field"] = vel_summary
    admiss = MeshAdmissibility(mesh_data)
    return mesh_data, pde, admiss


def obs_ctx(mesh_data, pde):
    return sinit._ObsTraceContext(
        mesh_data,
        ro.ETA1,
        float(pde["dt"] * pde["jprint"]),
        ro.OBJECTIVE_TIME_RULE,
    )


def unit_state(mesh_data, pde, cx, cy):
    src = sinit._gaussian_sum_source([(cx, cy, 1.0)], pde["sigma2"])
    return gt.run_forward(mesh_data, pde, sinit._DUMMY_Z, source_func=src)


def true_state(mesh_data, pde, control=TRUE_CONTROL):
    return gt.run_forward(mesh_data, pde, np.asarray(control, dtype=np.float64))
