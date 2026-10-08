"""Check nodal-velocity advection against constant-velocity assembly on App4."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
SOLVER_DIR = PROJECT_ROOT / "Src" / "Solvers"

if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from adv_diff_forward import _assemble_system
from mesh_data_config import build_mesh_data
from pde_params import PDEParams


def zero_source(_x, _y):
    return 0.0


def main() -> None:
    mesh_data = build_mesh_data()
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)
    connections = np.asarray(mesh_data["connections"], dtype=int)

    vx = 1.0
    vy = 0.3
    common = {
        "k": 0.0,
        "velocity": (vx, vy),
        "dt": 1.0e-3,
        "tfinal": 1.0e-3,
        "ngauss": 3,
    }
    constant_params = PDEParams(**common)
    velocity_nodes = np.column_stack(
        [
            np.full(coord.shape[0], vx, dtype=np.float64),
            np.full(coord.shape[0], vy, dtype=np.float64),
        ]
    )
    nodal_params = PDEParams(**common, velocity_nodes=velocity_nodes)

    constant_matrix, _, _ = _assemble_system(
        npoin=coord.shape[0],
        nelem=connections.shape[0],
        coord=coord,
        connections=connections,
        source_func=zero_source,
        params=constant_params,
    )
    nodal_matrix, _, _ = _assemble_system(
        npoin=coord.shape[0],
        nelem=connections.shape[0],
        coord=coord,
        connections=connections,
        source_func=zero_source,
        params=nodal_params,
    )

    diff = nodal_matrix - constant_matrix
    max_abs = float(np.max(np.abs(diff)))
    fro_norm = float(np.linalg.norm(diff))
    ref_norm = float(np.linalg.norm(constant_matrix))
    rel_fro = fro_norm / ref_norm if ref_norm > 0.0 else fro_norm
    print(f"constant velocity: vx={vx}, vy={vy}")
    print(f"matrix shape: {constant_matrix.shape}")
    print(f"max_abs_diff: {max_abs:.16e}")
    print(f"fro_diff: {fro_norm:.16e}")
    print(f"rel_fro_diff: {rel_fro:.16e}")

    tolerance = 1.0e-12
    if max_abs > tolerance:
        raise SystemExit(
            "nodal constant-velocity assembly does not match constant mode: "
            f"max_abs_diff={max_abs:.16e} > {tolerance:.1e}"
        )


if __name__ == "__main__":
    main()
