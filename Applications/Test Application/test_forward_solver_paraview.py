"""Run forward advection-diffusion solver on Test Application mesh and export ParaView time series.

Usage:
    python3 test_forward_solver_paraview.py

Outputs:
    Applications/Test Application/forward_output/
      - solution.pvd                 (open this in ParaView)
      - solution_0000.vtu, ...       (time-step data files)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import meshio
import numpy as np


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
SOLVER_DIR = PROJECT_ROOT / "Src" / "Solvers"

if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

_FORWARD_SOLVER_FILE = SOLVER_DIR / "adv_diff_forward.py"
_FORWARD_SOLVER_SPEC = importlib.util.spec_from_file_location(
    "adv_diff_forward",
    _FORWARD_SOLVER_FILE,
)
if _FORWARD_SOLVER_SPEC is None or _FORWARD_SOLVER_SPEC.loader is None:
    raise ImportError(f"Could not load forward solver module from {_FORWARD_SOLVER_FILE}")
_FORWARD_SOLVER_MODULE = importlib.util.module_from_spec(_FORWARD_SOLVER_SPEC)
_FORWARD_SOLVER_SPEC.loader.exec_module(_FORWARD_SOLVER_MODULE)
adv_diff_dyn = _FORWARD_SOLVER_MODULE.adv_diff_dyn

from mesh_data_config import build_mesh_data


def source_func(x: float, y: float) -> float:
    """Smooth source term for a stable forward test run."""
    xc = 0.25
    yc = 0.2
    sigma2 = 0.01
    return float(np.exp(-((x - xc) ** 2 + (y - yc) ** 2) / sigma2))


def write_pvd(pvd_path: Path, entries: list[tuple[float, str]]) -> None:
    """Write a ParaView collection file that indexes all VTU time steps."""
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>',
    ]
    for t, rel_file in entries:
        lines.append(f'    <DataSet timestep="{t:.12g}" group="" part="0" file="{rel_file}"/>')
    lines.extend(['  </Collection>', '</VTKFile>'])
    pvd_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    mesh_data = build_mesh_data()

    coord = np.asarray(mesh_data["coord"], dtype=np.float64)
    connections = np.asarray(mesh_data["connections"], dtype=int)

    npoin = coord.shape[0]
    nelem = connections.shape[0]

    # Domain size is known from generate_mesh.py (1.0 x 0.4 rectangle).
    lx = 1.0
    ly = 0.4

    # Keep a stable explicit-time run for smoke testing and visualization.
    ngauss = 3
    k = 1e-5
    vx = 2.0
    vy = 0.0
    dt = 1.0e-4
    tfinal = 0.02
    jprint = 1

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        unknown_out = adv_diff_dyn(
            npoin=npoin,
            nelem=nelem,
            ngauss=ngauss,
            lx=lx,
            ly=ly,
            k=k,
            vx=vx,
            vy=vy,
            dt=dt,
            tfinal=tfinal,
            jprint=jprint,
            coord=coord,
            connections=connections,
            source_func=source_func,
            mesh_data=mesh_data,
            dirichlet_tags=["left"],
        )

    out_dir = APP_DIR / "forward_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("solution_*.vtu"):
        stale.unlink()

    if coord.shape[1] == 2:
        points = np.column_stack([coord, np.zeros(npoin)])
    else:
        points = coord

    cells = [("triangle", connections)]

    nframes = unknown_out.shape[0]
    times = np.arange(nframes, dtype=np.float64) * dt * jprint

    pvd_entries: list[tuple[float, str]] = []
    for iframe in range(nframes):
        vtu_name = f"solution_{iframe:04d}.vtu"
        vtu_path = out_dir / vtu_name

        mesh = meshio.Mesh(
            points=points,
            cells=cells,
            point_data={"u": unknown_out[iframe]},
        )
        meshio.write(vtu_path, mesh)
        pvd_entries.append((float(times[iframe]), vtu_name))

    pvd_path = out_dir / "solution.pvd"
    write_pvd(pvd_path, pvd_entries)

    print(f"Forward solve complete. Output directory: {out_dir}")
    print(f"Open this file in ParaView: {pvd_path}")
    print("ParaView will automatically detect the time series from the .pvd collection.")


if __name__ == "__main__":
    main()
