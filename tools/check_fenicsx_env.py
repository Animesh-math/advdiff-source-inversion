#!/usr/bin/env python3
"""Check whether the local environment can import FEniCSx-related packages."""

from __future__ import annotations

import importlib
import sys
from typing import Any, Dict, Optional


PACKAGES = (
    "dolfinx",
    "ufl",
    "basix",
    "ffcx",
    "mpi4py",
    "petsc4py",
    "gmsh",
    "meshio",
)


def version_for(module: Any) -> str:
    version = getattr(module, "__version__", None)
    if version:
        return str(version)

    if module.__name__ == "petsc4py":
        try:
            from petsc4py import PETSc

            return ".".join(str(part) for part in PETSc.Sys.getVersion())
        except Exception as exc:  # pragma: no cover - diagnostic path
            return f"unknown (PETSc version unavailable: {type(exc).__name__}: {exc})"

    return "unknown"


def import_package(name: str) -> Optional[Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        print(f"{name}: FAILED ({type(exc).__name__}: {exc})")
        return None

    print(f"{name}: OK version={version_for(module)}")
    return module


def run_dolfinx_mesh_check(imports: Dict[str, Optional[Any]]) -> bool:
    if imports.get("dolfinx") is None:
        print("dolfinx unit-square mesh check: SKIPPED (dolfinx import failed)")
        return False

    try:
        from mpi4py import MPI

        from dolfinx import mesh

        domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
        dim = domain.topology.dim
        num_cells = domain.topology.index_map(dim).size_local
        num_vertices = domain.topology.index_map(0).size_local
    except Exception as exc:
        print(f"dolfinx unit-square mesh check: FAILED ({type(exc).__name__}: {exc})")
        return False

    print(
        "dolfinx unit-square mesh check: OK "
        f"topology_dim={dim} local_vertices={num_vertices} local_cells={num_cells}"
    )
    return True


def main() -> int:
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")

    imports = {name: import_package(name) for name in PACKAGES}
    run_dolfinx_mesh_check(imports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
