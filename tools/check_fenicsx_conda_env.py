#!/usr/bin/env python3
"""Verify the separate conda/mamba FEniCSx environment."""

from __future__ import annotations

import importlib
import sys
from typing import Any


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


def package_version(module: Any) -> str:
    version = getattr(module, "__version__", None)
    if version:
        return str(version)

    if module.__name__ == "petsc4py":
        from petsc4py import PETSc

        return ".".join(str(part) for part in PETSc.Sys.getVersion())

    return "unknown"


def main() -> int:
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")

    for name in PACKAGES:
        module = importlib.import_module(name)
        print(f"{name}: version={package_version(module)}")

    from mpi4py import MPI

    from dolfinx import mesh

    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    topology_dim = domain.topology.dim
    local_vertices = domain.topology.index_map(0).size_local
    local_cells = domain.topology.index_map(topology_dim).size_local

    print("unit-square mesh: OK")
    print(f"topology dimension: {topology_dim}")
    print(f"local vertices: {local_vertices}")
    print(f"local cells: {local_cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
