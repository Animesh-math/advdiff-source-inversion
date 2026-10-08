# App4 Navier-Stokes Velocity Solve

This folder contains a standalone FEniCSx solve that generates a geometry-aware
steady velocity field for Test Application4.  It does not modify the inverse
optimizer or the source-identification objective.

## PDE

The script solves the steady incompressible Navier-Stokes equations on the
existing App4 L-shaped city domain:

```text
(v . grad) v - nu * Laplace(v) + grad(p) = 0
div(v) = 0
```

There is no body force.  The default parameters are `Umax = 1.0` and
`nu = 0.10`.  The viscosity is intentionally moderate for the first solve on
this nonconvex domain with internal building holes.

## Boundary Conditions

The left boundary is the inflow:

```text
v = (4 * Umax * y * (1 - y), 0)
```

This is a parabolic inflow profile.  It is zero at `y = 0` and `y = 1`, and it
reaches `Umax` at `y = 0.5`.

The right boundary is the outflow.  The script imposes:

```text
p = 0
```

This fixes the pressure reference level on the outflow.  The velocity is not
prescribed there, so the flow can leave naturally.

All other boundaries are no-slip walls:

```text
v = (0, 0)
```

This includes the bottom wall, top wall, inner L-corner boundary, and all five
building boundaries.

## Numerical Method

The script uses Taylor-Hood finite elements:

```text
velocity: continuous P2 vector elements
pressure: continuous P1 scalar elements
```

The nonlinear Navier-Stokes term is solved by Picard fixed-point iteration.
At Picard iteration `k`, the advecting velocity `w` is frozen from the previous
iterate and the script solves:

```text
(w . grad) v - nu * Laplace(v) + grad(p) = 0
div(v) = 0
```

Each linearized saddle-point system is solved with PETSc LU.  The script prints
the L2 velocity update norm and relative update at each Picard iteration.

## How To Run

Run from the repository root:

```text
conda run -n fenicsx-env python "Applications/Test Application4/Navier Stokes Solve/solve_app4_navier_stokes.py"
```

or from this folder:

```text
conda run -n fenicsx-env python solve_app4_navier_stokes.py
```

Do not use the inverse-solver environment for this script.  It imports DOLFINx, UFL, Basix, PETSc,
MPI, Gmsh, and FEniCSx I/O packages from `fenicsx-env`.

## Outputs

The script writes generated files under:

```text
Applications/Test Application4/output/navier_stokes/
```

Expected files:

```text
app4_navier_stokes_velocity.bp
app4_navier_stokes_pressure.bp
app4_navier_stokes_velocity_nodes.npz
```

The `.bp` files are ParaView-readable ADIOS2/VTX files.  The `.npz` file
contains:

```text
coords
velocity_nodes
```

`coords` is copied from the existing App4 transport mesh in
`mesh_data_config.py`.  `velocity_nodes[i]` is the Navier-Stokes velocity
evaluated at `coords[i]`, so the node ordering matches the inverse transport
solver's existing mesh ordering.  The script uses DOLFINx collision-tree point
evaluation and reports any fallback used for boundary nodes.
