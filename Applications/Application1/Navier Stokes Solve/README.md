# Application1 Navier-Stokes Velocity Solve (DC Navy Yard)

Standalone FEniCSx steady incompressible Navier-Stokes solve that produces a
geometry-aware ambient-wind velocity field on the real DC Navy Yard mesh
(Taylor-Hood P2/P1, Picard fixed-point iteration, PETSc MUMPS LU). It does not touch the inverse/optimization code.

## PDE

Incompressible **kinematic** form (no body force):

```text
(v . grad) v - nu * Laplace(v) + grad(p) = 0
div(v) = 0
```

The pressure is kinematic (p/rho); density does not enter the equations.

## Single source of truth: the wind

Everything is driven by one vector in `../navier_stokes_config.py`:

```python
WIND_VELOCITY = (2.07, -0.36)   # m/s, +x = east, +y = north
```

To run a different wind, edit **only** this vector and rerun. `NU`, `RHO`,
`FLUX_TOLERANCE`, and the Picard controls also live in that config module.

The default `(2.07, -0.36)` is a measured wind of ~4.1 knots FROM 280 degrees
true, expressed as the Cartesian velocity the air moves toward.

## Boundary conditions (dynamic, Option 2)

The four outer rectangle edges carry these VERIFIED cardinal tags (from
`DC_mesh/build_dc_mesh.py`, confirmed geometrically at runtime):

```text
tag 11  bottom edge (y = y_min)  SOUTH   outward normal (0, -1)
tag 12  top    edge (y = y_max)  NORTH   outward normal (0, +1)
tag 13  left   edge (x = x_min)  WEST    outward normal (-1, 0)
tag 14  right  edge (x = x_max)  EAST    outward normal (+1, 0)
```

Each edge is classified from `flux = WIND_VELOCITY . outward_normal`:

```text
flux < -tol  -> inflow      Dirichlet  u = WIND_VELOCITY   (uniform far field)
flux >  tol  -> outflow     pressure outlet  p = 0         (do-nothing)
|flux| <= tol -> tangential  natural / do-nothing (slip)
```

For the default wind this gives **west + north = inflow**, **east + south =
outflow**. Changing the wind reclassifies automatically; no solver or mesh edit
is needed. A zero wind, or a wind with no inflow or no outflow boundary, raises
an error instead of running.

All 21 building walls (tags 101..121) are no-slip `u = (0, 0)`. The uniform wind
is never applied on building boundaries, so air does not pass through buildings.

## Numerical method

Taylor-Hood elements (velocity P2 vector, pressure P1). The nonlinear term is
handled by Picard iteration: the advecting velocity `w` is frozen from the
previous iterate (initial guess = uniform wind), and each linear Oseen
saddle-point system is solved with PETSc MUMPS LU. The `p = 0` outflow condition
also fixes the pressure reference.

`NU` is an effective/eddy kinematic viscosity chosen for a stable laminar solve
on this coarse urban mesh, **not** the molecular viscosity of air (which would
diverge here). It is exposed in the config and printed at runtime.

## How to run

From the repository root, with the FEniCSx conda environment (`fenicsx-env`, see
`environment-fenicsx.yml`; this imports DOLFINx, so do NOT use the inverse
environment):

```text
conda run -n fenicsx-env python \
    "Applications/Application1/Navier Stokes Solve/solve_application1_navier_stokes.py"
```

Optional overrides: `--nu`, `--max-picard-iter`, `--picard-tol` (defaults come
from the config module).

## Outputs

All written under `../output/navier_stokes/`, overwritten each run:

```text
application1_navier_stokes_velocity_nodes.npz   primary nodal velocity (see schema)
application1_navier_stokes_velocity.bp          ParaView velocity (ADIOS2/VTX)
application1_navier_stokes_pressure.bp          ParaView pressure  (ADIOS2/VTX)
application1_navier_stokes_run_metadata.json    full run metadata
application1_convergence_history.csv/.png       Picard convergence history
application1_velocity_magnitude.png             velocity magnitude + quiver
application1_streamlines.png                     streamlines
application1_boundary_conditions.png            BC classification preview
```

The NPZ uses the shared schema (`coords`, `velocity_nodes`) so the advection-diffusion
velocity loader can consume it without a new format:

```text
coords          float64 (num_nodes, 3)   Application1 transport node coordinates
velocity_nodes  float64 (num_nodes, 2)   [vx, vy] at coords[i], same node order
```

`coords` equals `mesh_data_config.build_mesh_data()["coord"]`, so
`velocity_nodes[i]` is the Navier-Stokes velocity at transport node `i`.
