# Advection-Diffusion PDECO

Deterministic PDE-constrained optimization for **source identification in
transient advection–diffusion**: recover the locations and intensities of
several Gaussian sources from concentration measurements, on 2D domains ranging
from a rectangle to a real city layout (Washington DC, Navy Yard) with building
footprints as impermeable obstacles and a Navier–Stokes wind field.

The implementation is plain NumPy/SciPy: linear triangular finite elements, an
RK4 forward solver, a fully discrete backward-in-time adjoint for exact
gradients of the reduced objective, and L-BFGS-B. The DC application adds an
adjoint-sensitivity initializer (beam search over insertion sensitivities with
a joint bounded amplitude fit), sigma-continuation, obstacle-aware feasibility,
boundary or point-sensor observations, and adaptive inference of the number of
sources.

## Repository layout

```text
Src/
  Solvers/                     geometry-agnostic FE forward/adjoint solvers
    adv_diff_forward.py          transient forward solver
    adv_diffusion_adjoint.py     backward-in-time adjoint solver
    advection_assembly.py        advection operator assembly
    linear_fe_data.py            P1 triangle data
    quadrature_data.py           triangle / interval quadrature
    observation_operator.py      point (mesh-node) sensor observations
    pde_params.py                PDE parameter helpers
  Optimizers/optimizer.py      SciPy L-BFGS-B wrapper

Applications/
  Test Application/            forward / adjoint / gradient checks on a rectangle
  Test Application2/           two-source identification on a rectangle (reference)
  Test Application3/           rectangle with a circular obstacle
  Test Application4/           L-shaped city domain with building holes + NS velocity
  Application1/                DC Navy Yard domain (main application; see its README)

tools/                         environment checks and Hessian diagnostics
docs/                          method notes (OSQP amplitude-solver validation)
```

Each application folder is self-contained: a mesh, a `mesh_data_config.py`
loader, the source model, and a `run_optimization.py` command-line driver.

## Installation

Two Python environments are used, because DOLFINx is only needed to generate
velocity fields:

1. **Inverse solver** (all optimization drivers):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   python tools/check_inverse_env.py
   ```

2. **FEniCSx** (only for the Navier–Stokes velocity solves in
   `Applications/*/Navier Stokes Solve/` and for rebuilding the DC mesh):

   ```bash
   mamba env create -f environment-fenicsx.yml
   conda run -n fenicsx-env python tools/check_fenicsx_conda_env.py
   ```

## Quick start (Application1, DC Navy Yard)

Velocity fields are generated, not committed. From the repository root:

```bash
# 1. Steady Navier–Stokes wind field (FEniCSx environment), written to
#    Applications/Application1/output/navier_stokes/
conda run -n fenicsx-env python \
    "Applications/Application1/Navier Stokes Solve/solve_application1_navier_stokes.py"

# 2. Forward-solve smoke test (inverse environment)
cd Applications/Application1
python run_optimization.py --mode location_and_intensity --stage smoke \
    --case-name app1_smoke --tfinal 6 --max-paraview-frames 5

# 3. Full source identification
python run_optimization.py --help
```

All outputs (ParaView `.vtu`/`.pvd` sequences, optimization histories, JSON
summaries) go to `Applications/<app>/output/`, which is git-ignored.
`Applications/Application1/README.md` documents every command-line option, the
exact commands for gradient/Hessian Taylor tests, and the experiment drivers.

## Tests

Regression and verification scripts live in
`Applications/Application1/test scripts/` and run as plain Python scripts (no
test runner required), e.g.

```bash
cd "Applications/Application1/test scripts"
python test_sensitivity_scoring.py
python test_amplitude_solver_osqp.py
python gradient_test_navier_stokes_velocity.py   # Taylor test, expect slope ≈ 2
```

The scripts in `test scripts/diagnostics/` are one-off studies and report
generators kept for reproducibility; several expect output directories produced
by earlier runs.

## Data

The DC building footprints are derived from the Open Data DC *Building
Footprints* dataset (CC BY 4.0). See
`Applications/Application1/DC_mesh/DATA_SOURCES.md`.

## License

The code is released under the MIT License (see `LICENSE`). Third-party data
keeps its original license, as noted above.
