# Advection–Diffusion Source Inversion

Deterministic PDE-constrained optimization for **source identification in
transient advection–diffusion**: recover the centres and release rates of
several Gaussian sources from point-sensor concentration time series, in 2D
domains with impermeable obstacles and a precomputed steady Navier–Stokes flow.

**Method.** Linear (P1) triangular finite elements with a lumped mass matrix,
explicit RK4 time stepping, a fully discrete backward-in-time adjoint (exact
gradients of the discrete objective), an adjoint-sensitivity initializer (beam
search over source-insertion sensitivities with a joint bounded amplitude fit
solved by OSQP), and local refinement by L-BFGS-B with optional Gauss–Newton
diagonal scaling. Plain NumPy/SciPy; FEniCSx is needed only to build meshes and
solve for the velocity fields.

**Applications.**

* **Application1 — urban domain** (Washington DC, Navy Yard; building footprints
  as holes; wind field from steady Navier–Stokes). Sensor-layout,
  prescribed-width, five-source, random-start, mesh-dependence and gradient
  experiments. → [`Applications/Application1/paper_experiments/`](Applications/Application1/paper_experiments/README.md)
* **Application2 — three-bed hospital room** (idealized 2D plan with solid beds,
  partitions and a bathroom; ceiling supplies, natural door/exhaust outlets).
  Fine-mesh synthetic data, fixed 1 % measurement noise, coarse-mesh
  reconstruction of three two-source cases. → [`Applications/Application2/paper_experiments/`](Applications/Application2/paper_experiments/README.md)

## Repository layout

```text
Src/Solvers/            FE forward solver (RK4), discrete adjoint, advection assembly,
                        P1/quadrature data, node-sensor observation operator
Src/Objective/          gradient_test.py: shared reduced objective, forward wrapper and
                        adjoint gradient used by both applications (imported as `gt`)
Src/Optimizers/         SciPy L-BFGS-B wrapper
Applications/Application1/
  run_optimization.py   driver: data generation, initialization, optimization, reports
  sensitivity_initialization.py, source_insertion_sensitivity.py   initializer
  post_init_optimizers.py   L-BFGS-B / GN-diag scaling / Gauss-Newton options
  mesh_admissibility.py, final_feasibility.py, multi_source_utils.py, mesh_data_config.py
  DC_mesh/              committed mesh + GIS sources (build_dc_mesh.py)
  Navier Stokes Solve/  FEniCSx wind-field solve
  paper_experiments/    paper reference runs, random-start / gradient scripts, saved results
  run_*.sh, test scripts/   batch drivers and verification tests
Applications/Application2/
  room_geometry_config.py, navier_stokes_config.py, transport_config.py   all settings
  Room_mesh/, Navier Stokes Solve/   mesh and flow (FEniCSx)
  generate_observations.py   fine-mesh data + fixed noise;  run_inversion.py   reconstruction
  run_application2_*.sh      pipelines;  report_*.py, paper_figures_inversion_batch.py   post-processing
  paper_experiments/    paper settings and saved results
Applications/*/supplied/    meshes and velocity fields used for the paper (see below)
tools/                  environment checks, install_supplied_assets.py
docs/                   OSQP amplitude-solver validation note
```

All generated files go to `Applications/<app>/output/` (git-ignored). Comments that
mention "App4" refer to an earlier rectangular/L-shaped development application
from which Application1 was ported; it is not part of this repository.

## Installation

Two environments are used.

1. **Inverse environment** (all inversion, data generation and post-processing):

   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt          # or requirements-verified.txt for the exact versions
   python tools/check_inverse_env.py
   ```

   Verified: Python 3.13.5, numpy 2.5.2, scipy 1.18.1, matplotlib 3.11.1,
   meshio 5.3.5, osqp 1.1.3 (Linux x86_64).

2. **FEniCSx environment** (only to regenerate meshes and velocity fields):

   ```bash
   mamba env create -f environment-fenicsx.yml
   conda run -n fenicsx-env python tools/check_fenicsx_conda_env.py
   ```

   Verified: DOLFINx 0.11.0, gmsh 4.15.2, PETSc 3.25.6, MPICH 5.0.1, Python 3.14.6.

## Quick start (about one minute, inverse environment only)

```bash
python tools/install_supplied_assets.py                 # copies supplied meshes/fields into output/ (verifies SHA-256)
cd Applications/Application1
python run_optimization.py --mode location_and_intensity --stage smoke \
    --case-name app1_smoke --tfinal 6 --max-paraview-frames 5    # forward solves + exports, ~5 s
cd "test scripts"
python gradient_test_node_sensors.py --tfinal 10                # Taylor test, expect "rate R2" ≈ 2 and PASS, ~15 s
```

## Supplied assets and regeneration

`supplied_assets_manifest.json` lists every supplied file with its SHA-256 and
the SHA-256 of the file used in the paper runs; `tools/install_supplied_assets.py`
copies them to where the drivers look and never overwrites a different file.

| asset | supplied | regenerate (FEniCSx) |
|---|---|---|
| App1 mesh `DC_mesh/dc_navy_yard_mesh.msh` | committed | `python DC_mesh/build_dc_mesh.py` (needs the raw footprint download, see `DC_mesh/DATA_SOURCES.md`) |
| App1 wind field (nodal NPZ) | `Applications/Application1/supplied/` | `python "Applications/Application1/Navier Stokes Solve/solve_application1_navier_stokes.py"` |
| App1 refined meshes/fields (mesh study) | not supplied | `run_mesh_dependence_study.sh` builds them |
| App2 coarse + fine meshes and their velocity fields | `Applications/Application2/supplied/` | `FX=<fenicsx python> bash Applications/Application2/run_application2_setup.sh run` (refuses to overwrite an existing version folder; set `APP2_OUT=<dir>` to write elsewhere) |
| App2 observations (clean, noise, noisy) | not supplied | `python generate_observations.py …` (~2 min; reproduces the paper data bit for bit from the supplied fields) |

The App1 field supplied here is byte-identical to the one used in the paper.
The App2 files differ from the originals only in embedded path strings
(numerical arrays identical; hashes of both in the manifest). A regenerated mesh
or field (other gmsh/DOLFINx versions) may differ in the last digits; Application2
stores mesh hashes in each velocity file and refuses a field that does not belong
to its mesh, so meshes and fields cannot be mixed silently.

## Reproducing the paper

| experiment | guide section | main command(s) | saved results |
|---|---|---|---|
| App1 sensor layout | [App1 guide](Applications/Application1/paper_experiments/README.md#sensor-layout) | `paper_experiments/run_reference.py init_* / restart_*` | `…/reference_results/gn_diag_default_check_20261004_161718/` |
| App1 prescribed width | [App1 guide](Applications/Application1/paper_experiments/README.md#prescribed-width) | `run_reference.py width_*`, `run_width_study.sh` | `…/width_study_B_L2_20261003_222835/` |
| App1 five sources | [App1 guide](Applications/Application1/paper_experiments/README.md#five-source-reference) | `run_reference.py M5_S4_gndiag` | `…/post_init_scaling_validation_20261004_125952/` |
| App1 random starts (baseline, matched budget) | [App1 guide](Applications/Application1/paper_experiments/README.md#random-start-comparisons) | `paper_experiments/random_start_*/` | `…/random_start_*` |
| App1 mesh dependence | [App1 guide](Applications/Application1/paper_experiments/README.md#mesh-dependence) | `run_mesh_dependence_study.sh` | `…/mesh_refinement_study/` |
| App1 gradient checks | [App1 guide](Applications/Application1/paper_experiments/README.md#gradient-verification) | `paper_experiments/gradient_tests/run_paper_gradient_tests.py` | `…/paper_gradient_tests_20261112_174559/` |
| App2 three-bed room (clean and noisy) | [App2 guide](Applications/Application2/paper_experiments/README.md) | `generate_observations.py`, `run_inversion.py` / `run_application2_inversion_batch.sh` | `…/first_full_sequential_20261117/` |

Full experiments are expensive (minutes to hours per run; the App1 random-start
baseline is 50 runs). The guides give the recorded run times.

## Tests

Verification scripts run as plain Python (no test runner), from
`Applications/Application1/test scripts/`:

```bash
for t in test_*.py; do python "$t" || echo "FAILED: $t"; done    # 18 scripts, ~1 h sequentially
python test_boundary_mode_unchanged.py --tfinal 10    # objective/gradient bit-identical to the previous release
python gradient_test_navier_stokes_velocity.py        # Taylor test with the Navier-Stokes field
```

Measured at 8 BLAS threads: 16 scripts take 1 s to 2.5 min each; `test_second_source_reselection.py` (~19 min) and `test_sensitivity_refine_coherence.py` (~32 min) are the slow ones.

`test scripts/diagnostics/` holds one-off studies kept for provenance; several
need output folders from earlier runs. Application2 has its own checks:
`validate_readiness.py --dataset-dir …` and `run_inversion.py --stage wiring`.

## Data

The DC building footprints are derived from the Open Data DC *Building
Footprints* dataset (CC BY 4.0); see `Applications/Application1/DC_mesh/DATA_SOURCES.md`.
The room layout of Application2 is an idealized adaptation inspired by
Löhner et al., [arXiv:2006.13792](https://arxiv.org/abs/2006.13792) (see its README).

## License

The code is released under the MIT License (see `LICENSE`). Third-party data
keep their original license, as noted above.
