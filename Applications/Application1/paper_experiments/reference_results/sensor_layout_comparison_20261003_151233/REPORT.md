# Sensor-layout comparison: Application1 (DC Navy Yard)

Experiment directory: `Applications/Application1/output/sensor_layout_comparison_20261003_151233`  
Generated: 2026-10-03 16:53:38 EDT (automatic, no analysis)  
Commit: `40c9e9e8e896704d1f105ef204a198f546c0a585`  host: `<host>`

## Settings (identical for all six cases except sensor placement)

```text
--mode location_and_intensity --stage full --num-sources 2 --bound-case mesh_domain --init-mode sensitivity --num-starts 1 --random-seed 0 --sensitivity-score predicted_reduction --sensitivity-top-k 24 --sensitivity-beam-width 3 --sensitivity-refine-centers --sensitivity-refine-max-fev 60 --sensitivity-coherence-cap 0.3 --sensitivity-geometry-aware-branches --sensitivity-obstacle-aware-diversity --sensitivity-refine-pair --sensitivity-pair-refine-max-fev 60 --sensitivity-reselect-second --sensitivity-reselect-threshold 0.25 --sensitivity-reselect-max-passes 2 --sensitivity-save-plots --amplitude-solver osqp --post-init-method lbfgsb --feasibility-mode raw --tfinal 150 --amplitude-floor 0.1 --max-iter 150 --max-paraview-frames 100 --observation-mode nodes --sensor-weights equal --sensor-equal-weight 1.0 --velocity-field-file Applications/Application1/output/navier_stokes/application1_navier_stokes_velocity_nodes.npz
```
- source model: f = sum_i A_i exp(-|x-c_i|^2 / sigma2)  (repo convention: sigma2 == rho)
- rho = sigma2 (fixed, all stages; no continuation): 400 m^2
- diffusivity k: 10 m^2/s
- dt: 0.1 s
- tfinal: 150 s
- eta1: 100
- lambda_reg: 0
- objective time rule: exclude_first
- observation: discrete node sensors, 25 per layout, equal weights W = I (w = 1.0)
- noise: none (noiseless synthetic data, generated per layout by the same forward model)
- amplitude bounds: [0.1, 2.0] (amplitude floor 0.1)
- center bounds: mesh bounding box + fluid-mesh admissibility (raw feasibility mode)
- velocity: nodal Navier-Stokes field output/navier_stokes/application1_navier_stokes_velocity_nodes.npz
- dirichlet: homogeneous Dirichlet on 'left' (inflow) nodes only; no sensor there
- initialization: adjoint sensitivity beam (predicted_reduction, top-K 24, beam 3) + refinement, OSQP joint amplitude fit
- local optimizer: L-BFGS-B, max-iter 150, ftol 1e-14, gtol 1e-9, maxls 20 (driver constants)
- random seed: 0 (recorded; not consumed by the sensitivity path per docs)
- measurement-boundaries: not passed (only labels/plot context in nodes+equal mode)

## Source configurations

- **A**: c06 (manuscript discrete-sensor example): mid-domain wake + upper-right street — truth `[162.0, 300.0, 1.4, 270.0, 600.0, 0.95]` (final_sweep c06_nearwall_b13; node_sensors_right_bottom_top_c06)
- **B**: c07 (lower domain): wake between buildings 15/19 + south-east corner below building 21 — truth `[100.0, 210.0, 1.3, 270.0, 60.0, 0.85]` (final_sweep c07_lower_domain)

## Results

| case | exit | J after init | J final | iters | nfev/njev | stop | init fwd/adj | opt fwd/adj | init s | opt s | total s | init max pos err (m) | max pos err (m) | max amp err | terminal feasible | J0(both) | J0(S1) | J0(S2) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_L1_boundary | 0 | 1.639e-01 | 4.237e-17 | 77 | 89/89 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH | 482/2 | 89/89 | 500 | 352 | 866 | 0.134 | 3.62e-09 | 1.93e-10 | True | 3.800e+05 | 6.112e+03 | 3.739e+05 |
| A_L2_mixed | 0 | 1.837e+01 | 4.018e-17 | 145 | 173/173 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH | 427/5 | 173/173 | 435 | 645 | 1092 | 27.4 | 2.02e-08 | 1.48e-09 | True | 4.719e+05 | 4.507e+05 | 2.129e+04 |
| A_L3_weak | 0 | 4.454e-01 | 8.914e-08 | 140 | 175/175 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH | 508/5 | 175/175 | 524 | 750 | 1289 | 265 | 169 | 1.05 | True | 9.172e+00 | 9.172e+00 | 8.914e-08 |
| B_L1_boundary | 0 | 6.413e-01 | 1.032e-16 | 123 | 141/141 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH | 507/2 | 141/141 | 520 | 574 | 1108 | 20.5 | 1e-07 | 2.87e-09 | True | 5.623e+05 | 7.267e+01 | 5.622e+05 |
| B_L2_mixed | 0 | 5.274e-01 | 5.126e-20 | 60 | 71/71 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH | 395/3 | 71/71 | 402 | 304 | 721 | 0.313 | 1.64e-10 | 3.68e-12 | True | 1.090e+06 | 9.230e+05 | 1.671e+05 |
| B_L3_weak | 0 | 1.078e-06 | 4.045e-09 | 35 | 39/39 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH | 515/2 | 39/39 | 526 | 166 | 707 | 22.4 | 22.4 | 0.402 | True | 5.103e+02 | 5.103e+02 | 4.085e-06 |

Solve counts are actual PDE solves (forward/adjoint) counted at `gt.run_forward` / `gt.rk4_discrete_source_adjoint`; nfev/njev are optimizer evaluation counts (cached responses do not trigger solves). Synthetic data generation and driver overhead (forward smoke/export solves, final evaluation) are listed per case in `summary.json` under `pde_solves`. J0 = objective at the zero control for that layout (observation signal energy, eta1-weighted), per source from `observation_signals.json` (diagnostic solves, separate).

## Failures / budget-limited terminations

- none

## Figures

- overview: `overview_reconstructions.png`, `overview_layouts.png`
- A_L1_boundary: layout `Applications/Application1/output/sensor_layout_comparison_20261003_151233/cases/A_L1_boundary/layout_A_L1_boundary.png`, reconstruction `Applications/Application1/output/sensor_layout_comparison_20261003_151233/cases/A_L1_boundary/reconstruction_A_L1_boundary.png`, convergence `Applications/Application1/output/sensor_layout_comparison_20261003_151233/cases/A_L1_boundary/convergence_A_L1_boundary.png`, sensitivity-init 2 PNG(s) in `cases/A_L1_boundary/full/sensitivity_init/`
- A_L2_mixed: layout `Applications/Application1/output/sensor_layout_comparison_20261003_151233/cases/A_L2_mixed/layout_A_L2_mixed.png`, reconstruction `Applications/Application1/output/sensor_layout_comparison_20261003_151233/cases/A_L2_mixed/reconstruction_A_L2_mixed.png`, convergence `Applications/Application1/output/sensor_layout_comparison_20261003_151233/cases/A_L2_mixed/convergence_A_L2_mixed.png`, sensitivity-init 2 PNG(s) in `cases/A_L2_mixed/full/sensitivity_init/`
- A_L3_weak: layout `Applications/Application1/output/sensor_layout_comparison_20261003_151233/cases/A_L3_weak/layout_A_L3_weak.png`, reconstruction `Applications/Application1/output/sensor_layout_comparison_20261003_151233/cases/A_L3_weak/reconstruction_A_L3_weak.png`, convergence `Applications/Application1/output/sensor_layout_comparison_20261003_151233/cases/A_L3_weak/convergence_A_L3_weak.png`, sensitivity-init 2 PNG(s) in `cases/A_L3_weak/full/sensitivity_init/`
- B_L1_boundary: layout `Applications/Application1/output/sensor_layout_comparison_20261003_151233/cases/B_L1_boundary/layout_B_L1_boundary.png`, reconstruction `Applications/Application1/output/sensor_layout_comparison_20261003_151233/cases/B_L1_boundary/reconstruction_B_L1_boundary.png`, convergence `Applications/Application1/output/sensor_layout_comparison_20261003_151233/cases/B_L1_boundary/convergence_B_L1_boundary.png`, sensitivity-init 2 PNG(s) in `cases/B_L1_boundary/full/sensitivity_init/`
- B_L2_mixed: layout `Applications/Application1/output/sensor_layout_comparison_20261003_151233/cases/B_L2_mixed/layout_B_L2_mixed.png`, reconstruction `Applications/Application1/output/sensor_layout_comparison_20261003_151233/cases/B_L2_mixed/reconstruction_B_L2_mixed.png`, convergence `Applications/Application1/output/sensor_layout_comparison_20261003_151233/cases/B_L2_mixed/convergence_B_L2_mixed.png`, sensitivity-init 2 PNG(s) in `cases/B_L2_mixed/full/sensitivity_init/`
- B_L3_weak: layout `Applications/Application1/output/sensor_layout_comparison_20261003_151233/cases/B_L3_weak/layout_B_L3_weak.png`, reconstruction `Applications/Application1/output/sensor_layout_comparison_20261003_151233/cases/B_L3_weak/reconstruction_B_L3_weak.png`, convergence `Applications/Application1/output/sensor_layout_comparison_20261003_151233/cases/B_L3_weak/convergence_B_L3_weak.png`, sensitivity-init 2 PNG(s) in `cases/B_L3_weak/full/sensitivity_init/`

Per-case: `cases/<id>/case_result.json`, `recovery_report.txt`, `run_instrumentation.json`, `batch_status.json`, `run.log`, `full/` (driver outputs, ParaView states, `source_paths/`, histories).
