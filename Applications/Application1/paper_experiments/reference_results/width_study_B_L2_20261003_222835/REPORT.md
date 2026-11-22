# Fixed-width study: Application1 B_L2 (discrete sensors)

Experiment directory: `Applications/Application1/output/width_study_B_L2_20261003_222835`  
Generated: 2026-10-03 23:40:22 EDT (automatic, no analysis)  
Commit: `cf1ec1dc284b4fc4ed28f04f1bf57b8cfcce424e`  host: `<host>`

- Application1 fixed-width study (B_L2, discrete sensors): five independent reconstructions at rho in [400.0, 300.0, 200.0, 100.0, 50.0] m^2, FIXED PEAK AMPLITUDES
- width study at fixed PEAK amplitudes: the true amplitudes 1.30 / 0.85 are the Gaussian peak values at every width and are not rescaled to preserve total source strength (the integrated strength A*pi*rho scales with rho)
- independence: each width is a separate process: synthetic data generated at that width, full sensitivity-beam initialization from scratch, then L-BFGS-B; no warm start from another width, no truth initialization, no --sigma2-continuation
- width parameter: rho of psi_c(x) = exp(-|x-c|^2/rho) is run_optimization.PDE['sigma2'] (NOT a noise variance; data are noiseless). Set per run by test scripts/width_study_run_case.py --rho before main(); driver default 400.0 m^2
- truth: `[100.0, 210.0, 1.3, 270.0, 60.0, 0.85]`; 25 sensors (B_L2 mixed layout), W = I
- local optimizer: L-BFGS-B (--post-init-method lbfgsb), 6 unknowns jointly, max 500 iterations, {'ftol': 1e-14, 'gtol': 1e-09, 'maxls': 20}; scipy default maxfun 15000 (driver passes none) > 500 x (20+1) = 10500, so the iteration limit is the binding budget; identical for all widths
- initialization: adjoint sensitivity beam (predicted_reduction, top-K 24, beam width 3), center refinement (max-fev 60), pair refinement (max-fev 60), second-source reselection (threshold 0.25, max 2 passes, refine max-fev 0), coherence cap 0.3, geometry-aware branches, obstacle-aware diversity, OSQP joint amplitude fit; --sensitivity-min-separation and --min-source-separation unset (None); runs at the case width (no continuation)

Common driver arguments:
```text
--mode location_and_intensity --stage full --num-sources 2 --bound-case mesh_domain --init-mode sensitivity --num-starts 1 --random-seed 0 --sensitivity-score predicted_reduction --sensitivity-top-k 24 --sensitivity-beam-width 3 --sensitivity-refine-centers --sensitivity-refine-max-fev 60 --sensitivity-coherence-cap 0.3 --sensitivity-geometry-aware-branches --sensitivity-obstacle-aware-diversity --sensitivity-refine-pair --sensitivity-pair-refine-max-fev 60 --sensitivity-reselect-second --sensitivity-reselect-threshold 0.25 --sensitivity-reselect-max-passes 2 --sensitivity-save-plots --amplitude-solver osqp --post-init-method lbfgsb --feasibility-mode raw --tfinal 150 --amplitude-floor 0.1 --max-iter 500 --max-paraview-frames 100 --observation-mode nodes --sensor-weights equal --sensor-equal-weight 1.0 --velocity-field-file Applications/Application1/output/navier_stokes/application1_navier_stokes_velocity_nodes.npz
```

- diffusivity PDE['k']: 10.0 m^2/s
- dt PDE['dt']: 0.1 s
- tfinal (--tfinal): 150 s
- eta1: 100.0
- lambda_reg: 0.0
- objective time rule: exclude_first
- gradient time rule: trapezoid
- J0: J at the zero control with the driver objective = 1/2 eta1 ||d||_h^2 (sensitivity_init_summary zero_source_objective); unweighted 1/2||d||_h^2 = J0/eta1
- noise: none
- amplitude bounds: [0.1, 2.0] (--amplitude-floor 0.1)
- center bounds: mesh bounding box + fluid-mesh admissibility (--feasibility-mode raw)
- velocity: Applications/Application1/output/navier_stokes/application1_navier_stokes_velocity_nodes.npz
- mesh: Applications/Application1 DC mesh (build_mesh_data), unchanged

## Results

| case | rho (m²) | exit | width ok | J0 | J after init | J final | J final/J0 | iters | nfev/njev | stop | data fwd | init fwd/adj | opt fwd/adj | init s | opt s | total s | init max pos err (m) | pos errs (m) | amp errs | max pos err (m) | max amp err | feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rho400 | 400 | 0 | True | 1.090e+06 | 5.274e-01 | 5.126e-20 | 4.702e-26 | 60 | 71/71 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH | 1 | 395/3 | 71/71 | 404 | 296 | 714 | 0.313 | 1.33e-11, 1.64e-10 | 1.33e-13, 3.68e-12 | 1.64e-10 | 3.68e-12 | True |
| rho300 | 300 | 0 | True | 6.967e+05 | 3.188e-01 | 2.858e-15 | 4.103e-21 | 63 | 71/71 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH | 1 | 519/3 | 71/71 | 562 | 317 | 894 | 0.19 | 7.59e-09, 3.26e-08 | 5.49e-11, 8.75e-10 | 3.26e-08 | 8.75e-10 | True |
| rho200 | 200 | 0 | True | 3.718e+05 | 4.309e-01 | 3.136e-16 | 8.435e-22 | 60 | 67/67 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH | 1 | 524/3 | 67/67 | 593 | 299 | 908 | 0.438 | 5.5e-09, 2.57e-09 | 1.39e-10, 1.06e-10 | 5.5e-09 | 1.39e-10 | True |
| rho100 | 100 | 0 | True | 1.223e+05 | 5.171e-01 | 4.933e-17 | 4.032e-22 | 65 | 73/73 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH | 1 | 505/2 | 73/73 | 567 | 337 | 920 | 0.863 | 2.93e-09, 6.3e-09 | 1.07e-11, 2.7e-10 | 6.3e-09 | 2.7e-10 | True |
| rho050 | 50 | 0 | True | 3.648e+04 | 2.866e-02 | 4.600e-16 | 1.261e-20 | 52 | 59/59 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH | 1 | 493/2 | 59/59 | 539 | 278 | 833 | 0.21 | 6.05e-09, 5.5e-08 | 4.78e-11, 3.31e-09 | 5.5e-08 | 3.31e-09 | True |

### Controls

| case | initialized control | final control |
|---|---|---|
| rho400 | [100.0883, 209.9508, 1.2986, 269.6868, 60.0110, 0.8469] | [100.0000, 210.0000, 1.3000, 270.0000, 60.0000, 0.8500] |
| rho300 | [99.9086, 209.9603, 1.3022, 269.9338, 60.1782, 0.8411] | [100.0000, 210.0000, 1.3000, 270.0000, 60.0000, 0.8500] |
| rho200 | [100.1994, 210.0497, 1.2962, 269.6526, 60.2665, 0.8331] | [100.0000, 210.0000, 1.3000, 270.0000, 60.0000, 0.8500] |
| rho100 | [100.1508, 209.9250, 1.2969, 270.0631, 60.8608, 0.8099] | [100.0000, 210.0000, 1.3000, 270.0000, 60.0000, 0.8500] |
| rho050 | [99.8184, 209.9695, 1.2986, 270.1068, 60.1809, 0.8418] | [100.0000, 210.0000, 1.3000, 270.0000, 60.0000, 0.8500] |

### Width-dependent initializer rules

- rho400: {'e_folding_radius_sqrt_rho_m': 20.0, 'screen_radius_m = max(min_separation or 0, 2 sqrt(rho)); min_separation=None': 40.0, 'branch_dedup_radius_m = max(2, 0.25 sqrt(rho))': 5.0, 'reselect_center_shift_trigger_m = threshold 0.25 * sqrt(rho)': 5.0}
- rho300: {'e_folding_radius_sqrt_rho_m': 17.320508075688775, 'screen_radius_m = max(min_separation or 0, 2 sqrt(rho)); min_separation=None': 34.64101615137755, 'branch_dedup_radius_m = max(2, 0.25 sqrt(rho))': 4.330127018922194, 'reselect_center_shift_trigger_m = threshold 0.25 * sqrt(rho)': 4.330127018922194}
- rho200: {'e_folding_radius_sqrt_rho_m': 14.142135623730951, 'screen_radius_m = max(min_separation or 0, 2 sqrt(rho)); min_separation=None': 28.284271247461902, 'branch_dedup_radius_m = max(2, 0.25 sqrt(rho))': 3.5355339059327378, 'reselect_center_shift_trigger_m = threshold 0.25 * sqrt(rho)': 3.5355339059327378}
- rho100: {'e_folding_radius_sqrt_rho_m': 10.0, 'screen_radius_m = max(min_separation or 0, 2 sqrt(rho)); min_separation=None': 20.0, 'branch_dedup_radius_m = max(2, 0.25 sqrt(rho))': 2.5, 'reselect_center_shift_trigger_m = threshold 0.25 * sqrt(rho)': 2.5}
- rho050: {'e_folding_radius_sqrt_rho_m': 7.0710678118654755, 'screen_radius_m = max(min_separation or 0, 2 sqrt(rho)); min_separation=None': 14.142135623730951, 'branch_dedup_radius_m = max(2, 0.25 sqrt(rho))': 2.0, 'reselect_center_shift_trigger_m = threshold 0.25 * sqrt(rho)': 1.7677669529663689}

Solve counts are actual PDE solves counted at `gt.run_forward` / `gt.rk4_discrete_source_adjoint`; nfev/njev are L-BFGS-B evaluation counts (cached evaluator responses do not trigger solves). 'data fwd' = forward solves before the initializer (synthetic data at that width); driver overhead solves (smoke/export, final evaluation) are under `pde_solves` in `summary.json`. J0 = zero-source objective at that width (driver objective, eta1-weighted); J0/eta1 = 1/2||d||_h^2 is in the CSV/JSON. `n/a` = unavailable (e.g. the run failed before that stage).

## Failures / budget-limited terminations

- none

## Figures

- overview: `overview_reconstructions.png`
- rho400: reconstruction `Applications/Application1/output/width_study_B_L2_20261003_222835/cases/rho400/reconstruction_rho400.png`, convergence `Applications/Application1/output/width_study_B_L2_20261003_222835/cases/rho400/convergence_rho400.png`, layout `Applications/Application1/output/width_study_B_L2_20261003_222835/cases/rho400/layout_rho400.png`, sensitivity-init 2 PNG(s) in `cases/rho400/full/sensitivity_init/`
- rho300: reconstruction `Applications/Application1/output/width_study_B_L2_20261003_222835/cases/rho300/reconstruction_rho300.png`, convergence `Applications/Application1/output/width_study_B_L2_20261003_222835/cases/rho300/convergence_rho300.png`, layout `Applications/Application1/output/width_study_B_L2_20261003_222835/cases/rho300/layout_rho300.png`, sensitivity-init 2 PNG(s) in `cases/rho300/full/sensitivity_init/`
- rho200: reconstruction `Applications/Application1/output/width_study_B_L2_20261003_222835/cases/rho200/reconstruction_rho200.png`, convergence `Applications/Application1/output/width_study_B_L2_20261003_222835/cases/rho200/convergence_rho200.png`, layout `Applications/Application1/output/width_study_B_L2_20261003_222835/cases/rho200/layout_rho200.png`, sensitivity-init 2 PNG(s) in `cases/rho200/full/sensitivity_init/`
- rho100: reconstruction `Applications/Application1/output/width_study_B_L2_20261003_222835/cases/rho100/reconstruction_rho100.png`, convergence `Applications/Application1/output/width_study_B_L2_20261003_222835/cases/rho100/convergence_rho100.png`, layout `Applications/Application1/output/width_study_B_L2_20261003_222835/cases/rho100/layout_rho100.png`, sensitivity-init 2 PNG(s) in `cases/rho100/full/sensitivity_init/`
- rho050: reconstruction `Applications/Application1/output/width_study_B_L2_20261003_222835/cases/rho050/reconstruction_rho050.png`, convergence `Applications/Application1/output/width_study_B_L2_20261003_222835/cases/rho050/convergence_rho050.png`, layout `Applications/Application1/output/width_study_B_L2_20261003_222835/cases/rho050/layout_rho050.png`, sensitivity-init 2 PNG(s) in `cases/rho050/full/sensitivity_init/`

Per case: `cases/<id>/case_result.json`, `width_check.json`, `recovery_report.txt`, `run_instrumentation.json`, `resolved_driver_args.json`, `batch_status.json`, `run.log`, `full/` (driver outputs, histories, `source_paths/`, ParaView states).
