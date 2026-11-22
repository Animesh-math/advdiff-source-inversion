# Five-source reconstruction (M = 5): Application1, L1_plus_S4 sensors

Experiment directory: `Applications/Application1/output/five_source_M5_S4sensors_20261003_204754`  
Generated: 2026-10-03 22:29:17 EDT (automatic, no analysis)  
Commit: `cf1ec1dc284b4fc4ed28f04f1bf57b8cfcce424e`  host: `<host>`  batch state: `finished`

## Status

- exit status: **0**
- stopping reason: `STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT` (success flag False)
- budget-limited: True  (max iterations 500, objective-evaluation budget maxfun 15000)
- terminal feasibility (all centers fluid-admissible): True

## Objectives and counts

| quantity | value |
|---|---|
| J after initialization | 8.434e+02 |
| J final | 3.471e+01 |
| final gradient norm | 3.801e+02 |
| L-BFGS-B iterations | 500 |
| objective evaluations (nfev) | 560 |
| gradient evaluations (njev) | 560 |
| PDE solves, synthetic data + setup (fwd/adj) | 1/0 |
| PDE solves, initialization (fwd/adj) | 3069/12 |
| PDE solves, local optimization (fwd/adj) | 560/560 |
| PDE solves, driver overhead: smoke/export/final eval (fwd/adj) | 4/2 |
| initialization time (s) | 3128 |
| local optimization time (s) | 2918 |
| total wall time (s) | 6064 |

PDE solves are actual `gt.run_forward` / `gt.rk4_discrete_source_adjoint` calls; nfev/njev count optimizer evaluator calls, some of which are served from the evaluator cache without a solve (so opt solves can be < nfev + njev).

## Sources (permutation-matched)

| true S | truth (x, y, A) | initialized | final | init pos err (m) | final pos err (m) | final amp err |
|---|---|---|---|---|---|---|
| S1 | (110.00, 660.00, 0.9000) | (117.84, 661.36, 0.7992) | (109.53, 659.75, 0.9141) | 7.95 | 0.533 | 0.0141 |
| S2 | (270.00, 600.00, 1.3500) | (267.11, 601.73, 1.4343) | (270.27, 600.00, 1.3471) | 3.37 | 0.274 | 0.00286 |
| S3 | (200.00, 425.00, 0.7000) | (189.59, 426.93, 0.7165) | (192.86, 424.51, 0.7086) | 10.6 | 7.16 | 0.00858 |
| S4 | (100.00, 210.00, 1.6000) | (100.00, 209.40, 1.6104) | (99.98, 209.99, 1.6006) | 0.599 | 0.0271 | 0.000575 |
| S5 | (275.00, 120.00, 1.1000) | (275.30, 120.08, 1.0929) | (275.02, 119.99, 1.0996) | 0.31 | 0.0223 | 0.000435 |

max position error 7.16 m, max amplitude error 0.0141, init max position error 10.6 m.

Observation signal per source (J at zero control with only that source on, from `observation_signals.json`, diagnostic solves outside the run): S1=7.818e+04, S2=7.551e+05, S3=4.734e+04, S4=1.399e+07, S5=2.788e+06

## Settings

```text
--mode location_and_intensity --stage full --num-sources 5 --bound-case mesh_domain --init-mode sensitivity --num-starts 1 --random-seed 0 --sensitivity-score predicted_reduction --sensitivity-top-k 24 --sensitivity-beam-width 3 --sensitivity-refine-centers --sensitivity-refine-max-fev 60 --sensitivity-coherence-cap 0.3 --sensitivity-geometry-aware-branches --sensitivity-obstacle-aware-diversity --sensitivity-refine-pair --sensitivity-pair-refine-max-fev 60 --sensitivity-reselect-second --sensitivity-reselect-threshold 0.25 --sensitivity-reselect-max-passes 2 --sensitivity-save-plots --amplitude-solver osqp --post-init-method lbfgsb --feasibility-mode raw --tfinal 150 --amplitude-floor 0.1 --max-iter 500 --max-paraview-frames 100 --observation-mode nodes --sensor-weights equal --sensor-equal-weight 1.0 --velocity-field-file Applications/Application1/output/navier_stokes/application1_navier_stokes_velocity_nodes.npz
```
- source model: f = sum_i A_i exp(-|x-c_i|^2 / sigma2); code name PDE['sigma2'] is the width parameter rho (run_optimization.PDE)
- rho = PDE['sigma2'] (fixed; --sigma2-continuation NOT passed): 400.0 m^2
- diffusivity PDE['k']: 10.0 m^2/s
- dt PDE['dt']: 0.1 s
- tfinal (--tfinal): 150 s
- eta1: 100.0
- lambda_reg: 0.0
- objective time rule: exclude_first
- gradient time rule: trapezoid
- noise: none; synthetic observations generated ONCE from the true control by the driver (same forward model), never regenerated
- amplitude bounds: [0.1, 2.0] (--amplitude-floor 0.1)
- center bounds: mesh bounding box + fluid-mesh admissibility (--feasibility-mode raw)
- velocity: Applications/Application1/output/navier_stokes/application1_navier_stokes_velocity_nodes.npz
- dirichlet: homogeneous Dirichlet on 'left' (inflow) nodes only; no sensor there
- local optimizer: {"method": "L-BFGS-B (--post-init-method lbfgsb), all 15 unknowns jointly", "max_iterations": 500, "options": {"ftol": 1e-14, "gtol": 1e-09, "maxls": 20, "tol": 1e-08}, "objective_evaluation_budget_maxfun": 15000, "budget_note": "maxfun is scipy's default 15000 (the driver passes no maxfun); worst case at maxls=20 is about 500 x (20 + 1) = 10500 evaluations < 15000, so the iteration limit is the binding budget", "stopping_tolerances": "production driver constants (unchanged)"}
- sensors: 31 nodes, layout `L1_plus_S4` (boundary-only: 25 outer-boundary nodes (right 13, bottom 6, top 6), the existing node_sensors_right_bottom_top_c06 set + 6 interior fluid nodes along the S4 plume (wake between buildings 15/16/12/13)), ids [641, 644, 647, 650, 653, 656, 659, 662, 665, 668, 671, 674, 677, 582, 585, 588, 591, 594, 596, 693, 690, 687, 684, 681, 679, 928, 1506, 2478, 1438, 2680, 2244] (coordinates in `experiment_config.json`)

## Files

- reconstruction map: `Applications/Application1/output/five_source_M5_S4sensors_20261003_204754/cases/M5_L1_plus_S4/reconstruction_M5_L1_plus_S4.png`
- convergence history: `Applications/Application1/output/five_source_M5_S4sensors_20261003_204754/cases/M5_L1_plus_S4/convergence_M5_L1_plus_S4.png`
- layout: `Applications/Application1/output/five_source_M5_S4sensors_20261003_204754/cases/M5_L1_plus_S4/layout_M5_L1_plus_S4.png`; sensitivity-init plots: 5 in `cases/M5_L1_plus_S4/full/sensitivity_init/`
- per-case: `cases/M5_L1_plus_S4/case_result.json`, `recovery_report.txt`, `run_instrumentation.json`, `run.log`, `full/` (optimization_history.csv, source_paths/, optimization_summary.json, sensitivity_init_summary.json, ParaView states)
