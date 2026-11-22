# App1 mesh refinement study (post-initialization)

Generated 2026-10-05 17:36:46 EDT by `scripts/mesh_study_report.py` (automatic; no interpretation).

Experiment: `Applications/Application1/output/mesh_refinement_study`  
Commit: `f168b3fbb1bee92f1eab871f5e95efb75998160e`

## Protocol

- Two documented successful two-source discrete-sensor cases from `sensor_layout_comparison_20261003_151233`: **A_L1** (A_L1_boundary) and **B_L2** (B_L2_mixed). True sources, peak amplitudes, sensor coordinates (25 each) and equal weights W = I are kept; boundary sensors are point observations (no boundary-integral objective).
- Mesh levels original, refine_2x, refine_4x (target = ratio of the original TRIANGLE count; `original` = the committed mesh, unchanged), refined meshes built from the original boundary polylines (every original boundary vertex kept, same tags/names) with interior sensor points embedded; sensors mapped by coordinate (snap tolerance 1e-6 m).
- Navier-Stokes solved once per refined mesh (unmodified solver, same wind/nu/BCs; FEniCSx env on this host reproduces the committed original-mesh field to ~1e-14), reused for both cases.
- Noiseless data generated on each mesh by the same forward model, fixed during that inversion.
- Start control = saved post-initialization control (initializer output on the ORIGINAL mesh), identical on all meshes; the sensitivity initializer is NOT rerun. All costs are post-initialization.
- L-BFGS-B with GN-diag scaling recomputed on each mesh at the start control (D = sqrt(diag H_GN), floor max(D)/1e4), maxcor 30, ftol 1e-14, gtol 1e-9, maxls 20, max-iter 200, raw feasibility, amplitude floor 0.1; identical for every run.
- The original mesh is run with the same harness, settings and common dt as the refined meshes.

## Meshes

| mesh | nodes | triangles | tri. ratio | node ratio | h min / median / max (m) | min angle (deg) |
|---|---|---|---|---|---|---|
| original | 2795 | 4936 | 1.000 | 1.000 | 2.472 / 8.569 / 25.448 | 10.7 |
| refine_2x | 5639 | 9975 | 2.021 | 2.018 | 1.332 / 5.839 / 19.401 | 21.8 |
| refine_4x | 10638 | 19648 | 3.981 | 3.806 | 0.852 / 4.200 / 15.057 | 26.0 |

h = longest triangle edge. Shared per-mesh preparation (not part of any run's cost):

| mesh | mesh build (s) | Navier-Stokes (s) | Picard it. | velocity rel. L2 diff vs original field |
|---|---|---|---|---|
| original | n/a | 3.4 | 37 | 1.850e-14 |
| refine_2x | 1.2 | 4.8 | 35 | 3.510e-02 |
| refine_4x | 1.2 | 7.6 | 35 | 3.133e-02 |

## Time step

RK4 stability (ARPACK spectrum of M_L^-1 K with each mesh's velocity): common dt = **0.0125 s**, 12000 steps (default 0.1 s; changed: True). Rule: largest candidate with dt * rho(M_L^-1 K) <= 2.5 (90 % of the RK4 real-axis limit 2.785) and max |R_RK4(-dt*lambda)| <= 1 over the computed extreme eigenvalues, on ALL meshes of the study (one common dt).

| mesh | spectral radius | dt x rho at chosen dt | dt x rho at 0.1 s |
|---|---|---|---|
| original | 18.29 | 0.229 | 1.829 |
| refine_2x | 68.69 | 0.859 | 6.869 |
| refine_4x | 158.3 | 1.979 | 15.829 |

## Results (6 runs)

| run | status | iters | nfev/njev | post-init fwd/adj | GN fwd/adj | GN s | post-init s | J0 | J start | J final | J/J0 | max pos err (m) | max amp err | feasible | GN cond unscaled -> scaled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_L1__original | converged | 22 | 24/24 | 31/30 | 7/6 | 103 | 578 | 3.799e+05 | 1.638e-01 | 6.062e-18 | 1.60e-23 | 7.60e-10 | 5.00e-11 | True | 4.63e+06 -> 4.99e+02 |
| B_L2__original | converged | 23 | 24/24 | 31/30 | 7/6 | 116 | 591 | 1.089e+06 | 5.271e-01 | 5.970e-19 | 5.48e-25 | 1.92e-10 | 3.73e-12 | True | 3.27e+05 -> 3.32e+02 |
| A_L1__refine_2x | converged | 23 | 25/25 | 32/31 | 7/6 | 564 | 2746 | 3.529e+05 | 1.591e-01 | 5.990e-20 | 1.70e-25 | 5.51e-11 | 8.96e-13 | True | 4.52e+06 -> 4.92e+02 |
| B_L2__refine_2x | converged | 26 | 27/27 | 35/33 | 7/6 | 562 | 3018 | 8.841e+05 | 4.000e-01 | 9.666e-19 | 1.09e-24 | 1.59e-10 | 5.63e-12 | True | 3.27e+05 -> 2.10e+02 |
| A_L1__refine_4x | converged | 23 | 25/25 | 32/31 | 7/6 | 5830 | 28911 | 3.533e+05 | 1.570e-01 | 7.243e-20 | 2.05e-25 | 1.33e-10 | 7.23e-12 | True | 4.53e+06 -> 4.88e+02 |
| B_L2__refine_4x | converged | 21 | 24/24 | 32/30 | 7/6 | 5893 | 28385 | 8.398e+05 | 4.072e-01 | 6.865e-21 | 8.17e-27 | 2.61e-11 | 5.49e-13 | True | 3.21e+05 -> 2.43e+02 |

Stopping statuses:

- `A_L1__original`: exit 0, CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
- `B_L2__original`: exit 0, CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
- `A_L1__refine_2x`: exit 0, CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
- `B_L2__refine_2x`: exit 0, CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
- `A_L1__refine_4x`: exit 0, CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
- `B_L2__refine_4x`: exit 0, CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL

Solve counts are actual forward/adjoint PDE solves; nfev/njev are optimizer evaluations (some served from the evaluator cache). post-init = GN-diag construction + L-BFGS-B (+ the initial objective at z0). Per-run data generation is 1 forward solve (`data_generation_*` columns).

## Earlier original-mesh runs (historical, dt 0.1 s; not equivalent to the `original` level above)

| case | source | iters | J final | max pos err (m) | fwd/adj | differences |
|---|---|---|---|---|---|---|
| B_L2 | diagscale_m30 | 23 | 1.08e-18 | 2.52e-10 | 24/24 | standalone restart script (not the driver), max 200 iterations, other processes running (timings not comparable), dt 0.1 |
| A_L1 | P_A_L1_boundary__gn-diag | 29 | 3.20e-17 | 1.08e-08 | 39/38 | maxcor 10 (not 30), max-iter 150, ~50 runs concurrently (timings not comparable), dt 0.1 |
| B_L2 | P_B_L2_mixed__gn-diag | 31 | 2.86e-16 | 3.62e-09 | 40/39 | maxcor 10 (not 30), max-iter 150, ~50 runs concurrently, dt 0.1 |

Every original-mesh reference used dt = 0.1 s (1500 steps); the refined runs use the common dt above, so absolute costs are not comparable to them even apart from the listed differences.

## Files

- `report/data/mesh_study_runs.{csv,json}`, `report/data/mesh_study_baselines.json`
- `report/table_mesh_study.tex`
- `report/figures/fig_{iterations,pde_solves,runtime}_vs_mesh.{pdf,png}` (+ `report/data/fig_*.csv`)
- `report/figures/reconstruction_<run>.{pdf,png}`, `report/figures/convergence_<case>.{pdf,png}`
- per run: `cases/<run>/run_instrumentation.json`, `cases/<run>/full/` (driver outputs), `logs/<run>.log`
- shared: `meshes/`, `velocity/`, `stability/dt_selection.json`, `config/`, `env/`, `manifest.json`
