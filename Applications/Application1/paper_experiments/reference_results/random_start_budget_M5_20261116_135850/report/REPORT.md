# Five-source random-start benchmark at matched PDE-sweep budget

Generated 2026-11-16 16:03:58. 10/10 starts finished.

## Setup
- Reference (ours): `Applications/Application1/output/post_init_scaling_validation_20261004_125952/cases/M5_S4_gndiag`. paper five-source result: M = 5, 31 node sensors (L1 boundary 25 + S4 interior 6), sensitivity initialization + L-BFGS-B with --post-init-scaling gn-diag; converged. NOT the earlier stalled unscaled run five_source_M5_S4sensors_20261003_204754.
- Not used: the stalled unscaled run `Applications/Application1/output/five_source_M5_S4sensors_20261003_204754/cases/M5_L1_plus_S4/full/optimization_summary.json` (STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT, J = 34.7).
- Budget unit: PDE sweep: one complete forward solve OR one complete adjoint solve = 1 unit. Reference counts from `Applications/Application1/output/post_init_scaling_validation_20261004_125952/cases/M5_S4_gndiag/run_instrumentation.json (actual gt.run_forward / gt.rk4_discrete_source_adjoint calls)`:
  initialization 3069 fwd + 12 adj; local refinement 84 fwd + 82 adj (= 1 fwd objective at the initialized control + GN-diag 16 fwd/15 adj + L-BFGS-B 67 fwd/67 adj). **Total 3153 fwd + 94 adj = 3247 sweeps.**
- the paper's 3,069 forward sweeps are the initialization forward count only; the initialization also used 12 adjoint sweeps and local refinement 84 fwd + 82 adj.
- Allocation: floor(3247/10) = 324 each; remainder 7 -> +1 sweep to r01..r07 (predetermined run order). No reallocation of unused allowance.
- Enforcement: PDE-evaluation level inside the driver's budgeted block (objective at the start, GN-diag construction, L-BFGS-B): before every forward or adjoint solve the guard checks the remaining allowance; during L-BFGS-B a new objective evaluation (forward) is only started when at least 2 sweeps remain, so that its gradient (adjoint) can also be completed. Otherwise the run stops cleanly (no solve started) and returns the best feasible fully evaluated point (objective AND gradient computed, all source centers in the fluid mesh). The GN-diag construction is only started if its full cost plus one evaluation fits.
- Excluded for both methods: data generation (1 fwd); driver reporting overhead (reference 4 fwd/2 adj; per start in the table).
- Starts: mesh_admissibility.MeshAdmissibility.sample_control_in_domain(num_sources=5, amplitude_bounds, rng=np.random.default_rng(seed), min_separation=None) (the sampler behind --init-mode mesh_random); centers independent, uniform with respect to area over the meshed fluid region (triangle chosen with probability proportional to area, then uniform in the triangle); building holes have zero probability; point-in-mesh check guards round-off; amplitudes independent uniform on [0.1, 2.0] = production L-BFGS-B amplitude box; seed = 2026111600 + r, r = 1..10 (fresh seeds, not those of random_start_baseline_20261114_224229). Truth never used.
- Optimizer per start: same driver path as the reference (GN-diag computed at each start and charged to that start, L-BFGS-B maxcor 10, ftol 1e-14, gtol 1e-9, maxls 20, raw feasibility, amplitude box [0.1, 2]); no sensitivity initialization, beam search or center refinement.
- J0 = objective of the zero source against the same synthetic data = 1.775e+07.

## Comparison
| method | fwd | adj | sweeps | iterations | objective | J/J0 | max center err (m) | max amp err | feasible | stop |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours (init + refinement) | 3153 | 94 | 3247 | 62 | 7.772e-15 | 4.379e-22 | 1.916e-08 | 1.945e-10 | True | converged |
| Best random start r05 -- cost of ALL 10 starts | 1629 | 1612 | 3241 | 115 (all: 1114) | 7.16e+04 | 0.00404 | 314 | 1 | True | budget_exhausted |

Ours split: initialization 3069 fwd / 12 adj; local refinement 84 fwd / 82 adj. Random starts: GN-diag setup 160 fwd / 150 adj in total; unused allowance 6 sweeps (not reallocated). Best start's own cost 163 fwd / 161 adj.

The best random start is selected by the lowest objective only; the center/amplitude errors are computed afterwards with the permutation that minimizes total center error (`run_optimization.match_sources_by_permutation`).

## Per start
| run | seed | allow | used fwd/adj | unused | GN fwd/adj | iters | evals | stop | J start | best J | J/J0 | feasible | max ctr err (m) | max amp err | overhead fwd/adj |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r01 | 2026111601 | 325 | 163/162 | 0 | 16/15 | 95 | 147 | budget_exhausted | 1.741e+07 | 2.793e+06 | 0.157 | True | 195 | 0.673 | 4/2 |
| r02 | 2026111602 | 325 | 163/161 | 1 | 16/15 | 122 | 146 | budget_exhausted | 5.586e+06 | 2.140e+06 | 0.121 | True | 291 | 1.25 | 5/3 |
| r03 | 2026111603 | 325 | 163/161 | 1 | 16/15 | 117 | 146 | budget_exhausted | 2.487e+07 | 1.678e+07 | 0.946 | True | 325 | 1.5 | 5/3 |
| r04 | 2026111604 | 325 | 163/161 | 1 | 16/15 | 104 | 146 | budget_exhausted | 1.786e+07 | 1.678e+07 | 0.946 | True | 244 | 1 | 4/2 |
| r05 | 2026111605 | 325 | 163/161 | 1 | 16/15 | 115 | 146 | budget_exhausted | 1.787e+07 | 7.16e+04 | 0.00404 | True | 314 | 1 | 4/2 |
| r06 | 2026111606 | 325 | 163/161 | 1 | 16/15 | 117 | 146 | budget_exhausted | 1.937e+07 | 1.404e+07 | 0.791 | True | 291 | 1.5 | 4/2 |
| r07 | 2026111607 | 325 | 163/162 | 0 | 16/15 | 128 | 147 | budget_exhausted | 8.801e+06 | 2.995e+06 | 0.169 | True | 511 | 1.1 | 5/3 |
| r08 | 2026111608 | 324 | 162/161 | 1 | 16/15 | 93 | 146 | budget_exhausted | 1.694e+07 | 5.237e+05 | 0.0295 | True | 373 | 1.16 | 5/3 |
| r09 | 2026111609 | 324 | 163/161 | 0 | 16/15 | 116 | 146 | budget_exhausted | 1.655e+07 | 7.16e+04 | 0.00404 | True | 314 | 1 | 4/2 |
| r10 | 2026111610 | 324 | 163/161 | 0 | 16/15 | 107 | 146 | budget_exhausted | 1.734e+07 | 7.23e+04 | 0.00407 | True | 601 | 0.8 | 4/2 |

`stop`: budget_exhausted = the PDE-sweep guard ended the run (not convergence); optimizer_converged = SciPy L-BFGS-B success. `best J` = best feasible fully evaluated point (objective + gradient), which is what the driver reports as final control on a budget stop. The `overhead` columns are driver re-evaluation/export solves outside the budget (excluded for both methods).

## Figures
- `reconstruction_comparison.png` (+ .pdf)
- `reconstruction_per_start.png` (+ .pdf)
- `best_objective_vs_sweeps.png` (+ .pdf)
- Cumulative plot order: r01, r02, ..., r10 (run index, fixed before launch). The reference curve shows only its two recorded endpoints (after initialization, after L-BFGS-B); per-sweep intermediate values were not recorded for the saved run.

## Scope
- One five-source configuration, noiseless synthetic (inverse-crime) data, one sensor layout, ten random starts. This is a cost-matched comparison against a single saved run, not a statistical study.
