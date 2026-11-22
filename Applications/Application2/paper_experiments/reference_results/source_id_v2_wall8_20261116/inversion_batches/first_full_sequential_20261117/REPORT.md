# Application2 source identification: first full inversion batch

Dataset `source_id_v2_wall8_20261116` (`Applications/Application2/output/source_id_v2_wall8_20261116`), sensor layout `wall_8` with 8 sensors (S1, S2, S3, S4, S5, S6, S7, S8), flow mesh/velocity sha256 `3536cc16a57f` / `b3f206216661`. Coarse-mesh inversion of fine-mesh observations; two unknown centres and amplitudes, prescribed width rho = 0.04 m^2; W = I; amplitude bounds [0.1, 2.0]. Sensitivity initializer (OSQP, M = 2 preset, room-scale tolerance shim) then GN-diagonal-scaled L-BFGS-B (maxcor 30, max-iter 200). Settings per run: `resolved_settings.json`.

The true controls were read only by this report, after the runs, to score recovery.

## Classification rules (fixed before the runs)

- Optimizer outcome: `FAILED` (non-zero exit or no summary), `ITERATION_LIMIT` (stopped at max-iter or an evaluation limit), `CONVERGED` (L-BFGS-B success flag), `STOPPED_OTHER` (other termination, e.g. abnormal line search; see message), `NOT_FINISHED`.
- Recovery: `RECOVERED` iff the outcome is CONVERGED or STOPPED_OTHER, both final centres lie in the fluid mesh, and both permutation-matched centre errors are <= one e-folding radius sqrt(rho) = 0.20 m. Amplitude errors are reported, not thresholded. An ITERATION_LIMIT or FAILED run is never counted as recovered.

## 1. Run status

| run | exit | outcome | recovery | L-BFGS-B its | nfev / njev | termination message |
|---|---|---|---|---|---|---|
| case_12_clean | 0 | **CONVERGED** | **RECOVERED** | 62 | 71 / 71 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| case_12_noisy | 0 | **CONVERGED** | **RECOVERED** | 45 | 58 / 58 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| case_13_clean | 0 | **CONVERGED** | **RECOVERED** | 22 | 27 / 27 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| case_13_noisy | 0 | **CONVERGED** | **RECOVERED** | 23 | 34 / 34 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| case_23_clean | 0 | **CONVERGED** | **RECOVERED** | 22 | 25 / 25 | CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL |
| case_23_noisy | 0 | **CONVERGED** | **RECOVERED** | 21 | 28 / 28 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |

## 2. Controls `[x1, y1, A1, x2, y2, A2]` and objectives

| run | initial control | final control | true control | J(0) | J(initial) | J(final) | J(final)/J(0) | noise-only J |
|---|---|---|---|---|---|---|---|---|
| case_12_clean | (2.2258, 5.0220, 1.1048, 5.6352, 5.2390, 0.6833) | (2.4724, 5.1756, 0.9877, 5.4826, 5.1811, 0.6901) | (2.475, 5.200, 1.000, 5.475, 5.200, 0.700) | 2.8605e+03 | 4.5107e-02 | 5.5344e-03 | 1.935e-06 | n/a |
| case_12_noisy | (2.3395, 5.0927, 1.0238, 5.5837, 5.2289, 0.6823) | (2.4764, 5.1778, 0.9869, 5.4771, 5.1785, 0.6908) | (2.475, 5.200, 1.000, 5.475, 5.200, 0.700) | 2.8607e+03 | 2.9760e-01 | 2.9033e-01 | 1.015e-04 | 2.8514e-01 |
| case_13_clean | (8.4605, 5.1350, 1.2832, 2.4940, 5.1856, 0.9839) | (8.4628, 5.1408, 1.2827, 2.5024, 5.1905, 0.9827) | (2.475, 5.200, 1.000, 8.500, 5.200, 1.300) | 5.0589e+03 | 1.5855e-02 | 1.4055e-02 | 2.778e-06 | n/a |
| case_13_noisy | (8.4616, 5.1369, 1.2829, 2.4940, 5.1856, 0.9839) | (8.4637, 5.1417, 1.2825, 2.5047, 5.1915, 0.9823) | (2.475, 5.200, 1.000, 8.500, 5.200, 1.300) | 5.0592e+03 | 5.1933e-01 | 5.1805e-01 | 1.024e-04 | 5.0387e-01 |
| case_23_clean | (8.4656, 5.1431, 1.2821, 5.4510, 5.1657, 0.6943) | (8.4643, 5.1416, 1.2823, 5.4490, 5.1626, 0.6946) | (5.475, 5.200, 0.700, 8.500, 5.200, 1.300) | 4.0911e+03 | 1.2062e-02 | 1.1896e-02 | 2.908e-06 | n/a |
| case_23_noisy | (8.4655, 5.1431, 1.2821, 5.4510, 5.1656, 0.6943) | (8.4639, 5.1415, 1.2824, 5.4512, 5.1638, 0.6943) | (5.475, 5.200, 0.700, 8.500, 5.200, 1.300) | 4.0914e+03 | 4.2077e-01 | 4.2060e-01 | 1.028e-04 | 4.0879e-01 |

Noise-only J = (eta1 dt/2) sum ||eps^n||^2 of the saved realization (clean runs: not applicable).

## 3. Iterations and PDE solve counts

Counts are actual forward / adjoint sweeps at the shared solver entry points (cache hits cost 0). Optimization includes the objective/gradient at the initial control and the final re-evaluation.

| run | init fwd | init adj | GN-diag scaling fwd | scaling adj | L-BFGS-B fwd | L-BFGS-B adj | total fwd | total adj | L-BFGS-B its |
|---|---|---|---|---|---|---|---|---|---|
| case_12_clean | 798 | 4 | 7 | 6 | 71 | 71 | 876 | 81 | 62 |
| case_12_noisy | 919 | 5 | 7 | 6 | 58 | 58 | 984 | 69 | 45 |
| case_13_clean | 510 | 2 | 7 | 6 | 28 | 28 | 545 | 36 | 22 |
| case_13_noisy | 638 | 3 | 7 | 6 | 34 | 34 | 679 | 43 | 23 |
| case_23_clean | 761 | 4 | 7 | 6 | 26 | 26 | 794 | 36 | 22 |
| case_23_noisy | 761 | 4 | 7 | 6 | 28 | 28 | 796 | 38 | 21 |

## 4. Final feasibility and permutation-matched recovery errors

| run | matched (true->est) | centre err 1 [m] | centre err 2 [m] | A err 1 (rel) | A err 2 (rel) | final in fluid mesh | clearances [m] | initial centre errs [m] |
|---|---|---|---|---|---|---|---|---|
| case_12_clean | [1, 2] | 0.0245 | 0.0204 | -0.0123 (-1.23%) | -0.0099 (-1.41%) | True | 0.472, 0.467 | 0.306, 0.165 |
| case_12_noisy | [1, 2] | 0.0223 | 0.0216 | -0.0131 (-1.31%) | -0.0092 (-1.31%) | True | 0.474, 0.473 | 0.173, 0.113 |
| case_13_clean | [2, 1] | 0.0290 | 0.0699 | -0.0173 (-1.73%) | -0.0173 (-1.33%) | True | 0.463, 0.448 | 0.024, 0.076 |
| case_13_noisy | [2, 1] | 0.0309 | 0.0686 | -0.0177 (-1.77%) | -0.0175 (-1.34%) | True | 0.464, 0.445 | 0.024, 0.074 |
| case_23_clean | [2, 1] | 0.0456 | 0.0684 | -0.0054 (-0.77%) | -0.0177 (-1.36%) | True | 0.464, 0.449 | 0.042, 0.066 |
| case_23_noisy | [2, 1] | 0.0433 | 0.0687 | -0.0057 (-0.82%) | -0.0176 (-1.35%) | True | 0.464, 0.451 | 0.042, 0.067 |

## 5. Clean versus noisy

| case | clean outcome / recovery | noisy outcome / recovery | max centre err clean / noisy [m] | |z_clean - z_noisy| centres [m] | amplitudes clean | amplitudes noisy |
|---|---|---|---|---|---|---|
| 12 | CONVERGED / RECOVERED | CONVERGED / RECOVERED | 0.0245 / 0.0223 | 0.0045, 0.0061 | 0.9877, 0.6901 | 0.9869, 0.6908 |
| 13 | CONVERGED / RECOVERED | CONVERGED / RECOVERED | 0.0699 / 0.0686 | 0.0026, 0.0013 | 0.9827, 1.2827 | 0.9823, 1.2825 |
| 23 | CONVERGED / RECOVERED | CONVERGED / RECOVERED | 0.0684 / 0.0687 | 0.0026, 0.0004 | 0.6946, 1.2823 | 0.6943, 1.2824 |

Centres are paired by the true source each one is matched to.

## 6. Sensor-trace fits at the final control

Relative residual ||C u(z_final) - d|| / ||d|| over n >= 1 (post-hoc coarse forward solves, not counted above).

| run | all sensors | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 |
|---|---|---|---|---|---|---|---|---|---|
| case_12_clean | 1.391e-03 | 2.47e-02 | 1.40e-03 | 1.82e-02 | 1.39e-03 | 9.38e-02 | 2.38e-02 | 1.17e-03 | 1.77e-03 |
| case_12_noisy | 1.007e-02 | 9.90e-01 | 4.55e-03 | 9.91e-01 | 6.35e-03 | 1.00e+00 | 9.96e-01 | 2.91e-02 | 3.28e-01 |
| case_13_clean | 1.667e-03 | 1.87e-02 | 1.43e-03 | 4.04e-01 | 4.78e-02 | 1.82e-02 | 1.80e-03 | 2.63e-03 | 1.36e-03 |
| case_13_noisy | 1.012e-02 | 9.94e-01 | 5.91e-03 | 1.00e+00 | 9.92e-01 | 9.94e-01 | 4.88e-03 | 6.49e-02 | 2.42e-02 |
| case_23_clean | 1.705e-03 | 6.25e-02 | 4.59e-02 | 3.51e-02 | 1.37e-03 | 1.86e-02 | 1.81e-03 | 1.43e-03 | 1.37e-03 |
| case_23_noisy | 1.014e-02 | 1.00e+00 | 9.95e-01 | 9.97e-01 | 7.60e-03 | 9.90e-01 | 4.47e-03 | 7.94e-02 | 2.09e-02 |

Data RMS per sensor (first run of each case) shows which sensors carry signal: 
- case 12: S1 0.00216, S2 3.56, S3 0.00196, S4 2.48, S5 0.000526, S6 0.00139, S7 0.528, S8 0.0446
- case 13: S1 0.00215, S2 3.55, S3 0.000526, S4 0.00241, S5 0.00242, S6 4.5, S7 0.317, S8 0.844
- case 23: S1 5.64e-06, S2 0.0019, S3 0.00144, S4 2.48, S5 0.00293, S6 4.5, S7 0.232, S8 0.886

## Figures

- `report/fig_objective_history.png`
- `report/fig_sources_paths_case_12.png`
- `report/fig_sources_paths_case_13.png`
- `report/fig_sources_paths_case_23.png`
- `report/fig_traces_case_12_clean.png`
- `report/fig_traces_case_12_noisy.png`
- `report/fig_traces_case_13_clean.png`
- `report/fig_traces_case_13_noisy.png`
- `report/fig_traces_case_23_clean.png`
- `report/fig_traces_case_23_noisy.png`

## Failures and limits

None: every run exited 0 and stopped before the iteration limit.
