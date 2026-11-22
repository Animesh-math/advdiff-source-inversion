# Random-start baseline (App1) — report

Generated 2026-11-15 15:05:37 — 50/50 runs finished.

Setup: `manifest.json`; starts and sampler: `config/starts.json`; per-run data: `runs.csv`, `summary.json`, `cases/<run>/`.

**Recovery criterion (fixed before launch):** recovered = all final centers inside the fluid mesh (raw feasibility) AND max matched center error <= 1.0 m AND max matched amplitude error <= 0.01. Matching: run_optimization.match_sources_by_permutation: label permutation minimizing the TOTAL center position error; amplitude errors taken under that same permutation (the convention of optimization_summary.json recovery_permutation_matched).
Recovery is truth-based and post hoc. It is reported separately from optimizer convergence (the L-BFGS-B termination message).

## Recovery and convergence per case

| case | finished | recovered | converged | conv. but not recovered | iter/eval cap | abnormal | failed | final infeasible | reference recovered |
|---|---|---|---|---|---|---|---|---|---|
| A_L1 | 10/10 | **1/10** | 10 | 9 | 0 | 0 | 0 | 3 | True |
| A_L2 | 10/10 | **2/10** | 10 | 8 | 0 | 0 | 0 | 1 | True |
| B_L1 | 10/10 | **1/10** | 9 | 8 | 1 | 0 | 0 | 3 | True |
| B_L2 | 10/10 | **0/10** | 10 | 10 | 0 | 0 | 0 | 5 | True |
| M5_S4 | 10/10 | **1/10** | 6 | 6 | 4 | 0 | 0 | 5 | True |

## Distributions over finished random starts (min / median / max)

| case | max center err (m) | max amp err | J/J0 | iterations | fwd setup+opt | adj setup+opt |
|---|---|---|---|---|---|---|
| A_L1 | 1.08e-08 / 331 / 621 | 2.23e-11 / 1.3 / 1.3 | 1.09e-21 / 0.0174 / 1 | 24 / 38 / 86 | 41 / 66.5 / 135 | 39 / 64.5 / 134 |
| A_L2 | 5.38e-09 / 201 / 602 | 3.71e-10 / 0.877 / 1.3 | 3.1e-22 / 0.05 / 0.996 | 38 / 45.5 / 137 | 54 / 83 / 175 | 53 / 81.5 / 174 |
| B_L1 | 2.49e-08 / 416 / 695 | 2.65e-10 / 1.2 / 1.2 | 9.35e-23 / 0.5 / 1 | 8 / 37.5 / 150 | 22 / 63 / 209 | 21 / 61 / 207 |
| B_L2 | 231 / 445 / 695 | 0.75 / 0.954 / 1.2 | 0.153 / 0.5 / 1 | 10 / 41 / 83 | 18 / 77 / 106 | 17 / 75.5 / 104 |
| M5_S4 | 0.000386 / 456 / 687 | 1.37e-06 / 0.8 / 1.25 | 2.36e-14 / 0.00536 / 0.161 | 71 / 436 / 500 | 103 / 536 / 597 | 101 / 534 / 596 |

## Sensitivity-initialized reference (paper run) and its full cost

Total = initialization + post-init block (GN-diag construction + L-BFGS-B). The GN share is not instrumented in the reference runs, so it stays inside the post-init column.
Data generation (1 forward) and driver output overhead are excluded from the totals, for both methods.

| case | iters (cap) | message | max center err (m) | max amp err | J/J0 | init fwd/adj | post-init fwd/adj | **total fwd/adj** | 10 random starts: setup+opt fwd/adj |
|---|---|---|---|---|---|---|---|---|---|
| A_L1 | 29 (150) | CONVERGENCE: RELATIVE REDUCTION OF F <=  | 1.08e-08 | 5.74e-10 | 8.43e-23 | 482/2 | 39/38 | **521/40** | 733/717 (10 runs) |
| A_L2 | 90 (150) | CONVERGENCE: RELATIVE REDUCTION OF F <=  | 2.03e-08 | 1.48e-09 | 9.72e-23 | 427/5 | 118/117 | **545/122** | 940/924 (10 runs) |
| B_L1 | 32 (150) | CONVERGENCE: RELATIVE REDUCTION OF F <=  | 5.24e-08 | 1.49e-09 | 5.52e-23 | 507/2 | 44/42 | **551/44** | 862/847 (10 runs) |
| B_L2 | 31 (150) | CONVERGENCE: RELATIVE REDUCTION OF F <=  | 3.62e-09 | 1.77e-10 | 2.63e-22 | 395/3 | 40/39 | **435/42** | 713/697 (10 runs) |
| M5_S4 | 62 (500) | CONVERGENCE: RELATIVE REDUCTION OF F <=  | 1.92e-08 | 1.94e-10 | 4.38e-22 | 3069/12 | 84/82 | **3153/94** | 4431/4412 (10 runs) |

## Best final objective after k starts (predetermined order r01..r10)

The best start is chosen by objective value only. Its recovery status is evaluated afterwards. Costs are cumulative setup+opt solves.

**A_L1**

| k | best J/J0 | best run | recovered | any of first k recovered | cum fwd | cum adj |
|---|---|---|---|---|---|---|
| 1 | 0.0161 | A_L1__r01 | False | False | 69 | 67 |
| 2 | 0.0161 | A_L1__r01 | False | False | 178 | 174 |
| 3 | 0.0161 | A_L1__r01 | False | False | 236 | 230 |
| 4 | 0.0161 | A_L1__r01 | False | False | 277 | 270 |
| 5 | 0.0161 | A_L1__r05 | False | False | 373 | 365 |
| 6 | 0.0161 | A_L1__r05 | False | False | 450 | 441 |
| 7 | 0.0161 | A_L1__r05 | False | False | 493 | 482 |
| 8 | 1.09e-21 | A_L1__r08 | True | True | 628 | 616 |
| 9 | 1.09e-21 | A_L1__r08 | True | True | 692 | 678 |
| 10 | 1.09e-21 | A_L1__r08 | True | True | 733 | 717 |

**A_L2**

| k | best J/J0 | best run | recovered | any of first k recovered | cum fwd | cum adj |
|---|---|---|---|---|---|---|
| 1 | 0.0451 | A_L2__r01 | False | False | 78 | 76 |
| 2 | 0.0451 | A_L2__r01 | False | False | 166 | 163 |
| 3 | 0.0451 | A_L2__r01 | False | False | 222 | 218 |
| 4 | 0.0451 | A_L2__r01 | False | False | 284 | 278 |
| 5 | 3.43e-21 | A_L2__r05 | True | True | 459 | 452 |
| 6 | 3.1e-22 | A_L2__r06 | True | True | 584 | 575 |
| 7 | 3.1e-22 | A_L2__r06 | True | True | 709 | 698 |
| 8 | 3.1e-22 | A_L2__r06 | True | True | 763 | 751 |
| 9 | 3.1e-22 | A_L2__r06 | True | True | 829 | 815 |
| 10 | 3.1e-22 | A_L2__r06 | True | True | 940 | 924 |

**B_L1**

| k | best J/J0 | best run | recovered | any of first k recovered | cum fwd | cum adj |
|---|---|---|---|---|---|---|
| 1 | 0.998 | B_L1__r01 | False | False | 29 | 28 |
| 2 | 0.000135 | B_L1__r02 | False | False | 122 | 120 |
| 3 | 0.000135 | B_L1__r02 | False | False | 144 | 141 |
| 4 | 0.000135 | B_L1__r02 | False | False | 252 | 247 |
| 5 | 7.54e-08 | B_L1__r05 | False | False | 461 | 454 |
| 6 | 7.54e-08 | B_L1__r05 | False | False | 500 | 491 |
| 7 | 7.54e-08 | B_L1__r05 | False | False | 555 | 544 |
| 8 | 9.35e-23 | B_L1__r08 | True | True | 763 | 751 |
| 9 | 9.35e-23 | B_L1__r08 | True | True | 834 | 820 |
| 10 | 9.35e-23 | B_L1__r08 | True | True | 862 | 847 |

**B_L2**

| k | best J/J0 | best run | recovered | any of first k recovered | cum fwd | cum adj |
|---|---|---|---|---|---|---|
| 1 | 0.154 | B_L2__r01 | False | False | 46 | 44 |
| 2 | 0.154 | B_L2__r01 | False | False | 116 | 113 |
| 3 | 0.153 | B_L2__r03 | False | False | 200 | 195 |
| 4 | 0.153 | B_L2__r03 | False | False | 262 | 255 |
| 5 | 0.153 | B_L2__r03 | False | False | 359 | 351 |
| 6 | 0.153 | B_L2__r03 | False | False | 377 | 368 |
| 7 | 0.153 | B_L2__r03 | False | False | 464 | 453 |
| 8 | 0.153 | B_L2__r03 | False | False | 516 | 504 |
| 9 | 0.153 | B_L2__r03 | False | False | 622 | 608 |
| 10 | 0.153 | B_L2__r03 | False | False | 713 | 697 |

**M5_S4**

| k | best J/J0 | best run | recovered | any of first k recovered | cum fwd | cum adj |
|---|---|---|---|---|---|---|
| 1 | 0.161 | M5_S4__r01 | False | False | 567 | 565 |
| 2 | 0.00403 | M5_S4__r02 | False | False | 1008 | 1004 |
| 3 | 0.00403 | M5_S4__r02 | False | False | 1111 | 1105 |
| 4 | 2.36e-14 | M5_S4__r04 | True | True | 1708 | 1701 |
| 5 | 2.36e-14 | M5_S4__r04 | True | True | 2214 | 2205 |
| 6 | 2.36e-14 | M5_S4__r04 | True | True | 2791 | 2780 |
| 7 | 2.36e-14 | M5_S4__r04 | True | True | 3384 | 3371 |
| 8 | 2.36e-14 | M5_S4__r04 | True | True | 3593 | 3578 |
| 9 | 2.36e-14 | M5_S4__r04 | True | True | 3848 | 3831 |
| 10 | 2.36e-14 | M5_S4__r04 | True | True | 4431 | 4412 |

## Per-run table

| run | seed | termination | iters | nfev/ngev | J0 → J_init → J_final | J/J0 | max center err (m) | max amp err | feasible | recovered | setup fwd/adj | opt fwd/adj |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_L1__r01 | 20262115 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 50 | 61/61 | 3.8e+05 → 1.33e+06 → 6.11e+03 | 0.0161 | 346 | 1.3 | False | False | 7/6 | 62/61 |
| A_L1__r02 | 20262116 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 46 | 101/101 | 3.8e+05 → 9.24e+05 → 6.89e+03 | 0.0181 | 323 | 1.3 | True | False | 7/6 | 102/101 |
| A_L1__r03 | 20262117 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 37 | 50/50 | 3.8e+05 → 3.99e+05 → 3.75e+05 | 0.986 | 621 | 0.85 | True | False | 7/6 | 51/50 |
| A_L1__r04 | 20262118 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 24 | 34/34 | 3.8e+05 → 3.4e+05 → 6.32e+03 | 0.0166 | 331 | 1.3 | True | False | 7/6 | 34/34 |
| A_L1__r05 | 20262119 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 59 | 89/89 | 3.8e+05 → 7.47e+05 → 6.11e+03 | 0.0161 | 154 | 1.3 | False | False | 7/6 | 89/89 |
| A_L1__r06 | 20262120 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 36 | 70/70 | 3.8e+05 → 6.04e+05 → 6.32e+03 | 0.0166 | 331 | 1.3 | True | False | 7/6 | 70/70 |
| A_L1__r07 | 20262121 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 32 | 35/35 | 3.8e+05 → 1.28e+06 → 3.81e+05 | 1 | 461 | 1.3 | False | False | 7/6 | 36/35 |
| A_L1__r08 | 20262122 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 86 | 128/128 | 3.8e+05 → 4.3e+05 → 4.16e-16 | 1.09e-21 | 1.08e-08 | 2.23e-11 | True | True | 7/6 | 128/128 |
| A_L1__r09 | 20262123 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 39 | 56/56 | 3.8e+05 → 7.37e+05 → 6.89e+03 | 0.0181 | 323 | 1.3 | True | False | 7/6 | 57/56 |
| A_L1__r10 | 20262124 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 25 | 33/33 | 3.8e+05 → 4.73e+05 → 1.44e+04 | 0.0378 | 403 | 1.3 | True | False | 7/6 | 34/33 |
| A_L2__r01 | 20263115 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 48 | 70/70 | 4.72e+05 → 5.09e+05 → 2.13e+04 | 0.0451 | 469 | 0.85 | False | False | 7/6 | 71/70 |
| A_L2__r02 | 20263116 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 43 | 81/81 | 4.72e+05 → 5.98e+06 → 2.59e+04 | 0.0548 | 602 | 0.85 | True | False | 7/6 | 81/81 |
| A_L2__r03 | 20263117 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 39 | 49/49 | 4.72e+05 → 4.06e+05 → 2.71e+04 | 0.0573 | 105 | 0.904 | True | False | 7/6 | 49/49 |
| A_L2__r04 | 20263118 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 38 | 54/54 | 4.72e+05 → 1.15e+06 → 4.7e+05 | 0.996 | 189 | 1.3 | True | False | 7/6 | 55/54 |
| A_L2__r05 | 20263119 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 137 | 168/168 | 4.72e+05 → 2.05e+05 → 1.62e-15 | 3.43e-21 | 5.38e-09 | 3.71e-10 | True | True | 7/6 | 168/168 |
| A_L2__r06 | 20263120 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 75 | 117/117 | 4.72e+05 → 3.13e+06 → 1.46e-16 | 3.1e-22 | 3e-08 | 2.14e-09 | True | True | 7/6 | 118/117 |
| A_L2__r07 | 20263121 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 94 | 117/117 | 4.72e+05 → 7.77e+06 → 2.13e+04 | 0.0451 | 319 | 1.01 | True | False | 7/6 | 118/117 |
| A_L2__r08 | 20263122 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 38 | 47/47 | 4.72e+05 → 4.11e+05 → 2.71e+04 | 0.0573 | 105 | 0.904 | True | False | 7/6 | 47/47 |
| A_L2__r09 | 20263123 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 41 | 58/58 | 4.72e+05 → 1.17e+07 → 2.99e+04 | 0.0633 | 379 | 0.904 | True | False | 7/6 | 59/58 |
| A_L2__r10 | 20263124 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 66 | 103/103 | 4.72e+05 → 5.36e+05 → 1.96e+04 | 0.0416 | 212 | 0.85 | True | False | 7/6 | 104/103 |
| B_L1__r01 | 20264115 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 21 | 22/22 | 5.62e+05 → 6.29e+05 → 5.61e+05 | 0.998 | 270 | 1.2 | True | False | 7/6 | 22/22 |
| B_L1__r02 | 20264116 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 76 | 86/86 | 5.62e+05 → 2.78e+06 → 76 | 0.000135 | 425 | 1.2 | False | False | 7/6 | 86/86 |
| B_L1__r03 | 20264117 | converged: CONVERGENCE: NORM OF PROJECTED GRADIEN | 8 | 15/15 | 5.62e+05 → 1.11e+06 → 5.64e+05 | 1 | 695 | 1.2 | True | False | 7/6 | 15/15 |
| B_L1__r04 | 20264118 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 61 | 100/100 | 5.62e+05 → 5.73e+05 → 919 | 0.00163 | 233 | 1.2 | True | False | 7/6 | 101/100 |
| B_L1__r05 | 20264119 | iteration_or_eval_limit: STOP: TOTAL NO. OF ITERATIONS REACHED  | 150 | 201/201 | 5.62e+05 → 5.92e+05 → 0.0424 | 7.54e-08 | 4.09 | 0.141 | True | False | 7/6 | 202/201 |
| B_L1__r06 | 20264120 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 26 | 31/31 | 5.62e+05 → 1.02e+06 → 5.62e+05 | 1 | 523 | 1.2 | False | False | 7/6 | 32/31 |
| B_L1__r07 | 20264121 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 31 | 47/47 | 5.62e+05 → 6.53e+05 → 5.61e+05 | 0.999 | 463 | 1.2 | True | False | 7/6 | 48/47 |
| B_L1__r08 | 20264122 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 134 | 201/201 | 5.62e+05 → 5.7e+05 → 5.26e-17 | 9.35e-23 | 2.49e-08 | 2.65e-10 | True | True | 7/6 | 201/201 |
| B_L1__r09 | 20264123 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 44 | 63/63 | 5.62e+05 → 6.37e+05 → 76 | 0.000135 | 425 | 1.2 | False | False | 7/6 | 64/63 |
| B_L1__r10 | 20264124 | converged: CONVERGENCE: NORM OF PROJECTED GRADIEN | 13 | 21/21 | 5.62e+05 → 6.08e+05 → 5.62e+05 | 1 | 407 | 1.2 | True | False | 7/6 | 21/21 |
| B_L2__r01 | 20265115 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 30 | 38/38 | 1.09e+06 → 1.02e+07 → 1.68e+05 | 0.154 | 695 | 0.75 | True | False | 7/6 | 39/38 |
| B_L2__r02 | 20265116 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 54 | 63/63 | 1.09e+06 → 1.01e+06 → 9.23e+05 | 0.847 | 231 | 1.16 | True | False | 7/6 | 63/63 |
| B_L2__r03 | 20265117 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 63 | 76/76 | 1.09e+06 → 1.65e+06 → 1.67e+05 | 0.153 | 614 | 0.75 | False | False | 7/6 | 77/76 |
| B_L2__r04 | 20265118 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 39 | 54/54 | 1.09e+06 → 2.67e+06 → 1.09e+06 | 1 | 251 | 1.2 | False | False | 7/6 | 55/54 |
| B_L2__r05 | 20265119 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 54 | 90/90 | 1.09e+06 → 3.17e+06 → 1.67e+05 | 0.153 | 251 | 0.75 | False | False | 7/6 | 90/90 |
| B_L2__r06 | 20265120 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 10 | 11/11 | 1.09e+06 → 1.1e+06 → 1.09e+06 | 1 | 424 | 1.2 | True | False | 7/6 | 11/11 |
| B_L2__r07 | 20265121 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 36 | 79/79 | 1.09e+06 → 1.95e+06 → 1.68e+05 | 0.154 | 695 | 0.75 | True | False | 7/6 | 80/79 |
| B_L2__r08 | 20265122 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 34 | 45/45 | 1.09e+06 → 1.34e+06 → 1.09e+06 | 1 | 614 | 1.16 | False | False | 7/6 | 45/45 |
| B_L2__r09 | 20265123 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 83 | 98/98 | 1.09e+06 → 1.4e+06 → 9.23e+05 | 0.847 | 428 | 1.2 | False | False | 7/6 | 99/98 |
| B_L2__r10 | 20265124 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 43 | 83/83 | 1.09e+06 → 2.33e+06 → 1.67e+05 | 0.153 | 463 | 0.75 | True | False | 7/6 | 84/83 |
| M5_S4__r01 | 20266115 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 464 | 550/550 | 1.77e+07 → 1.75e+07 → 2.86e+06 | 0.161 | 300 | 1 | False | False | 16/15 | 551/550 |
| M5_S4__r02 | 20266116 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 336 | 424/424 | 1.77e+07 → 1.76e+07 → 7.16e+04 | 0.00403 | 687 | 0.8 | True | False | 16/15 | 425/424 |
| M5_S4__r03 | 20266117 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 71 | 86/86 | 1.77e+07 → 2.55e+07 → 2.86e+06 | 0.161 | 669 | 1 | True | False | 16/15 | 87/86 |
| M5_S4__r04 | 20266118 | iteration_or_eval_limit: STOP: TOTAL NO. OF ITERATIONS REACHED  | 500 | 581/581 | 1.77e+07 → 1.74e+07 → 4.18e-07 | 2.36e-14 | 0.000386 | 1.37e-06 | True | True | 16/15 | 581/581 |
| M5_S4__r05 | 20266119 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 407 | 489/489 | 1.77e+07 → 1.07e+07 → 2.79e+06 | 0.157 | 246 | 1 | False | False | 16/15 | 490/489 |
| M5_S4__r06 | 20266120 | iteration_or_eval_limit: STOP: TOTAL NO. OF ITERATIONS REACHED  | 500 | 560/560 | 1.77e+07 → 2.56e+07 → 7.14e+04 | 0.00402 | 562 | 0.8 | False | False | 16/15 | 561/560 |
| M5_S4__r07 | 20266121 | iteration_or_eval_limit: STOP: TOTAL NO. OF ITERATIONS REACHED  | 500 | 576/576 | 1.77e+07 → 1.69e+07 → 7.14e+04 | 0.00402 | 552 | 0.696 | False | False | 16/15 | 577/576 |
| M5_S4__r08 | 20266122 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 168 | 192/192 | 1.77e+07 → 1.79e+07 → 2.86e+06 | 0.161 | 360 | 1.25 | True | False | 16/15 | 193/192 |
| M5_S4__r09 | 20266123 | converged: CONVERGENCE: RELATIVE REDUCTION OF F < | 215 | 238/238 | 1.77e+07 → 1.02e+07 → 7.14e+04 | 0.00402 | 323 | 0.8 | True | False | 16/15 | 239/238 |
| M5_S4__r10 | 20266124 | iteration_or_eval_limit: STOP: TOTAL NO. OF ITERATIONS REACHED  | 500 | 566/566 | 1.77e+07 → 1.84e+07 → 1.19e+05 | 0.00669 | 571 | 0.8 | False | False | 16/15 | 567/566 |

## Failures / exceptional terminations

- none

## Interpretation caveats

- A run that stops at the iteration cap is not proof that it could never recover with more iterations. Cap-limited runs are counted separately.
- Do not infer a cost advantage from local-optimizer counts alone. The reference total includes the initializer's forward solves, which dominate its cost.
- Noiseless inverse-crime data, one truth per case, ten starts per case.
- The 2-source references are a chain: the sensitivity initializer in `sensor_layout_comparison_20261003_151233`, then a gn-diag restart from its saved control in `gn_diag_default_check_20261004_161718`. The identical start is verified. The unscaled polish of the initializer run is not part of the paper pipeline and is excluded.
- Wall times are not compared (different thread counts and host load). Solve counts are the cost metric.

Figures: `overview_distributions.png/pdf`, `best_of_k.png/pdf`.
