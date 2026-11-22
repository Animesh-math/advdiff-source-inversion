# Paper gradient (Taylor-remainder) tests — Application 1

Created 2026-11-12 on `<host>`, repo HEAD `907552c` (see `git_commit.txt`).
Scope: gradient verification only. No initialization, center search, or L-BFGS-B was run;
no solver, manuscript, or repository code was modified (scripts live in `scripts/` here).

## Verdict

The expected rates are observed in all three cases:
**E0 = O(ε)** and **E1 = O(ε²)**, with fitted E1 slopes 1.989–1.996
and local rates 2.000 over 4–6 decades.

| Case | Source run | J_h(z) | g·q | E1 fitted slope (range) | E0 slope (first-order range) |
|---|---|---|---|---|---|
| B–L1, 25 boundary sensors | `gn_diag_default_check_20261004_161718/cases/P_B_L1_boundary__gn-diag` | 0.641293 | 3.887 | **1.989** (1 → 1e-8, all 17 pts) | 1.008 (3.2e-4 → 1e-8) |
| B–L2, 10 boundary + 15 interior | `gn_diag_default_check_20261004_161718/cases/P_B_L2_mixed__gn-diag` | 0.527434 | 28.17 | **1.996** (1 → 1e-8, all 17 pts) | 1.004 (1e-4 → 1e-8) |
| M5_S4, 5 sources, 31 sensors | `post_init_scaling_validation_20261004_125952/cases/M5_S4_gndiag` | 843.408 | 3416 | **1.995** (1 → 3.2e-8, 16 pts) | 1.003 (3.2e-3 → 1e-8) |

Local E1 rates between successive steps (ε from 1 to 1e-8):
- B–L1: 1.866, 1.961, 1.988, 1.996, 1.999, 2.000 ×8, 2.002, **1.948, 1.802**
- B–L2: 1.916, 1.972, 1.991, 1.997, 1.999, 2.000 ×9, 2.001, 1.984
- M5_S4: 1.955, 1.984, 1.995, 1.998, 1.999, 2.000 ×7, 2.002, 1.977, **1.902, 2.842**

**Resolved range vs. cancellation.**
- **Large steps:** at ε ≳ 1e-1, rates slightly below 2 come from higher-order terms, not from a gradient error.
- **Smallest steps:** below ε ≈ 1e-7, the remainder E1 (~1e-13 to 1e-12 absolute) approaches the
  floating-point resolution of J_h. That resolution is ~ε_mach·J_h, i.e. 1.4e-16 (B), 1.9e-13 (M5),
  amplified by the 1500-step time integration and the summation over sensors.
  This explains the B–L1 rates 1.95/1.80 and the M5 rates 1.90/2.84 at the last two steps.
- **Reporting:** no points were discarded. All points are in the tables and figures, and the
  "resolved" range in `summary.json` is the longest run of consecutive local rates in [1.8, 2.2].
- **E0 at large steps:** E0 shows slope ≈ 2 at large ε because the ε² term dominates there.
  It becomes first order once ε·|g·q| exceeds that term.

## Supplementary amplitude-block check

At the primary test points, ∂J/∂A ≈ 1e-9 for every source. The saved starting controls come from
the OSQP joint amplitude solve, so the amplitudes are already optimal for the given centers.
The primary g·q is therefore carried by the center components (split in `cases/*/result.json`).
The primary test still certifies that the near-zero amplitude gradient is correct, because a wrong
nonzero ∂J/∂A would add an O(ε) term to E1.

To exercise nonzero amplitude partials, I ran a supplementary test:
- **Test point:** a second saved nonstationary control from the same runs, the **iteration-1 iterate**
  in `full/optimization_history.csv`. J reproduces the history value to ≤5e-15 relative.
- **Direction:** **amplitude-only** (`supplement_amplitude/`, `figures/supplement_amplitude_taylor_3panel.*`).

| Case | J_h(z₁) | ‖∂J/∂A‖ | g·q | E1 fitted slope (range) | E0 slope |
|---|---|---|---|---|---|
| B–L1 | 0.50150 | 184.3 | −12.63 | **2.0005** (1 → 3.2e-8) | 0.994 |
| B–L2 | 0.35274 | 188.9 | 18.88 | **1.9988** (1 → 1e-8) | 1.004 |
| M5_S4 | 575.029 | 2.69e4 | 739.7 | **1.9974** (1 → 1e-7) | 1.006 |

The local rates are 2.000 except at the last 1–3 steps, which are in the cancellation regime:
- B–L1: 2.002, 2.02, 2.27
- B–L2: 1.94
- M5: 1.89, 1.48, 6.55

The M5 E1 at ε = 1e-8 is 4.7e-15, i.e. at the rounding level.

J is exactly quadratic in A for fixed centers, so here E1 is ½ε²qᵀHq plus rounding.
The B–L1 panel has a dip in E0 near ε ≈ 3e-3. It occurs because g·q < 0 while the quadratic term
is positive, so J(z+εq) − J(z) changes sign there. That is not an error.

## Setup (recovered from the records and verified)

- **Driver arguments** come from each run's `resolved_driver_args.json`. Each rebuilt quantity was
  checked against `full/optimization_summary.json` (all checks `True` in `cases/*/result.json`
  → `checks`):
  - mesh: 2795 nodes
  - sensor node IDs, coordinates, and weights (equal, w = 1)
  - velocity field: `output/navier_stokes/application1_navier_stokes_velocity_nodes.npz`, nodal
  - PDE: k = 10, dt = 0.1, tfinal = 150, jprint = 1, ngauss = 3, **σ² = 400 (ρ = 400)**,
    no σ² continuation
  - mode `location_and_intensity`, bound case `mesh_domain`, feasibility `raw`, amplitude floor 0.1
- **Objective and gradient:** production `run_optimization.build_evaluator` (the evaluator handed to
  L-BFGS-B) with η₁ = 100, λ = 0, objective time rule `exclude_first`, and the RK4 discrete-adjoint
  gradient. Everything is in **physical coordinates** [x, y, A] per source; the gn-diag scaling
  was not applied.
- **Synthetic data:** one forward solve at the true control, as in the driver.
- **Reproduction:** recomputed J(z) matches the recorded `initial_objective` with relative differences
  0, 1.0e-14, and 1.2e-15. ‖∇J‖ matches the physical-coordinate iteration-0 gradient norm of the
  companion `__none` runs (3.0232, 6.4483; relative difference ≤ 6e-15).
- **Code drift:** commits after the source runs changed only CLI parsing and post-init dispatch
  in `run_optimization.py`. The evaluator, objective, and solvers are unchanged.
- **Test point z:** the saved L-BFGS-B starting control (`initial_control`, = history iteration 0).
  It is nonstationary (‖∇J‖ = 3.02, 6.45, 820). It is neither the truth nor the recovered solution.
- **Direction:** q = D r with r ~ N(0, I) from `numpy.random.default_rng(seed)`, normalized to
  ‖r‖₂ = 1. D = diag(10 m, 10 m, 0.1) per source.
  - Seeds start at 0. A draw is rejected if any z + εq leaves the fluid mesh or the amplitude range
    [0.1, 2.0], or if |g·q| < 0.1‖Dg‖.
  - Seeds used: B–L1 0, B–L2 0, M5 2. For M5, seeds 0 and 1 were rejected for small |g·q|.
  - Largest step at ε = 1: 5.7–6.5 m in a center and 0.03–0.07 in amplitude.
  - No raw-infeasible evaluations occurred.
- **Steps:** ε = 10^(−k/2), k = 0…16 (1 → 1e-8, half-decade ratio). This follows the existing
  decade protocol of `test scripts/gradient_test_navier_stokes_velocity.py`, refined to half decades.

## Files

- **Figures**
  - `figures/taylor_gradient_tests_3panel.{pdf,png}` — the paper figure (PNG at 300 dpi)
  - `figures/taylor_gradient_test_{B_L1,B_L2,M5_S4}.{pdf,png}` — the individual panels
  - `figures/supplement_amplitude_taylor_3panel.{pdf,png}` — the supplementary check
- **Primary test data**
  - `cases/<case>/result.json`: recovered settings, checks, z, gradient, q, r, D, seed,
    rejected draws, and all raw and normalized errors and rates
  - `cases/<case>/taylor_table.csv` and `cases/<case>/control_direction.npz`
  - `all_cases_taylor_table.csv`, `summary.json`
- **Supplementary test data:** `supplement_amplitude/<case>/{result.json,taylor_table.csv}`
- **Scripts:** `scripts/run_paper_gradient_tests.py`, `scripts/run_amplitude_supplement.py`
- **Logs:** `logs/run.log`, `logs/supplement.log`, `logs/replot.log`
  (the primary figure was re-plotted after adjusting the reference-line placement; the data are unchanged)
- **Integrity:** `SHA256SUMS` lists checksums of all files.
