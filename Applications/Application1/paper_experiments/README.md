# Application1 (DC Navy Yard): reproducing the paper experiments

Everything here runs from `Applications/Application1` with the **inverse
environment** (`PY=python`), unless marked *FEniCSx*. All data are **noiseless
synthetic data** generated from the true control by the same forward model
(inverse crime); nothing in Application1 adds noise.

| file / folder | purpose |
|---|---|
| [`reference_runs.json`](reference_runs.json) | exact driver argv of every paper reference run (copied from each run's saved `run_instrumentation.json`), with source batch and code commit |
| [`run_reference.py`](run_reference.py) | runs one entry of `reference_runs.json` through the solve-counting wrapper |
| `random_start_baseline/`, `random_start_budget_M5/` | the two random-start batches (prepare → run → report) |
| `gradient_tests/` | Taylor-remainder gradient verification |
| `sensor_layout_figures/` | sensor-layout reconstruction figure |
| [`reference_results/`](reference_results) | saved reports, summaries, tables, frozen configs and per-run histories of the batches that produced the paper numbers (no figures; they regenerate) |
| `../run_*.sh`, `../test scripts/{sensor_layout,width_study,five_source,mesh_study}_*.py` | the original batch drivers (same code that produced the saved batches) |

## Common settings (all Application1 paper runs)

| setting | value | where |
|---|---|---|
| mesh | `DC_mesh/dc_navy_yard_mesh.msh`, 2795 nodes / 4936 P1 triangles, SHA-256 `a5bf5065…` | committed |
| velocity | steady Navier–Stokes nodal field, SHA-256 `dcdb22e4…` | `supplied/` or *FEniCSx* solve |
| transport | k = 10 m²/s, RK4 dt = 0.1 s, T = 150 s (mesh study: dt = 0.0125 s) | `run_optimization.PDE` |
| source | `f = Σ A_i exp(-|x-c_i|²/ρ)`, ρ = σ² = 400 m² fixed (no continuation; width study varies ρ) | `--true-control`, `PDE["sigma2"]` |
| objective | `J = (η₁ Δt/2) Σ_{n≥1} ‖C uⁿ − dⁿ‖²_W`, η₁ = 100, λ = 0, W = I (w = 1), n = 0 excluded | `ETA1`, `--sensor-weights equal` |
| observations | point sensors at mesh nodes (`--observation-mode nodes`); ids per case in `reference_runs.json` | |
| admissible set | centres in the meshed fluid region (`--bound-case mesh_domain`, `--feasibility-mode raw`); amplitudes [0.1, 2.0] (`--amplitude-floor 0.1`) | |
| initialization | adjoint-sensitivity beam search: `--sensitivity-score predicted_reduction --sensitivity-top-k 24 --sensitivity-beam-width 3`, centre refinement (60 fev), pair refinement (60 fev), coherence cap 0.3, geometry-aware branches, obstacle-aware diversity, second-source re-selection (threshold 0.25, 2 passes), OSQP joint amplitudes (`eps_abs = eps_rel = 1e-10`, verified KKT residual ≤ 1e-7) | flags in `reference_runs.json` |
| local refinement | SciPy L-BFGS-B, `ftol 1e-14`, `gtol 1e-9`, `maxls 20` (hard-coded), `maxcor` 10 (SciPy default) unless stated, `--post-init-scaling gn-diag` (Gauss–Newton diagonal, ratio cap 1e4) for the reported runs, `--max-iter` 150 (M = 2) / 500 (M = 5) | |

Solve counting: one **sweep** = one complete forward solve or one complete
adjoint solve. Counts come from wrappers that intercept `gt.run_forward` and
`gt.rk4_discrete_source_adjoint` (cached responses never reach them).

## Order of work

```bash
cd Applications/Application1
python ../../tools/install_supplied_assets.py          # or solve Navier–Stokes (top-level README)
python paper_experiments/run_reference.py --list        # 22 reference runs
python paper_experiments/run_reference.py init_A_L1_boundary
python paper_experiments/run_reference.py restart_A_L1_boundary
python paper_experiments/run_reference.py M5_S4_gndiag
```

Recorded wall times on a 192-core Linux host (machine- and load-dependent):
`init_*` / `width_init_*` 12–22 min, `restart_*` 50–200 min (that batch ran 50
jobs concurrently), `M5_S4_gndiag` 60 min.

`run_reference.py` writes `output/paper_reference_runs/cases/<run_id>/` (driver
output in `full/`, solve counts in `run_instrumentation.json`). Compare the
final values with `reference_results/<source_batch>/cases/<source_case>/full/optimization_summary.json`.
The downstream scripts read the reference runs either from the original batch
folders under `output/` (names below) or, with
`REFERENCE_RUNS_DIR=output/paper_reference_runs`, from regenerated ones.

## Experiment map

| paper experiment | runs / command | saved results (`reference_results/…`) | tables & figures |
|---|---|---|---|
| sensor layout (A, B × L1 boundary, L2 mixed, L3 weak) | `init_<cfg>_<layout>` then `restart_<cfg>_<layout>` | `sensor_layout_comparison_20261003_151233/` (stage 1, layouts), `gn_diag_default_check_20261004_161718/cases/P_*__gn-diag/` (reported) | `python paper_experiments/sensor_layout_figures/make_sensor_layout_reconstruction_figures.py` |
| prescribed width (B-L2, ρ = 400…50 m²) | `width_init_B_L2_rho*`, `width_restart_B_L2_rho*` (ρ = 400 is `restart_B_L2_mixed`) | `width_study_B_L2_20261003_222835/`, gn-diag restarts as above | `REPORT.md` there |
| five sources (M = 5, 31 sensors) | `M5_S4_gndiag` | `post_init_scaling_validation_20261004_125952/cases/M5_S4_gndiag/`; sensor/truth definition in `five_source_M5_S4sensors_20261003_204754/` | random-start figures below |
| random starts, 10 × 5 cases | `random_start_baseline/` | `random_start_baseline_20261114_224229/` | `test scripts/paper_figures_random_start.py` |
| random starts, M = 5, matched budget | `random_start_budget_M5/` | `random_start_budget_M5_20261116_135850/` | same script |
| mesh dependence (original, 2x, 4x) | `../run_mesh_dependence_study.sh` (*FEniCSx* + inverse) | `mesh_refinement_study/` (`report/table_mesh_study.tex`) | `test scripts/mesh_study_report.py` |
| gradient verification | `gradient_tests/run_paper_gradient_tests.py` | `paper_gradient_tests_20261112_174559/` | same script (`--plot-only`) |

### Sensor layout

Two truths, `A = [162, 300, 1.40 | 270, 600, 0.95]` and `B = [100, 210, 1.30 | 270, 60, 0.85]`,
three hand-chosen 25-sensor layouts each (node ids, requested/actual coordinates
and snap distances: `reference_results/sensor_layout_comparison_20261003_151233/layouts.json`).
The reported reconstruction is a **two-stage chain**: (1) the sensitivity
initializer of `init_*` (its own unscaled L-BFGS-B polish is *not* the reported
result), then (2) `restart_*`, which restarts the unmodified driver from the
stage-1 initialized control (embedded in its `--initial-control`) with gn-diag
L-BFGS-B. Stage-2 can therefore be rerun without stage 1. The batch driver
`../run_sensor_layout_comparison.sh <abs exp dir>` regenerates stage 1 together
with the observation-signal diagnostic and `REPORT.md`.

### Prescribed width

Case B-L2 at fixed **peak** amplitudes for ρ = 400, 300, 200, 100, 50 m², each an
independent run (own data, own sensitivity initialization, unscaled L-BFGS-B up
to 500 iterations, no continuation): `../run_width_study.sh prepare|run <abs exp dir>`
or the `width_init_*` entries. The gn-diag restarts from those initialized
controls are `width_restart_*`. Both are saved; which of the two the manuscript
tabulates must match `reference_results/…/REPORT.md` (see "Open points").

### Five-source reference

Truth `[110, 660, 0.90 | 270, 600, 1.35 | 200, 425, 0.70 | 100, 210, 1.60 | 275, 120, 1.10]`,
31 node sensors = the 25 L1 boundary nodes + 6 interior "S4" nodes
(`928, 1506, 2478, 1438, 2680, 2244`; chosen in `test scripts/five_source_prepare.py`).
Reported run: `M5_S4_gndiag` (converged in 62 iterations; initialization 3069 fwd
/ 12 adj, local refinement 84 fwd / 82 adj). The earlier unscaled run
`five_source_M5_S4sensors_20261003_204754` stopped at the 500-iteration limit
(J = 34.7) and is **not** the reference.

### Random-start comparisons

Both batches replace *only* the initialization of the reference argv by
`--init-mode manual --initial-control <start>` (plus `--case-name` and
`--max-paraview-frames 1`, export only); everything else is the reference argv
verbatim (`reference_results/<batch>/config/cases.json` / `case_config.json`).

* **Start generation.** `MeshAdmissibility.sample_control_in_domain` (the sampler of
  `--init-mode mesh_random`) with a fresh `numpy.random.default_rng(seed)` per run:
  each centre uniform with respect to **area** over the meshed fluid region
  (triangle chosen ∝ area, then uniform in it; buildings are holes and get zero
  probability), drawn independently with **no** minimum separation; then each
  amplitude uniform on [0.1, 2.0] (the L-BFGS-B box); all M centres first, then
  the M amplitudes, from the same stream. Every sampled start is run; the truth
  is never used to accept, reject or stop anything.
* **Seeds.** Baseline: `20261114 + 1000·k + r`, k = 1..5 for A-L1, A-L2, B-L1,
  B-L2, M5 and r = 1..10. Matched budget: `2026111600 + r`, r = 1..10. The starts
  themselves are frozen in `config/starts.{json,csv}`.
* **Feasibility.** Raw mode: an iterate whose centre leaves the fluid mesh is
  rejected by the evaluator (no projection); "feasible" final = all centres inside
  the mesh.
* **Optimizer.** The production local refinement: GN-diag scaling computed at
  *each* start (and charged to it), L-BFGS-B `maxcor 10`, `ftol 1e-14`,
  `gtol 1e-9`, `maxls 20`, `--max-iter` 150 (M = 2) / 500 (M = 5). Baseline runs
  stop on SciPy's convergence tests or the iteration cap.
* **Budget (matched batch only).** Total = the reference run's initialization +
  local refinement = 3069 + 84 forward and 12 + 82 adjoint = **3247 sweeps**
  (`config/budget.json`). Allocation: ⌊3247/10⌋ = 324 per start, the remainder 7
  to r01–r07 (325); unused allowance is not reallocated. Enforced at the PDE-solve
  level by `run_case.py`: before every forward or adjoint solve the remaining
  allowance is checked; a new L-BFGS-B evaluation starts only if ≥ 2 sweeps remain
  (so its adjoint can finish); GN-diag construction (n + 1 forward, n adjoint)
  only if it fits together with one evaluation. On exhaustion the run stops
  without starting a solve and returns its best **fully evaluated** (objective and
  gradient) feasible point. All ten starts stopped on the budget (3241 of 3247
  sweeps used).
* **Counted.** Reference: initialization (sensitivity screening, beam search,
  refinements, OSQP fits) + local refinement (objective at the initialized control,
  GN-diag construction, all L-BFGS-B evaluations incl. line search). Random start:
  objective at the start + GN-diag construction + L-BFGS-B. **Excluded for both**
  and reported separately: the one forward solve that generates the synthetic data,
  and driver reporting overhead (forward smoke/export at the start, objective/gradient
  smoke test, final re-evaluation and state export: reference 4 fwd / 2 adj). The
  sensor traces in the paper figures use extra post-hoc forward solves (21 for
  App1) that are in no count.
* **Selection and reporting.** The best random start is chosen by **lowest final
  objective only**; its centre/amplitude errors are computed afterwards with the
  source permutation minimizing total centre error
  (`run_optimization.match_sources_by_permutation`). The comparison table reports
  the best start's errors next to the **aggregate cost of all ten starts**
  (matched batch: best start `r05`, J = 7.16e4, max centre error 314 m, max
  amplitude error 1.0; all starts together 1629 fwd + 1612 adj). The baseline
  additionally applies a recovery criterion fixed before launch (all centres
  feasible, max matched centre error ≤ 1 m, max amplitude error ≤ 0.01), used only
  for post-hoc reporting (`config/recovery_criterion.json`).

```bash
# baseline (50 runs) -- EXP must be under Applications/Application1/output/
EXP=$PWD/output/random_start_baseline_$(date +%Y%m%d_%H%M%S)
mkdir -p "$EXP/scripts" && cp paper_experiments/random_start_baseline/* "$EXP/scripts/"
python "$EXP/scripts/prepare.py" --exp-dir "$EXP"          # samples + freezes starts, writes commands.tsv
setsid nohup bash "$EXP/scripts/run_batch.sh" "$EXP" 10 4 < /dev/null > "$EXP/batch_driver.log" 2>&1 &
# matched budget (10 runs)
EXP=$PWD/output/random_start_budget_M5_$(date +%Y%m%d_%H%M%S)
mkdir -p "$EXP/scripts" && cp paper_experiments/random_start_budget_M5/* "$EXP/scripts/"
python "$EXP/scripts/prepare.py" --exp-dir "$EXP" && bash "$EXP/scripts/launch.sh"
```

`prepare.py` reads the reference runs from `output/gn_diag_default_check_20261004_161718`,
`output/sensor_layout_comparison_20261003_151233` and
`output/post_init_scaling_validation_20261004_125952`, or from
`REFERENCE_RUNS_DIR`. Paper tables and figures (`--skip-traces` avoids the
post-hoc trace solves):

```bash
python "test scripts/paper_figures_random_start.py" [--skip-traces]       # reads output/<batches>
# from the shipped results instead (writes into the copy):
cp -r paper_experiments/reference_results /tmp/rr
RANDOM_START_RESULTS_ROOT=/tmp/rr python "test scripts/paper_figures_random_start.py" --skip-traces
```

### Mesh dependence

Question: is the post-initialization polish mesh-independent? Levels **original**
(the committed mesh, 4936 triangles), **refine_2x** (9975) and **refine_4x** (19648),
i.e. that multiple of the original *triangle count*, built by gmsh from the original
boundary polylines (`test scripts/mesh_study_make_meshes.py`). No other level was
run (an 8x attempt was aborted before producing results). Per level: Navier–Stokes
is solved on that mesh (*FEniCSx*; reproduces the committed original-mesh field to
~1e-14), the 25 sensors of each case are mapped by **coordinate**, and noiseless data
are generated on that mesh. All runs start from the **same** control: the saved
original-mesh initializer output of `init_A_L1_boundary` / `init_B_L2_mixed`
(values in `reference_results/mesh_refinement_study/config/study_config.json`); the
initializer is not rerun. One common dt = 0.0125 s (12000 steps) for every level,
the largest with dt·ρ(M⁻¹K) ≤ 2.5 on all meshes (`stability/dt_selection.json`).
Optimizer: gn-diag (cap 1e4), `maxcor 30`, `max-iter 200`, `ftol 1e-14`, `gtol 1e-9`.

```bash
PY=python FX=/path/to/fenicsx-env/bin/python \
  bash run_mesh_dependence_study.sh "$PWD/output/mesh_refinement_study_new"   # levels: original refine_2x refine_4x
```

### Gradient verification

Taylor remainders `E0 = |J(z+εq) − J(z)|`, `E1 = |J(z+εq) − J(z) − ε g·q|` at the
saved initialized controls of B-L1, B-L2 (gn-diag restarts) and M5, for
ε = 10^(−k/2), k = 0..16. Direction `q = D r/‖r‖`, `r ~ N(0, I)` from
`default_rng(seed)`, seeds from 0 upward, D = 10 m per centre coordinate and
0.1 per amplitude; a draw is rejected if `z + εq` leaves the admissible set or
`|g·q| < 0.1 ‖D g‖`. Expected: E1 slope 2 (saved: 1.989 / 1.996 / 1.995).

```bash
python paper_experiments/gradient_tests/run_paper_gradient_tests.py --out-dir output/paper_gradient_tests
python paper_experiments/gradient_tests/run_amplitude_supplement.py --help   # amplitude-only supplement
```

## Code versions behind the saved results

| batch | recorded code |
|---|---|
| sensor-layout stage 1 | `40c9e9e`, clean tree |
| width study, unscaled five-source run | `cf1ec1d`, clean tree |
| `M5_S4_gndiag` | `cf1ec1d` + uncommitted patch identical to `de5b069` (post-init scaling) |
| gn-diag restarts | `d5eea0e`, clean tree |
| mesh study | `f168b3f`, clean tree |
| gradient tests, random-start batches | `907552c`, clean tree |

These are commits of the development repository. Between `40c9e9e` and `907552c`
the solver, objective, initializer and `Src/` are unchanged; the only change is
the addition of `--post-init-scaling` / `--post-init-lbfgsb-maxcor`, whose defaults
reproduce the earlier behaviour (unscaled reruns reproduce saved controls to
≤ 1.6e-7). The code in this repository equals `907552c` except for comments, the
move of the shared objective module to `Src/Objective/gradient_test.py` (identical
numerics; `test scripts/test_boundary_mode_unchanged.py` checks bit-for-bit
equality against the previous release), and portable paths.

## Open points

* The manuscript source was not available when this folder was assembled. The
  mapping above follows the saved batch records (e.g. the sensor-layout figure
  script reads the gn-diag restarts). For the width study both the unscaled runs
  and the gn-diag restarts exist; check which one the manuscript tabulates.
* All Application1 results use the committed velocity field with the `p = 0`
  outlet condition. A natural-outlet variant was studied separately and is not
  part of these results.
