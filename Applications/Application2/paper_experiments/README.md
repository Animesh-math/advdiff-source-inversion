# Application2 (three-bed room): reproducing the paper experiment

Commands run from `Applications/Application2`; `PY` = inverse environment,
`FX` = FEniCSx environment. Every setting below is read from the version-controlled
configuration files (`room_geometry_config.py`, `navier_stokes_config.py`,
`transport_config.py`, `run_inversion.py` defaults) and was checked against the
saved batch (`reference_results/…/resolved_settings.json`).

## Setup at a glance

| item | value |
|---|---|
| geometry | main room [0,9]×[0,6] m ∪ bathroom annex [9,11]×[0,2] m; two 0.10 m partitions from the top wall; three **solid** beds [1,2], [4,5], [7,8] × [3.5,5.6] (holes in the mesh); bathroom wall with an open doorway; fluid area 50.98 m² |
| ventilation | three top-wall supplies (x ∈ [1.2,1.8], [4.2,4.8], [7.2,7.8]), downward parabola, mean 0.10 m/s; corridor door (y = 0, x ∈ [0.6,1.6]) and bathroom exhaust (x = 11, y ∈ [0.7,1.3]) are **natural** outlets `(ν∇v − pI)n = 0`; no slip elsewhere |
| flow model | steady incompressible Navier–Stokes, Taylor–Hood P2/P1, Picard (tol 1e-8, ≤ 100 its), MUMPS; constant **effective** viscosity ν = 2.3244e-3 m²/s (a modelling parameter) |
| meshes | **coarse** 4381 nodes / 7887 triangles (inversion); **fine** 15231 / 28772 (synthetic data); fine = half of every coarse size; each mesh has its **own** velocity field (52 Picard its each; door/exhaust 66 % / 34 % of the inflow) |
| transport | `u_t + v·∇u − kΔu = S`, k = 1e-2 m²/s (effective tracer diffusivity), u = 0 on the supplies, zero diffusive flux elsewhere; P1, lumped mass, RK4 |
| time | T = 300 s; dt = 0.015 s (coarse), 0.0025 s (fine, 6 sub-steps); observations at tₙ = n·0.015 s, n = 1..20000; n = 0 (zero initial state) excluded |
| sources | `S = Σ A_i exp(−|x − c_i|²/ρ)`, ρ = 0.04 m² (e-folding radius 0.20 m), A = peak rate; releases bed 1 (2.475, 5.2) A = 1.0, bed 2 (5.475, 5.2) A = 0.7, bed 3 (8.5, 5.2) A = 1.3; cases **12**, **13**, **23** (two sources each) |
| admissible controls | M = 2 known; centres anywhere inside a coarse fluid triangle (box [0,11]×[0,6]); amplitudes [0.1, 2.0] |
| sensors | `wall_8`: eight fixed wall-mounted points, coarse-mesh nodes 0.09–0.21 m from the wall (S1 (0.159, 5.100), S2 (2.831, 4.053), S3 (3.161, 5.085), S4 (5.831, 4.053), S5 (6.156, 5.078), S6 (8.843, 3.977), S7 (1.907, 0.090), S8 (10.509, 1.789)); the fine mesh is sampled at the **same coordinates** by P1 interpolation, the coarse model at the nodes |
| objective | `J = (η₁ Δt/2) Σₙ₌₁ᴺ ‖C Uⁿ − dⁿ‖²_W`, η₁ = 1, W = I, λ = 0 |

## Observation noise (`generate_observations.py`, `transport_config.py`)

```text
s      = 0.01 * sqrt( mean_{n=1..N, j=1..8} d_clean[n, j]^2 )        # global RMS, n = 0 excluded
eps    = numpy.random.default_rng(seed).normal(0, s, size=(N, 8))   # rows n = 1..N
noise[0] = 0,  noise[1:] = eps;   d_noisy = d_clean + noise           # no clipping
seed   = 2026111612 (case 12), 2026111613 (case 13), 2026111623 (case 23)
s      = 0.015439 / 0.020532 / 0.018464
```

The noise is drawn **once** when the dataset is generated, saved
(`case_XX/data/noise.npz`, `observations_noisy.npz`) and only read afterwards;
the reconstruction never redraws it. `generate_observations.py` refuses to
overwrite an existing dataset.

## Commands, in dependency order

```bash
cd Applications/Application2
# 1. meshes + flow: either the supplied files ...
python ../../tools/install_supplied_assets.py
#    ... or regenerate them (FEniCSx; < 1 min recorded; writes output/v3_bathroom_topsupply_20261116/)
FX=/path/to/fenicsx-env/bin/python bash run_application2_setup.sh run
# 2. dataset: fine-mesh observations, noise, truth-free inversion configs (~2 min)
python generate_observations.py --out output/source_id_v2_wall8_20261116
python validate_readiness.py --dataset-dir output/source_id_v2_wall8_20261116   # optional checks (~20 min)
# 3. one reconstruction (saved batch: six runs in 17 h at 96 BLAS threads, sequential)
python run_inversion.py --dataset-dir output/source_id_v2_wall8_20261116 --case 12 --data noisy \
    --stage full --out-dir output/source_id_v2_wall8_20261116/inversion_runs/case_12_noisy
#    or all six (sequential, preflight + post-hoc report):
PY=python bash run_application2_inversion_batch.sh launch my_batch
# 4. figures for a batch (12 post-hoc coarse forward solves for the traces)
python paper_figures_inversion_batch.py --batch-dir output/source_id_v2_wall8_20261116/inversion_batches/my_batch
```

`--data clean` / `--data noisy` selects the saved observation file; nothing else
differs. `--stage wiring` is a cheap check (one objective and gradient, ~6 min).
`run_application2_source_id_setup.sh run` performs step 2 together with the
settings-selection record, validation and `REPORT.md`.

## Reconstruction settings (`run_inversion.py` defaults, used for all six runs)

* Sensitivity initializer of Application1, unchanged: OSQP joint amplitudes,
  `top-k 24`, `beam 3`, centre refinement 60 fev, pair refinement 60 fev,
  coherence cap 0.3, no boundary buffer, no minimum separation.
* Room-scale tolerance shim: Application1's initializer hard-codes three DC-scale
  literals; `run_inversion.py` substitutes them at load time with Nelder–Mead
  `xatol = 0.005 m` (= 0.025·√ρ), `fatol = 1e-6·J(zero source)` and a
  de-duplication floor of 0.02 m (`--init-app1-literal-tolerances` disables it).
  The shim aborts if the Application1 source text it replaces is not found.
* Local refinement: L-BFGS-B with GN-diagonal scaling (ratio cap 1e4),
  `maxcor 30`, `max-iter 200`, `ftol 1e-14`, `gtol 1e-9`, `maxls 20`.
* Recovery classification (post hoc, fixed before the runs, in
  `report_inversion_batch.py`): RECOVERED = not failed, not at the iteration
  limit, centres in the fluid mesh, both permutation-matched centre errors
  ≤ 0.20 m.

## Saved results (`reference_results/`)

| path | content |
|---|---|
| `v3_bathroom_topsupply_20261116/report/` | flow report (Picard, outlet split, coarse vs fine) |
| `source_id_v2_wall8_20261116/` | dataset `REPORT.md`, `common/` (sensors, transport config, stability, settings selection), per-case `inversion_config.json`, `noise_record.json`, `truth_record.json`, validation results |
| `…/inversion_batches/first_full_sequential_20261117/` | `REPORT.md`, `report/summary.{csv,json}`, `resolved_settings.json`, per-run `inversion_summary.json`, `initialization.json`, `optimization_history.csv`, `paper_figures/FIGURES.md` |

Saved outcome: six of six runs converged before the iteration limit and are
RECOVERED, centre errors 0.020–0.070 m, amplitude errors −0.8 % to −1.8 %
(`REPORT.md`). Code: development commit `6f375e3` plus the then-untracked batch
scripts, all included here (checksums in `env/sha256_sources.txt`).

Regenerating the dataset from the supplied meshes and velocity fields with this
repository reproduces the saved clean data, noise and noisy data **bit for bit**
(checked for all three cases). The velocity *files* in `supplied/` differ from the
originals only in an embedded path string (`supplied_assets_manifest.json`), so the
dataset's recorded velocity SHA-256 changes accordingly; the arrays are identical.

## Not part of the paper experiment

`output/` versions `v1_nobeds_bottomsupply_20261116`, `v2_beds_bottomsupply_20261116`
and dataset `source_id_v1_20261116` (12-sensor `aisle_passage_12` layout) were
superseded and are not distributed. `wall_10` is a proposed, inactive layout.
Whether the remaining ~0.07 m centre offset (nearly identical for clean and noisy
data) comes from inverting fine-mesh data on the coarse mesh has not been tested.
