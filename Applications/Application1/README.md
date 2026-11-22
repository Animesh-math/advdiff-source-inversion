# Application1 — DC Navy Yard Source Identification

Application1 solves a deterministic transient advection–diffusion inverse
source-identification problem on the **DC Navy Yard** domain (building
footprints are holes in the mesh), driven by a steady **Navier–Stokes wind
field**. The pipeline: forward RK4 solver, fully-discrete RK4 adjoint gradient,
reduced objective, adjoint-sensitivity initialization, L-BFGS-B (optionally
GN-diagonal scaled), sigma-continuation, amplitude floor, and manual /
mesh_random / smart / adaptive initialization. The shared solvers in
`Src/Solvers/` and the reduced objective/gradient in
`Src/Objective/gradient_test.py` are geometry-agnostic (the mesh supplies all
geometry).

The inverse problem: recover the centers and amplitudes of M Gaussian sources
from the time-integrated concentration misfit at boundary segments or point
sensors. Control ordering is `[x1, y1, A1, x2, y2, A2, ...]`.

> **Paper experiments** (exact commands, settings, saved results):
> [`paper_experiments/README.md`](paper_experiments/README.md). This file is the
> full reference for the driver and its options.

## Coordinate units and scale (IMPORTANT)

The DC mesh uses a **LOCAL coordinate system in meters** whose origin is the
south-west corner of the outer computational domain:
`x ∈ [0, ≈302] m` and `y ∈ [0, ≈700] m`. The mesh was translated once from its
original projected (GIS) coordinates (`x ≈ 400033`, `y ≈ 133918`) by the
constant vector `(-400032.9721875289, -133918.0828324356)` — a pure rigid
translation, no scaling or rotation. `global = local + origin`; the origin is
recorded in `DC_mesh/dc_navy_yard_coordinate_origin.json` (and
`mesh_data_config.ORIGIN_X/ORIGIN_Y`) for reference only. The entire numerical
pipeline runs in local coordinates with no runtime origin subtraction.

All geometry-dependent parameters are in **physical meters**:

| Quantity | Application1 default (local meters) |
|---|---|
| Source center `x,y` | meters (`[0,302]×[0,700]`) |
| Gaussian `sigma2` (r²) | `400 m²` (e-folding 20 m) |
| Amplitude `A` | dimensionless `[0, 2]` (paper runs: `--amplitude-floor 0.1`) |
| Diffusivity `k` | `10 m²/s` (Pe ≈ 70) |
| Time step `dt` | `0.1 s` |
| Final time `tfinal` | `150 s` (`PDE["tfinal"]`; `--tfinal` overrides) |
| Smart min separation | `40 m` |
| Adaptive perturb radius | `15 m` |

The driver prints the runtime fluid-domain bounding box and every
geometry-dependent default at the start of each run (`=== Geometry /
Coordinate Scale ===`). Source `x,y` are in meters; amplitudes are
dimensionless emission strengths.

## Folder structure

```
Applications/Application1/
  DC_mesh/                         committed DC mesh + GIS inputs (do not delete)
    build_dc_mesh.py               rebuilds the mesh from the GeoJSON inputs (gmsh, geopandas)
    dc_navy_yard_mesh.msh          the Gmsh mesh used by every solver
    dc_navy_yard_*.geojson         domain + cleaned building footprints
    dc_navy_yard_building_metadata.json / _coordinate_origin.json
  Navier Stokes Solve/            standalone FEniCSx steady NS solve (see its README)
  output/                         generated, git-ignored
    navier_stokes/                PRIMARY VELOCITY (do not delete): *_velocity_nodes.npz
    <case-name>/<mode-folder>/     optimization runs, ParaView, history, summaries
    gradient_tests/  hessian_tests/  diagnostic outputs
  mesh_data_config.py             mesh loader + boundary tags (shared with NS solve)
  navier_stokes_config.py         NS wind/solver config (NS only)
  multi_source_utils.py           Gaussian source model, controls, bounds (DC units)
  mesh_admissibility.py           mesh point-location admissibility + sampling
  run_optimization.py             optimization driver (CLI)
  source_insertion_sensitivity.py / sensitivity_initialization.py   sensitivity initializer
  post_init_optimizers.py         post-initialization polish methods (--post-init-method)
  source_count_inference.py       adaptive source-count inference (--infer-num-sources)
  final_feasibility.py            final feasibility repair / certification
  hessian_utils.py                Hessian-vector products (diagnostic / Newton polish)
  inspect_mesh_nodes.py           offline mesh node-id lookup
  clean_outputs.py                safe output cleanup (never deletes velocity by default)
  run_*.sh                        batch experiment drivers
  test scripts/                   regression tests, Taylor gradient/Hessian tests,
                                  sensor-layout experiment helpers
    diagnostics/                  one-off diagnostic, sweep and report scripts
```

## Mesh, velocity, boundaries

- **Mesh:** `Applications/Application1/DC_mesh/dc_navy_yard_mesh.msh` (2795 nodes,
  4936 triangles). Facets: `dc_navy_yard_facets.xdmf`.
- **Velocity (primary):**
  `Applications/Application1/output/navier_stokes/application1_navier_stokes_velocity_nodes.npz`
  (keys `coords (2795,3)`, `velocity_nodes (2795,2)`, in the mesh node order;
  mean |u| ≈ 2.4 m/s, max ≈ 10.6 m/s; flows generally east/south). Auto-loaded
  by `run_optimization.py` unless `--constant-velocity` is given.
- **Boundary names and tags:**

  | Name | Tag | Cardinal | Role |
  |---|---|---|---|
  | `bottom` | 11 | south | outflow |
  | `top` | 12 | north | inflow |
  | `left` | 13 | west | **inflow (transport Dirichlet u=0)** |
  | `right` | 14 | east | outflow, **default observation** |
  | `building_1` … `building_21` | 101 … 121 | internal building walls | natural no-flux |

  Building numbering: `building_N` (N = 1..21) maps to tag `100 + N`. The
  `--measurement-boundaries` flag accepts these names and also zero-padded
  building aliases (`building_03` → `building_3`). Example selections:
  `--measurement-boundaries right`, `--measurement-boundaries building_3 building_7`,
  `--measurement-boundaries building_5 right`, `--measurement-boundaries right bottom`.
  (The building metadata JSON keys are zero-padded `building_01`..`building_21`;
  both forms map to the same tag.)

## Dependencies

The inverse-optimization pipeline runs in an ordinary Python virtual environment
(see `requirements.txt` at the repository root and "Exact commands" below).

**Required:** `numpy`, `scipy`, `matplotlib`, `meshio`.

**Optional, feature-gated:**

| Package | Gates | If missing |
|---|---|---|
| `osqp >= 1.0` | `--amplitude-solver osqp`, the DEFAULT joint amplitude solver of the sensitivity initialization | the driver exits at startup with the install command; `--amplitude-solver enumeration` needs no extra package |

Install the optional dependency into the same interpreter that runs
`run_optimization.py`:

```
"$PY" -m pip install 'osqp>=1.0'
```

Verify what the current interpreter can do (required + optional packages, plus a
one-QP OSQP smoke test):

```
"$PY" tools/check_inverse_env.py
```

The separate FEniCSx velocity-field environment has its own checker,
`tools/check_fenicsx_env.py` / `tools/check_fenicsx_conda_env.py`.

## Exact commands

Run everything from the repository root.

The inverse-optimization pipeline uses the Python interpreter of the inverse
environment (see `requirements.txt`):
```
PY="python3"   # or /path/to/venv/bin/python
```

### Navier–Stokes velocity rerun (FEniCSx environment, only if the velocity must be regenerated)
```
conda run -n fenicsx-env python \
    "Applications/Application1/Navier Stokes Solve/solve_application1_navier_stokes.py"
```

### Forward solve smoke test (writes true/initial state ParaView)
```
cd "Applications/Application1" && "$PY" run_optimization.py \
    --mode location_and_intensity --stage smoke --case-name app1_smoke \
    --tfinal 6 --max-paraview-frames 5
```

### Gradient (Taylor) test — expect R2 slope ≈ 2
```
cd "Applications/Application1" && "$PY" gradient_test_navier_stokes_velocity.py
```

### Hessian-vector verification (exact mode vs finite differences)
```
cd "Applications/Application1" && "$PY" hessian_test_navier_stokes_velocity.py
```

### Optimization (full recovery example, right+bottom outflow observation)
```
cd "Applications/Application1" && "$PY" run_optimization.py \
    --mode location_and_intensity --case-name app1_recovery \
    --measurement-boundaries right bottom \
    --true-control 37.0278 381.9172 1.10 267.0278 201.9172 0.85 \
    --initial-control 147.0278 381.9172 0.60 57.0278 561.9172 0.50 \
    --max-iter 150 --max-paraview-frames 50
```

Robust global recovery (multistart + globalizers):
```
cd "Applications/Application1" && "$PY" run_optimization.py \
    --mode location_and_intensity --case-name app1_recovery_adaptive \
    --measurement-boundaries right bottom top \
    --init-mode adaptive --random-seed 0 \
    --sigma2-continuation 1600 800 400 --amplitude-floor 0.05 \
    --max-iter 150 --max-paraview-frames 50
```

### Discrete mesh-node sensors (`--observation-mode nodes`)

The default is `--observation-mode boundary`: the historical boundary-integral
misfit over `--measurement-boundaries`, unchanged. `nodes` observes the FE state
at discrete mesh nodes, `C_s U^n = (U^n_j1, …, U^n_jNs)`, with the synthetic data
`d_s^n = C_s U_true^n` taken at exactly those nodes. The temporal quadrature is
the existing one, and the discrete adjoint is forced with the exact transpose
`eta1 * C_s^T W_s (C_s U^n − d_s^n)`.

Sensors by solver node id:
```
cd "Applications/Application1" && "$PY" run_optimization.py \
    --mode location_and_intensity --case-name app1_nodes_by_id \
    --observation-mode nodes --sensor-node-ids 2456,2697,644,965,1762,1127 \
    --measurement-boundaries left right bottom \
    --true-control 162 300 1.40 270 600 0.95 \
    --tfinal 150 --max-iter 150 --max-paraview-frames 50
```

Sensors by coordinate, snapped to the nearest mesh node (safer than trusting
ParaView point numbering):
```
cd "Applications/Application1" && "$PY" run_optimization.py \
    --mode location_and_intensity --case-name app1_nodes_by_coords \
    --observation-mode nodes \
    --sensor-coordinates "10,300;295,300;150,10;100,250;150,400;200,620" \
    --sensor-snap-tolerance 20 \
    --measurement-boundaries left right bottom \
    --true-control 162 300 1.40 270 600 0.95 \
    --tfinal 150 --max-iter 150 --max-paraview-frames 50
```

Node ids take precedence when both are supplied. Weighting is explicit:
`--sensor-weights equal` (default, `||r||_W^2 = w * sum_s r_s^2` with
`--sensor-equal-weight`, default `1.0`) or `--sensor-weights boundary_quadrature`,
the regression configuration whose node set is every node of the selected
measurement boundaries and whose `W_s` is the assembled P1 boundary observation
matrix — a symmetric matrix with off-diagonal edge coupling, **not** a diagonal
nodal weighting, because the boundary quadrature is not representable that way.
Resolved ids, coordinates and weights are printed and stored in
`optimization_summary.json` under `observation`.

Resolve ParaView-picked coordinates to solver nodes:
```
cd "Applications/Application1" && "$PY" inspect_mesh_nodes.py --nearest "100,250;150,400"
cd "Applications/Application1" && "$PY" inspect_mesh_nodes.py --node-ids 41,82,164
cd "Applications/Application1" && "$PY" inspect_mesh_nodes.py --list-boundary right --limit 20
```

### Sensor locations in ParaView (`sensors.vtu`)

Every `--observation-mode nodes` run writes `output/<case>/<mode-folder>/sensors.vtu`,
a point cloud with one vertex cell per sensor and the point fields `sensor_id`
(0-based, in sensor order), `node_id` (solver mesh node), `weight` (the diagonal
weight, or the `W_s` diagonal in `boundary_quadrature` mode), `row_sum_weight`
(the `W_s` row sum, which equals `weight` for diagonal weighting), and `x` / `y`.
Open it alongside `source_fields.vtu` and use Glyph → Sphere to see where the
sensors sit relative to the buildings and the recovered sources. The same ids,
coordinates and weights remain in `optimization_summary.json` under `observation`;
the `.vtu` only makes them visible. Boundary mode writes no `sensors.vtu` — it has
no discrete sensor set.

### Joint amplitude solver (`--amplitude-solver`)

For FIXED source centers the sensitivity initialization fits all amplitudes
jointly by variable projection. That fit is the box-constrained QP

```
min_{A_min <= a <= A_max}   0.5 a^T H a - a^T h
H[i,j] = <du_ci, du_cj>_obs + lambda*delta_ij        (Gram of the unit responses)
h[i]   = <du_ci, u_d>_obs
```

in the exact weighted boundary/node observation and time inner product the
objective itself uses. `--amplitude-solver` chooses how that QP is solved; it
changes nothing else — same `H`, same `h`, same observation and time weights,
same amplitude bounds, same `LAMBDA_REG` Tikhonov term.

```
--amplitude-solver osqp          (default) OSQP; no source-count cap
--amplitude-solver enumeration   exact 3^n active-set enumeration; n <= 8
```

`osqp` maps the same problem onto `min 0.5 x'Px + q'x` s.t. `l <= Ax <= u` with
`P = H` (explicitly symmetrized, since OSQP reads only the upper triangle),
`q = -h`, `A = I`, and `l`/`u` the amplitude bounds. `eps_abs = eps_rel = 1e-10`,
`max_iter = 20000`, polish on.

An answer is accepted on a **verified optimality residual, not on OSQP's status
label** — measurement showed the label is unreliable in both directions on
badly scaled Gram matrices (`solved` for a point with a ~1e-3 scaled box-KKT
residual; `solved inaccurate` for one optimal to ~1e-11). So the status must be
`solved` or `solved inaccurate`, the iterate must be within `1e-8` of the box
(then clipped onto it), and the scaled projected-gradient residual
`‖a − P(a − (Ha − h))‖_inf / max(1, ‖h‖_inf)` must be `<= 1e-7`
(`OSQP_KKT_TOL`). **No regularization is added and there is no fallback**: a
failure at any of those steps raises, reports the measured residual and
`cond(H)`, and names `--amplitude-solver enumeration` as the explicit
alternative. On extremely ill-conditioned problems (`cond(H) >= 1e22`) OSQP
refuses roughly 0.5% of the time where enumeration would still answer; the
refusal is loud, and no measured Application1 configuration reaches that regime.
OSQP's `verbose` flag does not suppress its polish notes, so they are muted at
the file-descriptor level (`OSQP_SUPPRESS_SOLVER_STDOUT`); the status and
iteration count are still read programmatically and reported in any error.

Cost per fit, measured (min-of-N): enumeration `0.113 / 0.387 / 1.218 / 3.815 /
11.389 / 36.025 / 104.958` ms at M = `2..8`; OSQP `0.35-0.46` ms **flat** for
M = `2..10`. The crossover is M = 3. At M = 2 OSQP costs `0.25` ms more per fit,
immaterial against the ~1 s forward solves a beam init pays hundreds of times —
so the default loses nothing today and is the only option that reaches M > 8.

`enumeration` is the original solver and stays available unchanged. It is exact
and dependency-free, but it enumerates `3^n` active sets and refuses `n > 8`
(`_ENUMERATION_MAX_SOURCES`); that cap is a property of the enumeration solver
only and does not apply to OSQP, which is what makes the intended 8-10 source
amplitude problems reachable.

The selection reaches the level expansion, the variable-projected
center-refinement trials, and the second-source re-selection (at the
initialization width and at `--sensitivity-reselect-at-target-width`). It does
NOT touch the closed-form single-source RANKING amplitude
`a*(c) = -G(c)/kappa(c)`, which is unchanged. It is recorded in
`sensitivity_init_summary.json` under `options.amplitude_solver`.

Validation (joint fitting only, no initialization / center search / full
optimization):
```
cd "Applications/Application1/test scripts" && "$PY" test_amplitude_solver_osqp.py
```
See `docs/amplitude_solver_osqp_validation.md` for the measured comparison on the
two-source case.

### Post-initialization optimizer (`--post-init-method`)

Initialization (`--init-mode` / `--initial-control`) produces the starting
control; `--post-init-method` chooses what polishes it. Initialization itself is
completely unaffected by this flag.

```
--post-init-method lbfgsb        (default) historical L-BFGS-B, existing runs unchanged
--post-init-method gauss_newton  bound-constrained trust-region Newton, Gauss-Newton Hessian
--post-init-method newton        the same driver with the exact reduced Hessian
```

`gauss_newton` and `newton` reuse the existing `hessian_utils` pipeline: the
dense reduced Hessian is assembled by coordinate Hessian-vector products, each
one tangent forward solve plus one incremental RK4 discrete adjoint solve, so a
`3M x 3M` Hessian costs `3M` solve pairs. `gauss_newton` keeps only the
residual-through-state term; `newton` (`hessian_utils` `mode="exact"`)
additionally contracts the base adjoint with `D^2 S(z)`. The Hessian is handed
to SciPy `trust-constr`, the one SciPy method that takes both a user Hessian and
the same bounds L-BFGS-B is given. They pay much more per iteration and are
meant for polishing an already-good initialization, not for global search.

Because they are strictly more expensive per iteration, they are only worth it
when the start is close enough that the iteration count collapses. Compare the
three from a saved initialization with:
```
cd "Applications/Application1/test scripts/diagnostics" && "$PY" diagnose_post_init_polish.py --max-iter 200
```

Polish from a saved sensitivity initialization with Gauss-Newton:
```
cd "Applications/Application1" && "$PY" run_optimization.py \
    --mode location_and_intensity --stage full --num-sources 2 \
    --case-name node_sensors_right_bottom_top_c06_gn_from_init \
    --observation-mode nodes --sensor-weights equal --sensor-equal-weight 1.0 \
    --sensor-node-ids 641,644,647,650,653,656,659,662,665,668,671,674,677 \
                      582,585,588,591,594,596 693,690,687,684,681,679 \
    --measurement-boundaries right bottom top \
    --feasibility-mode raw --amplitude-floor 0.1 \
    --true-control 162 300 1.40 270 600 0.95 \
    --initial-control 270.0042837605424 600.051691697479 0.9510994778671144 \
                      161.8662190981801 300.00986383727934 1.3940636135616904 \
    --post-init-method gauss_newton --max-iter 200
```

`optimization_summary.json` records `post_init_method` and a `post_init` block
(iterations, forward/adjoint solve counts, `nfev`/`njev`/`nhev`) for every
method. The Newton-type methods additionally write `post_init_summary.json` with
per-Hessian curvature diagnostics (eigenvalues, condition number, definiteness).

### Adaptive source-count inference (`--infer-num-sources`)

Infers M from the data instead of taking it from `--num-sources`, by driving the
existing sequential adjoint-sensitivity machinery one source at a time. For each
m it polishes the m-source model, scores the best next-source birth on the
residual, inserts and polishes the candidate, and accepts or rejects with a
recorded reason.

**Why a naive test is invalid.** The m-source model is nested inside the
(m+1)-source model (set the new amplitude to zero), so

```
J_{m+1}* <= J_m*     always, for any data and any spurious candidate
```

A decrease is a theorem, not evidence. Two consequences are enforced:

* **Amplitude lower bound 0 during inference.** Every candidate model is
  polished with amplitude lower bound `0`, never the production
  `--amplitude-floor`. With a floor a spurious source cannot shrink to zero — it
  pins at the floor and its non-zero amplitude looks like proof. The production
  floor is recorded as provenance only.
* **A threshold plus a non-degeneracy check.** A source is accepted only when the
  actual decrease exceeds `max(--source-count-abs-reduction-tol,
  --source-count-rel-reduction-tol * J_m)` **and** no source in the polished
  model has amplitude `<= --source-count-amp-tol`. The amplitude check uses the
  minimum over all m+1 amplitudes, so it is invariant to the relabelling and
  movement the polish is free to do.

The predicted decrease is the exact bounded 1-D quadratic
`DeltaJ(c, a) = a G(c) + 0.5 a^2 kappa(c)` minimized over `a` in `[0, A_max]`, so
a level is abandoned before paying for a polish when the best possible birth is
already negligible.

Stop reasons: `discrepancy_floor`, `max_sources_reached`,
`no_admissible_candidate`, `negligible_predicted_reduction`,
`rejected_insufficient_actual_reduction`, `rejected_amplitude_collapse`.

```
cd "Applications/Application1" && "$PY" run_optimization.py \
    --mode location_and_intensity --stage full \
    --infer-num-sources --max-sources 4 \
    --observation-mode nodes --sensor-weights equal --sensor-equal-weight 1.0 \
    --sensor-node-ids 641,644,647,650,653,656,659,662,665,668,671,674,677 \
                      582,585,588,591,594,596 693,690,687,684,681,679 \
    --measurement-boundaries right bottom top \
    --feasibility-mode raw --amplitude-floor 0.1 \
    --true-control 162 300 1.40 270 600 0.95 \
    --case-name node_sensors_right_bottom_top_c06_infer_m \
    --source-count-top-k 12 --source-count-polish-max-iter 150 --max-iter 200
```

`--num-sources` is ignored. `--true-control`, when given, is used only to report
permutation-matched recovery errors, and only when the inferred and true counts
agree. Results land in `source_count_inference_summary.json` (per-level
`J_before`, candidate, `predicted_decrease`, `J_after_candidate`,
`actual_decrease`, `accepted`, `reason`, amplitudes, polish cost; plus the final
inferred M, control, recovery errors and sensor metadata), and the inferred
control then runs through the ordinary pipeline under the production bounds.

Reference result on the recovered c06 case (25 node sensors, right+bottom+top):

| level | J_before | predicted ↓ | J_after | actual ↓ | min A | result |
|-------|----------|-------------|---------|----------|-------|--------|
| 1 | 3.800e+05 | 3.633e+05 | 6.112e+03 | 3.739e+05 | 0.9499994 | accept |
| 2 | 6.112e+03 | 5.358e+03 | 1.150e-14 | 6.112e+03 | 0.9500000 | accept |
| 3 | 1.150e-14 | 5.061e-15 | 6.332e-15 | 5.167e-15 | 9.699e-11 | **reject** |

Level 3 is the point of the design: the spurious third source *did* reduce J and
*did* clear the relative threshold (by ~450x), and was caught only by the
amplitude-collapse rule that the zero lower bound makes meaningful.

Tests:
```
cd "Applications/Application1/test scripts" && "$PY" test_source_count_inference.py
cd "Applications/Application1/test scripts" && "$PY" test_source_count_inference.py --unit-only
```

### Known weaknesses in source-count / optimizer feasibility reporting (2026-08-24, items 1/2/5 fixed 2026-08-24)

Originally documented as five open weaknesses. Items 1, 2 and 5 are now FIXED
(5 in the same pass that wrote this section; 1 and 2 in a follow-up pass the
same day — see "Source-count bookkeeping/reporting fixes" below). Items 3 and
4 remain open and are recorded here so they are not rediscovered from
scratch.

1. **~~Some reporting paths are not fully source-count aware.~~ FIXED.**
   `write_run_outputs()` used to derive `num_sources` (and therefore
   `control_dim`, component headers, and the reported optimized-control
   shape) from `true_control`, and called `match_sources_by_permutation`
   unconditionally. Both broke when `--infer-num-sources` returned an `M`
   different from `--true-control`'s `M` (real applications don't know the
   truth at all). Dimensions/headers now always follow `final_control`;
   permutation matching is wrapped in `safe_match_sources_by_permutation`,
   which returns `(None, reason)` instead of raising when the two `M`s
   disagree, at every call site that is scoring/reporting rather than
   internal-use. `match_sources_by_permutation` itself is unchanged and still
   raises for a direct caller that needs the hard guarantee.
2. **~~`source_count_inference_summary.json` can be written, then a
   crash.~~ FIXED.** The inferred control is now certified/repaired (when
   `--final-feasibility-polish` is set) and validated BEFORE
   `source_count_inference_summary.json` is written, not after. If it is
   still infeasible, the summary records why
   (`inferred_control_feasible`, `inferred_control_validation_error`,
   `skipped_final_optimization_reason`) and the driver returns gracefully
   instead of raising -- no crash, and no `optimization_summary.json` is
   expected for that run.
3. **`--feasibility-mode raw` can exploit Gaussian tails from inadmissible
   centers.** Raw mode is a valuable diagnostic/search mode — see the raw-mode
   recovery study above — precisely because it lets a source center cross a
   building wall during intermediate iterations. But a center deep inside a
   building or off the mesh entirely still contributes a real (if small)
   Gaussian tail into the fluid domain, so a `raw` search can settle into a
   basin whose "true" optimum uses inadmissible geometry to explain the data.
   Raw mode reports this rather than hiding it (the final-source diagnostic
   prints exact outside-mesh distances), but nothing certifies or repairs it.
4. **Projection mode can stall.** In `--feasibility-mode project`, many
   distinct invalid optimizer variables can map to the same projected fluid
   point (the projection is not injective near a wall), which can flatten the
   locally-observed objective and stall the line search — this is the
   documented mechanism behind the Case 3b `project`-mode
   `ABNORMAL_TERMINATION_IN_LNSRCH` stall (see the raw-mode diagnostic study
   above).
5. **Final accepted controls had no feasibility certification step.** Until
   this pass, nothing checked — let alone repaired — whether the FINAL
   accepted/reported control (from any feasibility mode) actually sits inside
   the fluid mesh with any real clearance. `penalty`/`project` modes usually
   end admissible by construction, and `raw` mode's own diagnostic only
   reports the failure. **This is the one weakness this pass fixes** — see
   `--final-feasibility-polish` immediately below.

**New design.** `raw` remains available for search/basin discovery — that
does not change. What changes is the final step: the final accepted source
control, when `--final-feasibility-polish` is requested, must pass a
feasibility-constrained repair/certification step before it is reported.
Item 4 above remains open (no projection-mode fix is part of this work).
No source-count RETRY logic and no change to the numerical model-order
acceptance rule are part of this work either -- see "Source-count
bookkeeping/reporting fixes" immediately below for what item 1/2 fixing
actually changed (reporting/bookkeeping only).

### Source-count bookkeeping/reporting fixes (2026-08-24)

Fixes items 1 and 2 above. `true_control` is a synthetic-benchmark SCORING
input only, everywhere in this codebase: it must never decide the reported
source count, control dimension, headers, or run validity, because a real
application does not know the truth at all.

* **`write_run_outputs()` is source-count aware.** `final_num_sources =
  infer_num_sources(final_control)` now drives `num_sources`, `control_dim`,
  and the history-CSV component headers -- never `true_control`.
  `optimization_summary.json` additionally reports `final_num_sources`,
  `initial_num_sources`, `true_num_sources` (scoring only) and
  `inferred_matches_true_count`.
* **`safe_match_sources_by_permutation(true_control, candidate_control)`**
  wraps `match_sources_by_permutation` for every scoring/reporting call site
  (final recovery, initial recovery, post-init summary recovery,
  source-count-inference recovery, the final console summary): it returns
  `(None, "Permutation recovery is undefined because true M=... but candidate
  M=...")` instead of raising when the counts differ, so a real count
  mismatch downgrades to an honest "undefined" report instead of crashing the
  run. `match_sources_by_permutation` itself still raises for any direct
  caller that needs the hard guarantee (e.g. a caller that has already
  checked the counts match).
* **Final feasibility now runs before the source-count-inference summary is
  written, and a still-infeasible inferred control no longer crashes the
  run.** When `--final-feasibility-polish` is set, the inferred control from
  `--infer-num-sources` is certified/repaired BEFORE
  `validate_control_for_region` and before `source_count_inference_summary.json`
  is written. The summary records `final_feasibility_polish`,
  `inferred_control_before_final_feasibility`,
  `inferred_control_after_final_feasibility`, `inferred_control_feasible`,
  and `inferred_control_validation_error`. If the control is still
  infeasible (polish disabled, or polish couldn't certify it in budget), the
  summary is written with `skipped_final_optimization_reason` set and the
  driver returns -- no `optimization_summary.json`, no crash, no traceback.
* **`max_sources_reached` is marked unresolved, not a clean decision.**
  `source_count_inference.py` adds `source_count_decision_status`
  (`"accepted"` when the discrepancy floor was reached, `"rejected_stop"`
  when a next-source candidate was tested and rejected on its merits,
  `"unresolved_max_sources"` when `--max-sources` cut the search off before
  the acceptance rule ran again) and `source_count_decision_converged`
  (`False` only for `unresolved_max_sources`) to
  `source_count_inference_summary.json`. This is reporting/status semantics
  only -- the numerical acceptance rule is unchanged, and a reported
  `inferred_num_sources` under `unresolved_max_sources` is a LOWER BOUND, not
  a decision.
* **Report scripts classify a missing final summary instead of calling it a
  generic optimizer failure.** Both `report_source_count_inference_sweep.py`
  and `report_source_count_overnight_sweep.py` still show the inference
  result (inferred `M`, true `M` if available, stop reason, decision status,
  final inference objective) when `optimization_summary.json` is missing, and
  classify WHY via `classify_missing_final_summary`: `validation_failure`,
  `infeasible_final_control` / `infeasible_final_control_after_polish`,
  `permutation_count_mismatch`, `max_sources_reached_unresolved`, or (only as
  a last resort, for summaries predating this fix that carry none of the new
  fields) `unknown_traceback` / `unknown_traceback_pre_source_count_fix`.

Verified against the real (pre-fix) overnight sweep archived in
`output/sc_overnight_*` (see "Verification" below): every one of the 6/8
cases missing `optimization_summary.json` is now classified as either
`max_sources_reached_unresolved` (2 cases) or `permutation_count_mismatch` (4
cases) -- confirming both the diagnosis and the fix, with no PDE solves
re-run.

### Final feasibility repair/certification (`--final-feasibility-polish`)

A final, short, opt-in certification/repair step that runs AFTER normal
optimization (and any `raw`-mode search/continuation) has already produced a
candidate control. It does not replace the main optimizer, does not change
`--infer-num-sources`, and does not use `--true-control` to decide anything —
off by default, so existing behavior (including `raw` mode's inadmissible
final centers) is unchanged unless the flag is given.

**Mechanism.**

* `mesh_admissibility.py` gains `MeshAdmissibility.signed_clearance(x, y)`:
  positive strictly inside the fluid domain, `0.0` on the boundary, negative
  outside it (building hole or off-mesh), built from the existing
  `is_point_in_domain` / `distance_to_boundary` / `nearest_boundary_point`
  machinery. (`nearest_domain_distance_sq` is deliberately NOT used for this —
  it is identically zero everywhere inside the fluid, so it cannot express
  "how much clearance is left before the nearest wall.")
* `final_feasibility.py` (new module) provides `final_feasibility_polish`: for
  a flat control `[x1, y1, A1, ..., xM, yM, AM]`, it certifies feasibility
  (every `signed_clearance(x_i, y_i) >= margin`) with **zero repair** if
  already satisfied; otherwise it projects any center OUTSIDE the fluid mesh
  once (`MeshAdmissibility.project_into_domain`) to form a starting point, then
  runs a short SciPy `SLSQP` solve with `signed_clearance(x_i, y_i) - margin
  >= 0` as an explicit inequality constraint per source, keeping the
  production amplitude bounds. The objective/gradient it polishes against are
  built with `feasibility_mode="raw"` (via the run's own `build_evaluator`)
  regardless of the run's actual `--feasibility-mode`, so every intermediate
  SLSQP iterate — feasible or not — has a well-defined true objective;
  admissibility is enforced only through the explicit clearance constraint,
  not by the evaluator.
* It returns the repaired control, a `feasible` flag, objective before/after,
  min/max signed clearance before/after, iteration count, the SciPy message,
  and whether the projection step actually moved a center.

**CLI flags** (all default to preserving existing behavior):

| flag | default | meaning |
|------|---------|---------|
| `--final-feasibility-polish` | off | enable the final repair/certification step |
| `--final-feasibility-margin` | `0.5` m | required signed clearance for every source center |
| `--final-feasibility-max-iter` | `50` | iteration cap for the repair solve (a short polish, not the main optimizer) |
| `--final-feasibility-method` | `slsqp` | SciPy constrained method (only `slsqp` today) |

**Integration.** In `run_optimization.py`, the polish runs once the
sigma2-continuation loop has produced its `final_control` (after the existing
`raw`-mode diagnostic / `validate_control_for_region` handling), using
`location_and_intensity`-style full bounds and evaluator regardless of the
run's own `--mode`, so it can move ANY source center to repair feasibility.
The repaired control then becomes the control the rest of the pipeline
reports (final objective/gradient/state, ParaView export, permutation-matched
recovery, `optimization_summary.json`). The result record is written to
`optimization_summary.json` under `final_feasibility_polish` (`null` when the
flag is not given).

```
cd "Applications/Application1" && "$PY" run_optimization.py \
    --mode location_and_intensity --stage full \
    --true-control 260 300 1.55 60 180 1.10 \
    --feasibility-mode raw \
    --final-feasibility-polish --final-feasibility-margin 0.5 \
    --final-feasibility-max-iter 50 \
    --case-name app1_final_feasibility_example
```

Test: `test scripts/test_final_feasibility_polish.py` runs the repair directly
(no CLI, small `max_iter=15`) from a two-source control whose first center is
placed deep inside a building hole, and asserts every repaired source center
ends inside the fluid mesh and the repair is certified feasible.
```
cd "Applications/Application1/test scripts" && "$PY" test_final_feasibility_polish.py
```

### Sensor-layout comparison (2 truths x 3 node-sensor layouts)

Six-case controlled experiment: truths c06 and c07, each with a boundary-only,
a mixed boundary + near-building, and a deliberately weak 25-sensor layout,
equal weights, fixed `sigma2 = 400` (no continuation), production sensitivity
preset + OSQP + L-BFGS-B. Settings are recorded in each experiment directory.
```
cd "Applications/Application1"
EXP="$PWD/output/sensor_layout_comparison_$(date +%Y%m%d_%H%M%S)"
"$PY" "test scripts/sensor_layout_prepare.py" --exp-dir "$EXP"     # freeze layouts + plots (no PDE solve)
setsid nohup ./run_sensor_layout_comparison.sh "$EXP" < /dev/null > "$EXP/batch_driver.log" 2>&1 &
tail -n 20 "$EXP/batch_progress.log"                                # status
```
The batch writes `manifest.json`, per-case `cases/<id>/` (logs, instrumentation
with phase-split PDE solve counts and timings, recovery report, layout /
reconstruction / convergence figures), `summary_partial.*` after each case, and
`summary.{json,csv}` + `REPORT.md` at the end. `run_sensor_layout_comparison.sh`
hard-codes the Linux workstation's absolute paths.

### Output cleanup (NEVER deletes the Navier–Stokes velocity by default)
```
cd "Applications/Application1" && "$PY" clean_outputs.py --dry-run --yes    # preview
cd "Applications/Application1" && "$PY" clean_outputs.py --yes              # delete generated outputs
cd "Applications/Application1" && "$PY" clean_outputs.py --case-name app1_recovery --yes
```
A full reset that also removes the expensive velocity requires the explicit
`--include-navier-stokes` flag (loudly warned).

## Notes and assumptions

- The shared forward/adjoint solvers impose the homogeneous Dirichlet
  (clean-inflow) condition on the boundary named **`left`** only (here the west
  wall; Application2 aliases its supply vents to this name). West is the dominant physical inflow. The
  north (`top`) inflow uses the natural do-nothing condition.
- Admissibility is decided purely by mesh point-location: a source center is
  admissible iff it lies in some fluid triangle, which exactly excludes the 21
  polygonal building holes and the exterior. No analytic building geometry is
  used.
- Sensor placement and `tfinal` strongly affect observability: with too short a
  `tfinal` the plume never reaches the outflow boundaries and sources become
  unobservable (every control collapses to the same zero-source misfit floor).
  Use the default `tfinal` (or larger) for real recovery.
