# Application2 — Three-bed room with bathroom (idealized 2D)

The room experiment uses a steady, two-dimensional airflow with a spatially
constant effective viscosity. It provides an idealized setting for evaluating
source identification rather than a validated prediction of hospital
ventilation.

The ventilation arrangement (ceiling supplies above the beds; air leaving
through the corridor door and a bathroom exhaust) is **inspired by** Fig. 24 and
Sec. 7.1 of Lohner et al., [arXiv:2006.13792](https://arxiv.org/pdf/2006.13792).
This three-bed room is an adaptation, not a reproduction; the paper's bathroom
ceiling exhaust is represented in 2D by a boundary outlet. The constant
effective viscosity is a modelling choice of this experiment, not something
taken from or justified by that paper or by the Chen–Xu model. No clinical
validation is claimed.

**Scope** (flow version `v3_bathroom_topsupply_20261116`, dataset
version `source_id_v2_wall8_20261116`): geometry, coarse and fine meshes, their
steady Navier–Stokes velocity fields, the source-identification **datasets**
(three two-source cases, eight wall-mounted sensors, clean and 1 % noisy
fine-mesh observations) and the coarse-mesh reconstructions (sensitivity
initialization + GN-diagonal-scaled L-BFGS-B) of all six case/data combinations.
**Paper reproduction, settings and saved results:
[`paper_experiments/README.md`](paper_experiments/README.md).** The meshes and
velocity fields used in the paper are supplied in `supplied/` (install with
`python tools/install_supplied_assets.py` from the repository root).

## Geometry (top-down plan, metres) — `room_geometry_config.py`

Reference drawing: `room_layout_reference.png` (not to scale). The coordinates
below define the computational geometry; `report/fig_geometry_vs_reference.png`
shows the two side by side.

- Fluid domain = main room `[0,9]×[0,6]` ∪ bathroom annex `[9,11]×[0,2]`,
  minus the solids:
  - partitions (0.10 m, from the top wall): `[2.95,3.05]×[3,6]`, `[5.95,6.05]×[3,6]`
    → bays 1–3 above a shared aisle `0 < y < 3`;
  - **solid beds** (holes in the mesh, heads toward the top wall):
    `[1,2]×[3.5,5.6]`, `[4,5]×[3.5,5.6]`, `[7,8]×[3.5,5.6]`;
  - bathroom wall (0.10 m): `[8.95,9.05]×[0,0.6]` and `[8.95,9.05]×[1.4,2]`;
    the gap `0.6 < y < 1.4` is an **open interior doorway** (fluid, no BC).
- Fluid area 50.98 m². Beds are an idealized 2D representation of impermeable
  obstacles; pillows/bedding are drawing only.
- Differences from the drawing (not substantive; same topology): the drawing
  places the bathroom's inner wall at x ≈ 8 m inside the 9 m span, whereas the
  coordinates put the annex entirely at x ≥ 9 with its wall on x = 9; the upper
  bathroom-wall piece protrudes 0.05 m into the main room, leaving a 5 cm wall
  step at y = 2, x ∈ [8.95, 9.0] (meshed and refined, not a slit).

| tag | name | condition |
|---|---|---|
| 1 | fluid | — |
| 21, 22, 23 | inlet_1..3: top wall y = 6, x ∈ [1.2,1.8], [4.2,4.8], [7.2,7.8] | Dirichlet, downward parabolic profile |
| 31 | outlet_door: corridor doorway, y = 0, x ∈ [0.6,1.6] | natural outflow `(ν∇v − pI)n = 0` |
| 32 | outlet_exhaust: bathroom exhaust, x = 11, y ∈ [0.7,1.3] | natural outflow `(ν∇v − pI)n = 0` |
| 41–44 | wall_bottom, wall_right, wall_top, wall_left (main room) | no slip |
| 45–47 | bath_bottom, bath_right, bath_top (annex) | no slip |
| 51, 52 | partition_1, partition_2 | no slip |
| 61, 62, 63 | bed_1, bed_2, bed_3 | no slip |
| 71, 72 | bath_wall_lower, bath_wall_upper (all exposed faces incl. doorway jambs, 5 cm step) | no slip |

All no-slip tags form `WALL_TAG_LIST`.

## Flow model and parameters — `navier_stokes_config.py`

```text
-ν Δv + (v·∇)v + ∇p = 0,   ∇·v = 0        (steady, kinematic p, vector-Laplacian viscosity)
```

- Taylor–Hood P2/P1, Picard iteration (iteration 1 = Stokes), MUMPS LU.
  Tolerance 1e-8 (relative L2 update), max 100 iterations. Same for both meshes.
- **Supplies:** downward parabola on each 0.60 m opening, zero at the endpoints,
  **mean** speed `INLET_MEAN_SPEED = 0.10 m/s` (peak 0.15 m/s); 0.06 m²/s per
  supply per metre depth.
- **Outlets:** natural `(ν∇v − pI)n = 0` on both the corridor door and the
  bathroom exhaust. No outlet pressure value, no outlet velocity, no pressure
  pinning or mean constraint: the door/exhaust split and any backflow are
  computed and reported. Passive natural outlets do not represent mechanical
  suction or a negative-pressure room.
- **Viscosity:** `NU = 2.3244e-3 m²/s`, spatially constant **effective**
  viscosity (modelling parameter); not molecular air viscosity
  (≈1.5e-5 m²/s), not a tracer diffusivity.
- Re_slot = U_mean·b/ν = 25.8 (b = 0.60 m); Re_room = U_mean·6 m/ν = 258.
- Approximate air changes ≈ 12.7 1/h on the 50.98 m² fluid area (11.2 1/h on
  the gross 58 m²), only under the assumptions in `navier_stokes_config.py`
  (height-averaged plan field, full-height openings, uniform supply).
- The parameters were selected on the v1 geometry (no beds, bottom supply;
  trials in `output/v1_nobeds_bottomsupply_20261116/trials_20261116_picard_viscosity/`)
  and are retained unchanged; stability has not been re-assessed for this layout.

## Meshes — `Room_mesh/build_room_mesh.py`

Same geometry and tags for both levels; fine = half of every coarse size
(bulk; supplies/outlets; partition, bed and bathroom-wall faces; opening
endpoints and every solid corner, incl. partition tips, bed corners, doorway
jambs and the 5 cm step). Gmsh, single-threaded, deterministic.

| level | purpose | lc_bulk | vertices | triangles | P2 velocity dofs | P1 pressure dofs |
|---|---|---|---|---|---|---|
| coarse | inversion | 0.20 m | 4381 | 7887 | 33302 | 4381 |
| fine | synthetic data generation | 0.10 m | 15231 | 28772 | 118472 | 15231 |

The build aborts unless: every boundary edge is tagged and each tag's length
matches the geometry; the mesh area equals 50.98 m²; the triangles form one
edge-connected component with exactly four boundary loops (outer + 3 beds); no
triangle centroid lies inside a solid or outside room ∪ annex; no sample point
inside a solid or above the annex is covered; every passage (around each bed,
the aisle, the bathroom doorway, the bathroom) is fully covered by triangles;
and DOLFINx reads back the same facet count per tag as Gmsh wrote.

## Running

```bash
bash Applications/Application2/run_application2_setup.sh launch   # detached (setsid nohup), survives SSH logout
bash Applications/Application2/run_application2_setup.sh status   # progress check
```

Output goes to `output/<OUTPUT_VERSION>/` (set in `room_geometry_config.py`);
the script refuses to run if that version's `coarse/` or `fine/` exists. It
builds both meshes, runs the two flow solves concurrently and independently (no
interpolation between meshes), then writes the report. Individual steps
(fenicsx-env python; default `--out-root` is the versioned folder):

```bash
FX=/path/to/fenicsx-env/bin/python   # or: conda activate fenicsx-env; FX=python
$FX Applications/Application2/Room_mesh/build_room_mesh.py --level all
$FX "Applications/Application2/Navier Stokes Solve/solve_application2_navier_stokes.py" --level coarse
$FX "Applications/Application2/Navier Stokes Solve/solve_application2_navier_stokes.py" --level fine
$FX Applications/Application2/report_application2_flow.py
```

## Outputs — `output/`

```text
v3_bathroom_topsupply_20261116/      CURRENT
  coarse/mesh/          application2_room_coarse.msh, .vtu, .xdmf/.h5, _facets.xdmf/.h5, _mesh_info.json
  coarse/navier_stokes/ application2_navier_stokes_velocity_nodes_coarse.npz   transport velocity (P1 nodes)
                        application2_navier_stokes_full_fields_coarse.npz      P2 velocity / P1 pressure dofs
                        application2_velocity_pressure_P1_coarse.vtu           VTK export (velocity, pressure, speed)
                        application2_velocity_pressure_P1_coarse.xdmf/.h5, application2_velocity_P2_coarse.bp,
                        application2_pressure_P1_coarse.bp, application2_probes_coarse.npz,
                        application2_picard_history_coarse.csv, application2_navier_stokes_run_metadata_coarse.json
  fine/...              same for the fine mesh
  report/               README.md (concise report), report_summary.json, figures: geometry + tags, geometry vs
                        reference drawing, meshes (+ zooms), element edges around the bed holes, |v| + streamlines,
                        velocity arrows, pressure, Picard, coarse vs fine
  logs/ env/            pipeline + solver logs; git revision/diff, package lists, host, source checksums
  RUN_STATE RUN_COMMANDS.txt LAUNCH.txt pipeline.pid
v2_beds_bottomsupply_20261116/       superseded: solid beds, bottom supply / top extraction (README_VERSION.md)
v1_nobeds_bottomsupply_20261116/     superseded: no beds (+ parameter-selection trials)
```

### ParaView

Paths relative to `output/v3_bathroom_topsupply_20261116/`:

| file | content | reader |
|---|---|---|
| `<level>/navier_stokes/application2_velocity_pressure_P1_<level>.vtu` | velocity (3-comp.), pressure, speed at vertices | XML Unstructured Grid Reader (built in) |
| `<level>/mesh/application2_room_<level>.vtu` | mesh + tagged boundary lines, cell data `physical_tag` | XML Unstructured Grid Reader (built in) |
| `<level>/navier_stokes/application2_velocity_pressure_P1_<level>.xdmf` (+ `.h5`) | P1 velocity, pressure | Xdmf3ReaderS |
| `<level>/mesh/application2_room_<level>_facets.xdmf` (+ `.h5`) | DOLFINx facet tags | Xdmf3ReaderS |
| `<level>/navier_stokes/application2_velocity_P2_<level>.bp`, `application2_pressure_P1_<level>.bp` (directories) | native P2 velocity / P1 pressure | ADIOS2VTXReader (ParaView ≥ 5.12 with ADIOS2) |

Quick view: open the velocity `.vtu`, Apply, colour by `speed`; Filters → Glyph
(orientation `velocity`) for arrows or Stream Tracer for streamlines. Open the
mesh `.vtu` with "Surface With Edges" and colour by `physical_tag` to see the
bed holes, partitions, bathroom wall and openings.

## Using a velocity field (mesh association)

```python
import mesh_data_config as mdc
md  = mdc.build_mesh_data("coarse")               # transport mesh_data (meshio order), current version
vel = mdc.load_velocity_nodes("coarse", md)        # (N, 2); raises if the field is not from this mesh
```

The velocity NPZ stores the mesh SHA-256, geometry hash, level and node
coordinates; `load_velocity_nodes` refuses any mismatch (older versions are
therefore rejected by the current code). For the tracer solve the
clean-air Dirichlet names are `mdc.INFLOW_BOUNDARY_NAMES = ["inlet_1", "inlet_2", "inlet_3"]`.
The shared transport solver steps a dense matrix, practical for the coarse
mesh; fine-mesh data generation uses the same assembled operator with a sparse
matrix-vector product (`transport_model.SparseTransport`). In scripts that also
import Application1 modules, use `transport_model.mdc` instead of a bare
`import mesh_data_config` (both applications have a module of that name).

## Source admissibility (inverse problem)

Beds, partitions and the bathroom wall are holes/notches, so they are never
admissible:

```python
ok = mdc.point_in_fluid_mesh(md, pts)                 # inside a fluid triangle (authoritative)
ok = mdc.point_in_fluid_mesh(md, pts, clearance=0.1)  # plus >= 0.1 m from every boundary segment
geo.admissible_source_mask(x, y, clearance)            # analytic version (room_geometry_config)
```

## Source-identification datasets — `transport_config.py`

Tracer transport on the steady flow field, Application1's discretization and
source model (P1, lumped mass, explicit RK4; `Src/Solvers`), Application1 and
`Src/` unmodified:

```text
du/dt + v·grad(u) − k Δu = S(x; z),  u(0) = 0
u = 0 on the supply vents;  k grad(u)·n = 0 on walls/partitions/beds/bathroom wall and on both outlets
S(x; z) = Σ_i A_i exp(−|x − c_i|² / ρ),   z = [x1, y1, A1, x2, y2, A2]
```

| setting | value | note |
|---|---|---|
| tracer diffusivity `k` | 1.0e-2 m²/s | effective; NOT the flow viscosity ν = 2.3244e-3 m²/s; smallest candidate with cell Péclet ≤ 1 on the coarse mesh |
| source width `ρ` (`SIGMA2`) | 0.04 m² | e-folding radius 0.20 m; ρ = 2σ²; Gaussian not normalized, A = peak release rate |
| horizon `T` | 300 s | about one nominal air-change time |
| time steps | coarse 0.015 s, fine 0.0025 s | RK4, dt·ρ(M⁻¹K) ≤ 2.5 checked per mesh and velocity field |
| observation interval | 0.015 s | every coarse step (the discrete adjoint needs `jprint = 1`), every 6th fine step |
| amplitude bounds | [0.1, 2.0] | |
| objective | J = (η₁Δt/2) Σₙ₌₁ᴺ ‖C Uⁿ − dⁿ‖²_W | η₁ = 1, W = I, n = 0 excluded; W configurable |

- **Releases** (one per bed, reused in every case; stationary tracer releases
  beside the bed heads, not resolved patient mouths): bed 1 `(2.475, 5.2)` A = 1.0,
  bed 2 `(5.475, 5.2)` A = 0.7, bed 3 `(8.5, 5.2)` A = 1.3. Cases `12`, `13`, `23`.
- **Sensors** (decision 2026-11-16): the eight-sensor wall-mounted layout `wall_8`,
  used for data generation and for the optimization, identical in all cases, never
  adjusted using the true sources or recovery outcomes. S1 left wall of bay 1 (upper),
  S2 partition 1 bay-1 face (lower), S3 partition 1 bay-2 face (upper), S4 partition 2
  bay-2 face (lower), S5 partition 2 bay-3 face (upper), S6 right wall of bay 3 (lower),
  S7 bottom aisle wall just right of the corridor door, S8 bathroom top wall near the
  exhaust. Upper/lower = 3/4 and 1/4 of the bed length (y = 5.075 / 4.025). Each wall
  point is moved 0.10 m into the fluid and snapped to the nearest interior coarse-mesh
  node (actual wall offsets 0.09–0.21 m, recorded in `common/sensors.csv`); the fine
  mesh is sampled at the same coordinates by P1 interpolation.
  *Proposed, not active:* `wall_10` adds two sensors on the bottom aisle wall, evenly
  spaced between S7 and the bathroom wall (needs a new `DATASET_VERSION`).
  *Superseded:* `aisle_passage_12`; dataset `output/source_id_v1_20261116/` was generated
  with it and is preserved with a `SUPERSEDED_SENSOR_LAYOUT.md` flag.
- **Noise**: independent Gaussian, s = 0.01 × global RMS of the clean samples in
  the objective, one saved realization per case (seeds in `transport_config.py`).
- The shared solvers hard-code the Dirichlet boundary name `"left"`;
  `transport_model.py` aliases it to the three supply vents (it is not the wall x = 0).

```bash
bash Applications/Application2/run_application2_source_id_setup.sh launch   # detached: selection record, data, validation, report
bash Applications/Application2/run_application2_source_id_setup.sh status
```

| file | role |
|---|---|
| `transport_config.py` | all transport / source / sensor / noise / objective settings and their rationale |
| `transport_model.py` | mesh_data + velocity adapter, sparse RK4 with the shared assembly, sensor snapping, P1 sampling |
| `generate_observations.py` | preflight identity checks, sensor fix, stability, fine-mesh data, noise |
| `run_inversion.py` | truth-free coarse-mesh inversion driver around Application1's initializer and GN-diag L-BFGS-B (`--stage wiring|init|full`) |
| `validate_readiness.py` | sampling, dataset, objective, Taylor, admissibility, wiring checks; mismatch diagnostic at the truth |
| `report_source_id.py` | figures, `REPORT.md`, CSV/JSON summaries |
| `export_paraview.py` | true fine-mesh concentration time series, release centres and sensors as `.pvd`/`.vtu` |
| `test scripts/select_transport_settings.py` | Péclet / RK4 / coarse-vs-fine evidence behind `k` and the time steps |

Output `output/source_id_v2_wall8_20261116/`: `common/`, `case_12|13|23/{config,truth,data,logs}`,
`validation/`, `report/`, `commands/inversion_commands.sh` (single-run commands; the paper runs used `run_application2_inversion_batch.sh`),
`REPORT.md`. Only `case_XX/config/inversion_config.json` and `data/observations_*.npz`
are inversion inputs; the true controls live in `case_XX/truth/`.

Application1's initializer contains three DC-scale literals (Nelder–Mead
`xatol = 0.5` m, `fatol = 1.0`, de-duplication floor 2 m). `run_inversion.py`
substitutes them at load time with room-scale values (`--init-refine-xatol`,
`--init-refine-fatol-rel`, `--init-dedup-floor`; `--init-app1-literal-tolerances`
disables this). Application1's file is not edited.
