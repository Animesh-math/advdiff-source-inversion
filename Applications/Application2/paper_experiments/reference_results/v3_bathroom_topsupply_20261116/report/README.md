# Application2 — three-bed room with bathroom: meshes and steady flow (v3_bathroom_topsupply_20261116)

Generated 2026-11-16T21:48:51.325701+00:00.

> The room experiment uses a steady, two-dimensional airflow with a spatially constant effective viscosity. It provides an idealized setting for evaluating source identification rather than a validated prediction of hospital ventilation.

The results below describe these particular computed fields only; they do not establish ventilation effectiveness.

Ventilation arrangement inspired by Fig. 24 and Sec. 7.1 of Lohner et al., arXiv:2006.13792; this three-bed room is an adaptation, not a reproduction, and the paper's bathroom ceiling exhaust is represented by a 2D boundary outlet. Not clinically validated.

**Geometry** (`room_geometry_config.py`, metres): main room [0,9]×[0,6] ∪ bathroom annex [9,11]×[0,2], minus two 0.10 m partitions [2.95,3.05]×[3,6] and [5.95,6.05]×[3,6], three **solid beds** [1,2]×[3.5,5.6], [4,5]×[3.5,5.6], [7,8]×[3.5,5.6] (idealized 2D impermeable obstacles; holes in the mesh), and a 0.10 m bathroom wall [8.95,9.05]×[0,0.6] ∪ [8.95,9.05]×[1.4,2] whose gap 0.6 < y < 1.4 is an open interior doorway. Fluid area 50.98 m². Supplies: three top-wall openings x ∈ [1.2,1.8], [4.2,4.8], [7.2,7.8] at y = 6, flow downward. Outlets: corridor doorway y = 0, x ∈ [0.6,1.6] and bathroom exhaust x = 11, y ∈ [0.7,1.3], both natural (ν∇v − pI)n = 0 with no prescribed pressure or velocity. All other boundaries are no-slip. Pillows drawn in figures are presentation only. The reference drawing `room_layout_reference.png` is not to scale; the coordinates define the geometry (see `fig_geometry_vs_reference.png`).

## Status

| level | status | Picard iterations | final rel. update | MUMPS null pivots | pressure Dirichlet dofs |
|---|---|---|---|---|---|
| coarse | **converged** | 52 | 9.80e-09 | 0 | 0 |
| fine | **converged** | 52 | 9.64e-09 | 0 | 0 |

## Parameters (navier_stokes_config.py)

- Supply profile: parabolic, downward, zero at the vent endpoints; **mean** speed 0.1 m/s, peak 0.15000000000000002 m/s (peak = 1.5 × mean); flux per inlet 0.060 m²/s per metre depth.
- ν = 0.0023244 m²/s — spatially constant effective kinematic viscosity (modelling parameter of an idealized 2D experiment); not molecular air viscosity, not a tracer diffusivity.
- Re_slot = U_mean·b/ν = 25.8 (b = 0.60 m vent width);  Re_room = U_mean·H/ν = 258 (H = 6 m room depth in plan). Molecular air viscosity (≈1.5e-5 m²/s) would give Re_slot ≈ 4000.
- Approximate air-change rate ≈ 12.7 1/h on the 51.0 m² fluid area (11.2 1/h on the gross 58 m² floor), valid only under these assumptions: plan-view field = height average; each opening spans the full room height; uniform supply/extract over the height; all air enters via the three inlets. Real terminals do not span the room height.
- Parameter selection: at U_mean = 0.30 m/s, ν = 5e-3 (Re_slot = 36) Picard did not converge on either mesh; Newton found the symmetric steady state but it is linearly unstable (leading eigenvalue ≈ −1.2e-3 1/s on both meshes). At the adopted Re_slot = 25.8 the steady state is linearly stable (leading eigenvalue +3.95e-3 1/s, coarse). Trials: `output/v1_nobeds_bottomsupply_20261116/trials_20261116_picard_viscosity/`. These trials used the earlier v1 geometry (no beds, bottom supply); the parameters are retained unchanged for this layout and its stability has not been re-assessed.
- Picard: tolerance 1e-08 (relative L2 update), max 100 iterations; iteration 1 is Stokes. Same for both levels.

## Meshes

| level | vertices | triangles | P2 velocity dofs | P1 pressure dofs | total | lc_bulk | lc_corner | min angle | edges per opening |
|---|---|---|---|---|---|---|---|---|---|
| coarse | 4381 | 7887 | 33302 | 4381 | 37683 | 0.2 | 0.025 | 28.5° | [14, 22] |
| fine | 15231 | 28772 | 118472 | 15231 | 133703 | 0.1 | 0.0125 | 33.2° | [28, 44] |

Transport (P1) nodes = mesh vertices. Mesh files and SHA-256 are in `<level>/mesh/*_mesh_info.json`.

### Geometry checks (Room_mesh/build_room_mesh.py; a failure aborts the build)

- **coarse**: 1 edge-connected component; 4 boundary loops (outer + 3 beds); triangle centroids inside solids: 0, outside room ∪ annex: 0; solid-interior samples covered by triangles: 0; Gmsh→DOLFINx tag transfer OK (cell tags [1]); edges per bed [104].
  Passages (clear width / median element size): bed_1_left 1.00 m ≈ 10 el., bed_1_right 0.95 m ≈ 12 el., bed_1_head 0.40 m ≈ 6 el., bed_1_foot 0.50 m ≈ 7 el., bed_2_left 0.95 m ≈ 12 el., bed_2_right 0.95 m ≈ 12 el., bed_2_head 0.40 m ≈ 6 el., bed_2_foot 0.50 m ≈ 6 el., bed_3_left 0.95 m ≈ 12 el., bed_3_right 1.00 m ≈ 10 el., bed_3_head 0.40 m ≈ 6 el., bed_3_foot 0.50 m ≈ 6 el., bathroom_doorway 0.80 m ≈ 22 el., bathroom_interior 1.95 m ≈ 19 el., aisle 3.00 m ≈ 15 el., aisle_to_bay3_beside_bathroom 1.00 m ≈ 6 el.; every passage sample point lies in a fluid triangle.
- **fine**: 1 edge-connected component; 4 boundary loops (outer + 3 beds); triangle centroids inside solids: 0, outside room ∪ annex: 0; solid-interior samples covered by triangles: 0; Gmsh→DOLFINx tag transfer OK (cell tags [1]); edges per bed [200].
  Passages (clear width / median element size): bed_1_left 1.00 m ≈ 19 el., bed_1_right 0.95 m ≈ 20 el., bed_1_head 0.40 m ≈ 12 el., bed_1_foot 0.50 m ≈ 12 el., bed_2_left 0.95 m ≈ 20 el., bed_2_right 0.95 m ≈ 20 el., bed_2_head 0.40 m ≈ 12 el., bed_2_foot 0.50 m ≈ 12 el., bed_3_left 0.95 m ≈ 20 el., bed_3_right 1.00 m ≈ 19 el., bed_3_head 0.40 m ≈ 12 el., bed_3_foot 0.50 m ≈ 11 el., bathroom_doorway 0.80 m ≈ 38 el., bathroom_interior 1.95 m ≈ 38 el., aisle 3.00 m ≈ 30 el., aisle_to_bay3_beside_bathroom 1.00 m ≈ 11 el.; every passage sample point lies in a fluid triangle.

## coarse: fluxes, divergence, natural-outflow check

| boundary | Taylor–Hood flux ∫v·n | P1 transport flux |
|---|---|---|
| inlet_1 | -0.060000 | -0.059652 |
| inlet_2 | -0.060000 | -0.059652 |
| inlet_3 | -0.060000 | -0.059652 |
| outlet_door | +0.118731 | +0.118203 |
| outlet_exhaust | +0.061269 | +0.060663 |
| all no-slip walls | +0.0e+00 | -7.1e-19 |
| **net** (normalized by inflow) | -1.67e-16 (9.3e-16) | -9.00e-05 (5.0e-04) |

Flux units m²/s per metre depth, outward normal (inflow < 0). Prescribed per inlet: -0.0600.
- Divergence, Taylor–Hood: ‖∇·v‖/‖∇v‖ = 5.66e-02 (∫∇·v = -8.4e-17); P1 transport: ‖∇·v‖/‖∇v‖ = 1.20e-01, max |∇·v| = 1.36e+00 1/s. Taylor–Hood is only weakly divergence-free; the P1 interpolant is not divergence-free.
- Natural outflow: max |momentum residual| on free outlet dofs 1.8e-13, interior 1.5e-12, vs Dirichlet reactions 2.0e-03 (ratio 8.7e-11).

**Outlet split and backflow** (computed, not prescribed):

| outlet | flux | share of inflow | mean u·n [m/s] | min u·n at dofs | backflow flux | backflow length | mean p | mean |traction·n| |
|---|---|---|---|---|---|---|---|---|
| outlet_door | +0.118731 | 65.96 % | +0.1187 | +0.00e+00 | +0.00e+00 | 0.000 m (0.0 %) | +1.84e-04 | 7.2e-05 |
| outlet_exhaust | +0.061269 | 34.04 % | +0.1021 | +0.00e+00 | +0.00e+00 | 0.000 m (0.0 %) | +3.28e-04 | 1.6e-04 |

Bathroom doorway (x = 9, 0.6 < y < 1.4): net flux into the bathroom +0.06127 (into +0.06127, back +0.00000). The outlets are passive natural-outflow boundaries: they do not model mechanical suction or a negative-pressure room.

- Wall nodes in transport field: max |v| = 5.2e-18.

## fine: fluxes, divergence, natural-outflow check

| boundary | Taylor–Hood flux ∫v·n | P1 transport flux |
|---|---|---|
| inlet_1 | -0.060000 | -0.059914 |
| inlet_2 | -0.060000 | -0.059914 |
| inlet_3 | -0.060000 | -0.059914 |
| outlet_door | +0.118714 | +0.118568 |
| outlet_exhaust | +0.061286 | +0.061103 |
| all no-slip walls | +0.0e+00 | -3.1e-19 |
| **net** (normalized by inflow) | -4.86e-17 (2.7e-16) | -7.02e-05 (3.9e-04) |

Flux units m²/s per metre depth, outward normal (inflow < 0). Prescribed per inlet: -0.0600.
- Divergence, Taylor–Hood: ‖∇·v‖/‖∇v‖ = 3.72e-02 (∫∇·v = -2.7e-17); P1 transport: ‖∇·v‖/‖∇v‖ = 6.87e-02, max |∇·v| = 2.14e+00 1/s. Taylor–Hood is only weakly divergence-free; the P1 interpolant is not divergence-free.
- Natural outflow: max |momentum residual| on free outlet dofs 4.7e-14, interior 4.4e-13, vs Dirichlet reactions 1.0e-03 (ratio 4.5e-11).

**Outlet split and backflow** (computed, not prescribed):

| outlet | flux | share of inflow | mean u·n [m/s] | min u·n at dofs | backflow flux | backflow length | mean p | mean |traction·n| |
|---|---|---|---|---|---|---|---|---|
| outlet_door | +0.118714 | 65.95 % | +0.1187 | +0.00e+00 | +0.00e+00 | 0.000 m (0.0 %) | +1.15e-04 | 2.8e-05 |
| outlet_exhaust | +0.061286 | 34.05 % | +0.1021 | +0.00e+00 | +0.00e+00 | 0.000 m (0.0 %) | +2.11e-04 | 7.4e-05 |

Bathroom doorway (x = 9, 0.6 < y < 1.4): net flux into the bathroom +0.06129 (into +0.06129, back +0.00000). The outlets are passive natural-outflow boundaries: they do not model mechanical suction or a negative-pressure room.

- Wall nodes in transport field: max |v| = 3.3e-18.

## Airflow by region (descriptive)

Weak flow: |v| < 0.010 m/s (= 0.1 × mean inlet speed). Metrics on the 0.05 m probe grid (area-weighted).

| level | region | mean |v| | max |v| | weak-area fraction | v_y<0 area fraction | net flux across y=3 |
|---|---|---|---|---|---|---|
| coarse | bay_1 | 0.033 | 0.148 | 0.21 | 0.95 | -0.0600 |
| coarse | bay_2 | 0.034 | 0.148 | 0.20 | 0.94 | -0.0600 |
| coarse | bay_3 | 0.033 | 0.148 | 0.21 | 0.94 | -0.0600 |
| coarse | aisle | 0.024 | 0.148 | 0.17 | 0.98 |  |
| coarse | bathroom | 0.033 | 0.123 | 0.44 | 0.47 |  |
| fine | bay_1 | 0.033 | 0.148 | 0.21 | 0.95 | -0.0600 |
| fine | bay_2 | 0.034 | 0.148 | 0.21 | 0.94 | -0.0600 |
| fine | bay_3 | 0.033 | 0.148 | 0.21 | 0.94 | -0.0600 |
| fine | aisle | 0.024 | 0.148 | 0.17 | 0.98 |  |
| fine | bathroom | 0.033 | 0.123 | 0.44 | 0.47 |  |

Net flux across y = 3 is per bay (negative = from the bay down into the aisle). The supplies blow downward, so v_y < 0 is the main flow direction in the bays; see the streamline figures for recirculation.

## Coarse vs fine (common physical points)

- Velocity: relative L2 difference 2.56e-03; max |Δv| 9.43e-04 m/s at [6.575, 5.575]; 95th percentile 2.01e-04 m/s.
- Pressure: relative L2 difference 1.33e-03.
- Lines: h_y1: rel 1.2e-03; h_y1.5: rel 1.2e-03; h_y3: rel 1.5e-03; h_y4.5: rel 1.1e-03; v_x1.5: rel 1.6e-03; v_x4.5: rel 2.4e-03; v_x7.5: rel 1.8e-03; v_x10: rel 1.3e-03; doorway_x9: rel 2.4e-03
- coarse minus fine P2/P1 fields evaluated at identical physical points (0.05 m cell-centred grid, solids and exterior excluded; 5 mm line samples); rel_l2 = ||dv|| / ||v_fine|| over the points

## ParaView

All paths below are relative to `Applications/Application2/output/v3_bathroom_topsupply_20261116/`. Quickest view: File → Open `<level>/navier_stokes/application2_velocity_pressure_P1_<level>.vtu` (reader picked automatically: XML Unstructured Grid Reader), Apply, colour by `speed`; Filters → Glyph (Orientation `velocity`, Scale `speed`) for arrows, or Filters → Stream Tracer (Vectors `velocity`, seed type Line) for streamlines. Open `<level>/mesh/application2_room_<level>.vtu`, colour by `physical_tag`, and use Representation 'Surface With Edges' to see the element edges around the bed holes and the tagged boundary lines.

| file | content | ParaView reader |
|---|---|---|
| `coarse/mesh/application2_room_coarse.vtu` | mesh + tagged boundary lines, cell data `physical_tag` (supplies 21–23, outlets 31–32, walls 41–47, partitions 51–52, beds 61–63, bathroom wall 71–72) | XML Unstructured Grid Reader (built in) |
| `coarse/mesh/application2_room_coarse_facets.xdmf` (+ `.h5`) | DOLFINx facet tags | Xdmf3ReaderS / Xdmf3ReaderT |
| `coarse/navier_stokes/application2_velocity_pressure_P1_coarse.vtu` | P1 velocity (3-component), pressure, speed at vertices | XML Unstructured Grid Reader (built in) |
| `coarse/navier_stokes/application2_velocity_pressure_P1_coarse.xdmf` (+ `.h5`) | P1 velocity, pressure | Xdmf3ReaderS / Xdmf3ReaderT |
| `coarse/navier_stokes/application2_velocity_P2_coarse.bp`, `application2_pressure_P1_coarse.bp` | native P2 velocity / P1 pressure (directories) | ADIOS2VTXReader (ParaView ≥ 5.12 with ADIOS2) |
| `fine/mesh/application2_room_fine.vtu` | mesh + tagged boundary lines, cell data `physical_tag` (supplies 21–23, outlets 31–32, walls 41–47, partitions 51–52, beds 61–63, bathroom wall 71–72) | XML Unstructured Grid Reader (built in) |
| `fine/mesh/application2_room_fine_facets.xdmf` (+ `.h5`) | DOLFINx facet tags | Xdmf3ReaderS / Xdmf3ReaderT |
| `fine/navier_stokes/application2_velocity_pressure_P1_fine.vtu` | P1 velocity (3-component), pressure, speed at vertices | XML Unstructured Grid Reader (built in) |
| `fine/navier_stokes/application2_velocity_pressure_P1_fine.xdmf` (+ `.h5`) | P1 velocity, pressure | Xdmf3ReaderS / Xdmf3ReaderT |
| `fine/navier_stokes/application2_velocity_P2_fine.bp`, `application2_pressure_P1_fine.bp` | native P2 velocity / P1 pressure (directories) | ADIOS2VTXReader (ParaView ≥ 5.12 with ADIOS2) |

Beds, partitions and the bathroom wall are holes/notches in every mesh/field file (no cells inside them). Choose 'Xdmf3ReaderS' when ParaView asks which XDMF reader to use.

## Figures

- geometry: `fig_geometry_boundary_tags.png`
- geometry_vs_reference: `fig_geometry_vs_reference.png`
- mesh_coarse: `fig_mesh_coarse.png`
- mesh_bed_holes_coarse: `fig_mesh_bed_holes_coarse.png`
- mesh_fine: `fig_mesh_fine.png`
- mesh_bed_holes_fine: `fig_mesh_bed_holes_fine.png`
- velocity_coarse: `fig_velocity_coarse.png`
- velocity_arrows_coarse: `fig_velocity_arrows_coarse.png`
- velocity_fine: `fig_velocity_fine.png`
- velocity_arrows_fine: `fig_velocity_arrows_fine.png`
- velocity_pair: `fig_velocity_coarse_vs_fine.png`
- pressure: `fig_pressure_coarse_vs_fine.png`
- picard: `fig_picard_convergence.png`
- comparison: `fig_coarse_fine_comparison.png`

## Provenance

Commands (RUN_COMMANDS.txt):

```text
# Application2 setup pipeline (2026-11-16T16:48:32-05:00); version v3_bathroom_topsupply_20261116; cwd .; OMP_NUM_THREADS=4
<fenicsx-env python> Applications/Application2/Room_mesh/build_room_mesh.py --level all --out-root Applications/Application2/output/v3_bathroom_topsupply_20261116
<fenicsx-env python> "Applications/Application2/Navier Stokes Solve/solve_application2_navier_stokes.py" --level coarse --out-root Applications/Application2/output/v3_bathroom_topsupply_20261116
<fenicsx-env python> "Applications/Application2/Navier Stokes Solve/solve_application2_navier_stokes.py" --level fine --out-root Applications/Application2/output/v3_bathroom_topsupply_20261116
<fenicsx-env python> Applications/Application2/report_application2_flow.py --out-root Applications/Application2/output/v3_bathroom_topsupply_20261116
```

- coarse: git b22a05a9353538d941d92d399942bc5c3580c910 (app1-osqp-amplitude-solver), dolfinx 0.11.0, basix 0.11.0, PETSc [3, 25, 6], numpy 2.5.3, scipy 1.18.1; command `<fenicsx-env python> Applications/Application2/Navier Stokes Solve/solve_application2_navier_stokes.py --level coarse --out-root Applications/Application2/output/v3_bathroom_topsupply_20261116`
- fine: git b22a05a9353538d941d92d399942bc5c3580c910 (app1-osqp-amplitude-solver), dolfinx 0.11.0, basix 0.11.0, PETSc [3, 25, 6], numpy 2.5.3, scipy 1.18.1; command `<fenicsx-env python> Applications/Application2/Navier Stokes Solve/solve_application2_navier_stokes.py --level fine --out-root Applications/Application2/output/v3_bathroom_topsupply_20261116`
- Environment captures: `conda_meta_fenicsx_env.txt`, `git_branch.txt`, `git_commit.txt`, `git_diff_tracked.patch`, `git_status.txt`, `host.txt`, `pip_list_fenicsx_env.txt`, `python_fenicsx_env.txt`, `sha256_application2_sources.txt`
