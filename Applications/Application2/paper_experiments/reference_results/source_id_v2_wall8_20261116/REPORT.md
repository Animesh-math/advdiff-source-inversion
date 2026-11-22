# Application2 source-identification datasets (source_id_v2_wall8_20261116)

Generated 2026-11-16 20:41:34 -0500. Pipeline state: `DONE_OK 2026-11-16T20:41:32-05:00`.

Scope: geometry/velocity confirmation, three two-source cases, fixed sensors, clean and noisy fine-mesh observations, and readiness checks of the coarse-mesh inverse problem. **No sensitivity initialization, centre search or optimization was run.** The releases are stationary tracer releases near the bed heads, not geometrically resolved patient mouths; the room is an idealized 2D test case.

## Status

- Readiness checks: **all passed** (48 of 48 passed).
- Datasets generated: case 12, case 13, case 23.

## 1. Geometry and velocity fields

Flow version `v3_bathroom_topsupply_20261116`; all 39 identity checks passed: True.

| level | role | nodes | triangles | mesh sha256 | velocity sha256 | Picard its. | final update |
|---|---|---|---|---|---|---|---|
| coarse | entire inversion | 4381 | 7887 | `3536cc16a57f9faf…` | `b3f2062166617eda…` | 52 | 9.80e-09 |
| fine | data generation | 15231 | 28772 | `5309186742fc17e1…` | `a4bd84897f73e659…` | 52 | 9.64e-09 |

Checked per level: mesh/geometry/coordinate hashes against the velocity file's association record; Navier–Stokes converged; ν and supply speed equal `navier_stokes_config`; natural outlet conditions; three bed holes and two partitions present, no triangle inside a solid; supply nodes on the top wall carrying the downward parabola; outlets at the corridor door and bathroom exhaust with no backflow; zero velocity on walls. Full list: `common/mesh_velocity_identifiers.json`.

Note: the flow metadata string boundary_conditions.inlets says 'parabolic upward profile' (stale wording from the earlier bottom-supply versions); the nodal velocity itself is verified above to be the DOWNWARD top-wall supply profile.

## 2. Transport model, sources and cases

```text
du/dt + v·grad(u) - k Laplace(u) = S(x; z),  u(x, 0) = 0
u = 0 on the three supply vents;  k grad(u)·n = 0 on walls, partitions, beds, bathroom wall and both outlets
S(x; z) = sum_i A_i exp(-|x - c_i|^2 / rho),   z = [x1, y1, A1, x2, y2, A2]
```

| quantity | value |
|---|---|
| tracer diffusivity k | 0.01 m²/s (effective; distinct from the flow viscosity ν = 0.0023244 m²/s) |
| source width ρ = σ² parameter | 0.04 m² (e-folding radius 0.20 m; standard deviation 0.141 m) |
| amplitude convention | A = peak release rate at the centre; Gaussian not normalized; total rate per metre depth = A·π·ρ = 0.1257·A |
| amplitude bounds | [0.1, 2.0] |
| observation horizon T | 300 s |
| time steps | coarse 0.015 s (20000 steps), fine 0.0025 s (120000 steps), explicit RK4 |
| observation interval | 0.015 s (every coarse step, every 6th fine step); 20000 times in the objective |
| objective | J = (η₁·Δt/2) Σₙ₌₁ᴺ ‖C Uⁿ − dⁿ‖²_W, η₁ = 1, W = I, n = 0 excluded |

| bed | release centre (x, y) [m] | amplitude | clearance to nearest solid |
|---|---|---|---|
| 1 | (2.475, 5.2) | 1 | 0.475 m (2.38 radii) |
| 2 | (5.475, 5.2) | 0.7 | 0.475 m (2.37 radii) |
| 3 | (8.5, 5.2) | 1.3 | 0.500 m (2.50 radii) |

Cases: **12** = beds 1 and 2; **13** = beds 1 and 3; **23** = beds 2 and 3.

- *diffusivity_k*: k = 1.0e-2 m^2/s is the smallest of the candidates {nu_eff = 2.3244e-3, 5e-3, 1e-2} for which the cell Peclet number |v| h / (2 k) is <= 1 in every triangle of the COARSE (inversion) mesh (max 0.97), the classical condition for a non-oscillatory unstabilized Galerkin P1 discretization; the shared solver has no stabilization.  Measured with all three releases active: k = nu_eff gives cell Pe up to 4.2 on the coarse mesh (46 % of the area above 1) and a small negative undershoot (min u / max u = -6.6e-4); k = 1e-2 gives a non-negative coarse solution and the smallest coarse-vs-fine field difference (1.6-1.9 % relative L2, against 2.4-3.0 % for k = nu_eff).  k is an effective (eddy) diffusivity, a modelling parameter like nu_eff but a different one (nu_eff / k = 0.23); it is far above molecular diffusion.
- *sigma2*: sqrt(rho) = 0.20 m is a compromise fixed by the 0.95-1.0 m bed-side passages: the passage centreline is 2.4 e-folding radii from the nearest solid (Gaussian value 0.4 % there), and the radius spans 1.3 mean element edges on the coarse mesh and 2.5 on the fine mesh at the release locations (Application1's default was 2.5 on its mesh; its width study went down to about 0.9).  The 3-point quadrature of the source integral agrees between the two meshes to 3e-4 and the coarse and fine concentration fields differ by 1.6-1.9 % (settings-selection record).  A wider Gaussian would be better resolved on the coarse mesh but would overlap the bed and the partition.
- *tfinal*: T = 300 s, about one nominal air-change time A_fluid / Q_supply = 50.98 / 0.18 = 283 s.
- *dt*: largest round steps with dt * rho(M^-1 K) <= 2.5 on each mesh with its own velocity field and an integer fine/coarse ratio, so that both meshes share the observation times.
- *amplitudes*: arbitrary distinct values within +-30 % of 1, strictly inside the bounds [0.1, 2.0].

Settings-selection evidence (all three releases active; full-field metrics only; `common/settings_selection/`):

| k [m²/s] | level | Pe max | Pe area-median | area Pe>1 | ρ(M⁻¹K) [1/s] | dt [s] | dt·ρ | steps | min u / max u | coarse–fine rel. L2 at t = 60, 120, 180, 240, 300 s |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.002324 | coarse | 4.17 | 0.92 | 45.9 % | 35.9 | 0.06 | 2.15 | 5000 | -6.6e-04 | 0.030, 0.025, 0.024, 0.025, 0.025 |
| 0.002324 | fine | 2.39 | 0.46 | 10.0 % | 193.9 | 0.012 | 2.33 | 25000 | -2.5e-11 |  |
| 0.005 | coarse | 1.94 | 0.43 | 6.7 % | 77.2 | 0.03 | 2.32 | 10000 | -2.4e-11 | 0.023, 0.021, 0.022, 0.022, 0.023 |
| 0.005 | fine | 1.11 | 0.21 | 0.1 % | 417.1 | 0.005 | 2.09 | 60000 | 0.0e+00 |  |
| 0.01 | coarse | 0.97 | 0.21 | 0.0 % | 154.4 | 0.015 | 2.32 | 20000 | 0.0e+00 | 0.016, 0.017, 0.017, 0.018, 0.019 |
| 0.01 | fine | 0.55 | 0.11 | 0.0 % | 834.1 | 0.0025 | 2.09 | 120000 | 0.0e+00 |  |

RK4 stability of the adopted steps (spectral radius of M⁻¹K per mesh and its own velocity field):

| level | ρ(M⁻¹K) [1/s] | dt [s] | dt·ρ (target ≤ 2.5, limit 2.785) | cell Pe max |
|---|---|---|---|---|
| coarse | 154.40 | 0.015 | 2.316 | 0.97 |
| fine | 834.09 | 0.0025 | 2.085 | 0.55 |

## 3. Sensors

eight wall-mounted sensors (user decision 2026-11-16): geometry only; each wall point moved 0.1 m into the fluid and snapped to the nearest coarse-mesh node that is not a boundary node; identical for all cases; not adjusted using true sources, signals or recovery outcomes.

![layout](report/fig_sources_sensors_layout.png)

| sensor | location | wall | x [m] | y [m] | coarse node | wall offset [m] | snap distance [m] |
|---|---|---|---|---|---|---|---|
| S1 | left exterior wall of Bay 1, beside the upper portion of the bed | wall_left | 0.1591 | 5.1000 | 1331 | 0.159 | 0.064 |
| S2 | Bay 1 face of the first partition, beside the lower portion of the bed | partition_1 | 2.8308 | 4.0531 | 3741 | 0.119 | 0.034 |
| S3 | Bay 2 face of the first partition, upper position | partition_1 | 3.1607 | 5.0845 | 4234 | 0.111 | 0.014 |
| S4 | Bay 2 face of the second partition, lower position | partition_2 | 5.8308 | 4.0531 | 3743 | 0.119 | 0.034 |
| S5 | Bay 3 face of the second partition, upper position | partition_2 | 6.1558 | 5.0784 | 4231 | 0.106 | 0.007 |
| S6 | right exterior wall of Bay 3, lower position | wall_right | 8.8430 | 3.9773 | 3833 | 0.157 | 0.074 |
| S7 | bottom wall of the shared aisle, immediately to the right of the corridor doorway | wall_bottom | 1.9074 | 0.0897 | 2663 | 0.090 | 0.013 |
| S8 | bathroom's upper wall, near the exhaust side | bath_top | 10.5086 | 1.7887 | 1382 | 0.211 | 0.112 |

Wall offset = distance of the sampling point from its wall (intended 0.1 m; the actual value is set by the first interior coarse node).

Proposed, **not active**: `wall_10`: PROPOSED, NOT ACTIVE: wall_8 plus two sensors evenly spaced on the bottom aisle wall between S7 and the bathroom wall. Coordinates in `common/proposed_sensor_layouts_NOT_ACTIVE.json`.

**Layout mismatch flag:** dataset `source_id_v1_20261116` was generated with the superseded layout `aisle_passage_12`. It is preserved unchanged and must not be mixed with this dataset.


Fine-mesh values are P1 interpolations at these same coordinates (`common/sensors.json`).

## 4. Observations

| case | beds | samples in objective | clean max | clean RMS | s_noise (absolute) | seed | realized ‖ε‖/‖d_clean‖ |
|---|---|---|---|---|---|---|---|
| 12 | 1, 2 | 160000 | 4.1270 | 1.5439 | 1.543931e-02 | 2026111612 | 0.009984 |
| 13 | 1, 3 | 160000 | 5.3111 | 2.0532 | 2.053230e-02 | 2026111613 | 0.009980 |
| 23 | 2, 3 | 160000 | 5.3141 | 1.8464 | 1.846419e-02 | 2026111623 | 0.009996 |

Noise: independent Gaussian, one common standard deviation per case, nominal level 0.01 of the global RMS over the samples in the objective; drawn once, saved in `case_XX/data/noise.npz`, never clipped.

![fields](report/fig_fine_concentration_final_time.png)

![traces 12](report/fig_sensor_traces_case_12.png)

![traces 13](report/fig_sensor_traces_case_13.png)

![traces 23](report/fig_sensor_traces_case_23.png)

## 5. Readiness checks

| id | result | check | detail |
|---|---|---|---|
| V1a | pass | sensor coordinates are the coarse-mesh node coordinates | 8 sensors |
| V1b | pass | no sensor on a clean-air Dirichlet node or outside the fluid |  |
| V1c | pass | fine-mesh P1 interpolation reproduces a linear field at the sensor coordinates | max error 1.8e-15 |
| V1d | pass | interpolation cells/weights equal those recorded at generation |  |
| V1e | pass | saved clean traces equal the interpolated saved fine-mesh fields at the snapshot times | max abs difference 0.0e+00 |
| V2a | pass | case 12: inversion configuration holds no truth field |  |
| V2b | pass | case 12: clean/noisy files reload identically and match the recorded hashes |  |
| V2c | pass | case 12: d_noisy = d_clean + noise exactly; n = 0 sample is zero and noise-free |  |
| V2d | pass | case 12: noise reproducible from the recorded seed; one common std | seed 2026111612, s_noise 1.543931e-02, realized relative noise norm 0.009984 |
| V2e | pass | case 12: realized relative noise norm consistent with the nominal level | 0.009984 vs nominal 0.01 |
| V2f | pass | case 12: negative noisy samples are kept (no clipping) | 46618 negative samples of 160000 |
| V2a | pass | case 13: inversion configuration holds no truth field |  |
| V2b | pass | case 13: clean/noisy files reload identically and match the recorded hashes |  |
| V2c | pass | case 13: d_noisy = d_clean + noise exactly; n = 0 sample is zero and noise-free |  |
| V2d | pass | case 13: noise reproducible from the recorded seed; one common std | seed 2026111613, s_noise 2.053230e-02, realized relative noise norm 0.009980 |
| V2e | pass | case 13: realized relative noise norm consistent with the nominal level | 0.009980 vs nominal 0.01 |
| V2f | pass | case 13: negative noisy samples are kept (no clipping) | 43830 negative samples of 160000 |
| V2a | pass | case 23: inversion configuration holds no truth field |  |
| V2b | pass | case 23: clean/noisy files reload identically and match the recorded hashes |  |
| V2c | pass | case 23: d_noisy = d_clean + noise exactly; n = 0 sample is zero and noise-free |  |
| V2d | pass | case 23: noise reproducible from the recorded seed; one common std | seed 2026111623, s_noise 1.846419e-02, realized relative noise norm 0.009996 |
| V2e | pass | case 23: realized relative noise norm consistent with the nominal level | 0.009996 vs nominal 0.01 |
| V2f | pass | case 23: negative noisy samples are kept (no clipping) | 45421 negative samples of 160000 |
| V5a | pass | mesh admissibility rejects beds, partitions, the bathroom wall and the exterior; accepts the fluid | 16 probe points; wrong: none |
| V5b | pass | control-level admissibility (validate_sources_in_domain) flags a centre inside a bed |  |
| V5c | pass | release centres in the fluid with >= 2 radii of clearance; radius / local mean edge >= 1 on the coarse mesh; source integral agrees between meshes | bed 1: clearance 2.38 radii, radius/h coarse 1.3 fine 2.5, quadrature integral coarse vs fine 2.5e-04; bed 2: clearance 2.37 radii, radius/h coarse 1.3 fine 2.5, quadrature integral coarse vs fine 2.5e-04; bed 3: clearance 2.50 radii, radius/h coarse 1.2 fine 2.2, quadrature integral coarse vs fine 1.0e-04 |
| V5d | pass | release amplitudes strictly inside the amplitude bounds | [1.0, 0.7, 1.3] in (0.1, 2.0) |
| V5e | pass | analytic Gaussian source derivatives (x, y, A) match central differences | max relative error 1.6e-10 |
| V3a | pass | case 12: evaluator objective == direct (eta1 dt / 2) sum_n>=1 r^T W r, W = I | J = 2.914758072331e+03, relative difference 1.6e-16 |
| V3b | pass | case 12: shared objective supports a non-uniform diagonal W | relative difference 0.0e+00 |
| V3b | pass | case 12: shared objective supports a full symmetric positive definite W | relative difference 0.0e+00 |
| V3c | pass | case 12: repeated evaluation gives the same objective; data file and data array unchanged | J = 2.914758072331e+03 twice |
| V3d | pass | sparse RK4 (data-generation stepping) reproduces the shared dense forward solver on the coarse mesh | max relative trace difference 2.7e-16 |
| V6 | pass | case 12 (clean): initializer and optimizer entry points wired to the coarse mesh, its velocity and this dataset | J = 2.914513e+03, /g/ = 2.595e+02, 4381 candidate centres {'bay_1': 878, 'bay_2': 1018, 'bay_3': 869, 'aisle': 1204, 'bathroom_and_doorway': 428} |
| V6 | pass | case 12 (noisy): initializer and optimizer entry points wired to the coarse mesh, its velocity and this dataset | J = 2.914758e+03, /g/ = 2.595e+02, 4381 candidate centres {'bay_1': 878, 'bay_2': 1018, 'bay_3': 869, 'aisle': 1204, 'bathroom_and_doorway': 428}; GN-diagonal scaling path: cond 3.26e+04 -> 3.51e+02 (7 forward / 6 adjoint sweeps) |
| V6 | pass | case 13 (clean): initializer and optimizer entry points wired to the coarse mesh, its velocity and this dataset | J = 6.254486e+03, /g/ = 6.781e+03, 4381 candidate centres {'bay_1': 878, 'bay_2': 1018, 'bay_3': 869, 'aisle': 1204, 'bathroom_and_doorway': 428} |
| V6 | pass | case 13 (noisy): initializer and optimizer entry points wired to the coarse mesh, its velocity and this dataset | J = 6.254936e+03, /g/ = 6.781e+03, 4381 candidate centres {'bay_1': 878, 'bay_2': 1018, 'bay_3': 869, 'aisle': 1204, 'bathroom_and_doorway': 428} |
| V6 | pass | case 23 (clean): initializer and optimizer entry points wired to the coarse mesh, its velocity and this dataset | J = 4.110438e+03, /g/ = 8.280e+02, 4381 candidate centres {'bay_1': 878, 'bay_2': 1018, 'bay_3': 869, 'aisle': 1204, 'bathroom_and_doorway': 428} |
| V6 | pass | case 23 (noisy): initializer and optimizer entry points wired to the coarse mesh, its velocity and this dataset | J = 4.110802e+03, /g/ = 8.280e+02, 4381 candidate centres {'bay_1': 878, 'bay_2': 1018, 'bay_3': 869, 'aisle': 1204, 'bathroom_and_doorway': 428} |
| V4 | pass | case 12 (clean data): directional Taylor test of the discrete-adjoint gradient | best two-step remainder rate 2.003, best FD-slope relative error 2.8e-06, J = 2.914513e+03, g.d = 1.321377e+02, distance to nearest true centre 2.54 m |
| V4 | pass | case 12 (noisy data): directional Taylor test of the discrete-adjoint gradient | best two-step remainder rate 2.003, best FD-slope relative error 2.8e-06, J = 2.914758e+03, g.d = 1.321217e+02, distance to nearest true centre 2.54 m |
| V4 | pass | case 13 (clean data): directional Taylor test of the discrete-adjoint gradient | best two-step remainder rate 2.000, best FD-slope relative error 1.4e-05, J = 6.254486e+03, g.d = -8.503837e+02, distance to nearest true centre 1.14 m |
| V4 | pass | case 13 (noisy data): directional Taylor test of the discrete-adjoint gradient | best two-step remainder rate 2.000, best FD-slope relative error 1.4e-05, J = 6.254936e+03, g.d = -8.503544e+02, distance to nearest true centre 1.14 m |
| V4 | pass | case 23 (clean data): directional Taylor test of the discrete-adjoint gradient | best two-step remainder rate 2.000, best FD-slope relative error 1.3e-06, J = 4.110438e+03, g.d = 1.193677e+02, distance to nearest true centre 2.50 m |
| V4 | pass | case 23 (noisy data): directional Taylor test of the discrete-adjoint gradient | best two-step remainder rate 2.000, best FD-slope relative error 1.3e-06, J = 4.110802e+03, g.d = 1.193712e+02, distance to nearest true centre 2.50 m |
| D | pass | case 12: coarse model at the true control vs clean fine-mesh data (diagnostic, not a pass/fail) | J = 4.886373e-01 (relative misfit norm 0.0131; noise-only objective 2.851442e-01; ratio 1.71) |
| D | pass | case 13: coarse model at the true control vs clean fine-mesh data (diagnostic, not a pass/fail) | J = 1.521038e+00 (relative misfit norm 0.0173; noise-only objective 5.038688e-01; ratio 3.02) |
| D | pass | case 23: coarse model at the true control vs clean fine-mesh data (diagnostic, not a pass/fail) | J = 1.352086e+00 (relative misfit norm 0.0182; noise-only objective 4.087888e-01; ratio 3.31) |

Taylor remainders |J(z + h d) − J(z) − h g·d| (rate 2 expected until round-off):

| case | data | h = 0.1 | h = 0.01 | h = 0.001 | h = 0.0001 | h = 1e-05 | best rate |
|---|---|---|---|---|---|---|---|
| 12 | clean | 3.74e-01 | 3.70e-03 | 3.70e-05 | 3.70e-07 | 3.70e-09 | 2.003 |
| 12 | noisy | 3.74e-01 | 3.70e-03 | 3.70e-05 | 3.70e-07 | 3.70e-09 | 2.003 |
| 13 | clean | 1.03e+01 | 1.15e-01 | 1.16e-03 | 1.16e-05 | 1.16e-07 | 2.000 |
| 13 | noisy | 1.03e+01 | 1.15e-01 | 1.16e-03 | 1.16e-05 | 1.16e-07 | 2.000 |
| 23 | clean | 1.24e-01 | 1.47e-03 | 1.49e-05 | 1.49e-07 | 1.50e-09 | 2.000 |
| 23 | noisy | 1.24e-01 | 1.47e-03 | 1.49e-05 | 1.49e-07 | 1.49e-09 | 2.000 |

### Discretization-mismatch diagnostic (truth used here only; not an inversion input)

| case | J(coarse at truth; clean data) | relative misfit norm | noise-only J | ratio | J(zero source) |
|---|---|---|---|---|---|
| 12 | 4.8864e-01 | 0.0131 | 2.8514e-01 | 1.71 | 2.8605e+03 |
| 13 | 1.5210e+00 | 0.0173 | 5.0387e-01 | 3.02 | 5.0589e+03 |
| 23 | 1.3521e+00 | 0.0182 | 4.0879e-01 | 3.31 | 4.0911e+03 |

The relative misfit norm is ‖C u_coarse(z*) − d_clean‖ / ‖d_clean‖ over the samples in the objective. A ratio above 1 means the coarse/fine model mismatch at the truth exceeds the added noise in the objective.

## 6. Inversion commands (prepared, NOT executed)

See `commands/inversion_commands.sh`. Each run reads only `case_XX/config/inversion_config.json` and the selected observation file.

## 7. Files

```text
common/     transport_config.json, mesh_velocity_identifiers.json, sensors.{json,csv}, stability.json,
            settings_selection/, generation_summary.{json,csv}
case_XX/    config/inversion_config.json (truth-free), truth/ (truth record, fine snapshots),
            data/ (observations_clean.npz, observations_noisy.npz, noise.npz, noise_record.json, csv), logs/
validation/ validation_results.{json,csv}, taylor_tests.json, discretization_mismatch.{json,csv},
            gaussian_resolution.json, wiring/<case>_<data>/wiring_check.json
commands/   inversion_commands.sh (not executed)
logs/ env/  pipeline and stage logs; git revision/diff, package lists, source checksums
```
