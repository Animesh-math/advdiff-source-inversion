"""Figures, summaries and REPORT.md for an Application2 source-identification dataset.

    PY report_source_id.py [--dataset-dir DIR]

Reads only what the pipeline saved; works on a partial dataset (missing parts
are reported as missing, so a failed run still gets a report).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import transport_config as tc  # noqa: E402
import transport_model as tm  # noqa: E402

geo = tm.geo
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SOLID_FILL, SOLID_EDGE = "#d9d8d2", "#898781"
SEQUENTIAL = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue_seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "legend.frameon": False,
})


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def draw_room(ax, labels: bool = True):
    pts, _ = geo.boundary_polygon()
    xs, ys = zip(*(pts + [pts[0]]))
    ax.plot(xs, ys, color=INK2, lw=1.0, zorder=3)
    for name, (x0, x1, y0, y1) in geo.SOLIDS.items():
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=SOLID_FILL, edgecolor=SOLID_EDGE, lw=0.8,
                               zorder=2))
        if labels and name.startswith("bed_"):
            ax.text(0.5 * (x0 + x1), 0.5 * (y0 + y1), f"bed {name[-1]}\n(solid)", ha="center", va="center",
                    color=INK2, fontsize=8, zorder=4)
    for a, b in geo.VENT_X_RANGES:
        ax.plot([a, b], [geo.INLET_Y] * 2, color=AQUA, lw=4, solid_capstyle="butt", zorder=4)
    ax.plot(geo.DOOR_X_RANGE, [0, 0], color=MUTED, lw=4, solid_capstyle="butt", zorder=4)
    ax.plot([geo.DOMAIN_X_MAX] * 2, geo.EXHAUST_Y_RANGE, color=MUTED, lw=4, solid_capstyle="butt", zorder=4)
    if labels:
        ax.text(4.5, 6.12, "supply vents (clean air, u = 0)", ha="center", va="bottom", color=INK2, fontsize=8)
        ax.text(1.1, -0.12, "corridor door (outlet)", ha="center", va="top", color=INK2, fontsize=8)
        ax.text(10.9, 0.55, "bathroom\nexhaust", ha="right", va="top", color=INK2, fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, 11.3)
    ax.set_ylim(-0.5, 6.5)
    ax.grid(False)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def fig_layout(out: Path, rep: Path, sensors: dict) -> str:
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    draw_room(ax)
    radius = np.sqrt(tc.SIGMA2)
    for bay, rel in tc.RELEASES.items():
        cx, cy = rel["center"]
        ax.add_patch(plt.Circle((cx, cy), radius, facecolor=ORANGE, alpha=0.18, edgecolor=ORANGE, lw=1.0, zorder=5))
        ax.plot(cx, cy, marker="*", ms=13, color=ORANGE, mec=SURFACE, mew=0.8, ls="none", zorder=7,
                label="release centre (truth; circle = e-folding radius 0.20 m)" if bay == 1 else None)
        ax.annotate(f"release {bay}\n({cx:g}, {cy:g}), A = {rel['amplitude']:g}", (cx, cy), xytext=(-8, 13),
                    textcoords="offset points", ha="right", va="bottom", fontsize=8, color=INK, zorder=8,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85, "pad": 1.5})
    xy = np.asarray(sensors["coordinates"])
    target = np.asarray([s["intended_xy"] for s in sensors["sensors"]])
    ax.plot(target[:, 0], target[:, 1], ls="none", marker="+", ms=9, color=MUTED, zorder=6,
            label="intended sampling point (geometry rule)")
    ax.plot(xy[:, 0], xy[:, 1], ls="none", marker="o", ms=8, color=BLUE, mec=SURFACE, mew=1.2, zorder=7,
            label="sensor sampling point (coarse-mesh node, used on both meshes)")
    for s in sensors["sensors"]:
        n = s.get("inward_normal") or [1.0, 0.0]
        ax.annotate(s["label"], s["xy"], xytext=(13 * n[0] + (0 if n[0] else 10), 13 * n[1] - 3),
                    textcoords="offset points", ha="center" if n[0] == 0 else ("left" if n[0] > 0 else "right"),
                    fontsize=9, fontweight="bold", color=INK, zorder=8)
    ax.legend(loc="upper left", bbox_to_anchor=(0.80, 1.0), fontsize=8)
    ax.set_title(f"Application2: sensor layout '{sensors['layout']}' ({sensors['num_sensors']} sensors) and the "
                 "fixed release locations (cases 12, 13, 23 use two of the three)", loc="left")
    path = rep / "fig_sources_sensors_layout.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path.name


def fig_traces(out: Path, rep: Path, case: str, sensors: dict) -> str | None:
    cdir = out / tc.case_dir_name(case)
    if not (cdir / "data" / "observations_noisy.npz").exists():
        return None
    clean = np.load(cdir / "data" / "observations_clean.npz")
    noisy = np.load(cdir / "data" / "observations_noisy.npz")["observations"]
    t, d = clean["times"], clean["observations"]
    stride = max(1, len(t) // 2000)
    ns = d.shape[1]
    nrows = int(np.ceil(ns / 4))
    fig, axes = plt.subplots(nrows, 4, figsize=(13.0, 2.5 * nrows + 0.4), sharex=True, squeeze=False)
    for ax in axes.ravel()[ns:]:
        ax.set_visible(False)
    for i, ax in enumerate(axes.ravel()[:ns]):
        ax.plot(t[::stride], noisy[::stride, i], color=ORANGE, lw=0.6, alpha=0.9, label="noisy (1 % global RMS)")
        ax.plot(t, d[:, i], color=BLUE, lw=1.6, label="clean (fine mesh)")
        s = sensors["sensors"][i]
        ax.set_title(f"{s['label']}  {s['name']}  ({s['xy'][0]:.2f}, {s['xy'][1]:.2f})", fontsize=8, loc="left")
        if i >= ns - 4:
            ax.set_xlabel("t [s]")
        if i % 4 == 0:
            ax.set_ylabel("concentration")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Case {case}: sensor traces (independent y-axes; the n = 0 sample is not in the objective)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = rep / f"fig_sensor_traces_case_{case}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def fig_fields(out: Path, rep: Path, sensors: dict) -> str | None:
    import matplotlib.tri as mtri

    cases = [c for c in tc.CASES if (out / tc.case_dir_name(c) / "truth" / "fine_state_snapshots.npz").exists()]
    if not cases:
        return None
    fig, axes = plt.subplots(1, len(cases), figsize=(5.6 * len(cases), 4.3), squeeze=False)
    xy = np.asarray(sensors["coordinates"])
    for ax, case in zip(axes[0], cases):
        snap = np.load(out / tc.case_dir_name(case) / "truth" / "fine_state_snapshots.npz")
        tri = mtri.Triangulation(snap["coord"][:, 0], snap["coord"][:, 1], snap["connections"])
        im = ax.tripcolor(tri, snap["states"][-1], shading="gouraud", cmap=SEQUENTIAL, vmin=0.0, zorder=1)
        draw_room(ax, labels=False)
        ax.plot(xy[:, 0], xy[:, 1], ls="none", marker="o", ms=4, color=SURFACE, mec=INK, mew=0.8, zorder=6)
        ax.set_title(f"case {case}: fine-mesh concentration at t = {snap['times'][-1]:g} s", loc="left")
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    fig.tight_layout()
    path = rep / "fig_fine_concentration_final_time.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, default=tm.DATASET_DIR)
    args = ap.parse_args()
    out = args.dataset_dir.resolve()
    rep = out / "report"
    rep.mkdir(parents=True, exist_ok=True)
    state = (out / "RUN_STATE").read_text().strip() if (out / "RUN_STATE").exists() else "(no RUN_STATE)"
    ident = load_json(out / "common" / "mesh_velocity_identifiers.json")
    sensors = load_json(out / "common" / "sensors.json")
    stab = load_json(out / "common" / "stability.json")
    sel = load_json(out / "common" / "settings_selection" / "settings_selection.json")
    gen = load_json(out / "common" / "generation_summary.json")
    val = load_json(out / "validation" / "validation_results.json")
    mis = load_json(out / "validation" / "discretization_mismatch.json")
    taylor = load_json(out / "validation" / "taylor_tests.json")

    figs = {}
    if sensors:
        figs["layout"] = fig_layout(out, rep, sensors)
        figs["fields"] = fig_fields(out, rep, sensors)
        for case in tc.CASES:
            figs[f"traces_{case}"] = fig_traces(out, rep, case, sensors)

    L = [f"# Application2 source-identification datasets ({tc.DATASET_VERSION})", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')}. Pipeline state: `{state}`.", "",
         "Scope: geometry/velocity confirmation, three two-source cases, fixed sensors, clean and noisy "
         "fine-mesh observations, and readiness checks of the coarse-mesh inverse problem. "
         "**No sensitivity initialization, centre search or optimization was run.** The releases are stationary "
         "tracer releases near the bed heads, not geometrically resolved patient mouths; the room is an "
         "idealized 2D test case.", ""]

    L += ["## Status", ""]
    if val:
        L.append(f"- Readiness checks: **{'all passed' if val['all_passed'] else 'FAILURES'}** "
                 f"({val['num_checks'] - val['num_failed']} of {val['num_checks']} passed).")
    else:
        L.append("- Readiness checks: **not available** (validation did not finish; see `logs/`).")
    L.append(f"- Datasets generated: {', '.join('case ' + r['case'] for r in gen['cases']) if gen else '**none**'}.")
    L.append("")

    L += ["## 1. Geometry and velocity fields", ""]
    if ident:
        L += [f"Flow version `{ident['flow_version']}`; all {len(ident['checks'])} identity checks passed: "
              f"{ident['all_passed']}.", "",
              "| level | role | nodes | triangles | mesh sha256 | velocity sha256 | Picard its. | final update |",
              "|---|---|---|---|---|---|---|---|"]
        for lv, role in (("coarse", "entire inversion"), ("fine", "data generation")):
            r = ident["levels"][lv]
            L.append(f"| {lv} | {role} | {r['n_nodes']} | {r['n_triangles']} | `{r['mesh_sha256'][:16]}…` | "
                     f"`{r['velocity_sha256'][:16]}…` | {r['picard_iterations']} | "
                     f"{r['picard_final_relative_update']:.2e} |")
        L += ["", "Checked per level: mesh/geometry/coordinate hashes against the velocity file's association record; "
              "Navier–Stokes converged; ν and supply speed equal `navier_stokes_config`; natural outlet conditions; "
              "three bed holes and two partitions present, no triangle inside a solid; supply nodes on the top wall "
              "carrying the downward parabola; outlets at the corridor door and bathroom exhaust with no backflow; "
              "zero velocity on walls. Full list: `common/mesh_velocity_identifiers.json`.", ""]
        note = ident["levels"]["coarse"].get("note_metadata_wording")
        if note:
            L += [f"Note: {note}.", ""]

    L += ["## 2. Transport model, sources and cases", "",
          "```text",
          "du/dt + v·grad(u) - k Laplace(u) = S(x; z),  u(x, 0) = 0",
          "u = 0 on the three supply vents;  k grad(u)·n = 0 on walls, partitions, beds, bathroom wall and both outlets",
          "S(x; z) = sum_i A_i exp(-|x - c_i|^2 / rho),   z = [x1, y1, A1, x2, y2, A2]",
          "```", "",
          "| quantity | value |", "|---|---|",
          f"| tracer diffusivity k | {tc.DIFFUSIVITY_K:g} m²/s (effective; distinct from the flow viscosity "
          f"ν = {tm.nsc.NU:g} m²/s) |",
          f"| source width ρ = σ² parameter | {tc.SIGMA2:g} m² (e-folding radius {np.sqrt(tc.SIGMA2):.2f} m; "
          f"standard deviation {np.sqrt(0.5 * tc.SIGMA2):.3f} m) |",
          "| amplitude convention | A = peak release rate at the centre; Gaussian not normalized; "
          f"total rate per metre depth = A·π·ρ = {np.pi * tc.SIGMA2:.4f}·A |",
          f"| amplitude bounds | [{tc.AMPLITUDE_BOUNDS[0]}, {tc.AMPLITUDE_BOUNDS[1]}] |",
          f"| observation horizon T | {tc.TFINAL:g} s |",
          f"| time steps | coarse {tc.DT['coarse']} s ({tc.NUM_STEPS['coarse']} steps), fine {tc.DT['fine']} s "
          f"({tc.NUM_STEPS['fine']} steps), explicit RK4 |",
          f"| observation interval | {tc.OBSERVATION_INTERVAL} s (every coarse step, every {tc.FINE_SUBSTEPS}th fine "
          f"step); {tc.NUM_OBSERVATION_TIMES} times in the objective |",
          f"| objective | J = (η₁·Δt/2) Σₙ₌₁ᴺ ‖C Uⁿ − dⁿ‖²_W, η₁ = {tc.ETA1:g}, W = I, n = 0 excluded |", ""]
    L += ["| bed | release centre (x, y) [m] | amplitude | clearance to nearest solid |", "|---|---|---|---|"]
    for bay, rel in tc.RELEASES.items():
        d = float(geo.distance_to_boundary(*rel["center"])[0])
        L.append(f"| {bay} | ({rel['center'][0]:g}, {rel['center'][1]:g}) | {rel['amplitude']:g} | {d:.3f} m "
                 f"({d / np.sqrt(tc.SIGMA2):.2f} radii) |")
    L += ["", "Cases: " + "; ".join(f"**{c}** = beds {b[0]} and {b[1]}" for c, b in tc.CASES.items()) + ".", ""]
    for key, text in tc.SETTINGS_RATIONALE.items():
        L.append(f"- *{key}*: {text}")
    L.append("")
    if sel:
        L += ["Settings-selection evidence (all three releases active; full-field metrics only; "
              "`common/settings_selection/`):", "",
              (out / "common" / "settings_selection" / "settings_selection.md").read_text()]
    if stab:
        L += ["RK4 stability of the adopted steps (spectral radius of M⁻¹K per mesh and its own velocity field):", "",
              "| level | ρ(M⁻¹K) [1/s] | dt [s] | dt·ρ (target ≤ %.1f, limit %.3f) | cell Pe max |" % (
                  stab["target_dt_rho"], stab["rk4_real_axis_limit"]), "|---|---|---|---|---|"]
        for lv, r in stab["levels"].items():
            L.append(f"| {lv} | {r['spectral_radius_Minv_K_per_s']:.2f} | {r['dt_s']} | "
                     f"{r['dt_times_spectral_radius']:.3f} | {r['cell_peclet']['max']:.2f} |")
        L.append("")

    L += ["## 3. Sensors", ""]
    if sensors:
        L += [sensors["selection_rule"], "", f"![layout]({figs['layout']})", "",
              "| sensor | location | wall | x [m] | y [m] | coarse node | wall offset [m] | snap distance [m] |",
              "|---|---|---|---|---|---|---|---|"]
        for s in sensors["sensors"]:
            L.append(f"| {s['label']} | {s['description']} | {s['wall'] or '-'} | {s['xy'][0]:.4f} | {s['xy'][1]:.4f} | "
                     f"{s['coarse_node_id']} | {s.get('wall_offset_m', float('nan')):.3f} | {s['snap_distance_m']:.3f} |")
        L += ["", "Wall offset = distance of the sampling point from its wall (intended "
              f"{tc.SENSOR_WALL_OFFSET:g} m; the actual value is set by the first interior coarse node).", ""]
        if tc.PROPOSED_SENSOR_LAYOUTS:
            L += ["Proposed, **not active**: " + "; ".join(f"`{k}`: {tc.SENSOR_LAYOUT_RULES[k]}"
                                                         for k in tc.PROPOSED_SENSOR_LAYOUTS)
                  + " Coordinates in `common/proposed_sensor_layouts_NOT_ACTIVE.json`.", ""]
        if tc.SUPERSEDED_DATASETS:
            L += ["**Layout mismatch flag:** " + "; ".join(
                f"dataset `{k}` was generated with the superseded layout `{v}`" for k, v in tc.SUPERSEDED_DATASETS.items())
                + ". It is preserved unchanged and must not be mixed with this dataset.", ""]
        L += ["", "Fine-mesh values are P1 interpolations at these same coordinates (`common/sensors.json`).", ""]

    L += ["## 4. Observations", ""]
    if gen:
        L += ["| case | beds | samples in objective | clean max | clean RMS | s_noise (absolute) | seed | "
              "realized ‖ε‖/‖d_clean‖ |", "|---|---|---|---|---|---|---|---|"]
        for r in gen["cases"]:
            L.append(f"| {r['case']} | {r['bays'][0]}, {r['bays'][1]} | {r['num_observations']} | "
                     f"{r['clean_max']:.4f} | {r['clean_rms']:.4f} | {r['s_noise']:.6e} | {r['seed']} | "
                     f"{r['realized_relative_noise_norm']:.6f} |")
        L += ["", f"Noise: independent Gaussian, one common standard deviation per case, nominal level "
              f"{gen['noise_level']:g} of the global RMS over the samples in the objective; drawn once, saved in "
              "`case_XX/data/noise.npz`, never clipped.", ""]
        if figs.get("fields"):
            L += [f"![fields]({figs['fields']})", ""]
        for case in tc.CASES:
            if figs.get(f"traces_{case}"):
                L += [f"![traces {case}]({figs[f'traces_{case}']})", ""]

    L += ["## 5. Readiness checks", ""]
    if val:
        L += ["| id | result | check | detail |", "|---|---|---|---|"]
        for r in val["checks"]:
            detail = r["detail"].replace("|", "/")
            L.append(f"| {r['id']} | {'pass' if r['passed'] else '**FAIL**'} | {r['name']} | {detail} |")
        L.append("")
    if taylor:
        L += ["Taylor remainders |J(z + h d) − J(z) − h g·d| (rate 2 expected until round-off):", "",
              "| case | data | " + " | ".join(f"h = {h:g}" for h in [s['h'] for s in taylor[0]['steps']])
              + " | best rate |", "|---|---|" + "---|" * (len(taylor[0]["steps"]) + 1)]
        for t in taylor:
            L.append(f"| {t['case']} | {t['data']} | " + " | ".join(f"{s['remainder']:.2e}" for s in t["steps"])
                     + f" | {t['best_two_step_rate']:.3f} |")
        L.append("")
    if mis and mis.get("cases"):
        L += ["### Discretization-mismatch diagnostic (truth used here only; not an inversion input)", "",
              "| case | J(coarse at truth; clean data) | relative misfit norm | noise-only J | ratio | J(zero source) |",
              "|---|---|---|---|---|---|"]
        for r in mis["cases"]:
            L.append(f"| {r['case']} | {r['objective_coarse_at_truth_vs_clean']:.4e} | "
                     f"{r['relative_misfit_norm_clean']:.4f} | {r['objective_noise_only']:.4e} | "
                     f"{r['mismatch_over_noise_objective_ratio']:.2f} | {r['objective_zero_source_clean']:.4e} |")
        L += ["", "The relative misfit norm is ‖C u_coarse(z*) − d_clean‖ / ‖d_clean‖ over the samples in the "
              "objective. A ratio above 1 means the coarse/fine model mismatch at the truth exceeds the added "
              "noise in the objective.", ""]

    L += ["## 6. Inversion commands (prepared, NOT executed)", "",
          "See `commands/inversion_commands.sh`. Each run reads only `case_XX/config/inversion_config.json` and the "
          "selected observation file.", "",
          "## 7. Files", "", "```text",
          "common/     transport_config.json, mesh_velocity_identifiers.json, sensors.{json,csv}, stability.json,",
          "            settings_selection/, generation_summary.{json,csv}",
          "case_XX/    config/inversion_config.json (truth-free), truth/ (truth record, fine snapshots),",
          "            data/ (observations_clean.npz, observations_noisy.npz, noise.npz, noise_record.json, csv), logs/",
          "validation/ validation_results.{json,csv}, taylor_tests.json, discretization_mismatch.{json,csv},",
          "            gaussian_resolution.json, wiring/<case>_<data>/wiring_check.json",
          "commands/   inversion_commands.sh (not executed)",
          "logs/ env/  pipeline and stage logs; git revision/diff, package lists, source checksums",
          "```", ""]
    (rep / "REPORT.md").write_text("\n".join(L))
    (out / "REPORT.md").write_text("\n".join(L).replace("](fig_", "](report/fig_"))

    summary = {"dataset_version": tc.DATASET_VERSION, "pipeline_state": state,
               "validation_all_passed": val["all_passed"] if val else None,
               "cases": gen["cases"] if gen else [], "mismatch": mis["cases"] if mis else [], "figures": figs}
    tm.write_json(rep / "summary.json", summary)
    if gen:
        by_case = {r["case"]: r for r in (mis["cases"] if mis else [])}
        with open(rep / "summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case", "beds", "num_observations", "clean_rms", "s_noise", "seed",
                        "realized_relative_noise_norm", "coarse_truth_relative_misfit", "mismatch_over_noise_ratio"])
            for r in gen["cases"]:
                m = by_case.get(r["case"], {})
                w.writerow([r["case"], "+".join(map(str, r["bays"])), r["num_observations"], r["clean_rms"],
                            r["s_noise"], r["seed"], r["realized_relative_noise_norm"],
                            m.get("relative_misfit_norm_clean"), m.get("mismatch_over_noise_objective_ratio")])
    print(f"report: {out / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
