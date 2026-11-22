"""Shared coarse/fine report for the Application2 meshes and Navier-Stokes fields.

Reads ``<out-root>/{coarse,fine}/{mesh,navier_stokes}`` (written by
Room_mesh/build_room_mesh.py and the NS solve) and writes figures, a JSON
summary and a Markdown report to ``<out-root>/report/``.  No PDE solves; needs
numpy, scipy, matplotlib, meshio (FEniCSx or inverse environment).

    $FX Applications/Application2/report_application2_flow.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import room_geometry_config as geo  # noqa: E402
import navier_stokes_config as cfg  # noqa: E402
import mesh_data_config as mdc  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

LEVELS = ("coarse", "fine")
WEAK_FRACTION = 0.10   # "weak flow": |u| < WEAK_FRACTION * INLET_MEAN_SPEED
TAG_COLORS = {"inlet": "#12a39a", "outlet": "#f07f13", "wall": "#1f2d4d", "partition": "#7f7f7f",
              "bed": "#3d7fc1", "bath_wall": "#7f7f7f"}
BED_FILL = "#cfe6fb"
SOLID_FILL = "0.55"
XLIM, YLIM = (-0.15, geo.DOMAIN_X_MAX + 0.15), (-0.15, geo.ROOM_Y_MAX + 0.15)


def boundary_kind(nm):
    for k in ("inlet", "outlet", "partition", "bed", "bath_wall"):
        if nm.startswith(k):
            return k
    return "wall"


def load_level(out_root: Path, level: str) -> dict | None:
    ns = out_root / level / "navier_stokes"
    meta_p = ns / f"application2_navier_stokes_run_metadata_{level}.json"
    info_p = mdc.mesh_info_file(level, out_root)
    if not info_p.exists():
        return None
    d = {"level": level, "mesh_info": json.loads(info_p.read_text()),
         "md": mdc.build_mesh_data(level, out_root)}
    if meta_p.exists():
        d["meta"] = json.loads(meta_p.read_text())
        d["vel"] = mdc.load_velocity_nodes(level, d["md"], out_root)   # association-checked
        with np.load(mdc.velocity_npz_file(level, out_root)) as z:
            d["p_nodes"] = z["pressure_nodes"]
        with np.load(ns / f"application2_probes_{level}.npz") as z:
            d["probes"] = {k: z[k] for k in z.files}
        d["history"] = np.loadtxt(ns / f"application2_picard_history_{level}.csv", delimiter=",", skiprows=1, ndmin=2)
    return d


# --------------------------------------------------------------------------- #
# drawing helpers
# --------------------------------------------------------------------------- #
def draw_outline(ax, color=None, lw=1.2):
    """Outer boundary of the fluid domain (room + annex with notches)."""
    pts, _ = geo.boundary_polygon()
    P = np.array(pts + [pts[0]])
    ax.plot(P[:, 0], P[:, 1], color=color or TAG_COLORS["wall"], lw=lw, zorder=4)


def draw_annotations(ax, beds=True, labels=True, vents=True, partitions=True, color="k", bed_fill=True,
                     pillows=True, outline=False):
    """Solids are drawn filled (they are holes in the mesh).  Pillows are presentation only."""
    if outline:
        draw_outline(ax)
    if partitions:
        for nm, (x0, x1, y0, y1) in {**geo.PARTITIONS, **geo.BATH_WALLS}.items():
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=SOLID_FILL, ec="k", lw=0.6, zorder=5))
    if beds:
        for bay, b in geo.BEDS.items():
            (x0, x1), (y0, y1) = b["x"], b["y"]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=BED_FILL if bed_fill else "none",
                                   ec=TAG_COLORS["bed"], lw=1.0, zorder=5))
            if pillows:  # presentation only, NOT an obstacle
                ax.add_patch(Rectangle((x0 + 0.12, y1 - 0.38), x1 - x0 - 0.24, 0.26,
                                       fc="white" if bed_fill else "none", ec=TAG_COLORS["bed"], ls=":", lw=0.7,
                                       zorder=6))
    if labels:
        for bay, rng in geo.BAYS.items():
            ax.text(0.5 * sum(rng["x"]), 3.12, f"Bay {bay}", ha="center", va="bottom", fontsize=9,
                    color=color, weight="bold", zorder=7)
        ax.text(4.5, 1.9, "Shared aisle", ha="center", fontsize=10, color=color, weight="bold", zorder=7)
        ax.text(10.0, 1.6, "Bathroom", ha="center", fontsize=9, color=color, weight="bold", zorder=7)
    if vents:
        for a, b in geo.VENT_X_RANGES:
            ax.plot([a, b], [geo.INLET_Y] * 2, color=TAG_COLORS["inlet"], lw=4, solid_capstyle="butt", zorder=8)
        ax.plot(geo.DOOR_X_RANGE, [0, 0], color=TAG_COLORS["outlet"], lw=4, solid_capstyle="butt", zorder=8)
        ax.plot([geo.DOMAIN_X_MAX] * 2, geo.EXHAUST_Y_RANGE, color=TAG_COLORS["outlet"], lw=4,
                solid_capstyle="butt", zorder=8)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")


def draw_geometry_panel(ax, md):
    coords = md["coord"]
    for nm, edges in md["boundary_edges_by_name"].items():
        kind = boundary_kind(nm)
        for e0, e1 in edges:
            ax.plot(coords[[e0, e1], 0], coords[[e0, e1], 1], color=TAG_COLORS[kind],
                    lw=5 if kind in ("inlet", "outlet") else 2.2, solid_capstyle="butt", zorder=4)
    draw_annotations(ax, vents=False)
    ac = dict(arrowstyle="-|>", lw=1.6)
    for k, ((a, b), nm) in enumerate(zip(geo.VENT_X_RANGES, geo.INLET_NAMES)):
        xm = 0.5 * (a + b)
        ax.annotate("", (xm, 5.62), xytext=(xm, 5.98), arrowprops={**ac, "color": TAG_COLORS["inlet"]}, zorder=9)
        ax.text(xm, 6.25, f"{nm} ({geo.INLET_TAGS[nm]})\nsupply ↓", ha="center", va="bottom", fontsize=8,
                color=TAG_COLORS["inlet"])
    xd = 0.5 * sum(geo.DOOR_X_RANGE)
    ax.annotate("", (xd, -0.45), xytext=(xd, -0.02), arrowprops={**ac, "color": TAG_COLORS["outlet"]})
    ax.text(xd, -0.55, f"outlet_door ({geo.OUTLET_TAGS['outlet_door']})\ncorridor doorway, natural outflow",
            ha="center", va="top", fontsize=8, color=TAG_COLORS["outlet"])
    ye = 0.5 * sum(geo.EXHAUST_Y_RANGE)
    ax.annotate("", (11.45, ye), xytext=(11.02, ye), arrowprops={**ac, "color": TAG_COLORS["outlet"]})
    ax.text(11.5, ye, f"outlet_exhaust ({geo.OUTLET_TAGS['outlet_exhaust']})\nbathroom exhaust,\nnatural outflow",
            ha="left", va="center", fontsize=8, color=TAG_COLORS["outlet"])
    for bay, b in geo.BEDS.items():
        (x0, x1), (y0, y1) = b["x"], b["y"]
        nm = f"bed_{bay}"
        ax.text(0.5 * (x0 + x1), 0.5 * (y0 + y1), f"{nm} ({geo.BED_TAGS[nm]})\n[{x0:g},{x1:g}]×\n[{y0:g},{y1:g}]\n"
                "solid", ha="center", va="center", fontsize=7, color=TAG_COLORS["bed"], zorder=9)
    for nm, (x0, x1, y0, y1) in geo.PARTITIONS.items():
        ax.text(x1 + 0.07, 4.6, f"{nm} ({geo.WALL_TAGS[nm]})", rotation=90, fontsize=7, va="center",
                color="0.35")
    ax.annotate(f"bathroom wall 0.10 m\nbath_wall_lower ({geo.WALL_TAGS['bath_wall_lower']}), "
                f"bath_wall_upper ({geo.WALL_TAGS['bath_wall_upper']})\ndoorway 0.6 < y < 1.4 (interior, open)",
                (9.0, 1.0), xytext=(6.2, 0.55), fontsize=7, color="0.25", ha="center",
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
    walls = ", ".join(f"{n} ({t})" for n, t in geo.WALL_TAGS.items()
                      if boundary_kind(n) == "wall")
    ax.text(0.0, -1.35, f"exterior no-slip walls: {walls}", fontsize=6.8, color=TAG_COLORS["wall"], va="top")
    ax.set_xlim(-0.4, 13.2)
    ax.set_ylim(-1.6, 7.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def fig_geometry(md, path):
    fig, ax = plt.subplots(figsize=(13, 7.4))
    draw_geometry_panel(ax, md)
    ax.set_title("Application2 three-bed room + bathroom: geometry and boundary tags (top-down, idealized 2D; "
                 "drawn from the mesh boundary)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_geometry_vs_reference(md, path):
    """Side-by-side check of the computational geometry against room_layout_reference.png."""
    ref = None
    for ext in ("png", "jpg", "jpeg"):
        p = APP_DIR / f"room_layout_reference.{ext}"
        if p.exists():
            ref = p
            break
    if ref is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(20, 7.2), gridspec_kw={"width_ratios": [1.0, 1.15]})
    axes[0].imshow(plt.imread(ref))
    axes[0].axis("off")
    axes[0].set_title(f"reference drawing ({ref.name}; not to scale)", fontsize=10)
    draw_geometry_panel(axes[1], md)
    axes[1].set_title("computational geometry (to scale, from the coarse mesh boundary)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return ref.name


def fig_mesh(d, path):
    md = d["md"]
    c = md["coord"]
    tri = mtri.Triangulation(c[:, 0], c[:, 1], md["connections"])
    info = d["mesh_info"]
    fig = plt.figure(figsize=(16, 7.6))
    ax = fig.add_axes([0.03, 0.08, 0.62, 0.84])
    ax.triplot(tri, lw=0.22, color="0.25")
    draw_annotations(ax, labels=False, partitions=True, bed_fill=False, pillows=False)
    ax.set_title(f"{d['level']} mesh: {info['n_vertices']} vertices, {info['n_triangles']} triangles "
                 f"(lc_bulk={info['sizes_m']['lc_bulk']} m); supplies teal, outlets orange")
    zooms = [((8.4, 9.6, 0.3, 2.3), "bathroom doorway + 5 cm step at (9, 2)"),
             ((0.35, 1.85, -0.05, 0.6), "corridor door outlet"),
             ((10.4, 11.05, 0.45, 1.55), "bathroom exhaust outlet"),
             ((2.55, 3.45, 2.65, 3.45), "partition-1 tip"),
             ((1.55, 2.4, 5.25, 6.05), "bed-1 head corner / inlet_1")]
    boxes = [[0.68, 0.55, 0.14, 0.38], [0.85, 0.55, 0.14, 0.38], [0.68, 0.08, 0.085, 0.38],
             [0.795, 0.08, 0.085, 0.38], [0.91, 0.08, 0.085, 0.38]]
    for ((x0, x1, y0, y1), title), box in zip(zooms, boxes):
        az = fig.add_axes(box)
        az.triplot(tri, lw=0.35, color="0.25")
        draw_annotations(az, labels=False, bed_fill=False, pillows=False)
        az.set_xlim(x0, x1)
        az.set_ylim(y0, y1)
        az.set_title(title, fontsize=7.5)
        az.tick_params(labelsize=6)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_mesh_bed_holes(d, path):
    """Element edges around the three bed holes (geometry verification)."""
    md = d["md"]
    c = md["coord"]
    tri = mtri.Triangulation(c[:, 0], c[:, 1], md["connections"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 7.2))
    for ax, (bay, b) in zip(axes, geo.BEDS.items()):
        (x0, x1), (y0, y1) = b["x"], b["y"]
        ax.triplot(tri, lw=0.35, color="0.3")
        nm = f"bed_{bay}"
        for e0, e1 in md["boundary_edges_by_name"][nm]:
            ax.plot(c[[e0, e1], 0], c[[e0, e1], 1], color=TAG_COLORS["bed"], lw=2.2, zorder=4)
        bx0, bx1 = geo.BAYS[bay]["x"]
        draw_annotations(ax, labels=False, beds=False)
        ax.set_xlim(bx0 - 0.15, bx1 + 0.15)
        ax.set_ylim(2.7, 6.1)
        ax.set_aspect("equal")
        ne = len(md["boundary_edges_by_name"][nm])
        ax.set_title(f"{nm} (tag {geo.BED_TAGS[nm]}): hole [{x0:g},{x1:g}]×[{y0:g},{y1:g}], "
                     f"{ne} boundary edges", fontsize=9)
        ax.text(0.5 * (x0 + x1), 0.5 * (y0 + y1), "no elements\n(solid bed)", ha="center", va="center",
                fontsize=9, color=TAG_COLORS["bed"])
        ax.set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    info = d["mesh_info"]
    fig.suptitle(f"{d['level']} mesh around the bed holes ({info['n_vertices']} vertices, {info['n_triangles']} "
                 "triangles): element edges grey, bed boundary edges brown, partitions grey fill", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def grid_field(probes, values):
    gx, gy, keep = probes["grid_x"], probes["grid_y"], probes["grid_keep_mask"]
    full = np.full(keep.shape, np.nan)
    full[keep] = values
    return np.ma.masked_invalid(full.reshape(len(gy), len(gx)))


def fig_velocity(levels, path_each, path_pair, vmax):
    for d, path in zip(levels, path_each):
        fig, ax = plt.subplots(figsize=(13, 7.0))
        _velocity_panel(ax, d, vmax, fig)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(len(levels), 1, figsize=(13, 6.6 * len(levels)))
    for ax, d in zip(np.atleast_1d(axes), levels):
        _velocity_panel(ax, d, vmax, fig)
    fig.tight_layout()
    fig.savefig(path_pair, dpi=140)
    plt.close(fig)


def _velocity_panel(ax, d, vmax, fig):
    md, vel, pr = d["md"], d["vel"], d["probes"]
    c = md["coord"]
    tri = mtri.Triangulation(c[:, 0], c[:, 1], md["connections"])
    sp = np.linalg.norm(vel, axis=1)
    tpc = ax.tripcolor(tri, sp, shading="gouraud", cmap="viridis", vmin=0, vmax=vmax)
    U = grid_field(pr, pr["grid_velocity"][:, 0])
    Vv = grid_field(pr, pr["grid_velocity"][:, 1])
    ax.streamplot(pr["grid_x"], pr["grid_y"], U.filled(np.nan), Vv.filled(np.nan), color="white",
                  density=2.2, linewidth=0.55, arrowsize=0.7, broken_streamlines=True)
    weak = WEAK_FRACTION * cfg.INLET_MEAN_SPEED
    ax.tricontour(tri, sp, levels=[weak], colors="#ff4fd8", linewidths=0.9)
    draw_annotations(ax, color="w")
    fig.colorbar(tpc, ax=ax, shrink=0.8, label="|v| [m/s]")
    ax.set_title(f"{d['level']}: |v| with streamlines (magenta: |v| = {weak:.3f} m/s weak-flow contour)",
                 fontsize=9.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def fig_arrows(d, path, vmax):
    """Velocity arrows (direction, length ~ |v|) on a 0.25 m subsample of the probe grid."""
    pr = d["probes"]
    P, V = pr["grid_points"], pr["grid_velocity"]
    step = 0.25
    sel = (np.abs(((P[:, 0] - 0.125) / step) - np.round((P[:, 0] - 0.125) / step)) < 1e-6) & \
          (np.abs(((P[:, 1] - 0.125) / step) - np.round((P[:, 1] - 0.125) / step)) < 1e-6)
    sp = np.linalg.norm(V[sel], axis=1)
    fig, ax = plt.subplots(figsize=(13, 7.0))
    draw_outline(ax)
    q = ax.quiver(P[sel, 0], P[sel, 1], V[sel, 0], V[sel, 1], sp, cmap="viridis", clim=(0, vmax),
                  angles="xy", scale_units="xy", scale=vmax / 0.35, width=0.0022, zorder=6)
    draw_annotations(ax)
    fig.colorbar(q, ax=ax, shrink=0.8, label="|v| [m/s]")
    ax.set_title(f"{d['level']}: velocity arrows on a {step} m grid (length ∝ |v|; {vmax:.3f} m/s ↔ 0.35 m)",
                 fontsize=9.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_pressure(levels, path, plim):
    fig, axes = plt.subplots(len(levels), 1, figsize=(13, 6.6 * len(levels)))
    for ax, d in zip(np.atleast_1d(axes), levels):
        c = d["md"]["coord"]
        tri = mtri.Triangulation(c[:, 0], c[:, 1], d["md"]["connections"])
        tpc = ax.tripcolor(tri, d["p_nodes"], shading="gouraud", cmap="RdBu_r", vmin=plim[0], vmax=plim[1])
        ax.tricontour(tri, d["p_nodes"], levels=15, colors="k", linewidths=0.3)
        draw_annotations(ax)
        fig.colorbar(tpc, ax=ax, shrink=0.8, label="kinematic pressure p [m²/s²]")
        ax.set_title(f"{d['level']}: pressure (level fixed by natural outflow, no pinning)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_picard(levels, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for d in levels:
        h = d["history"]
        ax.semilogy(h[:, 0], h[:, 2], "o-", ms=3, label=f"{d['level']} ({d['meta']['picard']['iterations']} it, "
                                                         f"{'converged' if d['meta']['picard']['converged'] else 'NOT converged'})")
    ax.axhline(cfg.PICARD_TOL, color="k", ls="--", lw=0.8, label=f"tolerance {cfg.PICARD_TOL:g}")
    ax.set_xlabel("Picard iteration")
    ax.set_ylabel("||u_k − u_{k−1}||_L2 / ||u_k||_L2")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"Picard convergence (nu = {cfg.NU:g} m²/s, U_mean = {cfg.INLET_MEAN_SPEED} m/s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def region_metrics(d):
    """Area-weighted (uniform probe grid) flow metrics per bay and the circulation zone."""
    pr = d["probes"]
    pts, v = pr["grid_points"], pr["grid_velocity"]
    sp = np.linalg.norm(v, axis=1)
    dA = (pr["grid_x"][1] - pr["grid_x"][0]) * (pr["grid_y"][1] - pr["grid_y"][0])
    weak = WEAK_FRACTION * cfg.INLET_MEAN_SPEED
    regions = {f"bay_{b}": (r["x"], r["y"]) for b, r in geo.BAYS.items()}
    regions["aisle"] = (geo.AISLE["x"], geo.AISLE["y"])
    regions["bathroom"] = (geo.BATHROOM["x"], geo.BATHROOM["y"])
    out = {}
    for nm, ((x0, x1), (y0, y1)) in regions.items():
        m = (pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)
        out[nm] = {
            "area_m2_probe": float(m.sum() * dA),
            "mean_speed": float(sp[m].mean()), "median_speed": float(np.median(sp[m])),
            "max_speed": float(sp[m].max()),
            "area_fraction_weak": float(np.mean(sp[m] < weak)),
            "area_fraction_downward_vy_lt_0": float(np.mean(v[m, 1] < 0)),
        }
    return out, weak


def compare_levels(c, f):
    pc, pf = c["probes"], f["probes"]
    assert np.array_equal(pc["grid_points"], pf["grid_points"]), "probe grids differ"
    dv = pc["grid_velocity"] - pf["grid_velocity"]
    nd = np.linalg.norm(dv, axis=1)
    sf = np.linalg.norm(pf["grid_velocity"], axis=1)
    k = int(np.argmax(nd))
    dp = pc["grid_pressure"] - pf["grid_pressure"]
    lines = {}
    for key in pc:
        if key.startswith("line_") and key.endswith("_velocity"):
            name = key[5:-9]
            a, b = pc[key], pf[key]
            lines[name] = {"rel_l2_diff": float(np.linalg.norm(a - b) / np.linalg.norm(b)),
                           "max_abs_diff": float(np.max(np.linalg.norm(a - b, axis=1)))}
    return {
        "grid_points": int(len(nd)),
        "velocity_rel_l2_diff_grid": float(np.linalg.norm(nd) / np.linalg.norm(sf)),
        "velocity_max_abs_diff": float(nd.max()),
        "velocity_max_abs_diff_location": pc["grid_points"][k].tolist(),
        "velocity_p95_abs_diff": float(np.percentile(nd, 95)),
        "velocity_median_abs_diff": float(np.median(nd)),
        "pressure_rel_l2_diff_grid": float(np.linalg.norm(dp) / np.linalg.norm(pf["grid_pressure"])),
        "pressure_max_abs_diff": float(np.abs(dp).max()),
        "lines": lines,
        "definition": "coarse minus fine P2/P1 fields evaluated at identical physical points (0.05 m cell-centred "
                      "grid, solids and exterior excluded; 5 mm line samples); rel_l2 = ||dv|| / ||v_fine|| over "
                      "the points",
    }, nd


def fig_comparison(c, f, nd, path):
    pc, pf = c["probes"], f["probes"]
    fig = plt.figure(figsize=(17, 10))
    ax0 = fig.add_axes([0.04, 0.55, 0.46, 0.4])
    m = ax0.pcolormesh(pc["grid_x"], pc["grid_y"], grid_field(pc, nd), cmap="magma", shading="auto")
    draw_annotations(ax0, labels=False, pillows=False)
    fig.colorbar(m, ax=ax0, label="|v_coarse − v_fine| [m/s]")
    ax0.set_title("coarse − fine velocity difference at common points")
    specs = [("h_y3", 1, "v_y along y = 3 (bay entrances)", 0),
             ("h_y1", 0, "v_x along y = 1 (aisle → doorway → bathroom)", 0),
             ("v_x1.5", None, "|v| along x = 1.5 (bay 1 axis, door below)", 1),
             ("v_x4.5", None, "|v| along x = 4.5 (bay 2 axis)", 1)]
    boxes = [[0.57, 0.55, 0.41, 0.4], [0.04, 0.06, 0.29, 0.38], [0.375, 0.06, 0.29, 0.38], [0.71, 0.06, 0.27, 0.38]]
    for (name, comp, title, axis), box in zip(specs, boxes):
        ax = fig.add_axes(box)
        for d, sty in ((c, "-"), (f, "--")):
            p = d["probes"][f"line_{name}_points"]
            v = d["probes"][f"line_{name}_velocity"]
            y = v[:, comp] if comp is not None else np.linalg.norm(v, axis=1)
            s_ = p[:, axis].copy()
            gap = np.where(np.diff(s_) > 0.02)[0] + 1        # break the curve across solids
            s_, y = np.insert(s_, gap, np.nan), np.insert(y, gap, np.nan)
            ax.plot(s_, y, sty, lw=1.1, label=d["level"])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x [m]" if axis == 0 else "y [m]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=mdc.OUTPUT_DIR)
    a = ap.parse_args()
    out_root = a.out_root.resolve()
    rep = out_root / "report"
    rep.mkdir(parents=True, exist_ok=True)
    data = [d for d in (load_level(out_root, lv) for lv in LEVELS) if d is not None]
    solved = [d for d in data if "meta" in d]
    figs = {}

    fig_geometry(data[0]["md"], rep / "fig_geometry_boundary_tags.png")
    figs["geometry"] = "fig_geometry_boundary_tags.png"
    if fig_geometry_vs_reference(data[0]["md"], rep / "fig_geometry_vs_reference.png"):
        figs["geometry_vs_reference"] = "fig_geometry_vs_reference.png"
    for d in data:
        fig_mesh(d, rep / f"fig_mesh_{d['level']}.png")
        figs[f"mesh_{d['level']}"] = f"fig_mesh_{d['level']}.png"
        fig_mesh_bed_holes(d, rep / f"fig_mesh_bed_holes_{d['level']}.png")
        figs[f"mesh_bed_holes_{d['level']}"] = f"fig_mesh_bed_holes_{d['level']}.png"

    summary = {"generated_utc": datetime.now(timezone.utc).isoformat(), "out_root": str(out_root),
               "parameters": cfg.as_dict(), "geometry_sha256": geo.geometry_hash(),
               "weak_flow_threshold_m_per_s": WEAK_FRACTION * cfg.INLET_MEAN_SPEED, "levels": {}}
    for d in data:
        mi = d["mesh_info"]
        entry = {"mesh": {k: mi[k] for k in ("n_vertices", "n_triangles", "taylor_hood_velocity_dofs_P2_vector",
                                             "taylor_hood_pressure_dofs_P1", "taylor_hood_total_dofs", "sizes_m",
                                             "min_angle_deg", "h_max_edge_m", "edges_across_each_opening",
                                             "msh_sha256", "msh_file", "bed_edges_per_bed",
                                             "topology_and_solids", "tag_transfer_gmsh_to_dolfinx",
                                             "vtu_file")}}
        entry["mesh"]["xdmf_files"] = mi.get("xdmf_files", [])
        if "meta" in d:
            mt = d["meta"]
            rm, _ = region_metrics(d)
            entry.update({"status": mt["status"], "picard": mt["picard"], "linear_system": mt["linear_system"],
                          "natural_outflow_check": mt["nonlinear_residual_natural_outflow_check"],
                          "taylor_hood": mt["taylor_hood_diagnostics"], "transport_P1": mt["transport_P1_diagnostics"],
                          "bay_flux_y3": mt["bay_flux_across_partition_tip_line_y3"], "regions": rm,
                          "doorway_flux": mt["bathroom_doorway_flux"],
                          "files": mt["files"], "provenance": mt["provenance"], "command": mt["command"]})
        summary["levels"][d["level"]] = entry

    if solved:
        vmax = max(float(np.linalg.norm(d["vel"], axis=1).max()) for d in solved)
        fig_velocity(solved, [rep / f"fig_velocity_{d['level']}.png" for d in solved],
                     rep / "fig_velocity_coarse_vs_fine.png", vmax)
        for d in solved:
            figs[f"velocity_{d['level']}"] = f"fig_velocity_{d['level']}.png"
            fig_arrows(d, rep / f"fig_velocity_arrows_{d['level']}.png", vmax)
            figs[f"velocity_arrows_{d['level']}"] = f"fig_velocity_arrows_{d['level']}.png"
        figs["velocity_pair"] = "fig_velocity_coarse_vs_fine.png"
        plim = (min(d["p_nodes"].min() for d in solved), max(d["p_nodes"].max() for d in solved))
        fig_pressure(solved, rep / "fig_pressure_coarse_vs_fine.png", plim)
        figs["pressure"] = "fig_pressure_coarse_vs_fine.png"
        fig_picard(solved, rep / "fig_picard_convergence.png")
        figs["picard"] = "fig_picard_convergence.png"
        summary["shared_color_scales"] = {"speed_m_per_s": [0.0, vmax], "pressure": list(map(float, plim))}
    if len(solved) == 2:
        comp, nd = compare_levels(solved[0], solved[1])
        summary["coarse_vs_fine"] = comp
        fig_comparison(solved[0], solved[1], nd, rep / "fig_coarse_fine_comparison.png")
        figs["comparison"] = "fig_coarse_fine_comparison.png"
    summary["figures"] = figs
    env_dir = out_root / "env"
    if env_dir.exists():
        summary["env_files"] = sorted(str(p) for p in env_dir.iterdir())
    cmd_file = out_root / "RUN_COMMANDS.txt"
    (rep / "report_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (rep / "README.md").write_text(render_markdown(summary, cmd_file))
    print(f"report written to {rep}")


def render_markdown(s, cmd_file: Path) -> str:
    P = s["parameters"]
    L = []
    add = L.append
    add(f"# Application2 — three-bed room with bathroom: meshes and steady flow ({geo.OUTPUT_VERSION})\n")
    add(f"Generated {s['generated_utc']}.\n")
    add(f"> {P['model_statement']}\n")
    add("The results below describe these particular computed fields only; they do not establish ventilation "
        "effectiveness.\n")
    add(f"{P['layout_attribution']}\n")
    add("**Geometry** (`room_geometry_config.py`, metres): main room [0,9]×[0,6] ∪ bathroom annex [9,11]×[0,2], "
        "minus two 0.10 m partitions [2.95,3.05]×[3,6] and [5.95,6.05]×[3,6], three **solid beds** " + ", ".join(
            f"[{v['x'][0]:g},{v['x'][1]:g}]×[{v['y'][0]:g},{v['y'][1]:g}]" for v in geo.BEDS.values())
        + " (idealized 2D impermeable obstacles; holes in the mesh), and a 0.10 m bathroom wall [8.95,9.05]×[0,0.6] "
        "∪ [8.95,9.05]×[1.4,2] whose gap 0.6 < y < 1.4 is an open interior doorway. Fluid area "
        f"{geo.fluid_area():.2f} m². Supplies: three top-wall openings x ∈ [1.2,1.8], [4.2,4.8], [7.2,7.8] at y = 6, "
        "flow downward. Outlets: corridor doorway y = 0, x ∈ [0.6,1.6] and bathroom exhaust x = 11, y ∈ [0.7,1.3], "
        "both natural (ν∇v − pI)n = 0 with no prescribed pressure or velocity. All other boundaries are no-slip. "
        "Pillows drawn in figures are presentation only. The reference drawing `room_layout_reference.png` is not to "
        "scale; the coordinates define the geometry (see `fig_geometry_vs_reference.png`).\n")
    add("## Status\n")
    add("| level | status | Picard iterations | final rel. update | MUMPS null pivots | pressure Dirichlet dofs |")
    add("|---|---|---|---|---|---|")
    for lv, e in s["levels"].items():
        if "status" in e:
            add(f"| {lv} | **{e['status']}** | {e['picard']['iterations']} | {e['picard']['final_relative_update']:.2e} "
                f"| {e['linear_system']['mumps_null_pivots_INFOG28']} | {e['linear_system']['pressure_dirichlet_dofs']} |")
        else:
            add(f"| {lv} | mesh only (no flow solve found) | | | | |")
    add("\n## Parameters (navier_stokes_config.py)\n")
    add(f"- Supply profile: parabolic, downward, zero at the vent endpoints; **mean** speed {P['inlet_mean_speed_m_per_s']} m/s, peak "
        f"{P['inlet_peak_speed_m_per_s']} m/s (peak = 1.5 × mean); flux per inlet "
        f"{P['inlet_mean_speed_m_per_s'] * geo.VENT_WIDTH:.3f} m²/s per metre depth.")
    add(f"- ν = {P['nu_m2_per_s']:g} m²/s — {P['nu_kind']}.")
    R = P["reynolds_numbers"]
    add(f"- Re_slot = U_mean·b/ν = {R['Re_slot_mean_speed_vent_width']:.1f} (b = 0.60 m vent width);  "
        f"Re_room = U_mean·H/ν = {R['Re_room_mean_speed_room_depth']:.0f} (H = 6 m room depth in plan). "
        f"Molecular air viscosity (≈1.5e-5 m²/s) would give Re_slot ≈ {R['Re_slot_if_molecular_air']:.0f}.")
    add(f"- Approximate air-change rate ≈ {P['approx_ach_full_height_openings']:.1f} 1/h on the "
        f"{P['approx_ach_area_basis_m2']:.1f} m² fluid area ({P['approx_ach_gross_room_area']:.1f} 1/h on the gross "
        f"{geo.gross_area():g} m² floor), valid only under these "
        f"assumptions: {P['approx_ach_assumptions']}. Real terminals do not span the room height.")
    add("- Parameter selection: at U_mean = 0.30 m/s, ν = 5e-3 (Re_slot = 36) Picard did not converge on either "
        "mesh; Newton found the symmetric steady state but it is linearly unstable (leading eigenvalue ≈ −1.2e-3 1/s "
        "on both meshes). At the adopted Re_slot = 25.8 the steady state is linearly stable (leading eigenvalue "
        "+3.95e-3 1/s, coarse). Trials: `output/v1_nobeds_bottomsupply_20261116/trials_20261116_picard_viscosity/`. "
        "These trials used the earlier v1 geometry (no beds, bottom supply); the parameters are retained unchanged "
        "for this layout and its stability has not been re-assessed.")
    add(f"- Picard: tolerance {P['picard_tol_relative_L2_update']:g} (relative L2 update), max "
        f"{P['max_picard_iter']} iterations; iteration 1 is Stokes. Same for both levels.\n")
    add("## Meshes\n")
    add("| level | vertices | triangles | P2 velocity dofs | P1 pressure dofs | total | lc_bulk | lc_corner | min angle | edges per opening |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for lv, e in s["levels"].items():
        m = e["mesh"]
        add(f"| {lv} | {m['n_vertices']} | {m['n_triangles']} | {m['taylor_hood_velocity_dofs_P2_vector']} | "
            f"{m['taylor_hood_pressure_dofs_P1']} | {m['taylor_hood_total_dofs']} | {m['sizes_m']['lc_bulk']} | "
            f"{m['sizes_m']['lc_corner']} | {m['min_angle_deg']['min']:.1f}° | "
            f"{sorted(set(m['edges_across_each_opening'].values()))} |")
    add("\nTransport (P1) nodes = mesh vertices. Mesh files and SHA-256 are in `<level>/mesh/*_mesh_info.json`.\n")
    add("### Geometry checks (Room_mesh/build_room_mesh.py; a failure aborts the build)\n")
    for lv, e in s["levels"].items():
        m = e["mesh"]
        t, tt = m["topology_and_solids"], m["tag_transfer_gmsh_to_dolfinx"]
        add(f"- **{lv}**: {t['edge_connected_components']} edge-connected component; {t['boundary_loops']} boundary "
            f"loops (outer + 3 beds); triangle centroids inside solids: {t['triangle_centroids_inside_solids']}, "
            f"outside room ∪ annex: {t['triangle_centroids_outside_union']}; "
            "solid-interior samples covered by triangles: "
            f"{sum(v['samples_inside_triangles'] for v in t['solid_interior_sampling'].values())}; Gmsh→DOLFINx "
            f"tag transfer {'OK' if tt['ok'] else 'FAILED'} (cell tags {tt['cell_tags']}); edges per bed "
            f"{sorted(set(m['bed_edges_per_bed'].values()))}.")
        add("  Passages (clear width / median element size): " + ", ".join(
            f"{k} {v['clear_width_m']:.2f} m ≈ {v['approx_elements_across']:.0f} el."
            for k, v in t["passages"].items()) + "; every passage sample point lies in a fluid triangle.")
    add("")
    for lv, e in s["levels"].items():
        if "taylor_hood" not in e:
            continue
        th, p1, nc = e["taylor_hood"], e["transport_P1"], e["natural_outflow_check"]
        add(f"## {lv}: fluxes, divergence, natural-outflow check\n")
        add("| boundary | Taylor–Hood flux ∫v·n | P1 transport flux |")
        add("|---|---|---|")
        for nm in list(geo.INLET_NAMES) + list(geo.OUTLET_NAMES):
            add(f"| {nm} | {th['flux_by_tag'][nm]:+.6f} | {p1['flux_by_tag'][nm]:+.6f} |")
        add(f"| all no-slip walls | {th['wall_flux_total']:+.1e} | "
            f"{sum(p1['flux_by_tag'][n] for n in geo.WALL_NAMES):+.1e} |")
        add(f"| **net** (normalized by inflow) | {th['net_boundary_flux']:+.2e} "
            f"({th['normalized_imbalance_abs_net_over_abs_inflow']:.1e}) | {p1['net_boundary_flux']:+.2e} "
            f"({p1['normalized_imbalance_abs_net_over_abs_inflow']:.1e}) |")
        add(f"\nFlux units m²/s per metre depth, outward normal (inflow < 0). Prescribed per inlet: "
            f"{th['prescribed_inflow_per_inlet']:+.4f}.")
        add(f"- Divergence, Taylor–Hood: ‖∇·v‖/‖∇v‖ = {th['div_L2_over_grad_L2']:.2e} (∫∇·v = {th['integral_div']:.1e}); "
            f"P1 transport: ‖∇·v‖/‖∇v‖ = {p1['div_L2_over_grad_L2']:.2e}, max |∇·v| = {p1['div_abs_max']:.2e} 1/s. "
            "Taylor–Hood is only weakly divergence-free; the P1 interpolant is not divergence-free.")
        add(f"- Natural outflow: max |momentum residual| on free outlet dofs {nc['momentum_max_abs_on_outlet_dofs']:.1e}, "
            f"interior {nc['momentum_max_abs_on_interior_free_dofs']:.1e}, vs Dirichlet reactions "
            f"{nc['momentum_max_abs_on_dirichlet_dofs_reaction']:.1e} (ratio {nc['outlet_over_reaction']:.1e}).")
        add("\n**Outlet split and backflow** (computed, not prescribed):\n")
        add("| outlet | flux | share of inflow | mean u·n [m/s] | min u·n at dofs | backflow flux | backflow length | "
            "mean p | mean |traction·n| |")
        add("|---|---|---|---|---|---|---|---|---|")
        for nm, o in th["outlets"].items():
            add(f"| {nm} | {o['flux']:+.6f} | {100 * o['share_of_total_inflow']:.2f} % | "
                f"{o['mean_normal_velocity']:+.4f} | {o['min_normal_velocity_at_P2_dofs']:+.2e} | "
                f"{o['backflow_flux_inward']:+.2e} | {o['backflow_length_m']:.3f} m "
                f"({100 * o['backflow_length_fraction']:.1f} %) | {o['mean_pressure']:+.2e} | "
                f"{o['mean_abs_normal_traction']:.1e} |")
        dw = e["doorway_flux"]
        add(f"\nBathroom doorway (x = 9, 0.6 < y < 1.4): net flux into the bathroom "
            f"{dw['net_flux_into_bathroom_+x']:+.5f} (into {dw['into_bathroom_part']:+.5f}, back "
            f"{dw['back_into_room_part']:+.5f}). The outlets are passive natural-outflow boundaries: they do not "
            "model mechanical suction or a negative-pressure room.\n")
        add(f"- Wall nodes in transport field: max |v| = {p1['wall_node_max_speed']:.1e}.\n")
    solved = {lv: e for lv, e in s["levels"].items() if "regions" in e}
    if solved:
        add("## Airflow by region (descriptive)\n")
        add(f"Weak flow: |v| < {s['weak_flow_threshold_m_per_s']:.3f} m/s (= {WEAK_FRACTION:g} × mean inlet speed). "
            "Metrics on the 0.05 m probe grid (area-weighted).\n")
        add("| level | region | mean |v| | max |v| | weak-area fraction | v_y<0 area fraction | net flux across y=3 |")
        add("|---|---|---|---|---|---|---|")
        for lv, e in solved.items():
            for rn, r in e["regions"].items():
                bf = ""
                if rn.startswith("bay_"):
                    bf = f"{e['bay_flux_y3'][rn[4:]]['net_upward_flux_across_y3']:+.4f}"
                add(f"| {lv} | {rn} | {r['mean_speed']:.3f} | {r['max_speed']:.3f} | {r['area_fraction_weak']:.2f} | "
                    f"{r['area_fraction_downward_vy_lt_0']:.2f} | {bf} |")
        add("\nNet flux across y = 3 is per bay (negative = from the bay down into the aisle). The supplies blow "
            "downward, so v_y < 0 is the main flow direction in the bays; see the streamline figures for "
            "recirculation.\n")
    if "coarse_vs_fine" in s:
        c = s["coarse_vs_fine"]
        add("## Coarse vs fine (common physical points)\n")
        add(f"- Velocity: relative L2 difference {c['velocity_rel_l2_diff_grid']:.2e}; max |Δv| "
            f"{c['velocity_max_abs_diff']:.2e} m/s at {np.round(c['velocity_max_abs_diff_location'], 3).tolist()}; "
            f"95th percentile {c['velocity_p95_abs_diff']:.2e} m/s.")
        add(f"- Pressure: relative L2 difference {c['pressure_rel_l2_diff_grid']:.2e}.")
        add("- Lines: " + "; ".join(f"{k}: rel {v['rel_l2_diff']:.1e}" for k, v in c["lines"].items()))
        add(f"- {c['definition']}\n")
    add("## ParaView\n")
    add(f"All paths below are relative to `Applications/Application2/output/{geo.OUTPUT_VERSION}/`. "
        "Quickest view: File → Open `<level>/navier_stokes/application2_velocity_pressure_P1_<level>.vtu` (reader "
        "picked automatically: XML Unstructured Grid Reader), Apply, colour by `speed`; Filters → Glyph "
        "(Orientation `velocity`, Scale `speed`) for arrows, or Filters → Stream Tracer (Vectors `velocity`, "
        "seed type Line) for streamlines. Open `<level>/mesh/application2_room_<level>.vtu`, colour by "
        "`physical_tag`, and use Representation 'Surface With Edges' to see the element edges around the bed holes "
        "and the tagged boundary lines.\n")
    add("| file | content | ParaView reader |")
    add("|---|---|---|")
    for lv in s["levels"]:
        add(f"| `{lv}/mesh/application2_room_{lv}.vtu` | mesh + tagged boundary lines, cell data `physical_tag` "
            "(supplies 21–23, outlets 31–32, walls 41–47, partitions 51–52, beds 61–63, bathroom wall 71–72) | "
            "XML Unstructured Grid Reader (built in) |")
        add(f"| `{lv}/mesh/application2_room_{lv}_facets.xdmf` (+ `.h5`) | DOLFINx facet tags | Xdmf3ReaderS / "
            "Xdmf3ReaderT |")
        add(f"| `{lv}/navier_stokes/application2_velocity_pressure_P1_{lv}.vtu` | P1 velocity (3-component), "
            "pressure, speed at vertices | XML Unstructured Grid Reader (built in) |")
        add(f"| `{lv}/navier_stokes/application2_velocity_pressure_P1_{lv}.xdmf` (+ `.h5`) | P1 velocity, pressure "
            "| Xdmf3ReaderS / Xdmf3ReaderT |")
        add(f"| `{lv}/navier_stokes/application2_velocity_P2_{lv}.bp`, `application2_pressure_P1_{lv}.bp` | "
            "native P2 velocity / P1 pressure (directories) | ADIOS2VTXReader (ParaView ≥ 5.12 with ADIOS2) |")
    add("\nBeds, partitions and the bathroom wall are holes/notches in every mesh/field file (no cells inside "
        "them). Choose 'Xdmf3ReaderS' when ParaView asks which XDMF reader to use.\n")
    add("## Figures\n")
    for k, v in s["figures"].items():
        add(f"- {k}: `{v}`")
    add("\n## Provenance\n")
    if cmd_file.exists():
        add("Commands (RUN_COMMANDS.txt):\n\n```text\n" + cmd_file.read_text().strip() + "\n```\n")
    for lv, e in solved.items():
        pv = e["provenance"]
        add(f"- {lv}: git {pv['git_commit']} ({pv['git_branch']}), dolfinx {pv['dolfinx']}, basix {pv['basix']}, "
            f"PETSc {pv['petsc4py']}, numpy {pv['numpy']}, scipy {pv['scipy']}; command `{e['command']}`")
    if "env_files" in s:
        add("- Environment captures: " + ", ".join(f"`{Path(p).name}`" for p in s["env_files"]))
    add("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
