"""Freeze the six-case sensor-layout comparison: sources, layouts, plots, manifest.

Two documented two-source truths x three 25-sensor layouts (boundary-only,
mixed boundary + near-building, deliberately weak).  Everything here is
geometry/velocity only: NO PDE solve, NO inverse run.  The output
(``layouts.json`` + per-case ``case_config.json`` + layout figures) is written
once, before any reconstruction, and is never revised afterwards.

Usage:
    PY sensor_layout_prepare.py --exp-dir <abs experiment dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_APP1_DIR = Path(__file__).resolve().parents[1]
if str(_APP1_DIR) not in sys.path:
    sys.path.insert(0, str(_APP1_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import run_optimization as ro  # noqa: E402
from mesh_admissibility import MeshAdmissibility  # noqa: E402
from mesh_data_config import build_mesh_data  # noqa: E402

# ---------------------------------------------------------------------------
# Source configurations (documented truths, the development log 2026-07-31
# final robustness sweep table).  A = c06 is also the 25-node-sensor
# (right+bottom+top) example of output/node_sensors_right_bottom_top_c06.
# ---------------------------------------------------------------------------
SOURCE_CONFIGS = {
    "A": {
        "label": "c06 (manuscript discrete-sensor example): mid-domain wake + upper-right street",
        "documented_as": "final_sweep c06_nearwall_b13; node_sensors_right_bottom_top_c06",
        "true_control": [162.0, 300.0, 1.40, 270.0, 600.0, 0.95],
    },
    "B": {
        "label": "c07 (lower domain): wake between buildings 15/19 + south-east corner below building 21",
        "documented_as": "final_sweep c07_lower_domain",
        "true_control": [100.0, 210.0, 1.30, 270.0, 60.0, 0.85],
    },
}

# Layout 1: the existing 25 right+bottom+top node sensors of
# node_sensors_right_bottom_top_c06 (README "Post-initialization optimizer"),
# used verbatim for both source configurations.
REFERENCE_BOUNDARY_IDS = [
    641, 644, 647, 650, 653, 656, 659, 662, 665, 668, 671, 674, 677,
    582, 585, 588, 591, 594, 596,
    693, 690, 687, 684, 681, 679,
]

# Layout 2: 10 outer-boundary nodes (subset of the reference set: 6 right,
# 2 bottom, 2 top) + 15 interior fluid points around buildings, in streets and
# wakes.  Same for both source configurations.
MIXED_BOUNDARY_IDS = [641, 647, 653, 662, 671, 677, 588, 596, 690, 684]
MIXED_INTERIOR_REQUESTS = [
    (285.0, 330.0, "wake between buildings 11 and 14"),
    (255.0, 300.0, "street between buildings 13 and 14"),
    (160.0, 255.0, "street between buildings 15 and 16"),
    (240.0, 205.0, "street between buildings 16 and 17"),
    (265.0, 185.0, "east of building 17"),
    (160.0, 100.0, "street between buildings 19 and 20"),
    (210.0, 40.0, "south of building 21"),
    (265.0, 100.0, "east of building 20"),
    (120.0, 300.0, "between buildings 15 and 12 (jet entrance)"),
    (200.0, 380.0, "main jet north of building 13"),
    (290.0, 470.0, "east of building 8"),
    (200.0, 490.0, "street between buildings 4 and 8"),
    (240.0, 590.0, "street north of building 4"),
    (110.0, 545.0, "street between buildings 3 and 4"),
    (70.0, 590.0, "street between building 1 and buildings 2/3"),
]

# Layout 3 (weak, config-specific): sensors chosen from the Navier-Stokes field
# and the building layout to sit UPSTREAM (west, on the inflow side) of the
# sources, or behind buildings / in pockets poorly connected to the sources'
# downstream transport paths.  Intended to provide weak observations; not
# assumed to make recovery impossible.
WEAK_REQUESTS = {
    "A": [
        (12.0, 30.0, "west strip near inflow, far upstream"),
        (12.0, 110.0, "west strip near inflow, far upstream"),
        (12.0, 230.0, "west strip near inflow, upstream of source 1"),
        (12.0, 300.0, "west strip near inflow, upstream of source 1"),
        (12.0, 370.0, "west strip near inflow, upstream"),
        (12.0, 440.0, "west strip near inflow, upstream"),
        (12.0, 560.0, "west strip near inflow, upstream of source 2"),
        (12.0, 650.0, "west strip near inflow, upstream of source 2"),
        (100.0, 690.0, "north-west, upstream of source 2"),
        (100.0, 640.0, "east of building 1, upstream of source 2"),
        (150.0, 690.0, "north edge, upstream/cross-stream of source 2"),
        (150.0, 630.0, "upstream of source 2"),
        (45.0, 592.0, "street west end between 1 and 2, upstream of source 2"),
        (110.0, 545.0, "between buildings 3 and 4, screened from source 2"),
        (95.0, 455.0, "courtyard among buildings 5/6/7/9"),
        (55.0, 490.0, "street between buildings 2 and 9"),
        (40.0, 60.0, "south-west corner jet, upstream"),
        (100.0, 20.0, "south of building 19, screened by 15/16/19"),
        (200.0, 30.0, "south of building 21, screened by 16/17/20/21"),
        (290.0, 30.0, "south-east corner, screened from both sources"),
        (160.0, 150.0, "street between 19 and 20, screened by 16"),
        (195.0, 205.0, "street between 16 and 17, behind building 16"),
        (255.0, 125.0, "east of building 20, screened"),
        (160.0, 60.0, "street between 19 and 20/21, screened"),
        (290.0, 190.0, "east of building 17, screened by 16/17"),
    ],
    "B": [
        (12.0, 300.0, "west strip near inflow, upstream of source 1"),
        (12.0, 370.0, "west strip near inflow, upstream"),
        (12.0, 440.0, "west strip near inflow, upstream"),
        (12.0, 560.0, "west strip near inflow, far upstream"),
        (12.0, 650.0, "west strip near inflow, far upstream"),
        (100.0, 690.0, "north-west, far from both sources"),
        (100.0, 640.0, "east of building 1, far from both sources"),
        (150.0, 690.0, "north edge, far from both sources"),
        (150.0, 630.0, "upper domain, far from both sources"),
        (45.0, 592.0, "street between 1 and 2, upper domain"),
        (110.0, 545.0, "between buildings 3 and 4, upper domain"),
        (95.0, 455.0, "courtyard among buildings 5/6/7/9"),
        (55.0, 490.0, "street between buildings 2 and 9"),
        (200.0, 490.0, "street between buildings 4 and 8"),
        (240.0, 590.0, "street north of building 4"),
        (290.0, 470.0, "east of building 8"),
        (275.0, 530.0, "east of building 4"),
        (185.0, 600.0, "street north of building 4"),
        (290.0, 650.0, "north-east, upper domain"),
        (220.0, 690.0, "north edge, upper domain"),
        (285.0, 330.0, "wake between 11 and 14, screened by 13/16 from source 1"),
        (255.0, 300.0, "street 13/14, cross-stream of both sources"),
        (200.0, 330.0, "wake north of building 13, screened by 16"),
        (160.0, 420.0, "street between 8 and 10 jet, north of all sources"),
        (240.0, 420.0, "main jet north of building 10, north of all sources"),
    ],
}

LAYOUT_INFO = {
    "L1_boundary": "boundary-only: 25 outer-boundary nodes (right 13, bottom 6, top 6), "
                   "the existing node_sensors_right_bottom_top_c06 set",
    "L2_mixed": "mixed: 10 outer-boundary nodes (right 6, bottom 2, top 2) + 15 interior "
                "fluid nodes around buildings / in streets and wakes",
    "L3_weak": "deliberately weak: 25 interior fluid nodes upstream of the sources or "
               "screened by buildings / poorly connected to their downstream paths "
               "(intended to provide weak observations)",
}


def build_context():
    mesh_data = build_mesh_data()
    velocity, vsummary = ro.load_velocity_field_file(
        str(ro.DEFAULT_VELOCITY_FIELD_FILE), mesh_data
    )
    adm = MeshAdmissibility(mesh_data)
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)
    bn = mesh_data["boundary_nodes_by_name"]
    boundary_ids = set()
    node_boundary = {}
    for name, ids in bn.items():
        for i in np.asarray(ids, dtype=int).tolist():
            boundary_ids.add(i)
            node_boundary.setdefault(i, []).append(name)
    dirichlet = set(np.asarray(bn["left"], dtype=int).tolist())
    return mesh_data, velocity, vsummary, adm, coord, boundary_ids, node_boundary, dirichlet


def snap_interior(coord, boundary_ids, x, y):
    """Nearest mesh node that is NOT on any boundary (outer or building wall)."""
    d = np.hypot(coord[:, 0] - x, coord[:, 1] - y)
    order = np.argsort(d)
    for i in order:
        if int(i) not in boundary_ids:
            return int(i), float(d[i])
    raise RuntimeError("no interior node found")


def sensor_record(idx, node, req, dist, coord, adm, node_boundary, dirichlet, kind, note):
    x, y = float(coord[node, 0]), float(coord[node, 1])
    return {
        "sensor_index": idx,
        "kind": kind,
        "note": note,
        "requested_xy": None if req is None else [float(req[0]), float(req[1])],
        "node_id": int(node),
        "actual_xy": [x, y],
        "snap_distance_m": float(dist),
        "on_boundaries": node_boundary.get(int(node), []),
        "is_homogeneous_dirichlet_node": int(node) in dirichlet,
        "fluid_admissible": bool(adm.is_point_in_domain(x, y))
        if not node_boundary.get(int(node)) else "boundary node (fluid closure)",
        "wall_distance_m": float(adm.distance_to_boundary(x, y))
        if not node_boundary.get(int(node)) else 0.0,
    }


def make_layout(kind_name, cfg, ctx):
    mesh_data, velocity, vsummary, adm, coord, boundary_ids, node_boundary, dirichlet = ctx
    sensors = []
    if kind_name == "L1_boundary":
        for i in REFERENCE_BOUNDARY_IDS:
            sensors.append(sensor_record(len(sensors), i, coord[i], 0.0, coord, adm,
                                         node_boundary, dirichlet, "outer_boundary",
                                         "reference right/bottom/top node"))
    elif kind_name == "L2_mixed":
        for i in MIXED_BOUNDARY_IDS:
            sensors.append(sensor_record(len(sensors), i, coord[i], 0.0, coord, adm,
                                         node_boundary, dirichlet, "outer_boundary",
                                         "subset of the reference boundary set"))
        for (x, y, note) in MIXED_INTERIOR_REQUESTS:
            node, dist = snap_interior(coord, boundary_ids, x, y)
            sensors.append(sensor_record(len(sensors), node, (x, y), dist, coord, adm,
                                         node_boundary, dirichlet, "interior_near_building", note))
    elif kind_name == "L3_weak":
        for (x, y, note) in WEAK_REQUESTS[cfg]:
            node, dist = snap_interior(coord, boundary_ids, x, y)
            sensors.append(sensor_record(len(sensors), node, (x, y), dist, coord, adm,
                                         node_boundary, dirichlet, "interior_weak", note))
    ids = [s["node_id"] for s in sensors]
    problems = []
    if len(ids) != 25:
        problems.append(f"{len(ids)} sensors, expected 25")
    if len(set(ids)) != len(ids):
        problems.append("duplicate node ids")
    if any(s["is_homogeneous_dirichlet_node"] for s in sensors):
        problems.append("sensor on homogeneous Dirichlet node")
    if any(s["fluid_admissible"] is False for s in sensors):
        problems.append("interior sensor outside fluid domain")
    if any(s["snap_distance_m"] > 20.0 for s in sensors):
        problems.append("snap distance > 20 m (driver tolerance)")
    if problems:
        raise SystemExit(f"layout {cfg}/{kind_name} invalid: {problems}")
    return sensors


def plot_layout(path, title, mesh_data, coord, velocity, true_control, sensors):
    fig, ax = plt.subplots(figsize=(6.2, 12.5))
    speed = np.linalg.norm(velocity, axis=1)
    tc = ax.tricontourf(coord[:, 0], coord[:, 1], mesh_data["connections"], speed,
                        levels=24, cmap="Blues", alpha=0.75)
    cb = fig.colorbar(tc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("|u| (m/s), Navier-Stokes field")
    sl = slice(None, None, 4)
    ax.quiver(coord[sl, 0], coord[sl, 1], velocity[sl, 0], velocity[sl, 1],
              color="0.25", scale=170, width=0.0018, alpha=0.8)
    for name in mesh_data["building_boundary_names"]:
        e = np.asarray(mesh_data["boundary_edges_by_name"][name])
        for a, b in e:
            ax.plot(coord[[a, b], 0], coord[[a, b], 1], color="k", lw=1.0)
        ids = np.asarray(mesh_data["boundary_nodes_by_name"][name])
        ax.text(coord[ids, 0].mean(), coord[ids, 1].mean(), name.split("_")[1],
                fontsize=6, ha="center", va="center", color="0.3")
    left = np.asarray(mesh_data["boundary_nodes_by_name"]["left"])
    ax.plot(coord[left, 0], coord[left, 1], color="tab:red", lw=3, alpha=0.5,
            label="left: inflow, homogeneous Dirichlet")
    styles = {"outer_boundary": ("s", "tab:orange"),
              "interior_near_building": ("^", "tab:green"),
              "interior_weak": ("D", "tab:purple")}
    seen = set()
    for s in sensors:
        m, col = styles[s["kind"]]
        lab = s["kind"] if s["kind"] not in seen else None
        seen.add(s["kind"])
        ax.plot(*s["actual_xy"], marker=m, color=col, mec="k", ms=7, ls="", label=lab)
    tc3 = np.asarray(true_control).reshape(-1, 3)
    for j, (x, y, a) in enumerate(tc3, start=1):
        ax.plot(x, y, "*", color="red", mec="k", ms=17, label="true source" if j == 1 else None)
        ax.annotate(f"S{j} A={a:g}", (x, y), xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color="darkred", weight="bold")
    ax.annotate("", xy=(70, 715), xytext=(10, 715),
                arrowprops=dict(arrowstyle="->", lw=2, color="k"), annotation_clip=False)
    ax.text(75, 712, "mean wind: west -> east", fontsize=8, clip_on=False)
    ax.set_xlim(-5, 307)
    ax.set_ylim(-5, 725)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=7)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", required=True)
    args = parser.parse_args()
    exp = Path(args.exp_dir).resolve()
    exp.mkdir(parents=True, exist_ok=True)
    ctx = build_context()
    mesh_data, velocity, vsummary, adm, coord, *_ = ctx

    for cfg, c in SOURCE_CONFIGS.items():
        for j, (x, y, a) in enumerate(np.asarray(c["true_control"]).reshape(-1, 3), 1):
            if not adm.is_point_in_domain(x, y):
                raise SystemExit(f"config {cfg} source {j} not admissible")
            c.setdefault("source_wall_clearance_m", []).append(
                float(adm.distance_to_boundary(x, y)))

    cases = []
    layouts = {}
    for cfg in SOURCE_CONFIGS:
        for lay in LAYOUT_INFO:
            case_id = f"{cfg}_{lay}"
            sensors = make_layout(lay, cfg, ctx)
            layouts[case_id] = sensors
            case_dir = exp / "cases" / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            fig_path = case_dir / f"layout_{case_id}.png"
            plot_layout(fig_path,
                        f"{case_id}: config {cfg} ({SOURCE_CONFIGS[cfg]['documented_as']})\n"
                        f"{LAYOUT_INFO[lay]}",
                        mesh_data, coord, velocity, SOURCE_CONFIGS[cfg]["true_control"],
                        sensors)
            rec = {
                "case_id": case_id,
                "source_config": cfg,
                "layout": lay,
                "layout_description": LAYOUT_INFO[lay],
                "true_control": SOURCE_CONFIGS[cfg]["true_control"],
                "sensor_node_ids": [s["node_id"] for s in sensors],
                "sensors": sensors,
                "layout_figure": str(fig_path),
            }
            (case_dir / "case_config.json").write_text(json.dumps(rec, indent=2) + "\n")
            cases.append(rec)

    blob = json.dumps({"source_configs": SOURCE_CONFIGS, "layout_info": LAYOUT_INFO,
                       "cases": cases}, indent=2)
    (exp / "layouts.json").write_text(blob + "\n")
    sha = hashlib.sha256(blob.encode()).hexdigest()
    (exp / "layouts.sha256").write_text(sha + "  layouts.json\n")
    for case in cases:
        s = case["sensors"]
        print(f"{case['case_id']:16s} n={len(s)} max_snap={max(x['snap_distance_m'] for x in s):5.2f} m "
              f"min_interior_wall_dist="
              f"{min([x['wall_distance_m'] for x in s if not x['on_boundaries']] or [float('nan')]):5.2f} m "
              f"ids={case['sensor_node_ids']}")
    print("layouts sha256", sha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
