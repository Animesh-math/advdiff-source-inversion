"""Build the manifest + job list for the overnight discrete-sensor source-count
inference sweep (see run_source_count_overnight_sweep.sh).

This script is the SINGLE declarative source of truth for the sweep cases:
case name, true control, discrete sensor node ids (or coordinates), the
measurement-boundary record, --max-sources, --source-count-top-k and
--source-count-polish-max-iter. It resolves sensor node ids from the live DC
mesh (so ids are guaranteed valid), validates every true-control source center
against MeshAdmissibility, reports wall/building clearance, and refuses to
write a manifest if any case reuses a previously-used sensor node id.

Usage:
    "$PY" build_source_count_overnight_sweep.py --out-dir <logdir>

Writes, under --out-dir:
    manifest.json   full structured record (one entry per case)
    cases.tsv       flat job list consumed by run_source_count_overnight_sweep.sh
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_APP1_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _APP1_DIR not in sys.path:
    sys.path.insert(0, _APP1_DIR)

from mesh_data_config import build_mesh_data
from mesh_admissibility import MeshAdmissibility


# ---------------------------------------------------------------------------
# Sensor node ids already used by prior Application1 runs, per
# the development log (App1 discrete mesh-node sensor observations,
# 2026-08-18, and App1 adaptive source-count inference, 2026-08-19) and the
# checked-in "test scripts" default configs. No output/ directory exists on
# this machine (fresh checkout on the SSH host), so there are no
# output/*/full/source_count_inference_summary.json or
# optimization_summary.json files to scan here; this hardcoded set is the
# complete historical record available and is treated as authoritative for
# "do not reuse" purposes.
# ---------------------------------------------------------------------------
PRIOR_USED_SENSOR_NODE_IDS = {
    # "old right/bottom/top 25-node set": node_sensors_right_bottom_top_c06 /
    # node_sensors_rbt_infer_m1_c01 / m2_c02 / m3_c01, and
    # test scripts/diagnose_post_init_polish.py, test_post_init_method_flag.py
    641, 644, 647, 650, 653, 656, 659, 662, 665, 668, 671, 674, 677,  # right
    582, 585, 588, 591, 594, 596,  # bottom
    693, 690, 687, 684, 681, 679,  # top
    # 18-node sparse smoke-test config (gradient_test_node_sensors.py /
    # output/node_sparse_smoke): coordinate-snapped, listed as resolved ids
    # in the 2026-08-18 phase log.
    2456, 2697, 2708, 644, 965, 2357, 677, 1651, 1665, 1458, 1762, 1127,
    334, 1574, 1228, 1804, 1932, 1950,
}

MIN_CLEARANCE_WARN_M = 10.0


def resolve_mesh():
    mesh_data = build_mesh_data()
    admissibility = MeshAdmissibility(mesh_data)
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)
    be = mesh_data["boundary_edges_by_name"]
    wall_ids = sorted(
        {int(i) for edges in be.values() for i in np.asarray(edges).flatten()}
    )
    wall_pts = coord[wall_ids][:, :2]
    return mesh_data, admissibility, coord, wall_pts


def clearance(wall_pts, x, y) -> float:
    d = np.hypot(wall_pts[:, 0] - x, wall_pts[:, 1] - y)
    return float(d.min())


def nearest_node(coord, x, y):
    d = np.hypot(coord[:, 0] - x, coord[:, 1] - y)
    idx = int(np.argmin(d))
    return idx, float(d[idx]), float(coord[idx, 0]), float(coord[idx, 1])


# ---------------------------------------------------------------------------
# Case declarations
# ---------------------------------------------------------------------------
# Sensor node ids below are grouped by the boundary/building family they come
# from purely for readability; the CLI receives one flat comma-separated
# --sensor-node-ids list.

CASE_A = {
    "case_name": "sc_overnight_A_close_pair_m3",
    "purpose": (
        "Three-source truth with two close sources (~15.8 m apart, inside "
        "one sigma e-folding radius of 20 m) plus one far source. Tests "
        "whether source-count inference separates the close pair instead of "
        "merging it into one accepted source, then correctly rejects a "
        "spurious 4th source."
    ),
    "true_control": [130.0, 285.0, 1.20, 145.0, 290.0, 1.05, 255.0, 610.0, 0.95],
    "expected_m": 3,
    "sensor_family": "left_wall_sparse + building_12 (near close pair) + right/top near far source",
    "sensor_node_ids": sorted(set(
        [640, 635, 630, 625, 620, 615, 610, 605, 600, 597]  # left wall, sparse
        + [133, 134, 135, 136, 445, 446]  # building_12 (hugs the close pair)
        + [672, 675, 678]  # right wall, near far source y~610
        + [680]  # top wall, near far source x~255
    )),
    "measurement_boundaries": ["left", "right", "top"],
    "max_sources": 5,
    "top_k": 12,
    "polish_max_iter": 150,
}

CASE_B1 = {
    "case_name": "sc_overnight_B1_building_facing",
    "purpose": (
        "Two-source case observed only by sensors on two building "
        "boundaries (building_13, building_16) -- family 1: "
        "building-facing sensors."
    ),
    "true_control": [195.0, 345.0, 1.15, 105.0, 600.0, 0.80],
    "expected_m": 2,
    "sensor_family": "building_13 + building_16 boundary nodes",
    "sensor_node_ids": sorted(set(
        [93, 96, 99, 102, 105, 108, 388, 391, 394, 397, 400]  # building_13
        + [86, 89, 92, 364, 367, 370, 373, 376, 379, 382, 385]  # building_16
    )),
    "measurement_boundaries": ["left"],
    "max_sources": 4,
    "top_k": 12,
    "polish_max_iter": 150,
}

CASE_B2 = {
    "case_name": "sc_overnight_B2_right_wall_plus_buildings",
    "purpose": (
        "Two-source case in the north end of the domain observed by a mix "
        "of right-wall outer-boundary sensors and building_1 / building_4 "
        "boundary sensors -- family 2: mixed right wall + building nodes."
    ),
    "true_control": [100.0, 640.0, 1.05, 250.0, 660.0, 0.90],
    "expected_m": 2,
    "sensor_family": "right wall (fresh ids) + building_1 + building_4 boundary nodes",
    "sensor_node_ids": sorted(set(
        [660, 663, 666, 669, 673, 676]  # right wall, fresh ids
        + [171, 174, 177, 495, 498, 501, 504, 507, 510, 513]  # building_1
        + [34, 37, 40, 43, 46, 49, 264, 267, 270, 273, 276, 279]  # building_4
    )),
    "measurement_boundaries": ["right"],
    "max_sources": 4,
    "top_k": 12,
    "polish_max_iter": 150,
}

CASE_B3 = {
    "case_name": "sc_overnight_B3_top_bottom_sparse_plus_buildings",
    "purpose": (
        "Two-source case observed by sparse top/bottom outer-boundary "
        "sensors plus building_8 / building_15 boundary sensors -- family "
        "3: top/bottom sparse nodes + selected building nodes."
    ),
    "true_control": [270.0, 405.0, 0.95, 75.0, 345.0, 1.10],
    "expected_m": 2,
    "sensor_family": "top + bottom (fresh ids) + building_8 + building_15 boundary nodes",
    "sensor_node_ids": sorted(set(
        [682, 685, 688, 691]  # top wall, fresh ids
        + [583, 587, 590, 593, 595]  # bottom wall, fresh ids
        + [0, 5, 10, 15, 223, 228, 233, 238, 243, 248, 253]  # building_8
        + [149, 152, 155, 158, 454, 457, 460, 463, 466, 469, 472]  # building_15
    )),
    "measurement_boundaries": ["top", "bottom"],
    "max_sources": 4,
    "top_k": 12,
    "polish_max_iter": 150,
}

CASE_B4 = {
    "case_name": "sc_overnight_B4_interior_coordinates_m3",
    "purpose": (
        "Three-source, well-separated truth observed by 16 interior-ish "
        "sensor locations chosen by coordinate and snapped to the nearest "
        "mesh node -- family 4: interior node sensors by coordinate."
    ),
    "true_control": [286.0, 318.0, 1.00, 170.0, 494.0, 0.80, 90.0, 26.0, 1.20],
    "expected_m": 3,
    "sensor_family": "16 interior-ish points, snapped to nearest mesh node",
    "sensor_coordinates": (
        "30,225;45,255;60,345;90,375;105,420;120,510;135,570;150,630;"
        "180,600;210,405;225,630;240,390;255,420;270,225;285,135;285,405"
    ),
    "measurement_boundaries": None,
    "max_sources": 5,
    "top_k": 12,
    "polish_max_iter": 150,
}

CASE_D = {
    "case_name": "sc_overnight_D_eight_source_stress",
    "purpose": (
        "Eight-source stress case: many admissible, separated sources "
        "observed by a rich mix of left/right outer-boundary sensors plus "
        "building_9 / building_19 / building_3 boundary sensors. May fail "
        "or undercount; still informative."
    ),
    "true_control": [
        30.0, 45.0, 0.65,
        270.0, 30.0, 1.55,
        120.0, 255.0, 0.85,
        270.0, 165.0, 1.35,
        105.0, 405.0, 0.70,
        270.0, 405.0, 1.20,
        135.0, 600.0, 1.05,
        255.0, 630.0, 1.50,
    ],
    "expected_m": 8,
    "sensor_family": "left + right wall (fresh ids) + building_9 + building_19 + building_3",
    "sensor_node_ids": sorted(set(
        [639, 631, 623, 616, 608, 601, 598]  # left wall, fresh ids
        + [643, 648, 651, 661, 664]  # right wall, fresh ids
        + [179, 182, 185, 188, 516, 519, 522, 525]  # building_9
        + [109, 112, 115, 118, 402, 405, 408, 411]  # building_19
        + [159, 161, 163, 165, 474, 477, 480, 483]  # building_3
    )),
    "measurement_boundaries": ["left", "right"],
    "max_sources": 10,
    "top_k": 16,
    "polish_max_iter": 150,
}

BASE_CASES = [CASE_A, CASE_B1, CASE_B2, CASE_B3, CASE_B4, CASE_D]

# Case C: cheaper per-level polish reruns of two representative base cases
# (one 3-source, one 2-source), same true control + same sensor ids/coords,
# only --source-count-polish-max-iter changes.
POLISH50_SOURCE_CASE_NAMES = [CASE_A["case_name"], CASE_B2["case_name"]]


def make_polish50_variant(base_case: dict) -> dict:
    variant = dict(base_case)
    variant["case_name"] = base_case["case_name"] + "_polish50"
    variant["purpose"] = (
        f"Cheaper per-level polish rerun of {base_case['case_name']} "
        "(--source-count-polish-max-iter 50 instead of 150), same true "
        "control and same sensor configuration, to compare inferred count, "
        "stop reason, final objective and recovery errors against the "
        "150-iteration case."
    )
    variant["polish_max_iter"] = 50
    variant["polish50_of"] = base_case["case_name"]
    return variant


def build_cases() -> list:
    cases = list(BASE_CASES)
    by_name = {c["case_name"]: c for c in BASE_CASES}
    for name in POLISH50_SOURCE_CASE_NAMES:
        cases.append(make_polish50_variant(by_name[name]))
    return cases


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_and_annotate(cases: list, mesh_data, admissibility, coord, wall_pts) -> list:
    npoin = coord.shape[0]
    errors = []
    seen_across_new_cases = {}

    for case in cases:
        name = case["case_name"]

        # --- true control admissibility + clearance -------------------
        tc = case["true_control"]
        if len(tc) % 3 != 0:
            errors.append(f"{name}: true_control length {len(tc)} not a multiple of 3")
            continue
        m = len(tc) // 3
        clearances = []
        for s in range(m):
            x, y, a = tc[3 * s], tc[3 * s + 1], tc[3 * s + 2]
            if not admissibility.is_point_in_domain(x, y):
                errors.append(
                    f"{name}: source {s + 1} center ({x}, {y}) is NOT admissible "
                    "(outside fluid mesh / inside a building)"
                )
            c = clearance(wall_pts, x, y)
            clearances.append(round(c, 2))
            if c < MIN_CLEARANCE_WARN_M:
                print(
                    f"WARNING: {name} source {s + 1} ({x}, {y}) has only "
                    f"{c:.2f} m wall/building clearance (< {MIN_CLEARANCE_WARN_M} m)"
                )
        case["wall_clearance_m"] = clearances
        case["num_true_sources"] = m

        # pairwise source separation, informational
        seps = []
        for i in range(m):
            for j in range(i + 1, m):
                dx = tc[3 * i] - tc[3 * j]
                dy = tc[3 * i + 1] - tc[3 * j + 1]
                seps.append(round(float(np.hypot(dx, dy)), 1))
        case["pairwise_source_separations_m"] = seps

        # --- sensor ids -------------------------------------------------
        if "sensor_node_ids" in case:
            ids = case["sensor_node_ids"]
            if len(ids) != len(set(ids)):
                errors.append(f"{name}: duplicate sensor node ids in the case definition")
            for i in ids:
                if i < 0 or i >= npoin:
                    errors.append(f"{name}: sensor node id {i} out of range [0, {npoin - 1}]")
            overlap = sorted(set(ids) & PRIOR_USED_SENSOR_NODE_IDS)
            if overlap:
                errors.append(
                    f"{name}: reuses previously-used sensor node ids {overlap}"
                )
            base_name = name[:-9] if name.endswith("_polish50") else name
            for i in ids:
                prev = seen_across_new_cases.get(i)
                if prev is not None and prev != name and prev != base_name and prev + "_polish50" != name:
                    print(
                        f"note: sensor node id {i} is used by both {prev} and "
                        f"{name} in this NEW sweep (not a prior-output reuse, "
                        "just an incidental overlap between two new configs)"
                    )
                seen_across_new_cases[i] = name
            case["num_sensors"] = len(ids)
            case["sensor_mode"] = "ids"
            coords_out = [
                {"node_id": i, "x": round(float(coord[i, 0]), 3), "y": round(float(coord[i, 1]), 3)}
                for i in ids
            ]
            case["sensor_coordinates_resolved"] = coords_out
        elif "sensor_coordinates" in case:
            case["sensor_mode"] = "coords"
            resolved = []
            ids = []
            for token in case["sensor_coordinates"].split(";"):
                x_str, y_str = token.split(",")
                x, y = float(x_str), float(y_str)
                idx, dist, nx, ny = nearest_node(coord, x, y)
                resolved.append(
                    {
                        "requested_x": x,
                        "requested_y": y,
                        "resolved_node_id": idx,
                        "resolved_x": round(nx, 3),
                        "resolved_y": round(ny, 3),
                        "snap_distance_m": round(dist, 3),
                    }
                )
                ids.append(idx)
            if len(ids) != len(set(ids)):
                errors.append(f"{name}: two requested coordinates snapped to the same node id")
            overlap = sorted(set(ids) & PRIOR_USED_SENSOR_NODE_IDS)
            if overlap:
                errors.append(
                    f"{name}: coordinate-snapped sensor node ids reuse previously-used ids {overlap}"
                )
            case["sensor_coordinates_resolved"] = resolved
            case["sensor_node_ids_resolved_preview"] = ids
            case["num_sensors"] = len(ids)
        else:
            errors.append(f"{name}: no sensor_node_ids or sensor_coordinates given")

    if errors:
        raise SystemExit("Sweep case validation FAILED:\n" + "\n".join(f"  - {e}" for e in errors))

    return cases


def write_outputs(cases: list, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "description": (
            "Overnight discrete-node-sensor source-count inference sweep "
            "(--observation-mode nodes --infer-num-sources). Amplitudes are "
            "not floored (no --amplitude-floor) so spurious sources can "
            "collapse to zero, per the nesting argument in "
            "source_count_inference.py."
        ),
        "prior_used_sensor_node_ids_avoided": sorted(PRIOR_USED_SENSOR_NODE_IDS),
        "prior_used_note": (
            "No output/ directory exists on this machine (fresh checkout); "
            "this list was taken from the development log ('App1 discrete "
            "mesh-node sensor observations' and 'App1 adaptive source-count "
            "inference' phase logs) and the checked-in test-script default "
            "sensor configs, which is the complete historical record "
            "available."
        ),
        "common_cli_flags": {
            "observation_mode": "nodes",
            "sensor_weights": "equal",
            "sensor_equal_weight": 1.0,
            "feasibility_mode": "raw",
            "amplitude_floor": None,
            "source_count_rel_reduction_tol": 1e-3,
            "source_count_abs_reduction_tol": 0.0,
            "source_count_amp_tol": 1e-3,
            "source_count_discrepancy_tol": 0.0,
            "source_count_polish_method": "lbfgsb",
            "max_paraview_frames": 50,
            "max_iter": 200,
            "mode": "location_and_intensity",
            "stage": "full",
        },
        "cases": cases,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    tsv_lines = []
    for case in cases:
        true_control_csv = ",".join(repr(v) for v in case["true_control"])
        if case["sensor_mode"] == "ids":
            sensor_value = ",".join(str(i) for i in case["sensor_node_ids"])
        else:
            sensor_value = case["sensor_coordinates"]
        mb = case.get("measurement_boundaries")
        mb_csv = ",".join(mb) if mb else "-"
        tsv_lines.append(
            "\t".join(
                str(x)
                for x in [
                    case["case_name"],
                    true_control_csv,
                    case["sensor_mode"],
                    sensor_value,
                    mb_csv,
                    case["max_sources"],
                    case["top_k"],
                    case["polish_max_iter"],
                ]
            )
        )
    (out_dir / "cases.tsv").write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")

    print(f"wrote {out_dir / 'manifest.json'}")
    print(f"wrote {out_dir / 'cases.tsv'}")
    print(f"{len(cases)} cases:")
    for case in cases:
        print(
            f"  {case['case_name']:45s} M_true={case['num_true_sources']} "
            f"sensors={case['num_sensors']:3d} max_sources={case['max_sources']} "
            f"top_k={case['top_k']} polish_max_iter={case['polish_max_iter']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    mesh_data, admissibility, coord, wall_pts = resolve_mesh()
    cases = build_cases()
    cases = validate_and_annotate(cases, mesh_data, admissibility, coord, wall_pts)
    write_outputs(cases, args.out_dir)


if __name__ == "__main__":
    main()
