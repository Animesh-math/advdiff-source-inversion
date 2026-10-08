"""Where was the correct geometric branch lost? (READ-ONLY, truth-aware)

Parses a ``--init-mode sensitivity`` run log and answers, stage by stage, whether
a first-source hypothesis in the TRUE source's street ever existed and where it
died:

    top-K screening -> beam pruning -> refinement -> second-source selection
    -> pair refinement -> retained branches

"In the true street" means: mutually visible from the true center through the
fluid domain (no building in between) and within ``--radius`` metres of it. The
truth is used only to classify what the initializer produced; nothing here feeds
back into the initializer.

Usage:
  cd /path/to/Advection-Diffusion-PDECO
  python3 \
    "Applications/Application1/test scripts/diagnostics/diagnose_branch_loss.py" \
    <run-log> --true-center X Y [--radius M]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

import sys as _sys_dc
from pathlib import Path as _Path_dc
_sys_dc.path.insert(0, str(_Path_dc(__file__).resolve().parents[1]))  # _diag_common
import _diag_common as dc
from mesh_admissibility import MeshAdmissibility

CAND = re.compile(
    r"^\s+cand \(\s*([-\d.]+),\s*([-\d.]+)\) G=\s*([-\d.eE+]+).*?"
    r"dJ_pred=\s*([-\d.eE+]+) rank=(\d+)"
)
LEVEL_HEADER = re.compile(r"^\[level (\d+)\] (root branch|branch \[(.*?)\]):")
KEPT = re.compile(r"^\s+\[level (\d+)\] kept branch (\d+): centers \[(.*?)\]")
REFINE = re.compile(
    r"^\s+\[level (\d+)\] refine branch: \(([-\d.]+),([-\d.]+)\) -> "
    r"\(([-\d.]+),([-\d.]+)\)\s+J ([-\d.eE+]+) -> ([-\d.eE+]+)"
)
DEDUP = re.compile(r"deduplicated (\d+) (?:refined|pair-refined) branch\(es\)")
PAIR = re.compile(r"^\s+branch (\d+): \[(.*?)\] amps")
PAIR_TO = re.compile(r"^\s+-> \[(.*?)\] amps \[(.*?)\] J=([-\d.eE+]+)")
RETAINED = re.compile(r"^\s+branch (\d+): \[(.*?)\] J_exact=([-\d.eE+]+)")


def parse_centers(text: str):
    return [
        (float(a), float(b))
        for a, b in re.findall(r"\(\s*([-\d.]+),\s*([-\d.]+)\)", text)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--true-center", type=float, nargs=2, required=True)
    parser.add_argument("--radius", type=float, default=60.0)
    args = parser.parse_args()

    mesh_data = dc.ro.configure_measurement_boundaries(
        dc.ro.build_mesh_data(), ["left", "right", "bottom"]
    )
    admiss = MeshAdmissibility(mesh_data)
    truth = np.asarray(args.true_center, dtype=np.float64)

    def in_true_street(center) -> bool:
        c = np.asarray(center, dtype=np.float64)
        if float(np.hypot(*(c - truth))) > args.radius:
            return False
        return bool(admiss.segment_in_domain(truth, c, spacing=0.5))

    def tag(center) -> str:
        c = np.asarray(center, dtype=np.float64)
        distance = float(np.hypot(*(c - truth)))
        visible = bool(admiss.segment_in_domain(truth, c, spacing=0.5))
        marker = "TRUE STREET" if in_true_street(c) else "           "
        return (f"{marker}  d(truth)={distance:8.2f} m  visible={str(visible):5}  "
                f"wall={admiss.distance_to_boundary(c[0], c[1]):7.3f} m")

    lines = args.log.read_text(errors="replace").splitlines()
    print(f"true center: ({truth[0]:.2f}, {truth[1]:.2f})   "
          f"street radius: {args.radius:g} m\n")

    level = None
    screened_level1 = []
    print("--- level 1 top-K screening ---")
    for line in lines:
        header = LEVEL_HEADER.match(line)
        if header:
            level = int(header.group(1))
            continue
        match = CAND.match(line)
        if match and level == 1:
            center = (float(match.group(1)), float(match.group(2)))
            rank = int(match.group(5))
            screened_level1.append((rank, center, float(match.group(4))))
            print(f"  rank {rank:>2}  ({center[0]:7.2f},{center[1]:7.2f})  "
                  f"dJ_pred={float(match.group(4)): .4e}  {tag(center)}")
    hits = [item for item in screened_level1 if in_true_street(item[1])]
    print(f"  -> {len(hits)} of {len(screened_level1)} screened candidates are in "
          f"the true street"
          + (f" (best rank {min(h[0] for h in hits)})" if hits else ""))

    print("\n--- level 1 beam pruning (kept branches) ---")
    kept_level1 = []
    for line in lines:
        match = KEPT.match(line)
        if match and int(match.group(1)) == 1:
            centers = parse_centers(match.group(3))
            kept_level1.append(centers[0])
            print(f"  branch {match.group(2)}: ({centers[0][0]:7.2f},"
                  f"{centers[0][1]:7.2f})  {tag(centers[0])}")
    kept_hits = [c for c in kept_level1 if in_true_street(c)]
    print(f"  -> {len(kept_hits)} of {len(kept_level1)} kept branches are in the "
          "true street")

    print("\n--- refinement moves ---")
    for line in lines:
        match = REFINE.match(line)
        if match:
            before = (float(match.group(2)), float(match.group(3)))
            after = (float(match.group(4)), float(match.group(5)))
            moved = float(np.hypot(after[0] - before[0], after[1] - before[1]))
            print(f"  ({before[0]:7.2f},{before[1]:7.2f}) -> "
                  f"({after[0]:7.2f},{after[1]:7.2f})  moved {moved:7.2f} m  "
                  f"J {float(match.group(6)):.4e} -> {float(match.group(7)):.4e}")
            print(f"      before: {tag(before)}")
            print(f"      after : {tag(after)}")

    print("\n--- deduplication ---")
    found = False
    for line in lines:
        match = DEDUP.search(line)
        if match:
            found = True
            print(f"  {line.strip()}")
    if not found:
        print("  no branches deduplicated")

    print("\n--- joint pair refinement ---")
    pending = None
    for line in lines:
        match = PAIR.match(line)
        if match:
            pending = (match.group(1), parse_centers(match.group(2)))
            continue
        match = PAIR_TO.match(line)
        if match and pending is not None:
            after = parse_centers(match.group(1))
            print(f"  branch {pending[0]}: "
                  + " ; ".join(
                      f"({b[0]:.2f},{b[1]:.2f})->({a[0]:.2f},{a[1]:.2f})"
                      for b, a in zip(pending[1], after))
                  + f"  J={float(match.group(3)):.4e}")
            for index, center in enumerate(after, start=1):
                print(f"      s{index}: {tag(center)}")
            pending = None

    print("\n--- retained branches handed to the optimizer ---")
    in_retained = False
    any_true_street = False
    for line in lines:
        if line.startswith("[retained branches]") or "branch multistart:" in line:
            in_retained = True
            print(f"  {line.strip()}")
            continue
        if in_retained:
            match = RETAINED.match(line)
            if match:
                centers = parse_centers(match.group(2))
                print(f"  branch {match.group(1)}: J_exact={float(match.group(3)):.4e}")
                for index, center in enumerate(centers, start=1):
                    hit = in_true_street(center)
                    any_true_street = any_true_street or hit
                    print(f"      s{index}: {tag(center)}")
            elif line.strip() and not line.startswith(" "):
                in_retained = False

    print("\n=== verdict ===")
    if not hits:
        print("  The true street was never screened: NO level-1 top-K candidate is "
              "in it. Loss point = INITIAL TOP-K SCREENING.")
    elif not kept_hits:
        print(f"  A true-street candidate was screened (best rank "
              f"{min(h[0] for h in hits)}) but did not survive the beam cut. "
              "Loss point = BEAM PRUNING (raise --sensitivity-beam-width or "
              "enforce geometric diversity in the beam selection).")
    elif not any_true_street:
        print("  A true-street branch survived screening and pruning but is absent "
              "from the retained branches. Loss point = REFINEMENT or "
              "DEDUPLICATION (see the sections above).")
    else:
        print("  A true-street branch reached the optimizer. Any remaining error is "
              "in the FINAL LOCAL OPTIMIZATION, not in branch selection.")


if __name__ == "__main__":
    main()
