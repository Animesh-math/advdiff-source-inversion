"""Focused tests for the obstacle-aware sensitivity diversity filter.

No PDE solves: the diversity filter is pure geometry plus a candidate ranking,
so every check below runs against the real Application1 DC mesh with synthetic
G values.  Nothing here uses a true control, a recovered control, or any
case-specific coordinates -- the candidate pairs are DISCOVERED from the mesh.

Verifies:
  1. two close candidates in open fluid  -> the lower-ranked one is suppressed;
  2. two equally close candidates separated by a building -> both survive;
  3. two candidates farther apart than the diversity radius -> both survive;
  4. obstacle-aware disabled -> byte-identical to the previous Euclidean filter;
  5. no truth coordinates or case-specific information enter the decision;
  6. the exact segment-crossing test agrees with dense sampling (geometry check).

Run:
  cd /path/to/Advection-Diffusion-PDECO
  python3 \
    "Applications/Application1/test scripts/test_obstacle_aware_diversity.py"
"""

from __future__ import annotations

import inspect
import sys

import numpy as np

import _diag_common as dc
import sensitivity_initialization as sinit
from mesh_admissibility import MeshAdmissibility

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def euclidean_reference(g_vals, candidate_idx, candidate_xy, k, radius):
    """The pre-change filter, reproduced verbatim as the regression oracle."""
    order = candidate_idx[np.argsort(g_vals[candidate_idx])]
    picked: list[int] = []
    for i in order:
        c = candidate_xy[i]
        if all(
            float(np.hypot(c[0] - candidate_xy[j][0], c[1] - candidate_xy[j][1]))
            >= radius
            for j in picked
        ):
            picked.append(int(i))
        if len(picked) >= k:
            break
    return picked


def find_pairs(coord, admiss, radius):
    """Discover an open-fluid pair and a building-separated pair from the mesh.

    Purely geometric search over mesh nodes; no source, truth, or case
    information is consulted.
    """
    rng = np.random.default_rng(0)
    n = coord.shape[0]
    open_pair = None
    blocked_pair = None
    for _ in range(200000):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        d = float(np.hypot(*(coord[i] - coord[j])))
        if not (0.35 * radius < d < 0.9 * radius):
            continue
        if not (
            admiss.is_point_in_domain(*coord[i])
            and admiss.is_point_in_domain(*coord[j])
        ):
            continue
        connected = admiss.segment_connects_in_domain(coord[i], coord[j])[0]
        if connected and open_pair is None:
            open_pair = (int(i), int(j), d)
        if (not connected) and blocked_pair is None:
            blocked_pair = (int(i), int(j), d)
        if open_pair is not None and blocked_pair is not None:
            break
    return open_pair, blocked_pair


def main() -> int:
    print("=== obstacle-aware sensitivity diversity ===")
    mesh_data = dc.ro.build_mesh_data()
    admiss = MeshAdmissibility(mesh_data)
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)[:, :2]
    radius = 80.0  # 2*sqrt(sigma2) at the production initialization width

    def callback(p, q):
        return bool(admiss.segment_connects_in_domain(p, q)[0])

    open_pair, blocked_pair = find_pairs(coord, admiss, radius)
    if open_pair is None or blocked_pair is None:
        print("could not discover both pair types on this mesh")
        return 1
    print(f"  discovered open-fluid pair  nodes {open_pair[0]},{open_pair[1]} "
          f"d={open_pair[2]:.2f} m")
    print(f"  discovered obstructed pair  nodes {blocked_pair[0]},{blocked_pair[1]} "
          f"d={blocked_pair[2]:.2f} m")

    # ---- 1. two close candidates in open fluid -> lower-ranked suppressed ----
    i, j, d_open = open_pair
    g = np.zeros(coord.shape[0])
    g[i], g[j] = -10.0, -5.0  # i ranks better (more negative)
    idx = np.array([i, j])
    log: dict = {}
    picked = sinit._topk_diverse(
        g, idx, coord, 8, radius, segment_is_admissible=callback, log=log
    )
    check(
        "1. open-fluid close pair: lower-ranked candidate suppressed",
        picked == [i] and log["n_suppressed_by_distance"] == 1
        and log["n_preserved_obstacle_separated"] == 0,
        f"d={d_open:.2f} m < r={radius} m, picked={picked}, "
        f"suppressed={log['n_suppressed_by_distance']}",
    )

    # ---- 2. equally close pair separated by a building -> both survive ------
    a, b, d_blocked = blocked_pair
    g = np.zeros(coord.shape[0])
    g[a], g[b] = -10.0, -5.0
    idx = np.array([a, b])
    log = {}
    picked = sinit._topk_diverse(
        g, idx, coord, 8, radius, segment_is_admissible=callback, log=log
    )
    events = log["events"]
    check(
        "2. building-separated close pair: both candidates survive",
        sorted(picked) == sorted([a, b])
        and log["n_preserved_obstacle_separated"] == 1
        and log["n_suppressed_by_distance"] == 0
        and len(events) == 1
        and events[0]["segment_in_fluid"] is False
        and events[0]["obstacle_encountered"] is True
        and events[0]["action"] == "retained_obstacle_separated",
        f"d={d_blocked:.2f} m < r={radius} m, picked={sorted(picked)}",
    )
    check(
        "2b. euclidean-only filter WOULD have suppressed that pair",
        euclidean_reference(g, idx, coord, 8, radius) == [a],
        "confirms the pair is inside the diversity radius",
    )

    # ---- 3. farther apart than the radius -> both survive, either way -------
    far_i, far_j = None, None
    for cand in range(coord.shape[0]):
        if not admiss.is_point_in_domain(*coord[cand]):
            continue
        if far_i is None:
            far_i = cand
            continue
        if float(np.hypot(*(coord[cand] - coord[far_i]))) > 1.5 * radius:
            far_j = cand
            break
    g = np.zeros(coord.shape[0])
    g[far_i], g[far_j] = -10.0, -5.0
    idx = np.array([far_i, far_j])
    log = {}
    picked_oa = sinit._topk_diverse(
        g, idx, coord, 8, radius, segment_is_admissible=callback, log=log
    )
    picked_eu = sinit._topk_diverse(g, idx, coord, 8, radius)
    d_far = float(np.hypot(*(coord[far_i] - coord[far_j])))
    check(
        "3. pair beyond the diversity radius: both survive, no segment test",
        sorted(picked_oa) == sorted([far_i, far_j])
        and picked_oa == picked_eu
        and log["n_close_comparisons"] == 0,
        f"d={d_far:.2f} m >= r={radius} m, comparisons={log['n_close_comparisons']}",
    )

    # ---- 4. disabled -> identical to the previous Euclidean-only filter -----
    rng = np.random.default_rng(20260729)
    admissible = np.array(
        [n for n in range(coord.shape[0]) if admiss.is_point_in_domain(*coord[n])]
    )
    identical = True
    detail = ""
    for trial in range(5):
        g_rand = rng.normal(size=coord.shape[0])
        for k in (1, 5, 24):
            new = sinit._topk_diverse(g_rand, admissible, coord, k, radius)
            ref = euclidean_reference(g_rand, admissible, coord, k, radius)
            if new != ref:
                identical = False
                detail = f"trial {trial} k={k}: {new[:5]} vs {ref[:5]}"
                break
        if not identical:
            break
    check(
        "4. obstacle-aware disabled reproduces the Euclidean-only filter exactly",
        identical,
        detail or "5 random G fields x k in {1,5,24} over all admissible nodes",
    )

    # a real screen-sized run, obstacle-aware ON, must retain >= the Euclidean one
    g_rand = rng.normal(size=coord.shape[0])
    log = {}
    oa = sinit._topk_diverse(
        g_rand, admissible, coord, 24, radius, segment_is_admissible=callback, log=log
    )
    eu = sinit._topk_diverse(g_rand, admissible, coord, 24, radius)
    check(
        "4b. obstacle-aware never retains fewer candidates than Euclidean",
        len(oa) >= len(eu)
        and log["n_retained"] == len(oa)
        and log["n_preserved_obstacle_separated"] >= 0,
        f"retained {len(oa)} vs {len(eu)}, obstacle-preserved "
        f"{log['n_preserved_obstacle_separated']}",
    )

    # ---- 5. no truth / case-specific information in the decision path ------
    src = "".join(
        inspect.getsource(fn)
        for fn in (
            sinit._topk_diverse,
            MeshAdmissibility.segment_crosses_boundary,
            MeshAdmissibility.segment_connects_in_domain,
        )
    )
    forbidden = ("true_control", "true_center", "TRUE_", "260", "300", "60.0", "180")
    hits = [tok for tok in forbidden if tok in src]
    sig = inspect.signature(sinit._topk_diverse).parameters
    check(
        "5. decision path is truth-free and case-free",
        not hits and not any("true" in p.lower() for p in sig),
        f"forbidden tokens found: {hits}" if hits else
        f"params={list(sig)}",
    )
    # symmetry: the geometric predicate cannot depend on candidate ordering
    sym_ok = True
    for _ in range(300):
        u, v = rng.integers(0, coord.shape[0], size=2)
        if callback(coord[u], coord[v]) != callback(coord[v], coord[u]):
            sym_ok = False
            break
    check("5b. connectivity predicate is symmetric in its two arguments", sym_ok)

    # ---- 6. exact crossing test vs dense sampling --------------------------
    agree = 0
    disagree = []
    for _ in range(400):
        u, v = rng.integers(0, coord.shape[0], size=2)
        if u == v:
            continue
        if not (
            admiss.is_point_in_domain(*coord[u])
            and admiss.is_point_in_domain(*coord[v])
        ):
            continue
        exact = admiss.segment_connects_in_domain(coord[u], coord[v])[0]
        sampled = admiss.segment_in_domain(coord[u], coord[v], spacing=0.5)
        if exact == sampled:
            agree += 1
        else:
            disagree.append(
                (float(coord[u][0]), float(coord[u][1]),
                 float(coord[v][0]), float(coord[v][1]), exact, sampled)
            )
    check(
        "6. exact crossing test agrees with 0.5 m dense sampling",
        len(disagree) == 0,
        f"{agree} agreeing pairs" if not disagree else f"{len(disagree)} disagreements: "
        f"{disagree[:3]}",
    )

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
