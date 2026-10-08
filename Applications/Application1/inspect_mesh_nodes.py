"""Small mesh-node lookup helper for choosing Application1 point sensors.

Purpose: ParaView point numbering is not guaranteed to match the solver's
internal node numbering, so sensor locations picked visually in ParaView are
safer expressed as COORDINATES and resolved here to exact solver node ids.

Typical workflow:

  1. inspect the mesh in ParaView and read off approximate sensor coordinates;
  2. resolve them here;
  3. pass the printed ids back to run_optimization.py as --sensor-node-ids
     (or pass the coordinates directly with --sensor-coordinates).

This is a read-only diagnostic. It builds no visualization and modifies nothing.

Examples
--------
    PY="python3"

    # nearest solver node to each requested coordinate
    "$PY" inspect_mesh_nodes.py --nearest "100,250;150,400;280,600"

    # coordinates and boundary membership of known node ids
    "$PY" inspect_mesh_nodes.py --node-ids 41,82,164,233

    # every node on a named outer boundary
    "$PY" inspect_mesh_nodes.py --list-boundary right --limit 20
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from mesh_data_config import build_mesh_data


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


obs_op = _load_module_from_file(
    "observation_operator_inspect",
    PROJECT_ROOT / "Src" / "Solvers" / "observation_operator.py",
)


def boundary_membership(mesh_data) -> dict[int, list[str]]:
    """Map node id -> the named boundaries it belongs to."""
    membership: dict[int, list[str]] = {}
    for name, nodes in mesh_data["boundary_nodes_by_name"].items():
        if name.startswith("__"):
            continue
        for node in np.asarray(nodes, dtype=int):
            membership.setdefault(int(node), []).append(name)
    return membership


def describe(node: int, coord: np.ndarray, membership: dict[int, list[str]]) -> dict:
    return {
        "node_id": int(node),
        "coordinate": [float(coord[node, 0]), float(coord[node, 1])],
        "boundaries": sorted(membership.get(int(node), [])),
    }


def parse_coordinates(text: str) -> np.ndarray:
    pairs = []
    for chunk in str(text).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p for p in chunk.replace(",", " ").split() if p]
        if len(parts) != 2:
            raise ValueError(f"Expected \"x,y\" pairs separated by ';'; got {chunk!r}")
        pairs.append((float(parts[0]), float(parts[1])))
    if not pairs:
        raise ValueError("No coordinates parsed")
    return np.asarray(pairs, dtype=np.float64)


def parse_ids(tokens) -> list[int]:
    values = []
    for token in tokens:
        for piece in str(token).replace(";", ",").split(","):
            piece = piece.strip()
            if piece:
                values.append(int(piece))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve Application1 mesh-node ids and coordinates."
    )
    parser.add_argument(
        "--nearest",
        metavar="COORDS",
        help='Requested sensor coordinates as "x1,y1;x2,y2;..." (local meters).',
    )
    parser.add_argument(
        "--node-ids",
        nargs="+",
        metavar="IDS",
        help="Solver node ids (comma and/or space separated) to describe.",
    )
    parser.add_argument(
        "--list-boundary",
        metavar="NAME",
        help="List the nodes of a named boundary (bottom, top, left, right, building_N).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Truncate --list-boundary output to this many nodes (0 = all).",
    )
    parser.add_argument(
        "--snap-tolerance",
        type=float,
        default=20.0,
        help="Report (do not fail) when a --nearest snapping distance exceeds this (meters).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the resolved records as JSON (for scripting).",
    )
    args = parser.parse_args()

    if not (args.nearest or args.node_ids or args.list_boundary):
        parser.error("nothing to do: pass --nearest, --node-ids or --list-boundary")

    mesh_data = build_mesh_data()
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)
    membership = boundary_membership(mesh_data)
    payload: dict[str, object] = {
        "mesh_file": mesh_data["mesh_file"],
        "num_nodes": int(coord.shape[0]),
    }

    if not args.json:
        print(f"mesh: {mesh_data['mesh_file']}")
        print(f"nodes: {coord.shape[0]}  triangles: {len(mesh_data['connections'])}")
        print(
            "bounding box: "
            f"x [{coord[:, 0].min():.4f}, {coord[:, 0].max():.4f}]  "
            f"y [{coord[:, 1].min():.4f}, {coord[:, 1].max():.4f}]"
        )

    if args.nearest:
        requested = parse_coordinates(args.nearest)
        # Report rather than reject here: this tool exists to diagnose a bad
        # ParaView pick, while the solver CLI enforces the tolerance.
        node_ids, records = obs_op.snap_coordinates_to_nodes(
            requested, coord, tolerance=float("inf")
        )
        resolved = []
        for record in records:
            info = describe(record["node_id"], coord, membership)
            info["requested"] = record["requested"]
            info["snap_distance"] = record["snap_distance"]
            info["within_tolerance"] = bool(
                record["snap_distance"] <= args.snap_tolerance
            )
            resolved.append(info)
        payload["nearest"] = resolved
        payload["sensor_node_ids"] = [int(i) for i in node_ids]
        if not args.json:
            print("\n=== Nearest Mesh Nodes ===")
            for info in resolved:
                flag = "" if info["within_tolerance"] else "   [ABOVE TOLERANCE]"
                print(
                    f"requested ({info['requested'][0]:.4f}, "
                    f"{info['requested'][1]:.4f}) -> node {info['node_id']} at "
                    f"({info['coordinate'][0]:.4f}, {info['coordinate'][1]:.4f}), "
                    f"distance {info['snap_distance']:.4f} m"
                    + (f", boundaries {info['boundaries']}" if info["boundaries"] else "")
                    + flag
                )
            print(
                "\n--sensor-node-ids "
                + ",".join(str(int(i)) for i in node_ids)
            )

    if args.node_ids:
        ids = obs_op.resolve_node_ids(parse_ids(args.node_ids), coord.shape[0])
        described = [describe(int(i), coord, membership) for i in ids]
        payload["node_ids"] = described
        if not args.json:
            print("\n=== Requested Node Ids ===")
            for info in described:
                print(
                    f"node {info['node_id']:5d} at ({info['coordinate'][0]:.4f}, "
                    f"{info['coordinate'][1]:.4f})"
                    + (f"  boundaries {info['boundaries']}" if info["boundaries"] else "")
                )

    if args.list_boundary:
        nodes_by_name = mesh_data["boundary_nodes_by_name"]
        if args.list_boundary not in nodes_by_name:
            available = sorted(n for n in nodes_by_name if not n.startswith("__"))
            parser.error(
                f"unknown boundary {args.list_boundary!r}; available: {available}"
            )
        nodes = np.asarray(nodes_by_name[args.list_boundary], dtype=int)
        shown = nodes if args.limit <= 0 else nodes[: args.limit]
        described = [describe(int(i), coord, membership) for i in shown]
        payload["boundary"] = {
            "name": args.list_boundary,
            "num_nodes": int(nodes.size),
            "nodes": described,
        }
        if not args.json:
            print(f"\n=== Boundary '{args.list_boundary}' ({nodes.size} nodes) ===")
            for info in described:
                print(
                    f"node {info['node_id']:5d} at ({info['coordinate'][0]:.4f}, "
                    f"{info['coordinate'][1]:.4f})"
                )
            if args.limit > 0 and nodes.size > args.limit:
                print(f"... {nodes.size - args.limit} more (raise --limit)")

    if args.json:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
