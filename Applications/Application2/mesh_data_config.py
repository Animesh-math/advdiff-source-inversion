"""Build the transport ``mesh_data`` dictionary for Application2 and load the
matching Navier-Stokes nodal velocity safely.

Mirrors ``Application1/mesh_data_config.py``: the Gmsh mesh is read with meshio
and returned as coordinates, P1 triangle connectivity and per-boundary node/edge
sets keyed by physical-group name -- the structure consumed by the shared
transport solvers in ``Src/Solvers``.  The ``coord`` order is the canonical
Application2 transport-node order for that mesh level and is the order of the
exported ``velocity_nodes``.

Two mesh levels exist (``"coarse"`` for eventual optimization, ``"fine"`` for
eventual synthetic data).  A velocity field must only ever be used on the mesh it
was computed on; ``load_velocity_nodes(level)`` enforces this by checking the
mesh SHA-256, the geometry hash, the level name and the node coordinates stored
in the velocity NPZ against the mesh being loaded.

Boundary names (see room_geometry_config.py):
    inlet_1..3 (top-wall supplies), outlet_door (corridor doorway, y = 0),
    outlet_exhaust (bathroom exhaust, x = 11), and the no-slip wall group
    wall_bottom/right/top/left, bath_bottom/right/top, partition_1..2,
    bed_1..3 (solid bed obstacles), bath_wall_lower/upper.
Beds, partitions and the bathroom wall are HOLES/notches in the mesh: no node or
triangle lies inside a solid.  Outputs live in the versioned folder
``output/<room_geometry_config.OUTPUT_VERSION>/``.  Candidate source centres for a future inverse problem must be checked
with ``point_in_fluid_mesh`` (mesh-level, authoritative) and/or
``room_geometry_config.admissible_source_mask`` (analytic, with clearance).
For a future tracer solve, the clean-air inflow Dirichlet names are
``INFLOW_BOUNDARY_NAMES`` (pass as ``dirichlet_tags``).  No observation or
control boundary is chosen here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import room_geometry_config as geo

try:
    import meshio
except ModuleNotFoundError:  # pragma: no cover
    meshio = None

APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "output" / geo.OUTPUT_VERSION
MESH_LEVELS = tuple(geo.MESH_LEVELS)

PHYSICAL_GROUPS = {"fluid": geo.FLUID_TAG, **geo.BOUNDARY_TAGS}
INFLOW_BOUNDARY_NAMES = list(geo.INLET_NAMES)
OUTFLOW_BOUNDARY_NAMES = list(geo.OUTLET_NAMES)
WALL_BOUNDARY_NAMES = list(geo.WALL_NAMES)
BED_BOUNDARY_NAMES = list(geo.BED_TAGS)
SOLID_NAMES = list(geo.SOLIDS)

VELOCITY_NPZ_NAME = "application2_navier_stokes_velocity_nodes_{level}.npz"


def mesh_file(level: str, out_root: Path = OUTPUT_DIR) -> Path:
    _check_level(level)
    return Path(out_root) / level / "mesh" / f"application2_room_{level}.msh"


def mesh_info_file(level: str, out_root: Path = OUTPUT_DIR) -> Path:
    return mesh_file(level, out_root).with_name(f"application2_room_{level}_mesh_info.json")


def velocity_npz_file(level: str, out_root: Path = OUTPUT_DIR) -> Path:
    _check_level(level)
    return Path(out_root) / level / "navier_stokes" / VELOCITY_NPZ_NAME.format(level=level)


def _check_level(level: str) -> None:
    if level not in MESH_LEVELS:
        raise ValueError(f"unknown mesh level {level!r}; expected one of {MESH_LEVELS}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def coords_fingerprint(coords: np.ndarray) -> str:
    """SHA-256 of the float64 (N,2) coordinate array (order-sensitive)."""
    c = np.ascontiguousarray(np.asarray(coords, dtype=np.float64)[:, :2])
    return hashlib.sha256(c.tobytes()).hexdigest()


def _cell_data(msh, cell_type):
    blocks = [block.data for block in msh.cells if block.type == cell_type]
    return np.concatenate(blocks, axis=0) if blocks else None


def build_mesh_data(level: str = "coarse", out_root: Path = OUTPUT_DIR) -> dict:
    """Return the transport mesh_data dict for one mesh level."""
    if meshio is None:
        raise ModuleNotFoundError("meshio is required (both environments provide it)")
    path = mesh_file(level, out_root)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; build it with Room_mesh/build_room_mesh.py")
    msh = meshio.read(path)
    coord = np.asarray(msh.points, dtype=np.float64)
    connections = _cell_data(msh, "triangle")
    lines = _cell_data(msh, "line")
    line_tags = np.concatenate([d for b, d in zip(msh.cells, msh.cell_data["gmsh:physical"])
                                if b.type == "line"]).astype(int)
    if connections is None or lines is None or len(lines) != len(line_tags):
        raise ValueError(f"{path.name}: missing triangles or tagged boundary lines")

    nodes_by_name, edges_by_name = {}, {}
    for name, tag in PHYSICAL_GROUPS.items():
        if name == "fluid":
            continue
        mask = line_tags == int(tag)
        if not np.any(mask):
            raise ValueError(f"{path.name}: boundary {name} (tag {tag}) has no edges")
        edges_by_name[name] = np.asarray(lines[mask], dtype=int)
        nodes_by_name[name] = np.unique(lines[mask].reshape(-1)).astype(int)

    return {
        "mesh_file": str(path),
        "mesh_level": level,
        "mesh_sha256": sha256_file(path),
        "geometry_sha256": geo.geometry_hash(),
        "coord": coord,
        "connections": np.asarray(connections, dtype=int),
        "dim": 2,
        "physical_groups": PHYSICAL_GROUPS.copy(),
        "inflow_boundary_names": INFLOW_BOUNDARY_NAMES,
        "outflow_boundary_names": OUTFLOW_BOUNDARY_NAMES,
        "wall_boundary_names": WALL_BOUNDARY_NAMES,
        "bed_boundary_names": BED_BOUNDARY_NAMES,
        "solids": {k: list(v) for k, v in geo.SOLIDS.items()},
        "boundary_lines": lines,
        "boundary_line_tags": line_tags,
        "boundary_nodes_by_name": nodes_by_name,
        "boundary_edges_by_name": edges_by_name,
    }


def point_in_fluid_mesh(mesh_data: dict, points, clearance: float = 0.0) -> np.ndarray:
    """Admissibility of candidate source centres: inside a fluid triangle of this mesh.

    Mirrors Application1's mesh-based rule (bed and partition interiors are holes,
    so they are never inside a triangle).  With ``clearance`` > 0 the analytic
    ``geo.admissible_source_mask`` is also required (distance from walls/solids).
    """
    import matplotlib.tri as mtri

    pts = np.atleast_2d(np.asarray(points, dtype=float))
    c = mesh_data["coord"]
    finder = mtri.Triangulation(c[:, 0], c[:, 1], mesh_data["connections"]).get_trifinder()
    ok = finder(pts[:, 0], pts[:, 1]) >= 0
    if clearance > 0.0:
        ok &= geo.admissible_source_mask(pts[:, 0], pts[:, 1], clearance)
    return ok


def load_velocity_nodes(level: str, mesh_data: dict | None = None, out_root: Path = OUTPUT_DIR):
    """Load the nodal velocity for ``level`` after verifying it belongs to that mesh.

    Raises ValueError on any mismatch (mesh hash, geometry hash, level, node
    count, coordinates).  Returns ``velocity_nodes`` with shape (num_nodes, 2) in
    the transport node order of ``build_mesh_data(level)``.
    """
    md = mesh_data if mesh_data is not None else build_mesh_data(level, out_root)
    if md.get("mesh_level") != level:
        raise ValueError(f"mesh_data is for level {md.get('mesh_level')!r}, not {level!r}")
    path = velocity_npz_file(level, out_root)
    with np.load(path, allow_pickle=False) as d:
        stored = {k: d[k] for k in d.files}
    meta = json.loads(str(stored["association_json"]))
    problems = []
    if meta.get("mesh_level") != level:
        problems.append(f"NPZ level {meta.get('mesh_level')!r} != {level!r}")
    if meta.get("mesh_sha256") != md["mesh_sha256"]:
        problems.append("mesh sha256 mismatch")
    if meta.get("geometry_sha256") != md["geometry_sha256"]:
        problems.append("geometry sha256 mismatch")
    coords = np.asarray(stored["coords"], dtype=np.float64)
    if coords.shape[0] != md["coord"].shape[0]:
        problems.append(f"node count {coords.shape[0]} != mesh {md['coord'].shape[0]}")
    elif not np.array_equal(coords[:, :2], md["coord"][:, :2]):
        problems.append("node coordinates differ from the mesh")
    vel = np.asarray(stored["velocity_nodes"], dtype=np.float64)
    if vel.shape != (md["coord"].shape[0], 2) or not np.all(np.isfinite(vel)):
        problems.append(f"velocity_nodes shape {vel.shape} or non-finite values")
    if problems:
        raise ValueError(f"{path}: velocity does not belong to the {level} mesh: " + "; ".join(problems))
    return vel


if __name__ == "__main__":
    for lv in MESH_LEVELS:
        try:
            md = build_mesh_data(lv)
        except FileNotFoundError as exc:
            print(exc)
            continue
        print(lv, "nodes", len(md["coord"]), "triangles", len(md["connections"]),
              {n: len(e) for n, e in md["boundary_edges_by_name"].items()})
