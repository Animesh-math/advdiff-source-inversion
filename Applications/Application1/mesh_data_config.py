"""Build the mesh_data dictionary for Application1 (DC Navy Yard).

This mirrors ``Test Application4/mesh_data_config.py``: it reads the committed
Gmsh mesh with ``meshio`` and returns coordinates, triangle connectivity, and
per-boundary node/edge sets keyed by physical-group name.  The ``coord`` array
here defines the canonical Application1 transport-mesh node ordering, which is
the ordering used for the exported Navier-Stokes nodal-velocity NPZ file (so the
future advection-diffusion solver can consume it without reordering).

The mesh is stored in a LOCAL coordinate system: every node was translated once
so the south-west corner of the outer domain is the origin (0, 0), giving x in
[0, ~302] m and y in [0, ~700] m.  The original projected (GIS) origin is
``(ORIGIN_X, ORIGIN_Y)`` (global = local + origin) and is recorded in
``dc_navy_yard_coordinate_origin.json``.

Cardinal-direction mapping of the four outer boundary tags was VERIFIED against
the DC mesh geometry (constant-coordinate edges of the domain bounding box):

    tag 11  ->  bottom edge, y = y_min  ->  SOUTH   (outward normal (0, -1))
    tag 12  ->  top    edge, y = y_max  ->  NORTH   (outward normal (0, +1))
    tag 13  ->  left   edge, x = x_min  ->  WEST    (outward normal (-1, 0))
    tag 14  ->  right  edge, x = x_max  ->  EAST    (outward normal (+1, 0))

The mesh's own physical names ("bottom_inflow", "top_outflow", ...) bake in an
assumed wind direction.  Inflow/outflow classification for the Navier-Stokes
solve is done dynamically from the wind vector and the geometric outward normal,
never from those names.
"""

from pathlib import Path

import numpy as np

try:
    import meshio
except ModuleNotFoundError:  # pragma: no cover - meshio always present in solve env
    meshio = None


APP_DIR = Path(__file__).resolve().parent
MESH_DIR = APP_DIR / "DC_mesh"
MESH_NAME = "dc_navy_yard_mesh"
MESH_FILE = MESH_DIR / f"{MESH_NAME}.msh"
BUILDING_METADATA_FILE = MESH_DIR / "dc_navy_yard_building_metadata.json"

# Physical group ids (see DC_mesh/build_dc_mesh.py).
DOMAIN_PHYSICAL_ID = 1
# Outer rectangle edges (geometric names; cardinal mapping verified above).
BOTTOM_PHYSICAL_ID = 11  # SOUTH, y = y_min
TOP_PHYSICAL_ID = 12     # NORTH, y = y_max
LEFT_PHYSICAL_ID = 13    # WEST,  x = x_min
RIGHT_PHYSICAL_ID = 14   # EAST,  x = x_max
OUTER_PHYSICAL_IDS = (
    BOTTOM_PHYSICAL_ID,
    TOP_PHYSICAL_ID,
    LEFT_PHYSICAL_ID,
    RIGHT_PHYSICAL_ID,
)

# Individual building walls: tags 101 .. 121 (building_1 .. building_21).
NUM_BUILDINGS = 21
BUILDING_TAG_OFFSET = 100
BUILDING_PHYSICAL_IDS = tuple(
    BUILDING_TAG_OFFSET + n for n in range(1, NUM_BUILDINGS + 1)
)

# The DC mesh has been translated into a LOCAL coordinate system whose origin is
# the south-west corner of the outer rectangular computational domain, i.e. every
# node was shifted once by (-ORIGIN_X, -ORIGIN_Y).  The mesh files on disk are
# already in local meters, so the whole numerical pipeline works in local
# coordinates with no runtime origin subtraction.
#
# ORIGIN_X/ORIGIN_Y are the ORIGINAL projected (GIS) coordinates of the local
# origin, kept for reference only (global = local + origin).  They are also
# recorded in DC_mesh/dc_navy_yard_coordinate_origin.json.
ORIGIN_X = 400032.9721875289
ORIGIN_Y = 133918.0828324356
COORDINATE_ORIGIN_FILE = MESH_DIR / "dc_navy_yard_coordinate_origin.json"

# Committed DC mesh fluid-domain bounding box in LOCAL meters (SW corner at the
# origin).  These are the exact min/max of the translated dc_navy_yard_mesh nodal
# coordinates and act as the single geometry-scale source of truth for the
# advection-diffusion optimization layer.  The optimization driver prints the
# runtime admissibility bounding box so this can be cross-checked at every run.
DOMAIN_X_MIN = 0.0
DOMAIN_X_MAX = 301.98544034478255
DOMAIN_Y_MIN = 0.0
DOMAIN_Y_MAX = 700.4646734384005
# "Width"/"height" are the physical spans in meters (unchanged by translation).
# The shared forward solver ignores lx/ly (geometry comes from the mesh); these
# exist for API parity with the Test Application4 driver signature and for legacy
# bounding-box helpers.
OUTER_WIDTH = DOMAIN_X_MAX - DOMAIN_X_MIN
OUTER_HEIGHT = DOMAIN_Y_MAX - DOMAIN_Y_MIN


def _outer_name(tag: int) -> str:
    return {
        BOTTOM_PHYSICAL_ID: "bottom",
        TOP_PHYSICAL_ID: "top",
        LEFT_PHYSICAL_ID: "left",
        RIGHT_PHYSICAL_ID: "right",
    }[tag]


def _building_name(tag: int) -> str:
    return "building_%d" % (tag - BUILDING_TAG_OFFSET)


def _build_physical_groups() -> dict[str, int]:
    groups = {"fluid": DOMAIN_PHYSICAL_ID}
    for tag in OUTER_PHYSICAL_IDS:
        groups[_outer_name(tag)] = tag
    for tag in BUILDING_PHYSICAL_IDS:
        groups[_building_name(tag)] = tag
    return groups


PHYSICAL_GROUPS = _build_physical_groups()
OUTER_BOUNDARY_NAMES = tuple(_outer_name(tag) for tag in OUTER_PHYSICAL_IDS)
BUILDING_BOUNDARY_NAMES = tuple(_building_name(tag) for tag in BUILDING_PHYSICAL_IDS)
OUTER_TAG_TO_NAME = {tag: _outer_name(tag) for tag in OUTER_PHYSICAL_IDS}
BUILDING_TAG_TO_NAME = {tag: _building_name(tag) for tag in BUILDING_PHYSICAL_IDS}


def _cell_data(msh, cell_type):
    blocks = [block.data for block in msh.cells if block.type == cell_type]
    if not blocks:
        return None
    return np.concatenate(blocks, axis=0)


def _boundary_nodes_from_lines_and_tags(lines, line_tags, physical_groups):
    nodes_by_name = {}
    for name, tag in physical_groups.items():
        if name == "fluid":
            continue
        mask = line_tags == int(tag)
        if np.any(mask):
            nodes_by_name[name] = np.unique(lines[mask].reshape(-1)).astype(int)
    return nodes_by_name


def _boundary_edges_from_lines_and_tags(lines, line_tags, physical_groups):
    edges_by_name = {}
    for name, tag in physical_groups.items():
        if name == "fluid":
            continue
        mask = line_tags == int(tag)
        if np.any(mask):
            edges_by_name[name] = np.asarray(lines[mask], dtype=int)
    return edges_by_name


def build_mesh_data():
    if meshio is None:
        raise ModuleNotFoundError(
            "meshio is required to build Application1 mesh data; run in an "
            "environment that provides it (e.g. fenicsx-env)."
        )
    if not MESH_FILE.exists():
        raise FileNotFoundError(
            f"{MESH_FILE} not found. Regenerate it with DC_mesh/build_dc_mesh.py."
        )

    msh = meshio.read(MESH_FILE)
    coord = np.asarray(msh.points, dtype=np.float64)
    connections = _cell_data(msh, "triangle")
    line_cells = _cell_data(msh, "line")
    if connections is None:
        raise ValueError(f"No triangle cells found in {MESH_FILE.name}")

    line_tags = None
    if "gmsh:physical" in msh.cell_data_dict:
        line_tags = msh.cell_data_dict["gmsh:physical"].get("line")
        if line_tags is not None:
            line_tags = np.asarray(line_tags, dtype=int)

    if line_cells is None or line_tags is None:
        raise ValueError(
            f"{MESH_FILE.name} is missing tagged boundary line cells; the DC mesh "
            "must carry physical facet tags 11-14 and 101-121."
        )
    if len(line_cells) != len(line_tags):
        raise ValueError("Boundary line cells and physical tags have inconsistent lengths")

    boundary_nodes_by_name = _boundary_nodes_from_lines_and_tags(
        line_cells, line_tags, PHYSICAL_GROUPS
    )
    boundary_edges_by_name = _boundary_edges_from_lines_and_tags(
        line_cells, line_tags, PHYSICAL_GROUPS
    )

    return {
        "mesh_file": str(MESH_FILE),
        "coord": coord,
        "connections": np.asarray(connections, dtype=int),
        "dim": 2,
        "physical_groups": PHYSICAL_GROUPS.copy(),
        "outer_boundary_names": list(OUTER_BOUNDARY_NAMES),
        "building_boundary_names": list(BUILDING_BOUNDARY_NAMES),
        "boundary_lines": line_cells,
        "boundary_line_tags": line_tags,
        "boundary_nodes_by_name": boundary_nodes_by_name,
        "boundary_edges_by_name": boundary_edges_by_name,
        # Physical inflow is the west ("left") wall; the shared forward/adjoint
        # solvers impose the homogeneous Dirichlet (clean-inflow) condition
        # there.  "right" (east) is an outflow wall and the default observation
        # boundary; both are overridable from the optimization CLI.
        "control_boundary": "right",
        "measurement_boundary": "right",
    }


if __name__ == "__main__":
    mesh_data = build_mesh_data()
    print("mesh_data keys:", list(mesh_data.keys()))
    print("number of points:", len(mesh_data["coord"]))
    print("number of elements:", len(mesh_data["connections"]))
    print(
        "boundary edge counts:",
        {name: len(edges) for name, edges in mesh_data["boundary_edges_by_name"].items()},
    )
