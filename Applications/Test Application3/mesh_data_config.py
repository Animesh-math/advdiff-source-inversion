"""Build the mesh_data dictionary for Test Application3."""

from pathlib import Path

import numpy as np

try:
    import meshio
except ModuleNotFoundError:
    meshio = None


APP_DIR = Path(__file__).resolve().parent
MESH_DIR = APP_DIR / "mesh"
MESH_NAME = "rectangle_with_hole"
MESH_FILE = MESH_DIR / f"{MESH_NAME}.msh"
VTK_FILE = MESH_DIR / f"{MESH_NAME}.vtk"

RECTANGLE_WIDTH = 1.0
RECTANGLE_HEIGHT = 0.4

CIRCLE_CENTER_X = 0.5
CIRCLE_CENTER_Y = 0.2
CIRCLE_RADIUS = 0.08

MESH_SIZE = 0.025

DOMAIN_PHYSICAL_ID = 1
LEFT_PHYSICAL_ID = 11
RIGHT_PHYSICAL_ID = 12
TOP_PHYSICAL_ID = 13
BOTTOM_PHYSICAL_ID = 14
OBSTACLE_PHYSICAL_ID = 15

PHYSICAL_GROUPS = {
    "domain": DOMAIN_PHYSICAL_ID,
    "left": LEFT_PHYSICAL_ID,
    "right": RIGHT_PHYSICAL_ID,
    "top": TOP_PHYSICAL_ID,
    "bottom": BOTTOM_PHYSICAL_ID,
    "obstacle": OBSTACLE_PHYSICAL_ID,
}


def _boundary_nodes_from_lines_and_tags(lines, line_tags, physical_groups):
    nodes_by_name = {}
    for name, tag in physical_groups.items():
        if name == "domain":
            continue
        mask = line_tags == int(tag)
        if np.any(mask):
            nodes = np.unique(lines[mask].reshape(-1))
            nodes_by_name[name] = nodes.astype(int)
    return nodes_by_name


def _boundary_edges_from_lines_and_tags(lines, line_tags, physical_groups):
    edges_by_name = {}
    for name, tag in physical_groups.items():
        if name == "domain":
            continue
        mask = line_tags == int(tag)
        if np.any(mask):
            edges_by_name[name] = np.asarray(lines[mask], dtype=int)
    return edges_by_name


def _all_boundary_edges_from_triangles(connections):
    edge_count = {}
    for tri in np.asarray(connections, dtype=int):
        tri_edges = (
            tuple(sorted((int(tri[0]), int(tri[1])))),
            tuple(sorted((int(tri[1]), int(tri[2])))),
            tuple(sorted((int(tri[2]), int(tri[0])))),
        )
        for edge in tri_edges:
            edge_count[edge] = edge_count.get(edge, 0) + 1
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    if not boundary_edges:
        return np.zeros((0, 2), dtype=int)
    return np.asarray(boundary_edges, dtype=int)


def _boundary_edges_fallback_from_nodes(boundary_nodes_by_name, connections):
    all_boundary_edges = _all_boundary_edges_from_triangles(connections)
    edges_by_name = {}
    for name, nodes in boundary_nodes_by_name.items():
        node_set = set(np.asarray(nodes, dtype=int).tolist())
        keep = [
            edge
            for edge in all_boundary_edges
            if int(edge[0]) in node_set and int(edge[1]) in node_set
        ]
        if keep:
            edges_by_name[name] = np.asarray(keep, dtype=int)
    return edges_by_name


def _boundary_nodes_fallback_from_coordinates(coord, physical_groups):
    x = coord[:, 0]
    y = coord[:, 1]
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    scale = max(xmax - xmin, ymax - ymin, 1.0)
    eps = 1.0e-6 * scale

    nodes_by_name = {}
    if "left" in physical_groups:
        nodes_by_name["left"] = np.where(np.abs(x - xmin) <= eps)[0].astype(int)
    if "right" in physical_groups:
        nodes_by_name["right"] = np.where(np.abs(x - xmax) <= eps)[0].astype(int)
    if "bottom" in physical_groups:
        nodes_by_name["bottom"] = np.where(np.abs(y - ymin) <= eps)[0].astype(int)
    if "top" in physical_groups:
        nodes_by_name["top"] = np.where(np.abs(y - ymax) <= eps)[0].astype(int)
    if "obstacle" in physical_groups:
        radius = np.hypot(x - CIRCLE_CENTER_X, y - CIRCLE_CENTER_Y)
        nodes_by_name["obstacle"] = np.where(
            np.abs(radius - CIRCLE_RADIUS) <= eps
        )[0].astype(int)
    return nodes_by_name


def _read_legacy_vtk_points_and_triangles(vtk_file: Path):
    text = vtk_file.read_text(encoding="utf-8")
    tokens = text.split()

    try:
        points_idx = tokens.index("POINTS")
        num_points = int(tokens[points_idx + 1])
        point_start = points_idx + 3
        point_values = np.asarray(
            tokens[point_start : point_start + 3 * num_points], dtype=np.float64
        )

        cells_idx = tokens.index("CELLS")
        num_cells = int(tokens[cells_idx + 1])
        cell_start = cells_idx + 3

        cell_types_idx = tokens.index("CELL_TYPES")
        cell_types = np.asarray(
            tokens[cell_types_idx + 2 : cell_types_idx + 2 + num_cells], dtype=int
        )
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Unsupported legacy VTK mesh format in {vtk_file}") from exc

    coord = point_values.reshape(num_points, 3)

    triangles = []
    cursor = cell_start
    for cell_type in cell_types:
        npts = int(tokens[cursor])
        cursor += 1
        nodes = np.asarray(tokens[cursor : cursor + npts], dtype=int)
        cursor += npts
        if int(cell_type) == 5:
            if npts != 3:
                raise ValueError(f"Expected triangle cell with 3 nodes in {vtk_file}")
            triangles.append(nodes)

    if not triangles:
        raise ValueError(f"No triangle cells found in {vtk_file}")

    return coord, np.asarray(triangles, dtype=int)


def _cell_data(msh, cell_type):
    blocks = [cell_block.data for cell_block in msh.cells if cell_block.type == cell_type]
    if not blocks:
        return None
    return np.concatenate(blocks, axis=0)


def build_mesh_data():
    if meshio is not None:
        if not MESH_FILE.exists():
            raise FileNotFoundError(
                f"{MESH_FILE.name} not found. Run generate_mesh.py first."
            )

        msh = meshio.read(MESH_FILE)
        coord = msh.points
        connections = _cell_data(msh, "triangle")
        line_cells = _cell_data(msh, "line")

        if connections is None:
            raise ValueError(f"No triangle cells found in {MESH_FILE.name}")
    else:
        if not VTK_FILE.exists():
            raise FileNotFoundError(
                f"Neither meshio nor {VTK_FILE.name} are available. "
                "Run generate_mesh.py first."
            )
        coord, connections = _read_legacy_vtk_points_and_triangles(VTK_FILE)
        line_cells = None
        msh = None

    line_tags = None
    if msh is not None and "gmsh:physical" in msh.cell_data_dict:
        line_tags = msh.cell_data_dict["gmsh:physical"].get("line")
        if line_tags is not None:
            line_tags = np.asarray(line_tags, dtype=int)

    if line_cells is not None and line_tags is not None:
        if len(line_cells) != len(line_tags):
            raise ValueError(
                "Boundary line cells and physical tags have inconsistent lengths"
            )
        boundary_nodes_by_name = _boundary_nodes_from_lines_and_tags(
            line_cells, line_tags, PHYSICAL_GROUPS
        )
        boundary_edges_by_name = _boundary_edges_from_lines_and_tags(
            line_cells, line_tags, PHYSICAL_GROUPS
        )
    else:
        boundary_nodes_by_name = _boundary_nodes_fallback_from_coordinates(
            coord, PHYSICAL_GROUPS
        )
        boundary_edges_by_name = _boundary_edges_fallback_from_nodes(
            boundary_nodes_by_name, connections
        )

    return {
        "mesh_file": str(MESH_FILE),
        "coord": coord,
        "connections": connections,
        "dim": 2,
        "physical_groups": PHYSICAL_GROUPS.copy(),
        "boundary_lines": line_cells,
        "boundary_line_tags": line_tags,
        "boundary_nodes_by_name": boundary_nodes_by_name,
        "boundary_edges_by_name": boundary_edges_by_name,
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
        {
            name: len(edges)
            for name, edges in mesh_data["boundary_edges_by_name"].items()
        },
    )
