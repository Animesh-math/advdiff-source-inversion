"""Build a simple mesh_data dictionary for the Test Application."""

from pathlib import Path

import numpy as np
import meshio


def _boundary_nodes_from_lines_and_tags(lines, line_tags, physical_groups):
    """Map named boundary tags to node ids using line-cell physical tags."""
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
    """Map named boundary tags to boundary edges using line-cell physical tags."""
    edges_by_name = {}
    for name, tag in physical_groups.items():
        if name == "domain":
            continue
        mask = line_tags == int(tag)
        if np.any(mask):
            edges = np.asarray(lines[mask], dtype=int)
            edges_by_name[name] = edges
    return edges_by_name


def _all_boundary_edges_from_triangles(connections):
    """Return unique boundary edges (edges that belong to exactly one triangle)."""
    edge_count = {}
    for tri in np.asarray(connections, dtype=int):
        tri_edges = (
            tuple(sorted((int(tri[0]), int(tri[1])))),
            tuple(sorted((int(tri[1]), int(tri[2])))),
            tuple(sorted((int(tri[2]), int(tri[0])))),
        )
        for e in tri_edges:
            edge_count[e] = edge_count.get(e, 0) + 1
    bnd = [e for e, c in edge_count.items() if c == 1]
    return np.asarray(bnd, dtype=int) if bnd else np.zeros((0, 2), dtype=int)


def _boundary_edges_fallback_from_nodes(boundary_nodes_by_name, connections):
    """Fallback edge extraction when line-tag data is unavailable in meshio."""
    all_bnd = _all_boundary_edges_from_triangles(connections)
    edges_by_name = {}
    for name, nodes in boundary_nodes_by_name.items():
        node_set = set(np.asarray(nodes, dtype=int).tolist())
        keep = [e for e in all_bnd if int(e[0]) in node_set and int(e[1]) in node_set]
        if keep:
            edges_by_name[name] = np.asarray(keep, dtype=int)
    return edges_by_name


def _boundary_nodes_fallback_from_coordinates(coord, physical_groups):
    """Fallback only when line-tag data is unavailable in meshio."""
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
    return nodes_by_name


def build_mesh_data():
    app_dir = Path(__file__).resolve().parent
    mesh_file = app_dir / "test_domain.msh"

    if not mesh_file.exists():
        raise FileNotFoundError("test_domain.msh not found. Run generate_mesh.py first.")

    msh = meshio.read(mesh_file)

    coord = msh.points
    connections = None
    line_cells = None
    for cell_block in msh.cells:
        if cell_block.type == "triangle":
            connections = cell_block.data
        elif cell_block.type == "line":
            line_cells = cell_block.data

    if connections is None:
        raise ValueError("No triangle cells found in test_domain.msh")

    physical_groups = {
        "domain": 1,
        "left": 11,
        "right": 12,
        "top": 13,
        "bottom": 14,
    }

    line_tags = None
    if "gmsh:physical" in msh.cell_data_dict and line_cells is not None:
        line_tags = msh.cell_data_dict["gmsh:physical"].get("line")

    if line_cells is not None and line_tags is not None:
        boundary_nodes_by_name = _boundary_nodes_from_lines_and_tags(
            line_cells, np.asarray(line_tags), physical_groups
        )
        boundary_edges_by_name = _boundary_edges_from_lines_and_tags(
            line_cells, np.asarray(line_tags), physical_groups
        )
    else:
        # NOTE: Current test mesh may not expose line-cell physical tags through meshio.
        boundary_nodes_by_name = _boundary_nodes_fallback_from_coordinates(
            coord, physical_groups
        )
        boundary_edges_by_name = _boundary_edges_fallback_from_nodes(
            boundary_nodes_by_name, connections
        )

    mesh_data = {
        "mesh_file": str(mesh_file),
        "coord": coord,
        "connections": connections,
        "dim": 2,
        "physical_groups": physical_groups,
        "boundary_lines": line_cells,
        "boundary_line_tags": line_tags,
        "boundary_nodes_by_name": boundary_nodes_by_name,
        "boundary_edges_by_name": boundary_edges_by_name,
        "control_boundary": "right",
        "measurement_boundary": "right",
    }
    return mesh_data


if __name__ == "__main__":
    data = build_mesh_data()
    print("mesh_data keys:", list(data.keys()))
    print("number of points:", len(data["coord"]))
    print("number of elements:", len(data["connections"]))
