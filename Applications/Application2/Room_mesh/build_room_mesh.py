"""Build the Application2 treatment-room meshes (coarse and/or fine) with Gmsh.

Both levels are generated from the SAME geometry and physical tags in
``room_geometry_config.py`` (main room UNION bathroom annex, minus partitions,
beds and the bathroom wall); only the characteristic sizes differ (fine = half
of every coarse size).  Partitions and the bathroom wall are notches of the
outer loop; the three beds are rectangular HOLES (tags 61-63).  Output per level,
under ``<out-root>/<level>/mesh/`` (default out-root:
``output/<OUTPUT_VERSION>``):

    application2_room_<level>.msh          Gmsh 4.1 ASCII, physical groups only
    application2_room_<level>.vtu          triangles + tagged boundary lines, cell data
                                           "physical_tag" (ParaView: built-in XML VTU reader)
    application2_room_<level>.xdmf/.h5     DOLFINx mesh (cells)            [optional]
    application2_room_<level>_facets.xdmf/.h5  DOLFINx facet tags          [optional]
    application2_room_<level>_mesh_info.json   counts, quality, geometry/tag checks, sha256

Checks (any failure aborts): every boundary edge tagged; per-tag lengths equal
the geometry; mesh area equals the fluid area; the triangulation is one
edge-connected component with exactly 1 + 3 boundary loops (outer + 3 beds); no
triangle centroid and no sample point inside a partition, bed or bathroom wall,
or in the exterior region above the annex; every clear passage (around each bed,
the aisle, the bathroom doorway and the bathroom) is fully covered by triangles;
DOLFINx reads back the same number of facets per tag as Gmsh wrote.

Run with fenicsx-env (needs gmsh; DOLFINx only for the XDMF export):

    FX=/path/to/fenicsx-env/bin/python    # or: conda activate fenicsx-env; FX=python
    $FX Applications/Application2/Room_mesh/build_room_mesh.py --level all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
APP_DIR = SCRIPT_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import room_geometry_config as geo  # noqa: E402

DEFAULT_OUT_ROOT = APP_DIR / "output" / geo.OUTPUT_VERSION


def mesh_paths(out_root: Path, level: str) -> dict:
    d = Path(out_root) / level / "mesh"
    stem = f"application2_room_{level}"
    return {
        "dir": d,
        "msh": d / f"{stem}.msh",
        "xdmf": d / f"{stem}.xdmf",
        "facets_xdmf": d / f"{stem}_facets.xdmf",
        "info": d / f"{stem}_mesh_info.json",
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_msh(level: str, msh_path: Path) -> None:
    import gmsh

    sizes = geo.MESH_LEVELS[level]
    points, names = geo.boundary_polygon()
    holes = geo.bed_hole_loops()
    gmsh.initialize(["-nt", "1"])
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.NumThreads", 1)  # deterministic
        gmsh.model.add(f"application2_room_{level}")
        geo_k = gmsh.model.geo
        ptags = [geo_k.addPoint(x, y, 0.0, sizes["lc_bulk"]) for (x, y) in points]
        ltags = []
        n = len(ptags)
        for i in range(n):
            ltags.append(geo_k.addLine(ptags[i], ptags[(i + 1) % n]))
        loop = geo_k.addCurveLoop(ltags)
        # bed holes (solid obstacles); every edge tagged with its bed name
        all_points, all_ptags = list(points), list(ptags)
        all_ltags, all_names = list(ltags), list(names)
        hole_loops = []
        for bed_name, hpts in holes:
            hp = [geo_k.addPoint(x, y, 0.0, sizes["lc_bulk"]) for (x, y) in hpts]
            hl = [geo_k.addLine(hp[i], hp[(i + 1) % 4]) for i in range(4)]
            hole_loops.append(geo_k.addCurveLoop(hl))
            all_points += hpts
            all_ptags += hp
            all_ltags += hl
            all_names += [bed_name] * 4
        surf = geo_k.addPlaneSurface([loop] + hole_loops)
        geo_k.synchronize()

        gmsh.model.addPhysicalGroup(2, [surf], geo.FLUID_TAG)
        gmsh.model.setPhysicalName(2, geo.FLUID_TAG, "fluid")
        by_name: dict[str, list[int]] = {}
        for lt, nm in zip(all_ltags, all_names):
            by_name.setdefault(nm, []).append(lt)
        for nm, tag in geo.BOUNDARY_TAGS.items():
            if nm not in by_name:
                raise RuntimeError(f"boundary name {nm} has no segments")
            gmsh.model.addPhysicalGroup(1, by_name[nm], tag)
            gmsh.model.setPhysicalName(1, tag, nm)

        # ---- size fields
        vent_curves = [lt for lt, nm in zip(all_ltags, all_names) if nm in geo.INLET_TAGS or nm in geo.OUTLET_TAGS]
        # partition faces, bed edges and the bathroom wall (incl. doorway jambs): lc_wall
        wall_curves = [lt for lt, nm in zip(all_ltags, all_names)
                       if nm.startswith(("partition", "bed_", "bath_wall"))]
        corner_pts = []
        for (cx, cy) in geo.corner_refinement_points():
            j = int(np.argmin([(px - cx) ** 2 + (py - cy) ** 2 for (px, py) in all_points]))
            assert abs(all_points[j][0] - cx) + abs(all_points[j][1] - cy) < 1e-12
            corner_pts.append(all_ptags[j])
        corner_pts = sorted(set(corner_pts))

        fields = []

        def threshold(dist_field, size_min, dmin, dmax):
            f = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(f, "InField", dist_field)
            gmsh.model.mesh.field.setNumber(f, "SizeMin", size_min)
            gmsh.model.mesh.field.setNumber(f, "SizeMax", sizes["lc_bulk"])
            gmsh.model.mesh.field.setNumber(f, "DistMin", dmin)
            gmsh.model.mesh.field.setNumber(f, "DistMax", dmax)
            return f

        fd = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(fd, "CurvesList", vent_curves)
        gmsh.model.mesh.field.setNumber(fd, "Sampling", 400)
        fields.append(threshold(fd, sizes["lc_vent"], *geo.REFINE_DIST["vent"]))

        fd = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(fd, "CurvesList", wall_curves)
        gmsh.model.mesh.field.setNumber(fd, "Sampling", 1000)
        fields.append(threshold(fd, sizes["lc_wall"], *geo.REFINE_DIST["wall"]))

        fd = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(fd, "PointsList", corner_pts)
        fields.append(threshold(fd, sizes["lc_corner"], *geo.REFINE_DIST["corner"]))

        fmin = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(fmin, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(fmin)

        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.RandomSeed", 1)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Laplace2D")

        msh_path.parent.mkdir(parents=True, exist_ok=True)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.write(str(msh_path))
    finally:
        gmsh.finalize()


def check_mesh(level: str, msh_path: Path) -> dict:
    """Validate the written mesh with meshio and collect statistics."""
    import meshio

    m = meshio.read(msh_path)
    pts = np.asarray(m.points[:, :2], dtype=float)
    tri = np.concatenate([b.data for b in m.cells if b.type == "triangle"])
    lines = np.concatenate([b.data for b in m.cells if b.type == "line"])
    ltags = np.concatenate([d for b, d in zip(m.cells, m.cell_data["gmsh:physical"]) if b.type == "line"])

    used = np.unique(tri)
    if len(used) != len(pts):
        raise RuntimeError(f"{len(pts) - len(used)} mesh points are not used by any triangle")

    a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    area = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    if np.any(np.abs(area) < 1e-14):
        raise RuntimeError("degenerate triangles")
    e = np.stack([np.linalg.norm(b - c, axis=1), np.linalg.norm(c - a, axis=1), np.linalg.norm(a - b, axis=1)], 1)
    hmax = e.max(1)
    # min interior angle per triangle
    cosA = (e[:, 1] ** 2 + e[:, 2] ** 2 - e[:, 0] ** 2) / (2 * e[:, 1] * e[:, 2])
    cosB = (e[:, 2] ** 2 + e[:, 0] ** 2 - e[:, 1] ** 2) / (2 * e[:, 2] * e[:, 0])
    cosC = (e[:, 0] ** 2 + e[:, 1] ** 2 - e[:, 2] ** 2) / (2 * e[:, 0] * e[:, 1])
    angles = np.degrees(np.arccos(np.clip(np.stack([cosA, cosB, cosC], 1), -1, 1)))
    min_angle = angles.min(1)

    # boundary: every boundary edge of the triangulation must be a tagged line
    edges = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    bnd = {tuple(x) for x in uniq[counts == 1]}
    tagged = {tuple(x) for x in np.sort(lines, axis=1)}
    if bnd != tagged:
        raise RuntimeError(f"boundary edges ({len(bnd)}) != tagged line cells ({len(tagged)})")

    # tag lengths vs geometry (outer loop + bed holes)
    seg_len = np.linalg.norm(pts[lines[:, 1]] - pts[lines[:, 0]], axis=1)
    expected = {}
    for p, q, nm in geo.all_boundary_segments():
        expected[nm] = expected.get(nm, 0.0) + float(np.linalg.norm(np.asarray(q) - np.asarray(p)))
    per_tag = {}
    for nm, tag in geo.BOUNDARY_TAGS.items():
        mask = ltags == tag
        L = float(seg_len[mask].sum())
        per_tag[nm] = {"tag": tag, "edges": int(mask.sum()), "length_m": L,
                       "expected_length_m": expected[nm], "length_error": abs(L - expected[nm])}
        if abs(L - expected[nm]) > 1e-9:
            raise RuntimeError(f"tag {nm}: length {L} != expected {expected[nm]}")
    total_area = float(np.abs(area).sum())
    if abs(total_area - geo.fluid_area()) > 1e-9:
        raise RuntimeError(f"mesh area {total_area} != fluid area {geo.fluid_area()}")

    topo = check_topology_and_solids(pts, tri, lines, ltags, uniq, counts, hmax)

    # vent-segment resolution (edges across each opening)
    vent_edges = {nm: per_tag[nm]["edges"] for nm in list(geo.INLET_NAMES) + list(geo.OUTLET_NAMES)}

    # P2/P1 Taylor-Hood dof counts on this triangulation (scalar P2 nodes = V + E)
    n_vertices = len(pts)
    n_edges = len(uniq)
    return {
        "level": level,
        "purpose": geo.MESH_LEVEL_PURPOSE[level],
        "sizes_m": geo.MESH_LEVELS[level],
        "refine_dist_m": geo.REFINE_DIST,
        "n_vertices": int(n_vertices),
        "n_triangles": int(len(tri)),
        "n_edges": int(n_edges),
        "n_boundary_edges": int(len(bnd)),
        "taylor_hood_velocity_dofs_P2_vector": int(2 * (n_vertices + n_edges)),
        "taylor_hood_pressure_dofs_P1": int(n_vertices),
        "taylor_hood_total_dofs": int(2 * (n_vertices + n_edges) + n_vertices),
        "transport_P1_nodes": int(n_vertices),
        "h_max_edge_m": {"min": float(hmax.min()), "median": float(np.median(hmax)), "max": float(hmax.max())},
        "min_angle_deg": {"min": float(min_angle.min()), "p01": float(np.percentile(min_angle, 1)),
                          "median": float(np.median(min_angle))},
        "triangle_area_m2_total": total_area,
        "fluid_area_m2_expected": geo.fluid_area(),
        "boundary_tags": per_tag,
        "edges_across_each_opening": vent_edges,
        "all_boundary_edges_tagged": True,
        "bed_edges_per_bed": {nm: per_tag[nm]["edges"] for nm in geo.BED_TAGS},
        "topology_and_solids": topo,
    }


def check_topology_and_solids(pts, tri, lines, ltags, uniq_edges, edge_counts, hmax) -> dict:
    """Connectivity, boundary loops, solid exclusion and passage coverage."""
    import matplotlib.tri as mtri
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    # (1) one edge-connected component of triangles
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    tri_id = np.tile(np.arange(len(tri)), 3)
    key = np.sort(edges, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    k_sorted, t_sorted = key[order], tri_id[order]
    same = np.all(k_sorted[1:] == k_sorted[:-1], axis=1)
    a, b = t_sorted[:-1][same], t_sorted[1:][same]
    adj = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(tri), len(tri)))
    n_comp, _ = connected_components(adj, directed=False)
    if n_comp != 1:
        raise RuntimeError(f"triangulation has {n_comp} edge-connected components (expected 1)")

    # (2) boundary loops: outer + one per bed
    bl = np.sort(lines, axis=1)
    vadj = coo_matrix((np.ones(len(bl)), (bl[:, 0], bl[:, 1])), shape=(len(pts), len(pts)))
    n_vcomp, vlab = connected_components(vadj, directed=False)
    bverts = np.unique(bl)
    loops = np.unique(vlab[bverts])
    if len(loops) != 1 + len(geo.BEDS):
        raise RuntimeError(f"{len(loops)} boundary loops (expected {1 + len(geo.BEDS)})")
    loop_tags = {}
    for lp in loops:
        on = vlab[bl[:, 0]] == lp
        loop_tags[int(lp)] = sorted(geo.TAG_TO_NAME[int(t)] for t in np.unique(ltags[on]))
    for nm in geo.BED_TAGS:
        if [nm] not in loop_tags.values():
            raise RuntimeError(f"bed {nm} is not a closed boundary loop of its own: {loop_tags}")

    # (3) no triangle inside a solid; (4) solid interiors not covered; (5) passages fully covered
    cen = pts[tri].mean(axis=1)
    n_cen_in_solid = int(geo.inside_solid(cen[:, 0], cen[:, 1]).sum())
    if n_cen_in_solid:
        raise RuntimeError(f"{n_cen_in_solid} triangle centroids lie inside a partition, bed or bathroom wall")
    n_cen_outside = int((~geo.inside_union(cen[:, 0], cen[:, 1])).sum())
    if n_cen_outside:
        raise RuntimeError(f"{n_cen_outside} triangle centroids lie outside main room UNION bathroom annex")
    finder = mtri.Triangulation(pts[:, 0], pts[:, 1], tri).get_trifinder()

    def sample(x0, x1, y0, y1, inset, n=41):
        gx = np.linspace(x0 + inset, x1 - inset, n)
        gy = np.linspace(y0 + inset, y1 - inset, n)
        X, Y = np.meshgrid(gx, gy)
        return X.ravel(), Y.ravel()

    solids = {}
    for nm, (x0, x1, y0, y1) in geo.SOLIDS.items():
        X, Y = sample(x0, x1, y0, y1, inset=1e-6, n=21 if nm.startswith("partition") else 41)
        hits = int((finder(X, Y) >= 0).sum())
        solids[nm] = {"interior_samples": int(X.size), "samples_inside_triangles": hits}
        if hits:
            raise RuntimeError(f"{hits} sample points inside solid {nm} are covered by triangles")
    exterior = {}
    for nm, (x0, x1, y0, y1) in geo.EXTERIOR_REGIONS.items():
        X, Y = sample(x0, x1, y0, y1, inset=1e-6)
        hits = int((finder(X, Y) >= 0).sum())
        exterior[nm] = {"samples": int(X.size), "samples_inside_triangles": hits}
        if hits:
            raise RuntimeError(f"{hits} sample points in exterior region {nm} are covered by triangles")
    pass_info = {}
    for nm, (x0, x1, y0, y1) in geo.passages().items():
        X, Y = sample(x0, x1, y0, y1, inset=1e-6)
        miss = int((finder(X, Y) < 0).sum())
        m = (cen[:, 0] > x0) & (cen[:, 0] < x1) & (cen[:, 1] > y0) & (cen[:, 1] < y1)
        width = geo.PASSAGE_CLEAR_WIDTH.get(nm, min(x1 - x0, y1 - y0))
        h_med = float(np.median(hmax[m])) if m.any() else float("nan")
        pass_info[nm] = {"rect": [x0, x1, y0, y1], "clear_width_m": width, "samples": int(X.size),
                         "samples_missing": miss, "triangles": int(m.sum()), "h_max_edge_median_m": h_med,
                         "approx_elements_across": width / h_med if m.any() else 0.0}
        if miss or not m.any():
            raise RuntimeError(f"passage {nm} is not fully covered by fluid triangles ({miss} samples missing)")
    return {"edge_connected_components": int(n_comp), "boundary_loops": len(loops),
            "boundary_loop_tags": list(loop_tags.values()),
            "triangle_centroids_inside_solids": n_cen_in_solid, "triangle_centroids_outside_union": n_cen_outside, "solid_interior_sampling": solids,
            "exterior_region_sampling": exterior,
            "passages": pass_info}


def write_vtu(paths: dict) -> str:
    """Mesh + boundary lines with integer cell data ``physical_tag`` (ParaView XML VTU)."""
    import meshio

    m = meshio.read(paths["msh"])
    cells, tags = [], []
    for blk, tg in zip(m.cells, m.cell_data["gmsh:physical"]):
        if blk.type in ("triangle", "line"):
            cells.append((blk.type, blk.data))
            tags.append(np.asarray(tg, dtype=np.int32))
    out = paths["msh"].with_suffix(".vtu")
    meshio.write(out, meshio.Mesh(m.points, cells, cell_data={"physical_tag": tags}), binary=True)
    return str(out)


def check_tag_transfer(msh_path: Path) -> dict:
    """Facet/cell tag counts as read by DOLFINx vs as written by Gmsh (meshio)."""
    import meshio
    from mpi4py import MPI
    from dolfinx.io import gmsh as dolfinx_gmsh

    m = meshio.read(msh_path)
    lt = np.concatenate([d for b, d in zip(m.cells, m.cell_data["gmsh:physical"]) if b.type == "line"])
    md = dolfinx_gmsh.read_from_msh(str(msh_path), MPI.COMM_SELF, rank=0, gdim=2)
    ft = md.facet_tags.values
    per = {}
    for nm, tag in geo.BOUNDARY_TAGS.items():
        per[nm] = {"tag": tag, "gmsh_lines": int((lt == tag).sum()), "dolfinx_facets": int((ft == tag).sum())}
    ct = np.unique(md.cell_tags.values)
    ok = all(v["gmsh_lines"] == v["dolfinx_facets"] > 0 for v in per.values()) and list(ct) == [geo.FLUID_TAG]
    extra = sorted(set(int(t) for t in np.unique(ft)) - set(geo.BOUNDARY_TAGS.values()))
    return {"ok": bool(ok and not extra), "per_tag": per, "cell_tags": [int(c) for c in ct],
            "unexpected_facet_tags": extra}


def write_xdmf(paths: dict) -> list[str]:
    from mpi4py import MPI
    from dolfinx import io
    from dolfinx.io import gmsh as dolfinx_gmsh

    md = dolfinx_gmsh.read_from_msh(str(paths["msh"]), MPI.COMM_SELF, rank=0, gdim=2)
    msh = md.mesh
    msh.topology.create_connectivity(1, 2)
    with io.XDMFFile(MPI.COMM_SELF, str(paths["xdmf"]), "w") as f:
        f.write_mesh(msh)
        if md.cell_tags is not None:
            f.write_meshtags(md.cell_tags, msh.geometry)
    with io.XDMFFile(MPI.COMM_SELF, str(paths["facets_xdmf"]), "w") as f:
        f.write_mesh(msh)
        f.write_meshtags(md.facet_tags, msh.geometry)
    return [str(paths["xdmf"]), str(paths["facets_xdmf"])]


def build_level(level: str, out_root: Path, xdmf: bool = True) -> dict:
    paths = mesh_paths(out_root, level)
    if paths["msh"].exists():
        raise FileExistsError(f"{paths['msh']} exists; refusing to overwrite (delete the level folder to rebuild)")
    print(f"[mesh:{level}] sizes {geo.MESH_LEVELS[level]}", flush=True)
    generate_msh(level, paths["msh"])
    info = check_mesh(level, paths["msh"])
    info["generated_utc"] = datetime.now(timezone.utc).isoformat()
    info["msh_file"] = str(paths["msh"])
    info["msh_sha256"] = sha256_file(paths["msh"])
    info["geometry_sha256"] = geo.geometry_hash()
    info["geometry"] = geo.geometry_dict()
    try:
        import gmsh
        info["gmsh_version"] = gmsh.__version__
    except Exception:  # noqa: BLE001
        pass
    info["vtu_file"] = write_vtu(paths)
    tt = check_tag_transfer(paths["msh"])
    info["tag_transfer_gmsh_to_dolfinx"] = tt
    if not tt["ok"]:
        paths["info"].write_text(json.dumps(info, indent=2) + "\n")
        raise RuntimeError(f"physical tags did not transfer to DOLFINx: {tt}")
    if xdmf:
        try:
            info["xdmf_files"] = write_xdmf(paths)
        except Exception as exc:  # noqa: BLE001  (the .msh is the primary format)
            info["xdmf_files"] = []
            info["xdmf_error"] = repr(exc)
            print(f"[mesh:{level}] XDMF export skipped: {exc}", flush=True)
    paths["info"].write_text(json.dumps(info, indent=2) + "\n")
    print(f"[mesh:{level}] vertices={info['n_vertices']} triangles={info['n_triangles']} "
          f"TH dofs v/p={info['taylor_hood_velocity_dofs_P2_vector']}/{info['taylor_hood_pressure_dofs_P1']} "
          f"min angle={info['min_angle_deg']['min']:.1f} deg  edges/opening={info['edges_across_each_opening']} "
          f"edges/bed={info['bed_edges_per_bed']}", flush=True)
    topo = info["topology_and_solids"]
    print(f"[mesh:{level}] components={topo['edge_connected_components']} boundary loops={topo['boundary_loops']} "
          f"centroids in solids={topo['triangle_centroids_inside_solids']} tag transfer ok={tt['ok']}; "
          "passages (width/h): " + ", ".join(f"{k}={v['approx_elements_across']:.1f}"
                                             for k, v in topo["passages"].items()), flush=True)
    print(f"[mesh:{level}] wrote {paths['msh']} (sha256 {info['msh_sha256'][:12]}...)", flush=True)
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", choices=["coarse", "fine", "all"], default="all")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                    help="Output root (default Applications/Application2/output/<OUTPUT_VERSION>).")
    ap.add_argument("--no-xdmf", action="store_true")
    a = ap.parse_args()
    levels = ["coarse", "fine"] if a.level == "all" else [a.level]
    for lv in levels:
        build_level(lv, a.out_root, xdmf=not a.no_xdmf)


if __name__ == "__main__":
    main()
