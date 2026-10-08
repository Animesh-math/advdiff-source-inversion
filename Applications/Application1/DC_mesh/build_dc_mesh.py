#!/usr/bin/env python3
"""Build a mesh-ready Washington DC (Navy Yard) 2D geometry and Gmsh mesh.

Omega = rectangular outer domain  minus  building-footprint holes.

Only building footprints are obstacles (holes). Parks / streets / soccer
fields are NOT obstacles.

The raw GIS data is visually fine but not automatically finite-element valid:
individually-valid polygons can still touch at a single point, sit a few
centimetres apart, or carry duplicate vertices. Those defects produce the
classic Gmsh failure

    There are N intersections in the 1D mesh (curves ...)
    Unable to recover the edge ... on curve ...

This script makes the polygon topology mesh-ready BEFORE calling Gmsh, then
generates the mesh. It keeps the geometry as real as possible and only falls
back to bounding-box / convex-hull replacement of a single obstacle as a last
resort, logging every such decision.

Run with the project environment:
    python3 DC_mesh/build_dc_mesh.py
"""

from __future__ import annotations

import os
import sys
import json
import itertools

import numpy as np

# geopandas / shapely / matplotlib are only needed for geometry cleanup and
# meshing. The `--convert-only` XDMF fallback runs in a different conda env
# (meshio + h5py) that need not have them, so keep these imports optional.
try:
    import geopandas as gpd
    from shapely.geometry import Polygon, MultiPolygon, Point, box  # noqa: F401
    from shapely.ops import unary_union
    try:
        from shapely import make_valid as _shapely_make_valid
    except Exception:  # pragma: no cover - older shapely
        _shapely_make_valid = None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # geometry stack absent (e.g. the XDMF-only fallback env)
    gpd = None
    Polygon = MultiPolygon = Point = box = None
    unary_union = None
    _shapely_make_valid = None
    plt = None

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))

DOMAIN_FILE = os.path.join(HERE, "dc_navy_yard_domain.geojson")
BUILDINGS_FILE = os.path.join(HERE, "dc_navy_yard_clean_buildings.geojson")

MESH_READY_BUILDINGS = os.path.join(HERE, "dc_navy_yard_mesh_ready_buildings.geojson")
MESH_READY_PREVIEW = os.path.join(HERE, "dc_navy_yard_mesh_ready_preview.png")
DEBUG_PREVIEW = os.path.join(HERE, "dc_navy_yard_mesh_debug.png")

MESH_MSH = os.path.join(HERE, "dc_navy_yard_mesh.msh")
MESH_XDMF = os.path.join(HERE, "dc_navy_yard_mesh.xdmf")
FACETS_XDMF = os.path.join(HERE, "dc_navy_yard_facets.xdmf")

BUILDING_METADATA = os.path.join(HERE, "dc_navy_yard_building_metadata.json")
COORD_ORIGIN_FILE = os.path.join(HERE, "dc_navy_yard_coordinate_origin.json")
BUILDING_LABELS_PNG = os.path.join(HERE, "dc_navy_yard_building_labels.png")

# Local meter-based CRS (District of Columbia, meters).
TARGET_CRS = "EPSG:26985"

# Cleanup parameters (meters).
MIN_OBSTACLE_AREA = 80.0        # drop tiny obstacles
MIN_BOUNDARY_CLEARANCE = 1.5    # drop / require clearance from outer boundary
CLOSE_EPS = 0.30                # morphological closing to weld point-touches
SIMPLIFY_TOL = 0.5              # light topology-preserving simplification
DEDUP_TOL = 0.05                # remove near-duplicate consecutive vertices
CONTAIN_SHRINK = 1.0            # strict-inside safety margin from rectangle

# Mesh size parameters (meters).
LC_OUTER = 22.0
LC_BUILDING = 10.0
MESH_SIZE_MIN = 4.0
MESH_SIZE_MAX = 22.0

# Physical group tags.
#
# Cells:
#   1   fluid
# Facets:
#   11  bottom_inflow   (rectangle edge at y = miny)
#   12  top_outflow     (rectangle edge at y = maxy)
#   13  left_wall       (rectangle edge at x = minx)
#   14  right_wall      (rectangle edge at x = maxx)
#   101, 102, ...       individual building walls (100 + building number)
TAG_FLUID = 1
TAG_BOTTOM_INFLOW = 11
TAG_TOP_OUTFLOW = 12
TAG_LEFT_WALL = 13
TAG_RIGHT_WALL = 14
BUILDING_TAG_OFFSET = 100  # building number N -> facet tag 100 + N

OUTER_WALLS = (
    # (side_key, tag, physical_name)
    ("bottom", TAG_BOTTOM_INFLOW, "bottom_inflow"),
    ("top", TAG_TOP_OUTFLOW, "top_outflow"),
    ("left", TAG_LEFT_WALL, "left_wall"),
    ("right", TAG_RIGHT_WALL, "right_wall"),
)

# Deterministic building numbering: buildings are grouped into approximately
# horizontal rows, rows are ordered north (largest centroid y) to south, and
# within each row buildings are ordered west (smallest centroid x) to east.
# ROW_TOL is the centroid-y band width (meters) used to cluster a row.
ROW_TOL = 30.0


# --------------------------------------------------------------------------- #
# Geometry cleanup helpers
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(f"[build_dc_mesh] {msg}", flush=True)


def make_valid_geom(geom):
    """Repair an invalid geometry via shapely.make_valid, else buffer(0)."""
    if geom is None or geom.is_empty:
        return geom
    if geom.is_valid:
        return geom
    if _shapely_make_valid is not None:
        try:
            return _shapely_make_valid(geom)
        except Exception:
            pass
    return geom.buffer(0)


def as_polygons(geom):
    """Return a list of Polygon parts from any geometry (drops non-polygons)."""
    if geom is None or geom.is_empty:
        return []
    gt = geom.geom_type
    if gt == "Polygon":
        return [geom]
    if gt == "MultiPolygon":
        return [g for g in geom.geoms if not g.is_empty]
    if gt == "GeometryCollection":
        out = []
        for g in geom.geoms:
            out.extend(as_polygons(g))
        return out
    return []  # points / lines are discarded


def fill_holes(poly: Polygon) -> Polygon:
    """Drop interior rings: building footprints are treated as solid obstacles."""
    if poly.interiors:
        return Polygon(poly.exterior)
    return poly


def dedup_ring(coords, tol: float):
    """Remove consecutive vertices closer than tol; keep ring closed."""
    coords = list(coords)
    if len(coords) < 4:
        return coords
    out = [coords[0]]
    for pt in coords[1:]:
        if np.hypot(pt[0] - out[-1][0], pt[1] - out[-1][1]) > tol:
            out.append(pt)
    # ensure closure
    if np.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) > tol:
        out.append(out[0])
    else:
        out[-1] = out[0]
    return out


def dedup_polygon(poly: Polygon, tol: float) -> Polygon:
    ext = dedup_ring(poly.exterior.coords, tol)
    if len(ext) < 4:
        return poly
    return Polygon(ext)


def close_weld(geom, eps: float):
    """Morphological closing to weld single-point contacts and hairline gaps."""
    if eps <= 0:
        return geom
    welded = geom.buffer(eps, join_style=2).buffer(-eps, join_style=2)
    return make_valid_geom(welded)


def clean_obstacles(buildings_gdf, domain_poly):
    """Full cleanup pipeline. Returns list of mesh-ready Polygons + a report."""
    report = {}

    # 1. repair + explode + keep polygons only
    raw = []
    for geom in buildings_gdf.geometry:
        raw.extend(as_polygons(make_valid_geom(geom)))
    report["after_explode"] = len(raw)

    # 2. fill interior rings (solid footprints), drop tiny obstacles
    filled = []
    n_had_holes = 0
    for p in raw:
        if p.interiors:
            n_had_holes += 1
        filled.append(fill_holes(p))
    report["buildings_with_interior_rings_filled"] = n_had_holes
    kept = [p for p in filled if p.area >= MIN_OBSTACLE_AREA]
    report["removed_tiny(<%.0fm2)" % MIN_OBSTACLE_AREA] = len(filled) - len(kept)

    # 3. drop obstacles touching / too close to the outer boundary
    bnd = domain_poly.boundary
    inside = []
    n_near_boundary = 0
    for p in kept:
        if (not domain_poly.contains(p)) or p.distance(bnd) < MIN_BOUNDARY_CLEARANCE:
            n_near_boundary += 1
            continue
        inside.append(p)
    report["removed_near_boundary"] = n_near_boundary

    # 4. merge overlapping/touching obstacles, then weld point contacts
    merged = unary_union(inside)
    merged = close_weld(merged, CLOSE_EPS)
    parts = as_polygons(merged)
    parts = [fill_holes(p) for p in parts]  # closing can never add real courtyards

    # 5. light simplify + dedup vertices
    cleaned = []
    for p in parts:
        s = p.simplify(SIMPLIFY_TOL, preserve_topology=True)
        s = make_valid_geom(s)
        for q in as_polygons(s):
            q = dedup_polygon(fill_holes(q), DEDUP_TOL)
            if q.is_valid and q.area >= MIN_OBSTACLE_AREA:
                cleaned.append(q)

    # 6. re-check pairwise; weld any residual touches/overlaps
    cleaned = resolve_residual_contacts(cleaned)

    # 7. guarantee strict containment inside the rectangle
    safe_domain = domain_poly.buffer(-CONTAIN_SHRINK)
    final = []
    n_clipped = 0
    for p in cleaned:
        if not safe_domain.contains(p):
            clip = p.intersection(safe_domain)
            clip = make_valid_geom(clip)
            polys = [q for q in as_polygons(clip) if q.area >= MIN_OBSTACLE_AREA]
            if not polys:
                continue
            n_clipped += 1
            final.extend(fill_holes(q) for q in polys)
        else:
            final.append(p)
    report["clipped_to_domain"] = n_clipped

    # final residual-contact pass after clipping
    final = resolve_residual_contacts(final)
    report["final_obstacles"] = len(final)
    return final, report


def resolve_residual_contacts(polys, max_passes: int = 6):
    """If any two obstacles still touch/intersect, weld them by closing."""
    polys = list(polys)
    for _ in range(max_passes):
        bad = None
        for i, j in itertools.combinations(range(len(polys)), 2):
            if polys[i].intersects(polys[j]):
                bad = (i, j)
                break
        if bad is None:
            return polys
        i, j = bad
        welded = close_weld(unary_union([polys[i], polys[j]]), CLOSE_EPS)
        new = [fill_holes(p) for p in as_polygons(welded)]
        polys = [p for k, p in enumerate(polys) if k not in (i, j)] + new
    log("WARNING: residual obstacle contacts persist after welding passes")
    return polys


# --------------------------------------------------------------------------- #
# Preview plotting
# --------------------------------------------------------------------------- #
def plot_geometry(domain_poly, obstacles, path, title, highlight=None):
    fig, ax = plt.subplots(figsize=(6, 12))
    dx, dy = domain_poly.exterior.xy
    ax.plot(dx, dy, "k-", lw=2)
    for idx, p in enumerate(obstacles):
        xs, ys = p.exterior.xy
        color = "red" if (highlight is not None and idx in highlight) else "#4c78a8"
        ax.fill(xs, ys, color=color, alpha=0.85, ec="k", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_building_metadata(building_info, path=BUILDING_METADATA):
    """Write per-building tag/centroid/area metadata keyed by building_NN."""
    meta = {}
    for b in building_info:
        meta["building_%02d" % b["building_number"]] = {
            "tag": b["tag"],
            "centroid": [round(b["centroid"][0], 4), round(b["centroid"][1], 4)],
            "area": round(b["area"], 4),
        }
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")


def plot_building_labels(domain_poly, obstacles, building_info, path=BUILDING_LABELS_PNG):
    """Plot every building and draw its building NUMBER at the centroid.

    The building number (not the physical facet tag) is drawn so the figure can
    be referenced in the paper when stating which buildings are used for
    measurements.
    """
    minx, miny, maxx, maxy = domain_poly.bounds
    fig, ax = plt.subplots(figsize=(6, 12))
    # outer rectangle actually meshed (the domain bounding box)
    ax.plot([minx, maxx, maxx, minx, minx],
            [miny, miny, maxy, maxy, miny], "k-", lw=2)
    for p in obstacles:
        xs, ys = p.exterior.xy
        ax.fill(xs, ys, color="#4c78a8", alpha=0.75, ec="k", lw=0.5)
    for b in building_info:
        cx, cy = b["centroid"]
        ax.text(cx, cy, str(b["building_number"]), ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("DC Navy Yard building numbering")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Deterministic building numbering
# --------------------------------------------------------------------------- #
def order_buildings(obstacles, row_tol: float = ROW_TOL):
    """Number buildings deterministically from their footprint geometry.

    Rule:
      * group buildings into approximately horizontal rows by centroid y,
        using ``row_tol`` as the band width;
      * order rows north -> south (largest centroid y first);
      * within each row order west -> east (smallest centroid x first).

    Returns a list of per-building dicts (one per obstacle), ordered by
    building number, each with keys:
        building_number, tag, obstacle_index, centroid, area
    The ordering depends only on the footprint geometry, so it is reproducible
    on every run.
    """
    centroids = [(p.centroid.x, p.centroid.y) for p in obstacles]
    areas = [float(p.area) for p in obstacles]

    # sort north -> south, then west -> east so that greedy row banding is
    # itself deterministic and rows read left-to-right.
    by_ns = sorted(range(len(obstacles)), key=lambda i: (-centroids[i][1], centroids[i][0]))

    rows = []
    current = []
    band_top = None
    for i in by_ns:
        cy = centroids[i][1]
        if band_top is None or (band_top - cy) <= row_tol:
            if band_top is None:
                band_top = cy
            current.append(i)
        else:
            rows.append(current)
            current = [i]
            band_top = cy
    if current:
        rows.append(current)

    building_info = []
    number = 0
    for row in rows:
        for i in sorted(row, key=lambda k: centroids[k][0]):  # west -> east
            number += 1
            building_info.append({
                "building_number": number,
                "tag": BUILDING_TAG_OFFSET + number,
                "obstacle_index": i,
                "centroid": [float(centroids[i][0]), float(centroids[i][1])],
                "area": areas[i],
            })
    return building_info


# --------------------------------------------------------------------------- #
# Gmsh meshing
# --------------------------------------------------------------------------- #
def build_gmsh_mesh(domain_poly, obstacles, out_msh):
    """Mesh rectangle minus building holes with OCC boolean cut.

    Tags the four rectangle edges separately (bottom/top/left/right, detected
    geometrically) and gives every building its own facet physical group using
    the deterministic :func:`order_buildings` numbering.

    Returns ``(num_nodes, num_tris, building_info)``. Raises on failure.
    """
    import gmsh

    minx, miny, maxx, maxy = domain_poly.bounds
    W, H = maxx - minx, maxy - miny

    building_info = order_buildings(obstacles)
    # obstacle index -> (building number, facet tag)
    obstacle_to_building = {
        b["obstacle_index"]: (b["building_number"], b["tag"]) for b in building_info
    }

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("dc_navy_yard")
        occ = gmsh.model.occ

        rect = occ.addRectangle(minx, miny, 0.0, W, H)

        bldg_tags = []
        for p in obstacles:
            coords = list(p.exterior.coords)[:-1]  # drop closing dup
            pts = [occ.addPoint(x, y, 0.0) for x, y in coords]
            lines = [
                occ.addLine(pts[k], pts[(k + 1) % len(pts)])
                for k in range(len(pts))
            ]
            loop = occ.addCurveLoop(lines)
            bldg_tags.append(occ.addPlaneSurface([loop]))

        # fluid = rectangle - buildings
        fluid, _ = occ.cut(
            [(2, rect)], [(2, t) for t in bldg_tags],
            removeObject=True, removeTool=True,
        )
        occ.synchronize()

        fluid_surfs = [t for (d, t) in fluid if d == 2]

        # Classify each boundary curve. Rectangle-perimeter curves are split
        # into the four named walls by which side their center of mass sits on;
        # every remaining curve is an interior building wall, assigned to the
        # nearest building footprint.
        tol = 1e-6 * max(W, H) + 1e-3
        wall_curves = {key: [] for key, _tag, _name in OUTER_WALLS}
        # building number -> list of curve tags
        building_curves = {b["building_number"]: [] for b in building_info}
        all_building_curves = []
        for (d, t) in gmsh.model.getBoundary(
            [(2, s) for s in fluid_surfs], oriented=False, combined=True
        ):
            if d != 1:
                continue
            cx, cy, _ = gmsh.model.occ.getCenterOfMass(1, t)
            side_dist = {
                "bottom": abs(cy - miny),
                "top": abs(cy - maxy),
                "left": abs(cx - minx),
                "right": abs(cx - maxx),
            }
            side, dmin = min(side_dist.items(), key=lambda kv: kv[1])
            if dmin < tol:
                wall_curves[side].append(t)
            else:
                # interior wall: nearest building footprint by boundary distance
                pt = Point(cx, cy)
                oi = min(
                    range(len(obstacles)),
                    key=lambda k: obstacles[k].exterior.distance(pt),
                )
                number, _bt = obstacle_to_building[oi]
                building_curves[number].append(t)
                all_building_curves.append(t)

        gmsh.model.addPhysicalGroup(2, fluid_surfs, TAG_FLUID)
        gmsh.model.setPhysicalName(2, TAG_FLUID, "fluid")
        for key, tag, name in OUTER_WALLS:
            if wall_curves[key]:
                gmsh.model.addPhysicalGroup(1, wall_curves[key], tag)
                gmsh.model.setPhysicalName(1, tag, name)
        for b in building_info:
            curves = building_curves[b["building_number"]]
            if curves:
                gmsh.model.addPhysicalGroup(1, curves, b["tag"])
                gmsh.model.setPhysicalName(
                    1, b["tag"], "building_%02d" % b["building_number"]
                )

        # size field: fine near building curves, coarse away
        gmsh.option.setNumber("Mesh.MeshSizeMin", MESH_SIZE_MIN)
        gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_SIZE_MAX)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay

        if all_building_curves:
            fdist = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(fdist, "CurvesList", all_building_curves)
            gmsh.model.mesh.field.setNumber(fdist, "Sampling", 200)
            fthr = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(fthr, "InField", fdist)
            gmsh.model.mesh.field.setNumber(fthr, "SizeMin", LC_BUILDING)
            gmsh.model.mesh.field.setNumber(fthr, "SizeMax", LC_OUTER)
            gmsh.model.mesh.field.setNumber(fthr, "DistMin", 5.0)
            gmsh.model.mesh.field.setNumber(fthr, "DistMax", 60.0)
            gmsh.model.mesh.field.setAsBackgroundMesh(fthr)
        else:
            gmsh.option.setNumber("Mesh.MeshSizeMax", LC_OUTER)

        gmsh.model.mesh.generate(2)

        node_tags, _, _ = gmsh.model.mesh.getNodes()
        num_nodes = len(node_tags)
        etypes, etags, _ = gmsh.model.mesh.getElements(2)
        num_tris = sum(len(t) for et, t in zip(etypes, etags) if et == 2)

        gmsh.write(out_msh)
        return num_nodes, num_tris, building_info
    finally:
        gmsh.finalize()


# Conda environment that has meshio + h5py for XDMF (HDF5) writing, used as a
# fallback when the running interpreter lacks h5py.
XDMF_FALLBACK_CONDA_ENV = "fenicsx-env"


def convert_to_xdmf(msh_path):
    """Convert .msh to fluid .xdmf + facet .xdmf, preserving physical tags.

    meshio's XDMF writer needs h5py. If h5py is missing in this interpreter,
    re-run only the conversion inside a conda env that has it.
    """
    try:
        import meshio  # noqa: F401
        import h5py  # noqa: F401
    except Exception as exc:
        log(f"h5py/meshio unavailable here ({exc}); "
            f"retrying XDMF conversion via conda env '{XDMF_FALLBACK_CONDA_ENV}'")
        import subprocess
        try:
            subprocess.run(
                ["conda", "run", "-n", XDMF_FALLBACK_CONDA_ENV,
                 "python", os.path.abspath(__file__), "--convert-only", msh_path],
                check=True,
            )
            return os.path.exists(MESH_XDMF)
        except Exception as exc2:  # noqa: BLE001
            log(f"XDMF conversion fallback failed ({exc2}); skipping XDMF")
            return False

    import meshio
    m = meshio.read(msh_path)

    def extract(cell_type):
        cells = [c for c in m.cells if c.type == cell_type]
        if not cells:
            return None
        data = np.concatenate([c.data for c in cells])
        key = "gmsh:physical"
        phys = None
        if key in m.cell_data_dict and cell_type in m.cell_data_dict[key]:
            phys = m.cell_data_dict[key][cell_type]
        return data, phys

    tri = extract("triangle")
    lin = extract("line")

    if tri is not None:
        cell_data = {"name_to_read": [tri[1]]} if tri[1] is not None else {}
        meshio.write(
            MESH_XDMF,
            meshio.Mesh(points=m.points[:, :2], cells=[("triangle", tri[0])],
                        cell_data=cell_data),
        )
        log(f"wrote {os.path.basename(MESH_XDMF)}")

    if lin is not None:
        cell_data = {"name_to_read": [lin[1]]} if lin[1] is not None else {}
        meshio.write(
            FACETS_XDMF,
            meshio.Mesh(points=m.points[:, :2], cells=[("line", lin[0])],
                        cell_data=cell_data),
        )
        log(f"wrote {os.path.basename(FACETS_XDMF)}")
    return True


# --------------------------------------------------------------------------- #
# Fallback: replace one problematic obstacle
# --------------------------------------------------------------------------- #
def oriented_bbox(poly: Polygon) -> Polygon:
    try:
        mrr = poly.minimum_rotated_rectangle
        if mrr.geom_type == "Polygon":
            return mrr
    except Exception:
        pass
    return box(*poly.bounds)


def mesh_with_fallback(domain_poly, obstacles):
    """Try meshing; on failure, replace obstacles by simpler proxies and log."""
    attempts = [("cleaned", obstacles)]

    # stage 2: stronger weld + separation
    stronger = resolve_residual_contacts(
        [p.buffer(0) for p in as_polygons(close_weld(unary_union(obstacles), CLOSE_EPS * 2))]
    )
    attempts.append(("stronger-weld", stronger))

    # stage 3: convex hulls
    attempts.append(("convex-hull", [p.convex_hull for p in obstacles]))

    # stage 4: oriented bounding boxes
    attempts.append(("oriented-bbox", [oriented_bbox(p) for p in obstacles]))

    last_exc = None
    for name, obs in attempts:
        obs = [p for p in obs if isinstance(p, Polygon) and p.area >= MIN_OBSTACLE_AREA]
        obs = resolve_residual_contacts(obs)
        try:
            n_nodes, n_tris, building_info = build_gmsh_mesh(domain_poly, obs, MESH_MSH)
            if name != "cleaned":
                log(f"FALLBACK USED: meshed with '{name}' obstacle proxies "
                    f"({len(obs)} obstacles). Geometry was simplified to reach a "
                    f"valid mesh; see log above.")
            return obs, n_nodes, n_tris, name, building_info
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log(f"mesh attempt '{name}' failed: {exc}")
    raise RuntimeError(f"all mesh attempts failed; last error: {last_exc}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    log("reading inputs")
    domain_gdf = gpd.read_file(DOMAIN_FILE)
    buildings_gdf = gpd.read_file(BUILDINGS_FILE)

    # ensure meter-based CRS
    for name, gdf in (("domain", domain_gdf), ("buildings", buildings_gdf)):
        if gdf.crs is None:
            log(f"WARNING: {name} has no CRS; assuming {TARGET_CRS}")
            gdf.set_crs(TARGET_CRS, inplace=True, allow_override=True)
        elif gdf.crs.to_string() != TARGET_CRS:
            log(f"reprojecting {name} from {gdf.crs.to_string()} to {TARGET_CRS}")
            gdf.to_crs(TARGET_CRS, inplace=True)

    domain_poly = make_valid_geom(unary_union(list(domain_gdf.geometry)))
    domain_poly = max(as_polygons(domain_poly), key=lambda p: p.area)

    minx, miny, maxx, maxy = domain_poly.bounds
    log(f"domain bounds x[{minx:.1f},{maxx:.1f}] y[{miny:.1f},{maxy:.1f}] "
        f"=> {maxx-minx:.1f} m x {maxy-miny:.1f} m")

    obstacles, report = clean_obstacles(buildings_gdf, domain_poly)
    log("cleanup report:")
    for k, v in report.items():
        log(f"    {k}: {v}")

    # --- LOCAL COORDINATE TRANSLATION -------------------------------------
    # Shift the whole geometry once so the south-west corner of the outer domain
    # becomes the origin (0, 0). This is a PURE rigid translation (no scaling,
    # no rotation): every node, building, and downstream artifact (mesh, facets,
    # metadata, plots) ends up in local meters, so the numerical pipeline never
    # subtracts an origin at runtime. The original projected (GIS) origin is
    # recorded in dc_navy_yard_coordinate_origin.json for reference; global =
    # local + origin. (The committed mesh was produced by this translation.)
    from shapely.affinity import translate as _translate
    origin_x, origin_y = minx, miny
    domain_poly = _translate(domain_poly, xoff=-origin_x, yoff=-origin_y)
    obstacles = [_translate(o, xoff=-origin_x, yoff=-origin_y) for o in obstacles]
    _lminx, _lminy, _lmaxx, _lmaxy = domain_poly.bounds
    with open(COORD_ORIGIN_FILE, "w") as _fh:
        json.dump({
            "description": ("Local coordinate origin for the DC Navy Yard mesh. "
                            "The mesh nodes were translated once so the south-west "
                            "corner of the outer domain is the local origin (0, 0). "
                            "global = local + origin. Reference-only; the numerical "
                            "pipeline runs in local meters."),
            "origin_x": origin_x,
            "origin_y": origin_y,
            "origin_units": "meters (projected, UTM-like)",
            "translation_vector_applied": [-origin_x, -origin_y],
            "local_bounding_box": {"x_min": _lminx, "x_max": _lmaxx,
                                   "y_min": _lminy, "y_max": _lmaxy},
            "scaling_applied": False,
            "rotation_applied": False,
        }, _fh, indent=2)
        _fh.write("\n")
    log(f"translated geometry to local origin ({origin_x:.4f}, {origin_y:.4f}); "
        f"local bounds x[0,{_lmaxx:.1f}] y[0,{_lmaxy:.1f}] (no scaling/rotation)")

    # write mesh-ready buildings + preview
    out_gdf = gpd.GeoDataFrame(
        {"obstacle_id": range(len(obstacles)),
         "area_m2": [round(p.area, 2) for p in obstacles]},
        geometry=obstacles, crs=TARGET_CRS,
    )
    out_gdf.to_file(MESH_READY_BUILDINGS, driver="GeoJSON")
    log(f"wrote {os.path.basename(MESH_READY_BUILDINGS)} ({len(obstacles)} obstacles)")
    plot_geometry(domain_poly, obstacles, MESH_READY_PREVIEW,
                  "DC Navy Yard mesh-ready geometry")
    log(f"wrote {os.path.basename(MESH_READY_PREVIEW)}")

    # mesh
    try:
        used_obs, n_nodes, n_tris, stage, building_info = mesh_with_fallback(
            domain_poly, obstacles)
    except Exception as exc:
        # write a debug plot highlighting overlapping/touching obstacles
        highlight = set()
        for i, j in itertools.combinations(range(len(obstacles)), 2):
            if obstacles[i].intersects(obstacles[j]):
                highlight.update((i, j))
        plot_geometry(domain_poly, obstacles, DEBUG_PREVIEW,
                      "DC mesh FAILED - highlighted bad obstacles", highlight)
        log(f"MESH FAILED: {exc}")
        log(f"wrote debug plot {os.path.basename(DEBUG_PREVIEW)}")
        raise

    # if a fallback changed the geometry, refresh the mesh-ready outputs
    if stage != "cleaned":
        out_gdf = gpd.GeoDataFrame(
            {"obstacle_id": range(len(used_obs)),
             "area_m2": [round(p.area, 2) for p in used_obs]},
            geometry=used_obs, crs=TARGET_CRS,
        )
        out_gdf.to_file(MESH_READY_BUILDINGS, driver="GeoJSON")
        plot_geometry(domain_poly, used_obs, MESH_READY_PREVIEW,
                      f"DC Navy Yard mesh-ready geometry ({stage})")
        obstacles = used_obs

    # building metadata JSON + numbered label plot (from the meshed geometry)
    write_building_metadata(building_info)
    log(f"wrote {os.path.basename(BUILDING_METADATA)} ({len(building_info)} buildings)")
    plot_building_labels(domain_poly, obstacles, building_info)
    log(f"wrote {os.path.basename(BUILDING_LABELS_PNG)}")

    xdmf_ok = convert_to_xdmf(MESH_MSH)

    # verification summary
    log("=" * 60)
    log("VERIFICATION SUMMARY")
    log(f"  final obstacles     : {len(obstacles)}")
    log(f"  domain size (m)     : {maxx-minx:.1f} x {maxy-miny:.1f}")
    log(f"  mesh stage used     : {stage}")
    log(f"  mesh nodes          : {n_nodes}")
    log(f"  mesh triangles      : {n_tris}")
    log(f"  .msh exists         : {os.path.exists(MESH_MSH)} ({os.path.basename(MESH_MSH)})")
    if xdmf_ok:
        log(f"  XDMF fluid exists   : {os.path.exists(MESH_XDMF)} ({os.path.basename(MESH_XDMF)})")
        log(f"  XDMF facets exists  : {os.path.exists(FACETS_XDMF)} ({os.path.basename(FACETS_XDMF)})")
    log("=" * 60)

    # boundary-tag verification listing
    print()
    print(f"Number of buildings: {len(building_info)}")
    print()
    print("Boundary tags:")
    for _key, tag, name in OUTER_WALLS:
        print(f"{tag} {name}")
    for b in building_info:
        print(f"{b['tag']} Building {b['building_number']}")
    return 0


if __name__ == "__main__":
    # Minimal entry point for the h5py-env XDMF fallback (see convert_to_xdmf).
    if len(sys.argv) >= 2 and sys.argv[1] == "--convert-only":
        msh = sys.argv[2] if len(sys.argv) >= 3 else MESH_MSH
        sys.exit(0 if convert_to_xdmf(msh) else 1)
    sys.exit(main())
