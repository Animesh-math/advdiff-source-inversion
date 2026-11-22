"""Geometry, boundary tags and mesh resolutions for Application2.

Application2 is an idealized two-dimensional, top-down (plan-view) three-bed
room with a bathroom annex.  This module is the SINGLE SOURCE OF TRUTH for the
geometry: the mesh builder, the Navier-Stokes solve, the mesh_data loader and
the report all import it, so the coarse and fine meshes share exactly the same
geometry and boundary tagging.

The ventilation arrangement (ceiling supplies above the beds, air leaving
through the corridor door and a bathroom exhaust) is inspired by Figure 24 and
Section 7.1 of Lohner et al., arXiv:2006.13792.  This three-bed room is an
adaptation, not a reproduction: the paper's bathroom ceiling exhaust is
represented in 2D by a boundary outlet on the bathroom's outer wall.  The
reference drawing is ``room_layout_reference.png`` (not to scale; the
coordinates below define the computational geometry).

All lengths are in metres; x left -> right, y bottom (corridor side) -> top.

Layout (version ``OUTPUT_VERSION``)
-----------------------------------
  * Fluid domain = main room (0,9)x(0,6)  UNION  bathroom annex (9,11)x(0,2),
    minus the solids below.
  * Two solid partitions (0.10 m) descending from the top wall:
        partition_1 = [2.95, 3.05] x [3, 6],   partition_2 = [5.95, 6.05] x [3, 6].
    They form bays 1-3; the lower part 0 < y < 3 is a shared aisle.
  * Three SOLID bed footprints (holes in the mesh; heads toward the top wall):
        bed_1 = [1, 2] x [3.5, 5.6],  bed_2 = [4, 5] x [3.5, 5.6],  bed_3 = [7, 8] x [3.5, 5.6].
  * Bathroom separating wall (0.10 m, finite thickness), with an open interior
    doorway 0.6 < y < 1.4 (an internal fluid passage, NOT a boundary):
        bath_wall_lower = [8.95, 9.05] x [0, 0.6],  bath_wall_upper = [8.95, 9.05] x [1.4, 2].
    The upper piece protrudes 0.05 m into the main room, leaving a 5 cm wall
    step at y = 2, x in [8.95, 9.0] (part of tag bath_wall_upper).
  * Openings:
        inlet_1..3      y = 6,  x in [1.2,1.8], [4.2,4.8], [7.2,7.8]   supply, flow DOWN into the room
        outlet_door     y = 0,  x in [0.6, 1.6]                         corridor doorway (natural outflow)
        outlet_exhaust  x = 11, y in [0.7, 1.3]                         bathroom exhaust (natural outflow)
  * Every other boundary segment (exterior walls, partitions, beds, bathroom
    wall) is a solid no-slip wall.  Beds are an idealized 2D representation of
    impermeable obstacles; pillows/bedding are drawing only.

This is an idealized 2D ventilation configuration for an inverse-problem test
case.  It is not a validated hospital ventilation design.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

OUTPUT_VERSION = "v3_bathroom_topsupply_20261116"

# --------------------------------------------------------------------------- #
# Room, annex and solids [m]
# --------------------------------------------------------------------------- #
ROOM_X_MIN, ROOM_X_MAX = 0.0, 9.0
ROOM_Y_MIN, ROOM_Y_MAX = 0.0, 6.0
ANNEX_X = (9.0, 11.0)
ANNEX_Y = (0.0, 2.0)
DOMAIN_X_MAX = ANNEX_X[1]
RECTANGLES = {"main_room": (ROOM_X_MIN, ROOM_X_MAX, ROOM_Y_MIN, ROOM_Y_MAX),
              "bathroom_annex": (*ANNEX_X, *ANNEX_Y)}

PARTITION_THICKNESS = 0.10
PARTITION_BOTTOM_Y = 3.0
PARTITIONS = {
    "partition_1": (2.95, 3.05, PARTITION_BOTTOM_Y, ROOM_Y_MAX),
    "partition_2": (5.95, 6.05, PARTITION_BOTTOM_Y, ROOM_Y_MAX),
}

BATH_WALL_X = (8.95, 9.05)
BATH_DOORWAY_Y = (0.6, 1.4)
BATH_WALLS = {
    "bath_wall_lower": (*BATH_WALL_X, ANNEX_Y[0], BATH_DOORWAY_Y[0]),
    "bath_wall_upper": (*BATH_WALL_X, BATH_DOORWAY_Y[1], ANNEX_Y[1]),
}

BAYS = {
    1: {"x": (ROOM_X_MIN, 2.95), "y": (PARTITION_BOTTOM_Y, ROOM_Y_MAX)},
    2: {"x": (3.05, 5.95), "y": (PARTITION_BOTTOM_Y, ROOM_Y_MAX)},
    3: {"x": (6.05, ROOM_X_MAX), "y": (PARTITION_BOTTOM_Y, ROOM_Y_MAX)},
}
AISLE = {"x": (ROOM_X_MIN, BATH_WALL_X[0]), "y": (ROOM_Y_MIN, PARTITION_BOTTOM_Y)}
BATHROOM = {"x": (BATH_WALL_X[1], ANNEX_X[1]), "y": ANNEX_Y}
DOORWAY = {"x": BATH_WALL_X, "y": BATH_DOORWAY_Y}

BED_WIDTH, BED_LENGTH = 1.0, 2.1
BED_Y_RANGE = (3.5, 5.6)
BED_CENTERS_X = (1.5, 4.5, 7.5)
BEDS = {
    bay: {"x": (cx - 0.5 * BED_WIDTH, cx + 0.5 * BED_WIDTH), "y": BED_Y_RANGE,
          "head_y": BED_Y_RANGE[1], "foot_y": BED_Y_RANGE[0]}
    for bay, cx in zip((1, 2, 3), BED_CENTERS_X)
}
BED_NAMES = {bay: f"bed_{bay}" for bay in BEDS}

SOLIDS = {**PARTITIONS, **{BED_NAMES[b]: (*BEDS[b]["x"], *BEDS[b]["y"]) for b in BEDS}, **BATH_WALLS}

# --------------------------------------------------------------------------- #
# Ventilation openings [m]
# --------------------------------------------------------------------------- #
VENT_WIDTH = 0.60                       # each supply opening
VENT_CENTERS_X = (1.5, 4.5, 7.5)
VENT_X_RANGES = tuple((c - 0.5 * VENT_WIDTH, c + 0.5 * VENT_WIDTH) for c in VENT_CENTERS_X)
INLET_Y = ROOM_Y_MAX                    # supplies on the top wall; flow is in -y
DOOR_X_RANGE = (0.6, 1.6)               # corridor doorway outlet on y = 0
EXHAUST_Y_RANGE = (0.7, 1.3)            # bathroom exhaust outlet on x = 11
# outward unit normals of the outlets
OUTLET_NORMALS = {"outlet_door": (0.0, -1.0), "outlet_exhaust": (1.0, 0.0)}

# --------------------------------------------------------------------------- #
# Gmsh physical tags
# --------------------------------------------------------------------------- #
FLUID_TAG = 1
INLET_TAGS = {"inlet_1": 21, "inlet_2": 22, "inlet_3": 23}
OUTLET_TAGS = {"outlet_door": 31, "outlet_exhaust": 32}
WALL_TAGS = {
    "wall_bottom": 41,      # main room y = 0 except the door
    "wall_right": 42,       # main room x = 9, 2 <= y <= 6
    "wall_top": 43,         # main room y = 6 except supplies / partition roots
    "wall_left": 44,        # x = 0
    "bath_bottom": 45,      # annex y = 0
    "bath_right": 46,       # annex x = 11 except the exhaust
    "bath_top": 47,         # annex y = 2
    "partition_1": 51,
    "partition_2": 52,
    "bed_1": 61,
    "bed_2": 62,
    "bed_3": 63,
    "bath_wall_lower": 71,  # all exposed faces of the lower separator piece
    "bath_wall_upper": 72,  # all exposed faces of the upper piece (incl. 5 cm step at y = 2)
}
BED_TAGS = {nm: WALL_TAGS[nm] for nm in BED_NAMES.values()}
BOUNDARY_TAGS = {**INLET_TAGS, **OUTLET_TAGS, **WALL_TAGS}
TAG_TO_NAME = {tag: name for name, tag in BOUNDARY_TAGS.items()}

INLET_NAMES = tuple(INLET_TAGS)
OUTLET_NAMES = tuple(OUTLET_TAGS)
WALL_NAMES = tuple(WALL_TAGS)
INLET_TAG_LIST = tuple(INLET_TAGS.values())
OUTLET_TAG_LIST = tuple(OUTLET_TAGS.values())
WALL_TAG_LIST = tuple(WALL_TAGS.values())
INLET_BAY = {"inlet_1": 1, "inlet_2": 2, "inlet_3": 3}

# --------------------------------------------------------------------------- #
# Mesh resolutions [m]  (fine = exactly half of every coarse size)
# --------------------------------------------------------------------------- #
#   lc_bulk   : bulk element size
#   lc_vent   : along supply and outlet segments
#   lc_wall   : along partition faces, bed edges and the bathroom wall
#   lc_corner : at opening endpoints and every solid corner (partition tips,
#               bed corners, doorway jambs, the 5 cm step)
MESH_LEVELS = {
    "coarse": {"lc_bulk": 0.20, "lc_vent": 0.05, "lc_wall": 0.07, "lc_corner": 0.025},
    "fine":   {"lc_bulk": 0.10, "lc_vent": 0.025, "lc_wall": 0.035, "lc_corner": 0.0125},
}
REFINE_DIST = {"vent": (0.05, 0.60), "wall": (0.05, 0.50), "corner": (0.02, 0.40)}
MESH_LEVEL_PURPOSE = {
    "coarse": "intended for eventual optimization (inverse solves)",
    "fine": "intended for eventual synthetic-data generation",
}


# --------------------------------------------------------------------------- #
# Boundary description
# --------------------------------------------------------------------------- #
def boundary_polygon():
    """Counter-clockwise outer boundary (fluid on the left) as (points, segment_names).

    Segment i runs points[i] -> points[i+1] (closing to points[0]).  Partitions and
    the bathroom wall are notches of this single loop; beds are separate holes.
    """
    Y1 = ROOM_Y_MAX
    (d0, d1), (e0, e1) = DOOR_X_RANGE, EXHAUST_Y_RANGE
    (bx0, bx1), (g0, g1) = BATH_WALL_X, BATH_DOORWAY_Y
    ax1, ay1 = ANNEX_X[1], ANNEX_Y[1]
    (i1a, i1b), (i2a, i2b), (i3a, i3b) = VENT_X_RANGES
    p1, p2 = PARTITIONS["partition_1"], PARTITIONS["partition_2"]
    yb = PARTITION_BOTTOM_Y
    seq = [
        ((0.0, 0.0), "wall_bottom"),
        ((d0, 0.0), "outlet_door"),
        ((d1, 0.0), "wall_bottom"),
        ((bx0, 0.0), "bath_wall_lower"),
        ((bx0, g0), "bath_wall_lower"),
        ((bx1, g0), "bath_wall_lower"),
        ((bx1, 0.0), "bath_bottom"),
        ((ax1, 0.0), "bath_right"),
        ((ax1, e0), "outlet_exhaust"),
        ((ax1, e1), "bath_right"),
        ((ax1, ay1), "bath_top"),
        ((bx1, ay1), "bath_wall_upper"),
        ((bx1, g1), "bath_wall_upper"),
        ((bx0, g1), "bath_wall_upper"),
        ((bx0, ay1), "bath_wall_upper"),     # 5 cm step along y = 2
        ((ROOM_X_MAX, ay1), "wall_right"),
        ((ROOM_X_MAX, Y1), "wall_top"),
        ((i3b, Y1), "inlet_3"),
        ((i3a, Y1), "wall_top"),
        ((p2[1], Y1), "partition_2"),
        ((p2[1], yb), "partition_2"),
        ((p2[0], yb), "partition_2"),
        ((p2[0], Y1), "wall_top"),
        ((i2b, Y1), "inlet_2"),
        ((i2a, Y1), "wall_top"),
        ((p1[1], Y1), "partition_1"),
        ((p1[1], yb), "partition_1"),
        ((p1[0], yb), "partition_1"),
        ((p1[0], Y1), "wall_top"),
        ((i1b, Y1), "inlet_1"),
        ((i1a, Y1), "wall_top"),
        ((0.0, Y1), "wall_left"),
    ]
    return [p for p, _ in seq], [n for _, n in seq]


def bed_hole_loops():
    """One closed rectangle per bed: list of (name, 4 corner points)."""
    out = []
    for bay, b in BEDS.items():
        (x0, x1), (y0, y1) = b["x"], b["y"]
        out.append((BED_NAMES[bay], [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
    return out


def all_boundary_segments():
    """Every boundary segment (outer loop + bed holes) as (p, q, name)."""
    points, names = boundary_polygon()
    segs = [(points[i], points[(i + 1) % len(points)], names[i]) for i in range(len(points))]
    for nm, pts in bed_hole_loops():
        segs += [(pts[i], pts[(i + 1) % 4], nm) for i in range(4)]
    return segs


def opening_endpoints():
    pts = []
    for a, b in VENT_X_RANGES:
        pts += [(a, INLET_Y), (b, INLET_Y)]
    pts += [(DOOR_X_RANGE[0], 0.0), (DOOR_X_RANGE[1], 0.0)]
    pts += [(ANNEX_X[1], EXHAUST_Y_RANGE[0]), (ANNEX_X[1], EXHAUST_Y_RANGE[1])]
    return pts


def corner_refinement_points():
    """Opening endpoints, all solid corners, and the 5 cm step corner (9, 2)."""
    pts = opening_endpoints()
    for x0, x1, y0, y1 in SOLIDS.values():
        pts += [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    pts.append((ROOM_X_MAX, ANNEX_Y[1]))
    return pts


def fluid_area() -> float:
    gross = sum((x1 - x0) * (y1 - y0) for x0, x1, y0, y1 in RECTANGLES.values())
    return gross - sum((x1 - x0) * (y1 - y0) for x0, x1, y0, y1 in SOLIDS.values())


def gross_area() -> float:
    return sum((x1 - x0) * (y1 - y0) for x0, x1, y0, y1 in RECTANGLES.values())


# --------------------------------------------------------------------------- #
# Point classification (probes, admissibility)
# --------------------------------------------------------------------------- #
def inside_solid(x, y, margin: float = 0.0):
    """True where (x, y) lies in a partition, bed or bathroom wall (enlarged by margin)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    m = np.zeros(np.broadcast(x, y).shape, dtype=bool)
    for x0, x1, y0, y1 in SOLIDS.values():
        m |= (x > x0 - margin) & (x < x1 + margin) & (y > y0 - margin) & (y < y1 + margin)
    return m


def inside_union(x, y, margin: float = 0.0):
    """True where (x, y) is strictly inside main room UNION annex (before subtracting solids).

    ``margin`` > 0 shrinks the domain away from its exterior boundary.  The
    interface x = 9, 0 < y < 2 is interior (the two rectangles are joined there).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    c = margin
    main = (x > ROOM_X_MIN + c) & (y > ROOM_Y_MIN + c) & (y < ROOM_Y_MAX - c) & (
        (x < ROOM_X_MAX - c) | ((x <= ROOM_X_MAX) & (y < ANNEX_Y[1] - c)))   # joined to the annex below y = 2
    annex = (x > ANNEX_X[0]) & (x < ANNEX_X[1] - c) & (y > ANNEX_Y[0] + c) & (y < ANNEX_Y[1] - c)
    return main | annex


def inside_fluid_domain(x, y, margin: float = 0.0):
    """Strictly inside the fluid domain; margin > 0 keeps points that far from every boundary
    (axis-aligned distance; adequate for this rectilinear geometry)."""
    return inside_union(x, y, margin) & ~inside_solid(x, y, margin)


def distance_to_boundary(x, y):
    """Euclidean distance from (x, y) to the nearest boundary segment (walls, solids, openings)."""
    x, y = np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float))
    P = np.stack([x, y], -1)[..., None, :]
    segs = all_boundary_segments()
    A = np.array([s[0] for s in segs], dtype=float)
    B = np.array([s[1] for s in segs], dtype=float)
    AB = B - A
    t = np.clip(np.sum((P - A) * AB, -1) / np.sum(AB * AB, -1), 0.0, 1.0)
    return np.min(np.linalg.norm(P - (A + t[..., None] * AB), axis=-1), axis=-1)


def admissible_source_mask(x, y, clearance: float = 0.0):
    """Analytic admissibility of candidate source centres (future inverse problem).

    True iff (x, y) is strictly inside the fluid domain (never inside a bed,
    partition or bathroom wall, never outside main room UNION annex) and at least
    ``clearance`` [m] from every boundary segment.  For the mesh-level check see
    ``mesh_data_config.point_in_fluid_mesh``.
    """
    ok = inside_fluid_domain(x, y)
    if clearance > 0.0:
        ok &= distance_to_boundary(x, y) >= clearance
    return ok


def passages():
    """Clear passages that must be fully fluid-covered (name -> (x0, x1, y0, y1))."""
    out = {}
    for bay, b in BEDS.items():
        (bx0, bx1), (by0, by1) = b["x"], b["y"]
        bay_x0, bay_x1 = BAYS[bay]["x"]
        out[f"bed_{bay}_left"] = (bay_x0, bx0, by0, by1)
        out[f"bed_{bay}_right"] = (bx1, bay_x1, by0, by1)
        out[f"bed_{bay}_head"] = (bx0, bx1, by1, ROOM_Y_MAX)
        out[f"bed_{bay}_foot"] = (bx0, bx1, PARTITION_BOTTOM_Y, by0)
    out["bathroom_doorway"] = (*BATH_WALL_X, *BATH_DOORWAY_Y)
    out["bathroom_interior"] = (BATH_WALL_X[1], ANNEX_X[1], *ANNEX_Y)
    out["aisle"] = (ROOM_X_MIN, BATH_WALL_X[0], ROOM_Y_MIN, PARTITION_BOTTOM_Y)
    out["aisle_to_bay3_beside_bathroom"] = (BEDS[3]["x"][1], ROOM_X_MAX, ANNEX_Y[1], PARTITION_BOTTOM_Y)
    return out


# Clear width of a passage when it is not min(width, height) of its rectangle.
PASSAGE_CLEAR_WIDTH = {"bathroom_doorway": BATH_DOORWAY_Y[1] - BATH_DOORWAY_Y[0]}

# Regions OUTSIDE the fluid domain that must contain no triangles.
EXTERIOR_REGIONS = {"beyond_annex_above_bathroom": (ROOM_X_MAX, ANNEX_X[1], ANNEX_Y[1], ROOM_Y_MAX)}


def geometry_dict() -> dict:
    points, names = boundary_polygon()
    return {
        "version": OUTPUT_VERSION,
        "units": "m",
        "union_rectangles": {k: list(v) for k, v in RECTANGLES.items()},
        "solids": {k: list(v) for k, v in SOLIDS.items()},
        "bathroom_doorway_y": list(BATH_DOORWAY_Y),
        "supply_x_ranges_y6": [list(r) for r in VENT_X_RANGES],
        "door_outlet_x_range_y0": list(DOOR_X_RANGE),
        "exhaust_outlet_y_range_x11": list(EXHAUST_Y_RANGE),
        "boundary_polygon": [list(p) for p in points],
        "boundary_segment_names": names,
        "hole_loops": {nm: [list(p) for p in pts] for nm, pts in bed_hole_loops()},
        "tags": {"fluid": FLUID_TAG, **BOUNDARY_TAGS},
        "bays": {str(k): v for k, v in BAYS.items()},
        "fluid_area_m2": fluid_area(),
    }


def geometry_hash() -> str:
    """SHA-256 of the canonical geometry description (tags included)."""
    return hashlib.sha256(json.dumps(geometry_dict(), sort_keys=True).encode()).hexdigest()
