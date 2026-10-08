"""Generate an L-shaped city-layout mesh with internal building holes."""

import math

import gmsh

from mesh_data_config import (
    BOTTOM_PHYSICAL_ID,
    BUILDING_1_CENTER,
    BUILDING_1_PHYSICAL_ID,
    BUILDING_1_RADIUS,
    BUILDING_2_BOUNDS,
    BUILDING_2_PHYSICAL_ID,
    BUILDING_3_BOUNDS,
    BUILDING_3_PHYSICAL_ID,
    BUILDING_4_CENTER,
    BUILDING_4_PHYSICAL_ID,
    BUILDING_4_RADIUS,
    BUILDING_5_BOUNDS,
    BUILDING_5_PHYSICAL_ID,
    DOMAIN_PHYSICAL_ID,
    INNER_CORNER_PHYSICAL_ID,
    LEFT_PHYSICAL_ID,
    MESH_DIR,
    MESH_FILE,
    MESH_NAME,
    MESH_SIZE,
    OUTER_HEIGHT,
    OUTER_WIDTH,
    REMOVED_X_MIN,
    REMOVED_Y_MIN,
    RIGHT_PHYSICAL_ID,
    TOP_PHYSICAL_ID,
    VTK_FILE,
)


def _add_rectangle(bounds):
    xmin, xmax, ymin, ymax = bounds
    return gmsh.model.occ.addRectangle(xmin, ymin, 0.0, xmax - xmin, ymax - ymin)


def _classify_boundary_curves(surface_tag):
    boundaries = {
        "bottom": [],
        "left": [],
        "right": [],
        "top": [],
        "inner_corner": [],
        "building_1": [],
        "building_2": [],
        "building_3": [],
        "building_4": [],
        "building_5": [],
    }
    tol = 1.0e-6 * max(OUTER_WIDTH, OUTER_HEIGHT, 1.0)

    for dim, tag in gmsh.model.getBoundary(
        [(2, surface_tag)], oriented=False, recursive=False
    ):
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)
        if abs(ymin) < tol and abs(ymax) < tol:
            boundaries["bottom"].append(tag)
        elif abs(xmin) < tol and abs(xmax) < tol:
            boundaries["left"].append(tag)
        elif abs(xmin - OUTER_WIDTH) < tol and abs(xmax - OUTER_WIDTH) < tol:
            boundaries["right"].append(tag)
        elif abs(ymin - OUTER_HEIGHT) < tol and abs(ymax - OUTER_HEIGHT) < tol:
            boundaries["top"].append(tag)
        elif (
            abs(xmin - REMOVED_X_MIN) < tol
            and abs(xmax - REMOVED_X_MIN) < tol
        ) or (
            abs(ymin - REMOVED_Y_MIN) < tol
            and abs(ymax - REMOVED_Y_MIN) < tol
        ):
            boundaries["inner_corner"].append(tag)
        elif math.hypot(xmid - BUILDING_1_CENTER[0], ymid - BUILDING_1_CENTER[1]) <= (
            BUILDING_1_RADIUS + tol
        ):
            boundaries["building_1"].append(tag)
        elif _point_in_rect_bbox(xmid, ymid, BUILDING_2_BOUNDS, tol):
            boundaries["building_2"].append(tag)
        elif _point_in_rect_bbox(xmid, ymid, BUILDING_3_BOUNDS, tol):
            boundaries["building_3"].append(tag)
        elif math.hypot(xmid - BUILDING_4_CENTER[0], ymid - BUILDING_4_CENTER[1]) <= (
            BUILDING_4_RADIUS + tol
        ):
            boundaries["building_4"].append(tag)
        elif _point_in_rect_bbox(xmid, ymid, BUILDING_5_BOUNDS, tol):
            boundaries["building_5"].append(tag)
        else:
            raise RuntimeError(
                "Could not classify boundary curve "
                f"{tag} with bounding box {(xmin, ymin, xmax, ymax)}"
            )

    return boundaries


def _point_in_rect_bbox(x, y, bounds, tol):
    xmin, xmax, ymin, ymax = bounds
    return xmin - tol <= x <= xmax + tol and ymin - tol <= y <= ymax + tol


def _add_physical_group(dim, entity_tags, physical_id, name):
    if not entity_tags:
        raise RuntimeError(f"No entities found for physical group '{name}'")
    gmsh.model.addPhysicalGroup(dim, entity_tags, tag=physical_id)
    gmsh.model.setPhysicalName(dim, physical_id, name)


def main() -> None:
    MESH_DIR.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add(MESH_NAME)

        points = [
            (0.0, 0.0, 0.0),
            (OUTER_WIDTH, 0.0, 0.0),
            (OUTER_WIDTH, REMOVED_Y_MIN, 0.0),
            (REMOVED_X_MIN, REMOVED_Y_MIN, 0.0),
            (REMOVED_X_MIN, OUTER_HEIGHT, 0.0),
            (0.0, OUTER_HEIGHT, 0.0),
        ]
        point_tags = [gmsh.model.occ.addPoint(*point) for point in points]
        line_tags = [
            gmsh.model.occ.addLine(point_tags[index], point_tags[(index + 1) % 6])
            for index in range(6)
        ]
        outer_loop = gmsh.model.occ.addCurveLoop(line_tags)
        city_surface = gmsh.model.occ.addPlaneSurface([outer_loop])

        building_surfaces = [
            gmsh.model.occ.addDisk(
                BUILDING_1_CENTER[0],
                BUILDING_1_CENTER[1],
                0.0,
                BUILDING_1_RADIUS,
                BUILDING_1_RADIUS,
            ),
            _add_rectangle(BUILDING_2_BOUNDS),
            _add_rectangle(BUILDING_3_BOUNDS),
            gmsh.model.occ.addDisk(
                BUILDING_4_CENTER[0],
                BUILDING_4_CENTER[1],
                0.0,
                BUILDING_4_RADIUS,
                BUILDING_4_RADIUS,
            ),
            _add_rectangle(BUILDING_5_BOUNDS),
        ]
        fluid_surfaces, _ = gmsh.model.occ.cut(
            [(2, city_surface)],
            [(2, tag) for tag in building_surfaces],
            removeObject=True,
            removeTool=True,
        )
        gmsh.model.occ.synchronize()

        if len(fluid_surfaces) != 1:
            raise RuntimeError(
                f"Expected one fluid surface after subtraction, found {fluid_surfaces}"
            )
        fluid_surface = fluid_surfaces[0][1]

        boundaries = _classify_boundary_curves(fluid_surface)
        _add_physical_group(2, [fluid_surface], DOMAIN_PHYSICAL_ID, "domain")
        _add_physical_group(1, boundaries["bottom"], BOTTOM_PHYSICAL_ID, "bottom")
        _add_physical_group(1, boundaries["left"], LEFT_PHYSICAL_ID, "left")
        _add_physical_group(1, boundaries["right"], RIGHT_PHYSICAL_ID, "right")
        _add_physical_group(1, boundaries["top"], TOP_PHYSICAL_ID, "top")
        _add_physical_group(
            1,
            boundaries["inner_corner"],
            INNER_CORNER_PHYSICAL_ID,
            "inner_corner",
        )
        _add_physical_group(
            1, boundaries["building_1"], BUILDING_1_PHYSICAL_ID, "building_1"
        )
        _add_physical_group(
            1, boundaries["building_2"], BUILDING_2_PHYSICAL_ID, "building_2"
        )
        _add_physical_group(
            1, boundaries["building_3"], BUILDING_3_PHYSICAL_ID, "building_3"
        )
        _add_physical_group(
            1, boundaries["building_4"], BUILDING_4_PHYSICAL_ID, "building_4"
        )
        _add_physical_group(
            1, boundaries["building_5"], BUILDING_5_PHYSICAL_ID, "building_5"
        )

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", MESH_SIZE)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", MESH_SIZE)
        gmsh.model.mesh.generate(2)

        gmsh.write(str(MESH_FILE))
        gmsh.write(str(VTK_FILE))

        fluid_area = gmsh.model.occ.getMass(2, fluid_surface)
        expected_area = (
            OUTER_WIDTH * OUTER_HEIGHT
            - (OUTER_WIDTH - REMOVED_X_MIN) * (OUTER_HEIGHT - REMOVED_Y_MIN)
            - math.pi * BUILDING_1_RADIUS**2
            - _rect_area(BUILDING_2_BOUNDS)
            - _rect_area(BUILDING_3_BOUNDS)
            - math.pi * BUILDING_4_RADIUS**2
            - _rect_area(BUILDING_5_BOUNDS)
        )
        if not math.isclose(
            fluid_area, expected_area, rel_tol=1.0e-8, abs_tol=1.0e-10
        ):
            raise RuntimeError(
                f"Fluid area {fluid_area} does not match expected area {expected_area}"
            )
    finally:
        gmsh.finalize()

    print(f"Mesh written to: {MESH_FILE}")
    print(f"ParaView mesh written to: {VTK_FILE}")
    print("Physical markers:")
    print(f"  {BOTTOM_PHYSICAL_ID} = bottom street boundary")
    print(f"  {LEFT_PHYSICAL_ID} = left street boundary")
    print(f"  {RIGHT_PHYSICAL_ID} = right/outflow or observation")
    print(f"  {TOP_PHYSICAL_ID} = upper-left street boundary")
    print(f"  {INNER_CORNER_PHYSICAL_ID} = re-entrant L-corner boundary")
    print(f"  {BUILDING_1_PHYSICAL_ID} = circular building near junction")
    print(f"  {BUILDING_2_PHYSICAL_ID} = lower-corridor rectangular building")
    print(f"  {BUILDING_3_PHYSICAL_ID} = right-corridor rectangular building")
    print(f"  {BUILDING_4_PHYSICAL_ID} = upper-left circular building")
    print(f"  {BUILDING_5_PHYSICAL_ID} = inner-corner rectangular building")
    print(f"  {DOMAIN_PHYSICAL_ID} = fluid street domain")


def _rect_area(bounds):
    xmin, xmax, ymin, ymax = bounds
    return (xmax - xmin) * (ymax - ymin)


if __name__ == "__main__":
    main()
