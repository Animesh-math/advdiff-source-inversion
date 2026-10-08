"""Generate a rectangular test mesh with an interior circular obstacle."""

import math

import gmsh

from mesh_data_config import (
    BOTTOM_PHYSICAL_ID,
    CIRCLE_CENTER_X,
    CIRCLE_CENTER_Y,
    CIRCLE_RADIUS,
    DOMAIN_PHYSICAL_ID,
    LEFT_PHYSICAL_ID,
    MESH_DIR,
    MESH_FILE,
    MESH_NAME,
    MESH_SIZE,
    OBSTACLE_PHYSICAL_ID,
    RECTANGLE_HEIGHT,
    RECTANGLE_WIDTH,
    RIGHT_PHYSICAL_ID,
    TOP_PHYSICAL_ID,
    VTK_FILE,
)


def _classify_boundary_curves(surface_tag):
    boundaries = {
        "left": [],
        "right": [],
        "top": [],
        "bottom": [],
        "obstacle": [],
    }
    tol = 1.0e-6 * max(RECTANGLE_WIDTH, RECTANGLE_HEIGHT, 1.0)

    for dim, tag in gmsh.model.getBoundary(
        [(2, surface_tag)], oriented=False, recursive=False
    ):
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
        if abs(xmin) < tol and abs(xmax) < tol:
            boundaries["left"].append(tag)
        elif (
            abs(xmin - RECTANGLE_WIDTH) < tol
            and abs(xmax - RECTANGLE_WIDTH) < tol
        ):
            boundaries["right"].append(tag)
        elif abs(ymin) < tol and abs(ymax) < tol:
            boundaries["bottom"].append(tag)
        elif (
            abs(ymin - RECTANGLE_HEIGHT) < tol
            and abs(ymax - RECTANGLE_HEIGHT) < tol
        ):
            boundaries["top"].append(tag)
        else:
            boundaries["obstacle"].append(tag)

    return boundaries


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

        rectangle = gmsh.model.occ.addRectangle(
            0.0, 0.0, 0.0, RECTANGLE_WIDTH, RECTANGLE_HEIGHT
        )
        disk = gmsh.model.occ.addDisk(
            CIRCLE_CENTER_X,
            CIRCLE_CENTER_Y,
            0.0,
            CIRCLE_RADIUS,
            CIRCLE_RADIUS,
        )
        fluid_surfaces, _ = gmsh.model.occ.cut(
            [(2, rectangle)],
            [(2, disk)],
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
        _add_physical_group(1, boundaries["left"], LEFT_PHYSICAL_ID, "left")
        _add_physical_group(1, boundaries["right"], RIGHT_PHYSICAL_ID, "right")
        _add_physical_group(1, boundaries["top"], TOP_PHYSICAL_ID, "top")
        _add_physical_group(1, boundaries["bottom"], BOTTOM_PHYSICAL_ID, "bottom")
        _add_physical_group(
            1, boundaries["obstacle"], OBSTACLE_PHYSICAL_ID, "obstacle"
        )

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", MESH_SIZE)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", MESH_SIZE)
        gmsh.model.mesh.generate(2)

        gmsh.write(str(MESH_FILE))
        gmsh.write(str(VTK_FILE))

        fluid_area = gmsh.model.occ.getMass(2, fluid_surface)
        expected_area = (
            RECTANGLE_WIDTH * RECTANGLE_HEIGHT - math.pi * CIRCLE_RADIUS**2
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
    print(f"  {LEFT_PHYSICAL_ID} = left/inflow")
    print(f"  {RIGHT_PHYSICAL_ID} = right/outflow or observation")
    print(f"  {TOP_PHYSICAL_ID} = top wall")
    print(f"  {BOTTOM_PHYSICAL_ID} = bottom wall")
    print(f"  {OBSTACLE_PHYSICAL_ID} = circular obstacle")
    print(f"  {DOMAIN_PHYSICAL_ID} = fluid domain")


if __name__ == "__main__":
    main()
