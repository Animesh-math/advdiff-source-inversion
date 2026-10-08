"""Generate a simple test mesh for PDE-constrained optimization experiments."""

from pathlib import Path

import gmsh


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    msh_path = out_dir / "test_domain.msh"
    vtk_path = out_dir / "test_domain.vtk"

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("test_domain")

    width = 1.0
    height = 0.4
    lc = 0.04
    rect = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, width, height)
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(2, [rect], tag=1)
    gmsh.model.setPhysicalName(2, 1, "domain")

    boundary_curves = gmsh.model.getBoundary([(2, rect)], oriented=False, recursive=False)
    left = []
    right = []
    bottom = []
    top = []
    tol = 1.0e-10

    for dim, tag in boundary_curves:
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
        if abs(xmin - 0.0) < tol and abs(xmax - 0.0) < tol:
            left.append(tag)
        elif abs(xmin - width) < tol and abs(xmax - width) < tol:
            right.append(tag)
        elif abs(ymin - 0.0) < tol and abs(ymax - 0.0) < tol:
            bottom.append(tag)
        elif abs(ymin - height) < tol and abs(ymax - height) < tol:
            top.append(tag)

    if left:
        gmsh.model.addPhysicalGroup(1, left, tag=11)
        gmsh.model.setPhysicalName(1, 11, "left")
    if right:
        gmsh.model.addPhysicalGroup(1, right, tag=12)
        gmsh.model.setPhysicalName(1, 12, "right")
    if top:
        gmsh.model.addPhysicalGroup(1, top, tag=13)
        gmsh.model.setPhysicalName(1, 13, "top")
    if bottom:
        gmsh.model.addPhysicalGroup(1, bottom, tag=14)
        gmsh.model.setPhysicalName(1, 14, "bottom")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.model.mesh.generate(2)

    gmsh.write(str(msh_path))
    gmsh.write(str(vtk_path))
    gmsh.finalize()

    print(f"Mesh written to: {msh_path}")
    print(f"Mesh written to: {vtk_path}")


if __name__ == "__main__":
    main()
