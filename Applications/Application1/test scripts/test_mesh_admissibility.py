"""Standalone diagnostic for Application1 mesh-based source admissibility.

Adapted from App4's ``test_mesh_admissibility.py`` to the DC Navy Yard
mesh (coordinates in meters).  Because the DC buildings are exact polygonal mesh
holes, admissibility is decided purely by mesh point location: a point is
admissible iff it lies inside some fluid triangle.  Negative tests therefore use
points known to be outside the outer DC boundary and the building-hole centroids
from the building metadata.  (One building has a non-convex footprint whose
centroid lies in open street; the diagnostic reports such cases rather than
failing on them.)
"""

import json
import os as _os
import sys as _sys
from pathlib import Path

import numpy as np

# This test/verification script lives in Application1/"test scripts"/; put the
# parent Application1 directory on sys.path so its local modules import cleanly.
_APP1_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _APP1_DIR not in _sys.path:
    _sys.path.insert(0, _APP1_DIR)

from mesh_admissibility import MeshAdmissibility
from mesh_data_config import BUILDING_METADATA_FILE, build_mesh_data


# The Application1 directory (parent of this "test scripts" folder), so output
# artifacts land under Application1/output as before the move.
APP_DIR = Path(__file__).resolve().parent.parent
SAMPLE_OUTPUT = APP_DIR / "output" / "mesh_admissibility_samples.vtu"

# Known fluid/street points in the DC domain (LOCAL meters), verified
# admissible.  (Same physical points as the GIS versions minus the origin.)
KNOWN_FLUID_POINTS = {
    "upper-left corridor": (57.0278, 561.9172),
    "mid-left corridor": (37.0278, 381.9172),
    "lower-left corridor": (47.0278, 201.9172),
    "central street": (147.0278, 381.9172),
    "south-east corridor": (267.0278, 201.9172),
    "east corridor": (287.0278, 431.9172),
}
# Points outside the outer DC boundary (LOCAL meters), verified inadmissible.
KNOWN_EXTERIOR_POINTS = {
    "far west of domain": (-1032.972, 81.917),
    "far south of domain": (147.028, -918.083),
    "far east of domain": (967.028, 381.917),
}


def export_samples(points: np.ndarray, path: Path) -> None:
    """Write sampled points as an ASCII VTK XML unstructured grid."""
    points_3d = np.column_stack([points, np.zeros(len(points))])
    point_values = " ".join(f"{value:.17g}" for value in points_3d.reshape(-1))
    connectivity = " ".join(str(index) for index in range(len(points)))
    offsets = " ".join(str(index) for index in range(1, len(points) + 1))
    cell_types = " ".join("1" for _ in points)  # VTK_VERTEX
    sample_ids = " ".join(str(index) for index in range(len(points)))
    text = f"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="{len(points)}" NumberOfCells="{len(points)}">
      <PointData Scalars="sample_id">
        <DataArray type="Int32" Name="sample_id" format="ascii">
          {sample_ids}
        </DataArray>
      </PointData>
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          {point_values}
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">
          {connectivity}
        </DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">
          {offsets}
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">
          {cell_types}
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_diagnostic(num_samples: int = 200, min_separation: float = 40.0) -> Path:
    mesh_data = build_mesh_data()
    admissibility = MeshAdmissibility(mesh_data)

    for label, point in KNOWN_FLUID_POINTS.items():
        if not admissibility.is_point_in_domain(*point):
            raise AssertionError(f"{label} {point}: expected admissible, got rejected")
    for label, point in KNOWN_EXTERIOR_POINTS.items():
        if admissibility.is_point_in_domain(*point):
            raise AssertionError(f"{label} {point}: expected rejected, got admissible")

    # Building holes: their centroids should be rejected. Non-convex footprints
    # whose centroid lands in open street are reported, not treated as failures.
    metadata = json.loads(BUILDING_METADATA_FILE.read_text(encoding="utf-8"))
    centroid_in_fluid = []
    for name, info in metadata.items():
        cx, cy = info["centroid"]
        if admissibility.is_point_in_domain(float(cx), float(cy)):
            centroid_in_fluid.append(name)
    rejected = len(metadata) - len(centroid_in_fluid)
    if rejected < len(metadata) - 2:
        raise AssertionError(
            f"Only {rejected}/{len(metadata)} building centroids rejected; "
            "expected nearly all building-hole centroids to be inadmissible"
        )

    # Exercise the brute-force locator even when Matplotlib is installed.
    original_trifinder = admissibility._trifinder
    admissibility._trifinder = None
    fluid_point = next(iter(KNOWN_FLUID_POINTS.values()))
    if not admissibility.is_point_in_domain(*fluid_point):
        raise AssertionError("Brute-force locator rejected a known fluid point")
    exterior_point = next(iter(KNOWN_EXTERIOR_POINTS.values()))
    if admissibility.is_point_in_domain(*exterior_point):
        raise AssertionError("Brute-force locator accepted an exterior point")
    admissibility._trifinder = original_trifinder

    rng = np.random.default_rng(0)
    samples = np.vstack(
        [admissibility.sample_point_in_domain(rng) for _ in range(num_samples)]
    )
    if not all(admissibility.is_point_in_domain(x, y) for x, y in samples):
        raise AssertionError("At least one sampled point lies outside the fluid mesh")

    control = admissibility.sample_control_in_domain(
        num_sources=2,
        amplitude_bounds=(0.0, 2.0),
        rng=rng,
        min_separation=min_separation,
    )
    if not admissibility.validate_sources_in_domain(control, num_sources=2):
        raise AssertionError("Sampled control contains an inadmissible source center")
    centers = control.reshape(2, 3)[:, :2]
    amplitudes = control.reshape(2, 3)[:, 2]
    if np.any(amplitudes < 0.0) or np.any(amplitudes > 2.0):
        raise AssertionError("Sampled control violates amplitude bounds")
    pair_distance = float(np.linalg.norm(centers[0] - centers[1]))
    if pair_distance < min_separation:
        raise AssertionError("Sampled control violates min_separation")

    export_samples(samples, SAMPLE_OUTPUT)
    print(f"mesh nodes: {len(admissibility.coordinates)}")
    print(f"mesh triangles: {len(admissibility.connections)}")
    print(f"point locator: {admissibility.locator_name}")
    if admissibility.trifinder_error is not None:
        print(f"trifinder fallback reason: {admissibility.trifinder_error}")
    print(f"known fluid points accepted: {len(KNOWN_FLUID_POINTS)}")
    print(f"known exterior points rejected: {len(KNOWN_EXTERIOR_POINTS)}")
    print(f"building centroids rejected: {rejected}/{len(metadata)}")
    if centroid_in_fluid:
        print(
            "building centroids in open street (non-convex footprints): "
            f"{centroid_in_fluid}"
        )
    print(f"sampled valid points: {len(samples)}")
    print(f"sampled control separation: {pair_distance:.6f} m (min {min_separation} m)")
    print(f"ParaView sample file: {SAMPLE_OUTPUT}")
    print("mesh admissibility diagnostic: PASS")
    return SAMPLE_OUTPUT


if __name__ == "__main__":
    run_diagnostic()
