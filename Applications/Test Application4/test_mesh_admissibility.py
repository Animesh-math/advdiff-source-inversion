"""Standalone diagnostic for App4 mesh-based source admissibility."""

from pathlib import Path

import numpy as np

from mesh_admissibility import MeshAdmissibility
from mesh_data_config import (
    BUILDING_1_CENTER,
    BUILDING_1_RADIUS,
    BUILDING_2_BOUNDS,
    BUILDING_3_BOUNDS,
    BUILDING_4_CENTER,
    BUILDING_4_RADIUS,
    BUILDING_5_BOUNDS,
    REMOVED_X_MIN,
    REMOVED_Y_MIN,
    build_mesh_data,
)


APP_DIR = Path(__file__).resolve().parent
SAMPLE_OUTPUT = APP_DIR / "output" / "mesh_admissibility_samples.vtu"


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


def _inside_rect(point, bounds):
    xmin, xmax, ymin, ymax = bounds
    x, y = point
    return xmin <= x <= xmax and ymin <= y <= ymax


def _inside_building_formula(point):
    x, y = point
    if np.hypot(x - BUILDING_1_CENTER[0], y - BUILDING_1_CENTER[1]) <= BUILDING_1_RADIUS:
        return True
    if _inside_rect(point, BUILDING_2_BOUNDS):
        return True
    if _inside_rect(point, BUILDING_3_BOUNDS):
        return True
    if np.hypot(x - BUILDING_4_CENTER[0], y - BUILDING_4_CENTER[1]) <= BUILDING_4_RADIUS:
        return True
    if _inside_rect(point, BUILDING_5_BOUNDS):
        return True
    return False


def _inside_removed_top_right(point):
    x, y = point
    return x >= REMOVED_X_MIN and y >= REMOVED_Y_MIN


def run_diagnostic(num_samples: int = 200) -> Path:
    mesh_data = build_mesh_data()
    admissibility = MeshAdmissibility(mesh_data)

    expected = {
        "valid lower horizontal corridor": ((0.20, 0.20), True),
        "valid upper-left corridor": ((0.20, 0.90), True),
        "outside L-shape": ((1.50, 0.25), False),
        "removed top-right rectangle": ((1.10, 0.80), False),
        "building 1 center": (BUILDING_1_CENTER, False),
        "building 2 center": (((0.78 + 0.96) / 2.0, (0.18 + 0.36) / 2.0), False),
        "building 3 center": (((1.12 + 1.24) / 2.0, (0.08 + 0.42) / 2.0), False),
        "building 4 center": (BUILDING_4_CENTER, False),
        "building 5 center": (((0.58 + 0.72) / 2.0, (0.56 + 0.72) / 2.0), False),
    }
    for label, (point, expected_value) in expected.items():
        actual = admissibility.is_point_in_domain(*point)
        if actual != expected_value:
            raise AssertionError(
                f"{label} {point}: expected {expected_value}, got {actual}"
            )

    # Exercise the brute-force locator even when Matplotlib is installed.
    original_trifinder = admissibility._trifinder
    admissibility._trifinder = None
    if not admissibility.is_point_in_domain(0.20, 0.20):
        raise AssertionError("Brute-force locator rejected a known fluid point")
    if admissibility.is_point_in_domain(*BUILDING_1_CENTER):
        raise AssertionError("Brute-force locator accepted building 1 center")
    edge_midpoint = np.mean(admissibility.triangle_vertices[0, :2], axis=0)
    if not admissibility.is_point_in_domain(*edge_midpoint):
        raise AssertionError("Brute-force locator rejected a triangle-edge point")
    admissibility._trifinder = original_trifinder

    rng = np.random.default_rng(0)
    samples = np.vstack(
        [admissibility.sample_point_in_domain(rng) for _ in range(num_samples)]
    )
    if not all(admissibility.is_point_in_domain(x, y) for x, y in samples):
        raise AssertionError("At least one sampled point lies outside the fluid mesh")
    if any(_inside_removed_top_right(point) for point in samples):
        raise AssertionError("At least one sampled point lies in the removed L region")
    if any(_inside_building_formula(point) for point in samples):
        raise AssertionError("At least one sampled point lies inside a building hole")

    control = admissibility.sample_control_in_domain(
        num_sources=5,
        amplitude_bounds=(0.0, 2.0),
        rng=rng,
        min_separation=0.05,
    )
    if not admissibility.validate_sources_in_domain(control, num_sources=5):
        raise AssertionError("Sampled control contains an inadmissible source center")
    centers = control.reshape(5, 3)[:, :2]
    amplitudes = control.reshape(5, 3)[:, 2]
    if np.any(amplitudes < 0.0) or np.any(amplitudes > 2.0):
        raise AssertionError("Sampled control violates amplitude bounds")
    pair_distances = [
        np.linalg.norm(centers[i] - centers[j])
        for i in range(len(centers))
        for j in range(i)
    ]
    if min(pair_distances) < 0.05:
        raise AssertionError("Sampled control violates min_separation")

    export_samples(samples, SAMPLE_OUTPUT)
    print(f"mesh nodes: {len(admissibility.coordinates)}")
    print(f"mesh triangles: {len(admissibility.connections)}")
    print(f"point locator: {admissibility.locator_name}")
    if admissibility.trifinder_error is not None:
        print(f"trifinder fallback reason: {admissibility.trifinder_error}")
    print(f"checked known points: {len(expected)}")
    print(f"sampled valid points: {len(samples)}")
    print("sampled control min separation: " f"{min(pair_distances):.6f}")
    print(f"ParaView sample file: {SAMPLE_OUTPUT}")
    print("mesh admissibility diagnostic: PASS")
    return SAMPLE_OUTPUT


if __name__ == "__main__":
    run_diagnostic()
