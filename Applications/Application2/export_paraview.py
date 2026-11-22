"""ParaView export of the TRUE fine-mesh concentration fields of an Application2 dataset.

    PY export_paraview.py [--dataset-dir DIR] [--cases 12 13 23] [--frame-interval 5.0]

Visualization only: re-runs the fine-mesh forward solve of each true
configuration (same operator, time step and source as the data generation),
checks that its sensor samples equal the saved clean observations, and writes

    case_XX/truth/paraview/concentration_case_XX.pvd      time series (open this)
    case_XX/truth/paraview/frames/concentration_NNNN.vtu  fine mesh + point data "concentration"
    case_XX/truth/paraview/release_centres_case_XX.vtu    true release centres (bay, amplitude)
    common/paraview/sensors.vtu                           the 12 sensors (index, coarse node id)

It reads the truth record, so its output lives under truth/ and is not an
inversion input.  The saved observation files are not touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import meshio
import numpy as np

import transport_config as tc
import transport_model as tm


def points3(xy) -> np.ndarray:
    xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
    return np.column_stack([xy, np.zeros(len(xy))])


def write_points(path: Path, xy, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(xy)
    meshio.write(path, meshio.Mesh(points3(xy), [("vertex", np.arange(n).reshape(-1, 1))], point_data=data))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, default=tm.DATASET_DIR)
    ap.add_argument("--cases", nargs="+", default=list(tc.CASES), choices=list(tc.CASES))
    ap.add_argument("--frame-interval", type=float, default=5.0, help="seconds between exported frames")
    args = ap.parse_args()
    out = args.dataset_dir.resolve()

    sensors = json.loads((out / "common" / "sensors.json").read_text())
    write_points(out / "common" / "paraview" / "sensors.vtu", sensors["coordinates"], {
        "sensor_index": np.arange(sensors["num_sensors"], dtype=np.int32),
        "coarse_node_id": np.asarray(sensors["node_ids"], dtype=np.int32)})

    md = tm.build_transport_mesh_data("fine")
    op = tm.SparseTransport(md, tm.build_pde("fine", md))
    P, _ = tm.interpolation_matrix(md, sensors["coordinates"])
    pts, cells = points3(md["coord"]), [("triangle", md["connections"])]
    dt, nsteps = tc.DT["fine"], tc.NUM_STEPS["fine"]
    every = max(1, int(round(args.frame_interval / dt)))
    steps = sorted(set(range(0, nsteps + 1, every)) | {nsteps})

    for case in args.cases:
        cdir = out / tc.case_dir_name(case)
        truth = json.loads((cdir / "truth" / "truth_record.json").read_text())
        z = np.asarray(truth["true_control"], dtype=np.float64)
        samples, snaps = op.solve(op.load_vector(tm.gaussian_source(z, tc.SIGMA2)), sample_every=tc.FINE_SUBSTEPS,
                                  sampler=P, snapshot_steps=steps)
        clean = np.load(cdir / "data" / "observations_clean.npz")["observations"]
        diff = float(np.max(np.abs(samples - clean)))
        if diff > 1e-12:
            raise RuntimeError(f"case {case}: re-solved sensor traces differ from the saved clean data by {diff:.2e}")
        pdir = cdir / "truth" / "paraview"
        (pdir / "frames").mkdir(parents=True, exist_ok=True)
        entries = []
        for i, s in enumerate(steps):
            name = f"frames/concentration_{i:04d}.vtu"
            meshio.write(pdir / name, meshio.Mesh(pts, cells, point_data={"concentration": snaps[s]}))
            entries.append((s * dt, name))
        lines = ['<?xml version="1.0"?>', '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
                 "  <Collection>"]
        lines += [f'    <DataSet timestep="{t:.6f}" group="" part="0" file="{name}"/>' for t, name in entries]
        lines += ["  </Collection>", "</VTKFile>"]
        (pdir / f"concentration_case_{case}.pvd").write_text("\n".join(lines) + "\n")
        src = truth["sources"]
        write_points(pdir / f"release_centres_case_{case}.vtu", [s["center_xy"] for s in src], {
            "bay": np.asarray([s["bay"] for s in src], dtype=np.int32),
            "amplitude": np.asarray([s["amplitude"] for s in src], dtype=np.float64)})
        print(f"case {case}: {len(steps)} frames (every {every * dt:g} s) -> {pdir}/concentration_case_{case}.pvd  "
              f"[sensor traces match saved clean data, max diff {diff:.1e}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
