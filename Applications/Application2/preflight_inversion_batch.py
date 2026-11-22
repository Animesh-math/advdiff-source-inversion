"""Truth-free preflight for an Application2 inversion batch (no PDE solve, no truth access).

    PY preflight_inversion_batch.py --batch-dir <dir> --dataset-dir <dataset> --expect-dataset-version V \
        --expect-layout wall_8 --expect-num-sensors 8 --run "12 clean" ... -- <extra run_inversion args>

For every run it parses the exact run_inversion.py argument list (resolving all
defaults), loads the configuration and the selected observation file through
run_inversion.load_dataset (sha256-checked), and verifies that the run uses the
expected dataset version, sensor layout and sensor count, the dataset's saved
sensor coordinates (common/sensors.json), W = I, the dataset's eta1, the
room-scale initializer shim and the GN-diagonal L-BFGS-B defaults.  It writes
<batch>/resolved_settings.json and exits non-zero on any failed check.  It never
opens case_XX/truth/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import run_inversion as ri
import transport_config as tc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-dir", type=Path, required=True)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--expect-dataset-version", required=True)
    ap.add_argument("--expect-layout", required=True)
    ap.add_argument("--expect-num-sensors", type=int, required=True)
    ap.add_argument("--run", action="append", required=True, help='"<case> <clean|noisy>"')
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="run_inversion.py args common to all runs, after --")
    a = ap.parse_args(argv)
    extra = [x for x in a.extra if x != "--"]
    dataset_dir = a.dataset_dir.resolve()
    sensors_common = json.loads((dataset_dir / "common" / "sensors.json").read_text())

    failures, runs = [], []

    def check(run, ok, what):
        run["checks"].append({"check": what, "passed": bool(ok)})
        if not ok:
            failures.append(f"{run['run_id']}: {what}")

    for spec in a.run:
        case, kind = spec.split()
        run_id = f"case_{case}_{kind}"
        out_dir = a.batch_dir.resolve() / "runs" / run_id
        argv_run = ["--dataset-dir", str(dataset_dir), "--case", case, "--data", kind, "--stage", "full",
                    "--out-dir", str(out_dir)] + extra
        args = ri.parse_args(argv_run)
        ds = ri.load_dataset(args.dataset_dir, args.case, args.data)
        cfg = ds["config"]
        run = {"run_id": run_id, "case": case, "data": kind, "run_inversion_argv": argv_run,
               "resolved_arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "inputs_read_by_optimization": {"inversion_config": str(ds["config_path"]),
                                               "observations_file": str(ds["path"]),
                                               "observations_file_sha256": ds["file_sha256"],
                                               "observations_sha256": ds["observations_sha256"]},
               "dataset_version": cfg["dataset_version"], "sensor_layout": cfg["sensors"]["layout"],
               "num_sensors": len(cfg["sensors"]["node_ids"]), "sensor_labels": cfg["sensors"]["labels"],
               "sensor_node_ids": cfg["sensors"]["node_ids"], "sensor_coordinates": cfg["sensors"]["coordinates"],
               "sensor_weights": cfg["objective"]["sensor_weights"], "eta1": cfg["objective"]["eta1"],
               "lambda_reg": cfg["objective"]["lambda_reg"], "time_rule": cfg["objective"]["time_rule"],
               "transport": cfg["transport"], "admissible_set": cfg["admissible_set"],
               "num_sources": cfg["num_sources"], "mesh": cfg["mesh"], "velocity": cfg["velocity"],
               "checks": []}
        noise_rec = json.loads((ds["case_dir"] / "data" / "noise_record.json").read_text())
        run["noise"] = {k: noise_rec.get(k) for k in ("seed", "nominal_noise_level", "s_noise_absolute",
                                                      "s_noise_definition", "generator",
                                                      "realized_relative_noise_norm")}
        run["noise"]["applies_to_this_run"] = kind == "noisy"

        check(run, args.dataset_dir.resolve() == dataset_dir, "dataset directory is the expected one")
        check(run, dataset_dir.name == a.expect_dataset_version, "dataset directory name == expected version")
        check(run, cfg["dataset_version"] == a.expect_dataset_version, "config dataset_version == expected")
        check(run, tc.DATASET_VERSION == a.expect_dataset_version, "transport_config.DATASET_VERSION == expected")
        check(run, cfg["sensors"]["layout"] == a.expect_layout, f"sensor layout == {a.expect_layout}")
        check(run, tc.ACTIVE_SENSOR_LAYOUT == a.expect_layout, "transport_config.ACTIVE_SENSOR_LAYOUT == expected")
        check(run, sensors_common["layout"] == a.expect_layout and sensors_common["active_layout"] is True,
              "common/sensors.json is the active expected layout")
        check(run, len(cfg["sensors"]["node_ids"]) == a.expect_num_sensors == sensors_common["num_sensors"],
              f"{a.expect_num_sensors} sensors in config and common/sensors.json")
        check(run, ds["observations"].shape[1] == a.expect_num_sensors,
              f"observation array has {a.expect_num_sensors} columns")
        check(run, np.array_equal(ds["node_ids"], cfg["sensors"]["node_ids"])
              and np.array_equal(ds["node_ids"], sensors_common["node_ids"]),
              "sensor node ids equal in data file, config and common/sensors.json")
        check(run, np.array_equal(np.asarray(ds["sensor_xy"]), np.asarray(cfg["sensors"]["coordinates"]))
              and np.array_equal(np.asarray(cfg["sensors"]["coordinates"]),
                                 np.asarray(sensors_common["coordinates"])),
              "sensor coordinates equal in data file, config and common/sensors.json")
        check(run, cfg.get("contains_truth") is False, "configuration marked truth-free")
        check(run, "truth" not in str(ds["path"]) and "truth" not in str(ds["config_path"]),
              "no optimization input lies under truth/")
        check(run, ds["path"].name == f"observations_{kind}.npz", f"observation file is observations_{kind}.npz")
        check(run, np.all(np.asarray(cfg["objective"]["sensor_weights"]) == 1.0)
              and args.sensor_weight is None and args.sensor_weights_file is None, "W = I (no override)")
        check(run, args.eta1 is None and cfg["objective"]["eta1"] == 1.0, "eta1 = 1 from the dataset")
        check(run, cfg["admissible_set"]["amplitude_bounds"] == list(tc.AMPLITUDE_BOUNDS),
              "amplitude bounds equal transport_config")
        check(run, args.initial_control is None, "sensitivity initialization (no manual start)")
        check(run, not args.init_app1_literal_tolerances, "room-scale initializer tolerance shim enabled")
        check(run, args.amplitude_solver == "osqp", "OSQP joint amplitude solver")
        check(run, args.post_init_scaling == "gn-diag", "GN-diagonal scaling")
        check(run, args.post_init_lbfgsb_maxcor == 30 and args.max_iter == 200, "L-BFGS-B maxcor 30, max-iter 200")
        check(run, not out_dir.exists(), "run output folder does not exist yet")
        run["all_checks_passed"] = all(c["passed"] for c in run["checks"])
        runs.append(run)
        print(f"{run_id}: dataset {cfg['dataset_version']}, layout {cfg['sensors']['layout']}, "
              f"{run['num_sensors']} sensors {cfg['sensors']['labels']}, data {ds['path']}, "
              f"checks {'PASS' if run['all_checks_passed'] else 'FAIL'}")

    record = {"dataset_dir": str(dataset_dir), "expected_dataset_version": a.expect_dataset_version,
              "expected_layout": a.expect_layout, "expected_num_sensors": a.expect_num_sensors,
              "common_extra_arguments": extra, "runs": runs, "all_passed": not failures, "failures": failures,
              "truth_accessed": False,
              "determinism": "no random numbers in the initializer or optimizer; the only seeds are the saved "
                             "noise seeds (recorded per run; the noise is read from disk, not regenerated)"}
    a.batch_dir.mkdir(parents=True, exist_ok=True)
    (a.batch_dir / "resolved_settings.json").write_text(json.dumps(record, indent=2, default=float))
    if failures:
        print("PREFLIGHT FAILED:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    print(f"preflight: all {sum(len(r['checks']) for r in runs)} checks passed for {len(runs)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
