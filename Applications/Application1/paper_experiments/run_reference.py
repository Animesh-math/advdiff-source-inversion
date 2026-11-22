"""Re-run one Application1 paper reference run from ``reference_runs.json``.

    python paper_experiments/run_reference.py --list
    python paper_experiments/run_reference.py restart_A_L1_boundary --dry-run
    python paper_experiments/run_reference.py restart_A_L1_boundary [--out-dir DIR]

Run from ``Applications/Application1`` with the inverse environment, after
``python ../../tools/install_supplied_assets.py`` (or after solving the
Navier-Stokes field yourself).  The run goes through the same solve-counting
wrapper the original batches used (``test scripts/sensor_layout_run_case.py``),
so ``<out-dir>/cases/<run_id>/run_instrumentation.json`` holds forward/adjoint
counts per phase that can be compared with ``total_counted_solves`` in
``reference_runs.json``.  Driver output: ``<out-dir>/cases/<run_id>/full/``.

Default ``--out-dir``: ``output/paper_reference_runs``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP1 = HERE.parent
WRAPPER = APP1 / "test scripts" / "sensor_layout_run_case.py"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--list", action="store_true", help="list run ids and exit")
    ap.add_argument("--out-dir", default=str(APP1 / "output" / "paper_reference_runs"))
    ap.add_argument("--dry-run", action="store_true", help="print the command only")
    args = ap.parse_args()

    runs = {r["run_id"]: r for r in json.loads((HERE / "reference_runs.json").read_text())["runs"]}
    if args.list or not args.run_id:
        for rid, r in runs.items():
            print(f"{rid:28s} {r['paper_role']}")
        return 0
    if args.run_id not in runs:
        raise SystemExit(f"unknown run id {args.run_id!r}; use --list")
    run = runs[args.run_id]
    driver_argv = list(run["driver_argv"])
    # --case-name only names the output folder; use the run id so the driver
    # output and the wrapper's instrumentation share cases/<run_id>/.
    if "--case-name" in driver_argv:
        driver_argv[driver_argv.index("--case-name") + 1] = args.run_id
    else:
        driver_argv += ["--case-name", args.run_id]
    cmd = [sys.executable, str(WRAPPER), "--exp-dir", str(Path(args.out_dir).resolve()),
           "--case-id", args.run_id, "--"] + driver_argv
    print(f"cd {shlex.quote(str(APP1))}")
    print(shlex.join(cmd))
    if args.dry_run:
        return 0
    os.environ.setdefault("MPLBACKEND", "Agg")
    return subprocess.call(cmd, cwd=APP1)


if __name__ == "__main__":
    raise SystemExit(main())
