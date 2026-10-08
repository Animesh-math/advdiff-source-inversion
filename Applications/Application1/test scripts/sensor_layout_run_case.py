"""Run ONE sensor-layout-comparison case through the unmodified App1 driver.

Thin instrumentation wrapper around ``run_optimization.main()``:

* redirects the driver's output root to ``<exp>/cases`` so the run lands in
  ``<exp>/cases/<case_id>/full`` (``--case-name`` = case id);
* counts EVERY ``gt.run_forward`` / ``gt.rk4_discrete_source_adjoint`` call
  (actual PDE solves; cached objective/gradient responses never reach these)
  and splits them by phase: synthetic data + setup (before the initializer),
  initializer, local optimization (the driver's own ``pio.solve_counter``
  block), and the remaining driver overhead (forward smoke/export solves,
  final objective/gradient/state);
* wall-clock times each phase.

No solver, objective, initializer or optimizer code is changed: the wrappers
only count and time calls and delegate to the originals.  Results are written
to ``<exp>/cases/<case_id>/run_instrumentation.json`` even on failure.

Usage:
    PY sensor_layout_run_case.py --exp-dir DIR --case-id ID -- <driver args>
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
import traceback
from pathlib import Path

_APP1_DIR = Path(__file__).resolve().parents[1]
if str(_APP1_DIR) not in sys.path:
    sys.path.insert(0, str(_APP1_DIR))


def main() -> int:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: --exp-dir DIR --case-id ID -- <driver args>")
    split = argv.index("--")
    own, driver_args = argv[:split], argv[split + 1:]
    exp_dir = Path(own[own.index("--exp-dir") + 1]).resolve()
    case_id = own[own.index("--case-id") + 1]
    case_dir = exp_dir / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    out_path = case_dir / "run_instrumentation.json"

    t_start = time.time()
    p_start = time.perf_counter()
    import run_optimization as ro  # noqa: E402
    import post_init_optimizers as pio  # noqa: E402
    import sensitivity_initialization as sinit  # noqa: E402

    gt = ro.gt
    ro.OUTPUT_ROOT = exp_dir / "cases"

    counts = {"forward": 0, "adjoint": 0}
    orig_fwd, orig_adj = gt.run_forward, gt.rk4_discrete_source_adjoint

    def counting_forward(*a, **k):
        counts["forward"] += 1
        return orig_fwd(*a, **k)

    def counting_adjoint(*a, **k):
        counts["adjoint"] += 1
        return orig_adj(*a, **k)

    gt.run_forward = counting_forward
    gt.rk4_discrete_source_adjoint = counting_adjoint

    rec: dict = {
        "case_id": case_id,
        "driver_argv": driver_args,
        "output_dir": str(case_dir / "full"),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t_start)),
        "phases": {},
        "local_opt_blocks": [],
        "notes": [
            "solve counts are actual calls to gt.run_forward / "
            "gt.rk4_discrete_source_adjoint (PDE solves); optimizer nfev/njev in "
            "optimization_summary.json post_init count evaluator calls, which may "
            "be served from the evaluator cache without a solve",
            "pre_init solves = synthetic observation generation for the true "
            "control (expected 1 forward solve; no continuation so no other width)",
        ],
    }

    def snap():
        return dict(counts), time.perf_counter()

    marks: dict = {}

    orig_init = sinit.build_sensitivity_initial_control

    def timed_init(*a, **k):
        marks["init_start"] = snap()
        try:
            return orig_init(*a, **k)
        finally:
            marks["init_end"] = snap()

    sinit.build_sensitivity_initial_control = timed_init

    orig_single = ro.run_single_optimization

    def timed_single(*a, **k):
        marks["single_start"] = snap()
        try:
            return orig_single(*a, **k)
        finally:
            marks["single_end"] = snap()

    ro.run_single_optimization = timed_single

    orig_counter = pio.solve_counter

    @contextlib.contextmanager
    def timed_counter(gt_module):
        c0, t0 = snap()
        with orig_counter(gt_module) as inner:
            try:
                yield inner
            finally:
                c1, t1 = snap()
                rec["local_opt_blocks"].append({
                    "seconds": t1 - t0,
                    "forward_solves": c1["forward"] - c0["forward"],
                    "adjoint_solves": c1["adjoint"] - c0["adjoint"],
                })

    pio.solve_counter = timed_counter

    # Record the fully resolved driver arguments (explicit + defaults).
    orig_parse = ro.parse_args

    def recording_parse(*a, **k):
        ns = orig_parse(*a, **k)
        rec["resolved_driver_args"] = {
            key: (val.tolist() if hasattr(val, "tolist") else val)
            for key, val in vars(ns).items()
        }
        (case_dir / "resolved_driver_args.json").write_text(
            json.dumps(rec["resolved_driver_args"], indent=2, default=str) + "\n",
            encoding="utf-8")
        return ns

    ro.parse_args = recording_parse

    status, err = 0, None
    sys.argv = ["run_optimization.py"] + driver_args
    try:
        ro.main()
    except SystemExit as exc:  # argparse / explicit exits
        status = int(exc.code) if isinstance(exc.code, int) else 1
        if status:
            err = f"SystemExit({exc.code})"
    except BaseException as exc:  # noqa: BLE001 - record and report every failure
        status = 1
        err = "".join(traceback.format_exception(exc))
        traceback.print_exc()
    finally:
        p_end = time.perf_counter()
        final = dict(counts)

        def delta(a, b):
            if a not in marks or b not in marks:
                return None
            (ca, ta), (cb, tb) = marks[a], marks[b]
            return {"seconds": tb - ta,
                    "forward_solves": cb["forward"] - ca["forward"],
                    "adjoint_solves": cb["adjoint"] - ca["adjoint"]}

        if "init_start" in marks:
            c, t = marks["init_start"]
            rec["phases"]["pre_init_setup_and_synthetic_data"] = {
                "seconds": t - p_start, "forward_solves": c["forward"],
                "adjoint_solves": c["adjoint"]}
        rec["phases"]["initialization"] = delta("init_start", "init_end")
        rec["phases"]["between_init_and_optimization"] = delta("init_end", "single_start")
        single = delta("single_start", "single_end")
        rec["phases"]["run_single_optimization_total"] = single
        lo = rec["local_opt_blocks"]
        lo_sum = {"seconds": sum(b["seconds"] for b in lo),
                  "forward_solves": sum(b["forward_solves"] for b in lo),
                  "adjoint_solves": sum(b["adjoint_solves"] for b in lo)} if lo else None
        rec["phases"]["local_optimization"] = lo_sum
        if single and lo_sum:
            rec["phases"]["driver_overhead_smoke_export_final_eval"] = {
                k: single[k] - lo_sum[k] for k in single}
        rec["total_counted"] = {"forward_solves": final["forward"],
                                "adjoint_solves": final["adjoint"]}
        rec["total_elapsed_seconds_in_process"] = p_end - p_start
        rec["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        rec["exit_status"] = status
        rec["error"] = err
        out_path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        print(f"[sensor_layout_run_case] instrumentation -> {out_path} (status {status})",
              flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
