#!/usr/bin/env bash
# Six-case sensor-layout comparison (2 source configs x 3 sensor layouts).
#
# Layouts are FROZEN beforehand by
#   "test scripts/sensor_layout_prepare.py" --exp-dir EXP
# which writes EXP/layouts.json (+ sha256) and EXP/cases/<id>/case_config.json.
# This script never edits a layout. Per case it runs the unmodified driver
# through "test scripts/sensor_layout_run_case.py" (solve counting/timing only),
# then "test scripts/sensor_layout_report.py case" refreshes the partial
# summary; at the end "... final" writes summary.{json,csv} and REPORT.md.
#
# Settings: the M=2 production sensitivity preset (the development log
# 2026-07-31 sweep) with node sensors, OSQP joint amplitude fit, L-BFGS-B, and
# NO sigma2 continuation (fixed rho = sigma2 = 400 m^2 throughout).
#
# Runs are SEQUENTIAL, each using the whole machine (same policy as
# run_m3_amplitude_solver_ab.sh). A failing case does not stop the batch.
#
# Launch (detached):
#   setsid nohup ./run_sensor_layout_comparison.sh EXP < /dev/null > EXP/batch_driver.log 2>&1 &

set -uo pipefail

EXP="${1:?usage: run_sensor_layout_comparison.sh <abs experiment dir>}"
APP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$APP/../.." && pwd)"
PY="${PY:-python3}"  # override with PY=/path/to/python
TS="$APP/test scripts"
VEL="$APP/output/navier_stokes/application1_navier_stokes_velocity_nodes.npz"
CASES=(A_L1_boundary A_L2_mixed A_L3_weak B_L1_boundary B_L2_mixed B_L3_weak)

cd "$APP" || exit 2
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$EXP/mplcache"
mkdir -p "$MPLCONFIGDIR"

COMMON=(
    --mode location_and_intensity
    --stage full
    --num-sources 2
    --bound-case mesh_domain
    --init-mode sensitivity
    --num-starts 1
    --random-seed 0
    --sensitivity-score predicted_reduction
    --sensitivity-top-k 24
    --sensitivity-beam-width 3
    --sensitivity-refine-centers
    --sensitivity-refine-max-fev 60
    --sensitivity-coherence-cap 0.3
    --sensitivity-geometry-aware-branches
    --sensitivity-obstacle-aware-diversity
    --sensitivity-refine-pair
    --sensitivity-pair-refine-max-fev 60
    --sensitivity-reselect-second
    --sensitivity-reselect-threshold 0.25
    --sensitivity-reselect-max-passes 2
    --sensitivity-save-plots
    --amplitude-solver osqp
    --post-init-method lbfgsb
    --feasibility-mode raw
    --tfinal 150
    --amplitude-floor 0.1
    --max-iter 150
    --max-paraview-frames 100
    --observation-mode nodes
    --sensor-weights equal
    --sensor-equal-weight 1.0
    --velocity-field-file "$VEL"
)

log() { echo "[$(date -Iseconds)] $*" | tee -a "$EXP/batch_progress.log"; }

log "batch start pid=$$ pgid=$(ps -o pgid= $$ | tr -d ' ') sid=$(ps -o sid= $$ | tr -d ' ') host=$(hostname)"
echo "$$" > "$EXP/batch.pid"
echo "running" > "$EXP/BATCH_STATE"

# ---------------------------------------------------------------- manifest
git -C "$REPO" diff HEAD > "$EXP/git_diff_tracked.patch" 2>/dev/null
git -C "$REPO" status --porcelain > "$EXP/git_status.txt" 2>/dev/null
mkdir -p "$EXP/scripts"
cp "$0" "$TS/sensor_layout_prepare.py" "$TS/sensor_layout_run_case.py" \
   "$TS/sensor_layout_report.py" "$EXP/scripts/"
"$PY" -m pip freeze > "$EXP/pip_freeze.txt" 2>/dev/null
{ lscpu; echo; free -g; echo; uname -a; } > "$EXP/hardware.txt" 2>&1

: > "$EXP/commands.sh"
for c in "${CASES[@]}"; do
    read -r TRUE IDS < <("$PY" - "$EXP/cases/$c/case_config.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(" ".join(repr(float(x)) for x in d["true_control"]), ",".join(map(str, d["sensor_node_ids"])))
EOF
)
    # TRUE has 6 numbers, IDS one comma list; re-split safely
    set -- $TRUE $IDS
    TC=("$1" "$2" "$3" "$4" "$5" "$6"); SID="$7"
    printf '%q ' "$PY" "$TS/sensor_layout_run_case.py" --exp-dir "$EXP" --case-id "$c" -- \
        "${COMMON[@]}" --true-control "${TC[@]}" --sensor-node-ids "$SID" --case-name "$c" \
        >> "$EXP/commands.sh"
    echo >> "$EXP/commands.sh"
done

"$PY" - "$EXP" "$REPO" "${COMMON[@]}" <<'EOF'
import json, os, platform, subprocess, sys, time
exp, repo, common = sys.argv[1], sys.argv[2], sys.argv[3:]
def sh(*a):
    try: return subprocess.run(a, capture_output=True, text=True, cwd=repo).stdout.strip()
    except Exception as e: return f"unavailable: {e!r}"
lay = json.load(open(os.path.join(exp, "layouts.json")))
vers = {}
for m in ("numpy", "scipy", "osqp", "matplotlib", "meshio"):
    try:
        mod = __import__(m); vers[m] = getattr(mod, "__version__", "?")
    except Exception as e: vers[m] = f"unavailable: {e!r}"
man = {
  "experiment": "Application1 sensor-layout comparison: 2 source configs x 3 layouts",
  "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
  "experiment_dir": exp,
  "case_ids": [c["case_id"] for c in lay["cases"]],
  "cases": [{k: c[k] for k in ("case_id", "source_config", "layout", "layout_description",
                               "true_control", "sensor_node_ids", "layout_figure")}
            for c in lay["cases"]],
  "source_configs": lay["source_configs"],
  "layout_info": lay["layout_info"],
  "layouts_sha256": open(os.path.join(exp, "layouts.sha256")).read().split()[0],
  "commands_file": os.path.join(exp, "commands.sh"),
  "commands": [l for l in open(os.path.join(exp, "commands.sh")).read().splitlines() if l.strip()],
  "common_driver_args": common,
  "physical_settings": {
    "source model": "f = sum_i A_i exp(-|x-c_i|^2 / sigma2)  (repo convention: sigma2 == rho)",
    "rho = sigma2 (fixed, all stages; no continuation)": "400 m^2",
    "diffusivity k": "10 m^2/s", "dt": "0.1 s", "tfinal": "150 s",
    "eta1": "100", "lambda_reg": "0", "objective time rule": "exclude_first",
    "observation": "discrete node sensors, 25 per layout, equal weights W = I (w = 1.0)",
    "noise": "none (noiseless synthetic data, generated per layout by the same forward model)",
    "amplitude bounds": "[0.1, 2.0] (amplitude floor 0.1)",
    "center bounds": "mesh bounding box + fluid-mesh admissibility (raw feasibility mode)",
    "velocity": "nodal Navier-Stokes field output/navier_stokes/application1_navier_stokes_velocity_nodes.npz",
    "dirichlet": "homogeneous Dirichlet on 'left' (inflow) nodes only; no sensor there",
    "initialization": "adjoint sensitivity beam (predicted_reduction, top-K 24, beam 3) + refinement, OSQP joint amplitude fit",
    "local optimizer": "L-BFGS-B, max-iter 150, ftol 1e-14, gtol 1e-9, maxls 20 (driver constants)",
    "random seed": "0 (recorded; not consumed by the sensitivity path per docs)",
    "measurement-boundaries": "not passed (only labels/plot context in nodes+equal mode)",
  },
  "git": {"commit": sh("git", "rev-parse", "HEAD"), "branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
          "status_porcelain": sh("git", "status", "--porcelain"),
          "tracked_diff_file": os.path.join(exp, "git_diff_tracked.patch"),
          "note": "experiment scripts are untracked new files; copies in scripts/"},
  "environment": {"hostname": platform.node(), "python": sys.executable,
                  "python_version": sys.version, "packages": vers,
                  "pip_freeze_file": os.path.join(exp, "pip_freeze.txt"),
                  "nproc": os.cpu_count(),
                  "thread_env": {k: os.environ.get(k) for k in (
                      "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                      "NUMEXPR_NUM_THREADS", "PYTHONUNBUFFERED")},
                  "thread_policy": "threads unset -> BLAS/OpenMP use all cores; cases run sequentially",
                  "hardware_file": os.path.join(exp, "hardware.txt")},
  "launch": {"mechanism": "setsid nohup, stdin </dev/null, output to files (no batch scheduler on host)",
             "batch_pid": os.getppid()},
}
json.dump(man, open(os.path.join(exp, "manifest.json"), "w"), indent=2)
print("manifest written")
EOF

# ---------------------------------------------------------------- signals
log "observation-signal diagnostics (6 forward solves, separate from runs)"
"$PY" "$TS/sensor_layout_report.py" signals --exp-dir "$EXP" > "$EXP/observation_signals.log" 2>&1
log "signals exit=$?"

# ---------------------------------------------------------------- cases
mapfile -t CMDS < "$EXP/commands.sh"
for i in "${!CASES[@]}"; do
    c="${CASES[$i]}"
    cdir="$EXP/cases/$c"
    runlog="$cdir/run.log"
    start_iso="$(date -Iseconds)"; start_s="$(date +%s)"
    printf '{"case_id": "%s", "state": "running", "start_time": "%s", "log": "%s"}\n' \
        "$c" "$start_iso" "$runlog" > "$cdir/batch_status.json"
    log "START $c"
    { echo "=== $c ==="; echo "start: $start_iso"; echo "command: ${CMDS[$i]}"; echo "---"; } > "$runlog"
    eval "${CMDS[$i]}" >> "$runlog" 2>&1 < /dev/null
    status=$?
    end_iso="$(date -Iseconds)"; end_s="$(date +%s)"
    { echo "---"; echo "end: $end_iso"; echo "exit_status: $status"; echo "wall_seconds: $((end_s - start_s))"; } >> "$runlog"
    printf '{"case_id": "%s", "state": "finished", "start_time": "%s", "end_time": "%s", "exit_status": %d, "wall_seconds": %d, "log": "%s"}\n' \
        "$c" "$start_iso" "$end_iso" "$status" "$((end_s - start_s))" "$runlog" > "$cdir/batch_status.json"
    log "DONE  $c exit=$status wall=$((end_s - start_s))s"
    "$PY" "$TS/sensor_layout_report.py" case --exp-dir "$EXP" --case-id "$c" >> "$EXP/report_steps.log" 2>&1
    log "case report $c exit=$?"
done

"$PY" "$TS/sensor_layout_report.py" final --exp-dir "$EXP" >> "$EXP/report_steps.log" 2>&1
log "final report exit=$? -> $EXP/REPORT.md"
echo "finished" > "$EXP/BATCH_STATE"
log "ALL DONE"
