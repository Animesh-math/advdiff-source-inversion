#!/usr/bin/env bash
# Overnight discrete-node-sensor source-count inference sweep for
# Applications/Application1. Reusable/reproducible: case definitions live in
# "test scripts/diagnostics/build_source_count_overnight_sweep.py", NOT in this file.
#
# Usage:
#   cd Applications/Application1
#   nohup ./run_source_count_overnight_sweep.sh > output/batch_logs/driver_<ts>.log 2>&1 &
#
# Optional env overrides:
#   PY            python interpreter to use (default: python3 on PATH)
#   CONCURRENCY   max simultaneous cases (default: 4)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY="${PY:-python3}"
CONCURRENCY="${CONCURRENCY:-4}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$SCRIPT_DIR/output/batch_logs/source_count_overnight_${TIMESTAMP}"
mkdir -p "$LOGDIR"

# Thread-oversubscription guards: at most 4 concurrent processes on a
# many-core machine, but each process must stay single-threaded in its BLAS/
# OpenMP layers or 4 concurrent runs would fight over all cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Writable matplotlib cache (ParaView/plot export touches matplotlib in this
# codebase's diagnostics; keep it out of any shared/readonly cache dir).
export MPLCONFIGDIR="$LOGDIR/mplcache"
mkdir -p "$MPLCONFIGDIR"

echo "[$(date -Iseconds)] building sweep manifest -> $LOGDIR"
"$PY" "$SCRIPT_DIR/test scripts/diagnostics/build_source_count_overnight_sweep.py" --out-dir "$LOGDIR"

CASES_TSV="$LOGDIR/cases.tsv"
echo "[$(date -Iseconds)] launching $(wc -l < "$CASES_TSV") cases, concurrency=$CONCURRENCY"
echo "[$(date -Iseconds)] logs under: $LOGDIR"

run_one_case() {
    local case_name="$1" true_control_csv="$2" sensor_mode="$3" sensor_value="$4"
    local mb_csv="$5" max_sources="$6" top_k="$7" polish_max_iter="$8"

    local log_file="$LOGDIR/${case_name}.log"
    local start_ts end_ts status
    start_ts="$(date -Iseconds)"

    IFS=',' read -r -a true_control_arr <<< "$true_control_csv"

    local -a extra_args=()
    if [[ "$sensor_mode" == "ids" ]]; then
        extra_args+=(--sensor-node-ids "$sensor_value")
    else
        extra_args+=(--sensor-coordinates "$sensor_value")
    fi
    if [[ "$mb_csv" != "-" ]]; then
        IFS=',' read -r -a mb_arr <<< "$mb_csv"
        extra_args+=(--measurement-boundaries "${mb_arr[@]}")
    fi

    {
        echo "=== $case_name ==="
        echo "start: $start_ts"
        echo "true_control: ${true_control_arr[*]}"
        echo "sensor_mode: $sensor_mode  sensor_value: $sensor_value"
        echo "max_sources=$max_sources top_k=$top_k polish_max_iter=$polish_max_iter"
        echo "---"
    } > "$log_file"

    set +e
    "$PY" run_optimization.py \
        --mode location_and_intensity \
        --stage full \
        --infer-num-sources \
        --max-sources "$max_sources" \
        --observation-mode nodes \
        --sensor-weights equal \
        --sensor-equal-weight 1.0 \
        "${extra_args[@]}" \
        --feasibility-mode raw \
        --source-count-top-k "$top_k" \
        --source-count-rel-reduction-tol 1e-3 \
        --source-count-abs-reduction-tol 0 \
        --source-count-amp-tol 1e-3 \
        --source-count-discrepancy-tol 0 \
        --source-count-polish-method lbfgsb \
        --source-count-polish-max-iter "$polish_max_iter" \
        --max-iter 200 \
        --max-paraview-frames 50 \
        --case-name "$case_name" \
        --true-control "${true_control_arr[@]}" \
        >> "$log_file" 2>&1
    status=$?
    set -e

    end_ts="$(date -Iseconds)"
    {
        echo "---"
        echo "end: $end_ts"
        echo "exit_status: $status"
    } >> "$log_file"
    echo "[$(date -Iseconds)] finished $case_name (exit $status) -> $log_file"
}
export -f run_one_case
export PY LOGDIR

# Simple, portable bash job pool: launch cases as background jobs and cap
# concurrency at CONCURRENCY using `wait -n`.
running=0
while IFS=$'\t' read -r case_name true_control_csv sensor_mode sensor_value mb_csv max_sources top_k polish_max_iter; do
    [[ -z "$case_name" ]] && continue
    run_one_case "$case_name" "$true_control_csv" "$sensor_mode" "$sensor_value" \
        "$mb_csv" "$max_sources" "$top_k" "$polish_max_iter" &
    running=$((running + 1))
    if (( running >= CONCURRENCY )); then
        wait -n
        running=$((running - 1))
    fi
done < "$CASES_TSV"

wait
echo "[$(date -Iseconds)] all cases finished. Logs: $LOGDIR"
