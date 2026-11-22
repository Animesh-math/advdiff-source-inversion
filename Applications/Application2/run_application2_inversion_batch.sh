#!/usr/bin/env bash
# Application2 source-identification INVERSION batch: cases 12, 13, 23 x clean, noisy (six runs, sequential).
# Each run: run_inversion.py --stage full with its defaults (sensitivity initializer, OSQP, M = 2 preset,
# room-scale tolerance shim, GN-diagonal-scaled L-BFGS-B, maxcor 30, max-iter 200), coarse mesh, W = I,
# saved fine-mesh observations of the dataset given below. No solver/driver option is changed.
#
#   bash Applications/Application2/run_application2_inversion_batch.sh launch <batch-name>   # detached (setsid nohup)
#   bash Applications/Application2/run_application2_inversion_batch.sh status <batch-name>
#   bash Applications/Application2/run_application2_inversion_batch.sh report <batch-name>   # regenerate report
#
# Output: <dataset>/inversion_batches/<batch-name>/ (refuses to reuse an existing folder):
#   resolved_settings.json   truth-free preflight: every run's resolved arguments, dataset, sensors, W, noise seed
#   RUN_COMMANDS.txt LAUNCH.txt RUN_STATE batch.pid  env/ (git revision, status, diff, untracked sources, checksums)
#   runs/<run>/   run_inversion.py outputs      logs/<run>.log   status/<run>.exit (rc, start, end)
#   REPORT.md report/   generated automatically after all six runs (scores against the truth, post hoc only)
# Runs are sequential; one run failing does not stop the next; each exit status is recorded.
set -uo pipefail

APP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$APP/../.." && pwd)"
DATASET_VERSION="${APP2_INV_DATASET_VERSION:-source_id_v2_wall8_20261116}"
EXPECT_LAYOUT="${APP2_INV_LAYOUT:-wall_8}"
EXPECT_NSENS="${APP2_INV_NUM_SENSORS:-8}"
DATASET="$APP/output/$DATASET_VERSION"
NAME="${2:-}"
[ -n "$NAME" ] || { echo "usage: $0 {launch|status|run|report} <batch-name>" >&2; exit 2; }
B="$DATASET/inversion_batches/$NAME"
PY="${PY:-python3}"  # set PY to the inverse-environment python
THREADS="${APP2_INV_THREADS:-96}"
RUNS=("12 clean" "12 noisy" "13 clean" "13 noisy" "23 clean" "23 noisy")

stamp() { date -Iseconds; }
state() { echo "$1 $(stamp)" > "$B/RUN_STATE"; echo "[$(stamp)] $1" >> "$B/logs/batch.log"; }

capture_env() {
  mkdir -p "$B/env"
  git -C "$REPO" rev-parse HEAD > "$B/env/git_commit.txt" 2>&1
  git -C "$REPO" rev-parse --abbrev-ref HEAD > "$B/env/git_branch.txt" 2>&1
  git -C "$REPO" status --porcelain > "$B/env/git_status.txt" 2>&1
  git -C "$REPO" diff HEAD > "$B/env/git_diff_tracked.patch" 2>&1
  "$PY" -m pip freeze > "$B/env/projenv_pip_freeze.txt" 2>&1 || true
  { hostname; uname -a; echo "nproc $(nproc)"; free -g | head -2; date -Iseconds;
    echo "threads_per_run=$THREADS runs_sequential=${#RUNS[@]}"; } > "$B/env/host.txt"
  ( cd "$REPO" && sha256sum Applications/Application2/*.py Applications/Application2/*.sh \
      Applications/Application1/*.py Src/Objective/gradient_test.py Src/Solvers/*.py \
      Src/Optimizers/*.py ) > "$B/env/sha256_sources.txt" 2>&1
  mkdir -p "$B/env/untracked_application2_sources"
  ( cd "$REPO" && git status --porcelain --untracked-files=all -- Applications/Application2 \
      | awk '$1=="??"{print substr($0,4)}' | grep -E '\.(py|sh)$' \
      | while read -r f; do cp -p "$f" "$B/env/untracked_application2_sources/"; done )
}

run_one() {   # run_one <case> <kind>
  local c="$1" k="$2" id="case_${1}_${2}"
  local start; start=$(stamp)
  "$PY" run_inversion.py --dataset-dir "$DATASET" --case "$c" --data "$k" --stage full \
      --out-dir "$B/runs/$id" > "$B/logs/$id.log" 2>&1
  local rc=$?
  echo "rc=$rc start=$start end=$(stamp)" > "$B/status/$id.exit"
  echo "[$(stamp)] $id exit=$rc" >> "$B/logs/batch.log"
}

run_batch() {
  cd "$APP" || exit 2
  export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
  export PYTHONUNBUFFERED=1 MPLCONFIGDIR="$B/.mplcache"
  mkdir -p "$B/logs" "$B/status" "$B/runs"
  for spec in "${RUNS[@]}"; do
    state "RUNS_RUNNING (sequential; current: $spec)"
    run_one $spec
  done
  local bad; bad=$(grep -L '^rc=0 ' "$B"/status/*.exit 2>/dev/null | xargs -r -n1 basename | sed 's/\.exit$//' | tr '\n' ' ')
  state "REPORT_RUNNING"
  "$PY" report_inversion_batch.py --batch-dir "$B" > "$B/logs/report.log" 2>&1
  local rrc=$?
  if [ -z "$bad" ] && [ $rrc -eq 0 ]; then state "DONE_OK (all runs exit 0; see REPORT.md for outcomes)"
  else state "DONE_WITH_FAILURES runs:[${bad}] report_rc=$rrc"; fi
}

case "$1" in
  launch)
    [ -d "$DATASET" ] || { echo "no dataset $DATASET" >&2; exit 2; }
    [ -e "$B" ] && { echo "refusing: $B exists" >&2; exit 1; }
    mkdir -p "$B/logs"
    cd "$APP" || exit 2
    pre=(); for spec in "${RUNS[@]}"; do pre+=(--run "$spec"); done
    if ! "$PY" preflight_inversion_batch.py --batch-dir "$B" --dataset-dir "$DATASET" \
        --expect-dataset-version "$DATASET_VERSION" --expect-layout "$EXPECT_LAYOUT" \
        --expect-num-sensors "$EXPECT_NSENS" "${pre[@]}" 2>&1 | tee "$B/logs/preflight.log"; then
      echo "PREFLIGHT_FAILED $(stamp)" > "$B/RUN_STATE"; echo "preflight failed; not launched" >&2; exit 1
    fi
    capture_env
    {
      echo "# Application2 inversion batch $NAME ($(stamp)); dataset $DATASET_VERSION; cwd $APP"
      echo "export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS PYTHONUNBUFFERED=1 MPLCONFIGDIR=$B/.mplcache"
      echo "$PY preflight_inversion_batch.py --batch-dir $B --dataset-dir $DATASET --expect-dataset-version $DATASET_VERSION --expect-layout $EXPECT_LAYOUT --expect-num-sensors $EXPECT_NSENS ${pre[*]}"
      for spec in "${RUNS[@]}"; do set -- $spec
        echo "$PY run_inversion.py --dataset-dir $DATASET --case $1 --data $2 --stage full --out-dir $B/runs/case_${1}_${2} > $B/logs/case_${1}_${2}.log 2>&1; echo rc=\$? > $B/status/case_${1}_${2}.exit"
      done
      echo "# runs execute one after another; a non-zero exit does not stop the next run"
      echo "$PY report_inversion_batch.py --batch-dir $B"
    } > "$B/RUN_COMMANDS.txt"
    setsid nohup bash "$APP/run_application2_inversion_batch.sh" run "$NAME" < /dev/null > "$B/logs/batch_driver.log" 2>&1 &
    echo $! > "$B/batch.pid"
    { echo "launched $(stamp) pid $(cat "$B/batch.pid") (setsid session leader; pgid = pid)";
      echo "command: setsid nohup bash $APP/run_application2_inversion_batch.sh run $NAME < /dev/null > $B/logs/batch_driver.log 2>&1 &";
      echo "dataset: $DATASET ($EXPECT_LAYOUT, $EXPECT_NSENS sensors)"; echo "output: $B"; } | tee "$B/LAUNCH.txt"
    ;;
  run)
    run_batch ;;
  report)
    cd "$APP" && MPLCONFIGDIR="$B/.mplcache" "$PY" report_inversion_batch.py --batch-dir "$B" ;;
  status)
    echo "output: $B"
    [ -f "$B/RUN_STATE" ] && echo "state : $(cat "$B/RUN_STATE")"
    if [ -f "$B/batch.pid" ]; then
      pid=$(cat "$B/batch.pid")
      if ps -p "$pid" > /dev/null 2>&1; then echo "batch pid $pid: running ($(pgrep -g "$pid" -f run_inversion.py | wc -l) run_inversion processes)"
      else echo "batch pid $pid: finished"; fi
    fi
    for spec in "${RUNS[@]}"; do set -- $spec; id="case_${1}_${2}"
      if [ -f "$B/status/$id.exit" ]; then s="$(cat "$B/status/$id.exit")"
      elif [ -f "$B/logs/$id.log" ]; then s="running; last: $(grep -E '^(Source level|\[20|At iterate|  \[level)' "$B/logs/$id.log" 2>/dev/null | tail -1 | cut -c1-150)"
      else s="queued"; fi
      echo "  $id: $s"
    done
    [ -f "$B/REPORT.md" ] && echo "report: $B/REPORT.md"
    exit 0
    ;;
  *) echo "usage: $0 {launch|status|run|report} <batch-name>" >&2; exit 2 ;;
esac
