#!/usr/bin/env bash
# Random-start baseline batch: run every line of EXP/commands.tsv (run_id \t cmd),
# PAR at a time, THREADS BLAS/OpenMP threads each. Resumable:
#   * a run with cases/<id>/DONE is skipped (DONE is written for failures too; its
#     first line holds the exit status). To retry a failed run, delete its DONE
#     and re-launch; the old folder is moved to aborted/ first.
#   * a run folder WITHOUT DONE (killed mid-run) is moved to aborted/<id>_<stamp>
#     and the run restarts from its saved start.
# Failures are logged in failures.log and never stop the remaining runs.
# At the end: scripts/report.py writes report/ (CSV/JSON, REPORT.md, plots).
set -uo pipefail
EXP="$1"; PAR="${2:-10}"; THREADS="${3:-4}"
PY="${PY:-python3}"  # inverse-environment python
APP="${APP1_DIR:-$(cd "$EXP/../.." && pwd)}"  # EXP lives in Applications/Application1/output/
cd "$APP" || exit 2
export PYTHONUNBUFFERED=1 MPLCONFIGDIR="$EXP/mplcache"
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
mkdir -p "$MPLCONFIGDIR" "$EXP/logs" "$EXP/aborted"
echo "running PAR=$PAR THREADS=$THREADS pid=$$ $(date -Iseconds)" > "$EXP/BATCH_STATE"
echo "[$(date -Iseconds)] BATCH START pid=$$ PAR=$PAR THREADS=$THREADS" >> "$EXP/batch_progress.log"
one() {
  IFS=$'\t' read -r id cmd <<< "$1"
  cdir="$EXP/cases/$id"; log="$EXP/logs/$id.log"
  if [[ -f "$cdir/DONE" ]]; then
    echo "[$(date -Iseconds)] SKIP  $id ($(head -1 "$cdir/DONE"))" >> "$EXP/batch_progress.log"; return 0
  fi
  if [[ -d "$cdir" ]]; then
    st=$(date +%Y%m%d_%H%M%S); mv "$cdir" "$EXP/aborted/${id}_$st"
    [[ -f "$log" ]] && mv "$log" "$EXP/aborted/${id}_$st.log"
    echo "[$(date -Iseconds)] MOVED incomplete $id -> aborted/${id}_$st" >> "$EXP/batch_progress.log"
  fi
  s=$(date +%s)
  echo "[$(date -Iseconds)] START $id" >> "$EXP/batch_progress.log"
  { echo "command: $cmd"; echo "threads: $OMP_NUM_THREADS"; echo ---; } > "$log"
  eval "$cmd" >> "$log" 2>&1 < /dev/null; rc=$?
  echo "exit_status: $rc wall_seconds: $(( $(date +%s) - s ))" >> "$log"
  if [[ ! -f "$cdir/DONE" ]]; then mkdir -p "$cdir"; printf 'exit_status %s\nwrapper did not finish (crash/kill)\n' "$rc" > "$cdir/DONE"; fi
  [[ $rc -ne 0 ]] && echo "[$(date -Iseconds)] $id exit=$rc (see logs/$id.log)" >> "$EXP/failures.log"
  echo "[$(date -Iseconds)] DONE  $id exit=$rc wall=$(( $(date +%s) - s ))s" >> "$EXP/batch_progress.log"
}
export -f one; export EXP
# Predetermined order: cases A_L1, A_L2, B_L1, B_L2, M5_S4; runs r01..r10.
tr '\n' '\0' < "$EXP/commands.tsv" | xargs -0 -P "$PAR" -I{} bash -c 'one "$@"' _ {}
echo "[$(date -Iseconds)] ALL RUNS FINISHED; generating report" >> "$EXP/batch_progress.log"
"$PY" "$EXP/scripts/report.py" --exp-dir "$EXP" > "$EXP/logs/report.log" 2>&1
echo "[$(date -Iseconds)] REPORT exit=$? (logs/report.log)" >> "$EXP/batch_progress.log"
echo "finished $(date -Iseconds)" > "$EXP/BATCH_STATE"
echo "[$(date -Iseconds)] ALL DONE" >> "$EXP/batch_progress.log"
