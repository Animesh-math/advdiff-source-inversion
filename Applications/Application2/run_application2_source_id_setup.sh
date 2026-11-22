#!/usr/bin/env bash
# Application2 source-identification DATASET pipeline (no inversion):
#   settings-selection record -> observation generation (fine mesh, 3 cases, clean + noisy)
#   -> readiness validation -> report.  Then writes the (NOT executed) inversion commands.
#
#   bash Applications/Application2/run_application2_source_id_setup.sh launch   # detached (survives SSH logout)
#   bash Applications/Application2/run_application2_source_id_setup.sh status   # progress check
#   bash Applications/Application2/run_application2_source_id_setup.sh run      # foreground
#   bash Applications/Application2/run_application2_source_id_setup.sh report   # regenerate the report only
#
# Output: Applications/Application2/output/<DATASET_VERSION>/  (DATASET_VERSION in transport_config.py;
# override the folder with $APP2_SID_OUT).  Refuses to run if any case_XX/ exists there: saved
# observations and noise are never regenerated in place (bump DATASET_VERSION instead).
# The report is generated even if an earlier stage fails; the failure is recorded in RUN_STATE,
# logs/pipeline.log and logs/<stage>.log.
set -uo pipefail

APP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$APP/../.." && pwd)"
VERSION="$(sed -n 's/^DATASET_VERSION = "\(.*\)"$/\1/p' "$APP/transport_config.py")"
[ -n "$VERSION" ] || { echo "DATASET_VERSION not found in transport_config.py" >&2; exit 2; }
OUT="${APP2_SID_OUT:-$APP/output/$VERSION}"
PY="${PY:-python3}"  # set PY to the inverse-environment python
THREADS="${APP2_SID_THREADS:-64}"

stamp() { date -Iseconds; }
state() { echo "$1 $(stamp)" > "$OUT/RUN_STATE"; echo "[$(stamp)] $1" >> "$OUT/logs/pipeline.log"; }

capture_env() {
  mkdir -p "$OUT/env"
  git -C "$REPO" rev-parse HEAD > "$OUT/env/git_commit.txt" 2>&1
  git -C "$REPO" rev-parse --abbrev-ref HEAD > "$OUT/env/git_branch.txt" 2>&1
  git -C "$REPO" status --porcelain > "$OUT/env/git_status.txt" 2>&1
  git -C "$REPO" diff HEAD > "$OUT/env/git_diff_tracked.patch" 2>&1
  "$PY" -m pip freeze > "$OUT/env/projenv_pip_freeze.txt" 2>&1 || true
  { hostname; uname -a; nproc; free -g | head -2; date -Iseconds; echo "threads=$THREADS"; } > "$OUT/env/host.txt"
  ( cd "$REPO" && sha256sum Applications/Application2/*.py Applications/Application2/*.sh \
      "Applications/Application2/test scripts"/*.py Applications/Application1/*.py \
      Src/Objective/gradient_test.py Src/Solvers/*.py Src/Optimizers/*.py ) \
      > "$OUT/env/sha256_sources.txt" 2>&1
  mkdir -p "$OUT/env/untracked_application2_sources"
  ( cd "$APP" && cp -p transport_config.py transport_model.py generate_observations.py run_inversion.py \
      validate_readiness.py report_source_id.py run_application2_source_id_setup.sh \
      "test scripts/select_transport_settings.py" "$OUT/env/untracked_application2_sources/" )
}

write_commands() {
  mkdir -p "$OUT/commands"
  {
    echo "#!/usr/bin/env bash"
    echo "# Application2 inversion commands for dataset $VERSION -- PREPARED, NOT EXECUTED by the setup pipeline."
    echo "# Each run: coarse mesh, Application1 sensitivity initialization (OSQP joint amplitudes, M = 2 production"
    echo "# preset) followed by GN-diagonal-scaled L-BFGS-B (maxcor 30, max-iter 200, raw feasibility, amplitude"
    echo "# bounds [0.1, 2.0]).  Inputs: case_XX/config/inversion_config.json + the selected observation file only."
    echo "# Output: $OUT/case_XX/inversion_<clean|noisy>/ (inversion_summary.json, initialization.json,"
    echo "#         optimization_history.csv, full.log); the driver refuses to overwrite an existing result folder."
    echo "# Rough cost per run on this host: one forward sweep ~10 s, one adjoint sweep ~48 s (20000 RK4 steps)."
    echo "# Launch detached, e.g.:  setsid nohup bash $OUT/commands/inversion_commands.sh clean < /dev/null > $OUT/logs/inversion_clean_driver.log 2>&1 &"
    echo "set -uo pipefail"
    echo "cd \"$APP\""
    echo "export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS PYTHONUNBUFFERED=1 MPLCONFIGDIR=\"$OUT/.mplcache\""
    echo "KIND=\"\${1:?usage: inversion_commands.sh clean|noisy}\""
    echo "case \"\$KIND\" in"
    for kind in clean noisy; do
      echo "  $kind)"
      for c in 12 13 23; do
        echo "    \"$PY\" run_inversion.py --dataset-dir \"$OUT\" --case $c --data $kind --stage full > \"$OUT/logs/inversion_case_${c}_${kind}.log\" 2>&1"
      done
      echo "    ;;"
    done
    echo "  *) echo \"usage: \$0 clean|noisy\" >&2; exit 2 ;;"
    echo "esac"
  } > "$OUT/commands/inversion_commands.sh"
  cat > "$OUT/commands/README.md" <<EOF
# Inversion commands (prepared, not executed)

\`\`\`bash
# clean-data inversions (cases 12, 13, 23, sequential)
setsid nohup bash $OUT/commands/inversion_commands.sh clean < /dev/null > $OUT/logs/inversion_clean_driver.log 2>&1 &
# noisy-data inversions
setsid nohup bash $OUT/commands/inversion_commands.sh noisy < /dev/null > $OUT/logs/inversion_noisy_driver.log 2>&1 &
\`\`\`

A single run: \`$PY run_inversion.py --case 12 --data noisy --stage full\` (from \`$APP\`).
\`--stage init\` runs the sensitivity initialization only; \`--stage wiring\` runs neither.
Options and defaults: \`$PY run_inversion.py --help\`. The clean and noisy runs of a case use the same
mesh, velocity, sensors, bounds and algorithm settings; only the observation file differs.
The \`full\` and \`init\` stages have not been executed yet; only \`wiring\` was exercised by the validation.
EOF
}

run_stage() {   # run_stage <STATE_NAME> <logfile> <command...>
  local name="$1" logf="$2"; shift 2
  state "${name}_RUNNING"
  "$@" > "$OUT/logs/$logf" 2>&1
  local rc=$?
  echo "[$(stamp)] $name exit=$rc" >> "$OUT/logs/pipeline.log"
  return $rc
}

run_pipeline() {
  mkdir -p "$OUT/logs"
  export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
  export PYTHONUNBUFFERED=1 MPLCONFIGDIR="$OUT/.mplcache"
  cd "$APP" || exit 2
  [ -d "$OUT/env" ] || capture_env
  {
    echo "# Application2 source-identification dataset pipeline ($(stamp)); dataset $VERSION; cwd $APP; threads $THREADS"
    echo "$PY \"test scripts/select_transport_settings.py\" --out $OUT/common/settings_selection"
    echo "$PY generate_observations.py --out $OUT"
    echo "$PY validate_readiness.py --dataset-dir $OUT"
    echo "$PY report_source_id.py --dataset-dir $OUT"
  } > "$OUT/RUN_COMMANDS.txt"
  write_commands

  local failed=""
  run_stage SELECTION settings_selection.log "$PY" "test scripts/select_transport_settings.py" \
      --out "$OUT/common/settings_selection" || failed="$failed selection"
  if run_stage GENERATION generate_observations_driver.log "$PY" generate_observations.py --out "$OUT"; then
    run_stage VALIDATION validate_readiness_driver.log "$PY" validate_readiness.py --dataset-dir "$OUT" \
        || failed="$failed validation"
  else
    failed="$failed generation(validation-skipped)"
  fi
  if [ -z "$failed" ]; then state "REPORT_RUNNING (all stages ok)"; else state "REPORT_RUNNING (failed:$failed)"; fi
  "$PY" report_source_id.py --dataset-dir "$OUT" > "$OUT/logs/report.log" 2>&1 || failed="$failed report"
  if [ -z "$failed" ]; then state "DONE_OK"; else state "FAILED:$failed (see logs/)"; fi
  "$PY" report_source_id.py --dataset-dir "$OUT" >> "$OUT/logs/report.log" 2>&1 || true   # report shows the final state
}

refuse_if_exists() {
  if ls -d "$OUT"/case_* > /dev/null 2>&1; then
    echo "refusing: $OUT already holds case_* datasets (saved observations/noise are never regenerated in place)" >&2
    exit 1
  fi
}

case "${1:-}" in
  run)
    refuse_if_exists
    mkdir -p "$OUT/logs"
    run_pipeline
    ;;
  launch)
    refuse_if_exists
    mkdir -p "$OUT/logs"
    capture_env
    setsid nohup bash "$APP/run_application2_source_id_setup.sh" run < /dev/null > "$OUT/logs/pipeline_driver.log" 2>&1 &
    echo $! > "$OUT/pipeline.pid"
    { echo "launched $(stamp) pid $(cat "$OUT/pipeline.pid") (setsid session leader; pgid = pid)";
      echo "command: setsid nohup bash $APP/run_application2_source_id_setup.sh run < /dev/null > $OUT/logs/pipeline_driver.log 2>&1 &";
      echo "output: $OUT"; } | tee "$OUT/LAUNCH.txt"
    ;;
  report)
    cd "$APP" && MPLCONFIGDIR="$OUT/.mplcache" "$PY" report_source_id.py --dataset-dir "$OUT"
    ;;
  status)
    echo "output: $OUT"
    [ -f "$OUT/RUN_STATE" ] && echo "state : $(cat "$OUT/RUN_STATE")" || echo "state : (no run)"
    if [ -f "$OUT/pipeline.pid" ]; then
      pid=$(cat "$OUT/pipeline.pid")
      if ps -p "$pid" > /dev/null 2>&1; then echo "pid $pid: running"; else echo "pid $pid: finished"; fi
    fi
    [ -f "$OUT/logs/pipeline.log" ] && { echo "--- pipeline.log"; tail -6 "$OUT/logs/pipeline.log"; }
    if [ -f "$OUT/logs/validate_readiness.log" ]; then
      echo "--- validation: $(grep -c '\[PASS\]' "$OUT/logs/validate_readiness.log") passed, $(grep -c '\[FAIL\]' "$OUT/logs/validate_readiness.log") failed so far"
      tail -2 "$OUT/logs/validate_readiness.log" | cut -c1-220
    elif [ -f "$OUT/logs/generate_observations.log" ]; then
      echo "--- generation"; tail -2 "$OUT/logs/generate_observations.log" | cut -c1-220
    fi
    [ -f "$OUT/REPORT.md" ] && echo "report: $OUT/REPORT.md"
    ;;
  *)
    echo "usage: $0 {launch|status|run|report}" >&2; exit 2 ;;
esac
