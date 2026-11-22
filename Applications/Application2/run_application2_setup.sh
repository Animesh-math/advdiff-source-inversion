#!/usr/bin/env bash
# Application2 setup pipeline: meshes (coarse + fine; three-bed room + bathroom annex,
# partitions, solid beds and bathroom wall, with geometry/tag checks) -> two
# independent steady Navier-Stokes solves (run concurrently) -> shared report.
#
#   bash Applications/Application2/run_application2_setup.sh launch   # detached (survives SSH logout)
#   bash Applications/Application2/run_application2_setup.sh status   # progress check
#   bash Applications/Application2/run_application2_setup.sh run      # foreground
#
# Everything is written to the VERSIONED folder
#   Applications/Application2/output/<OUTPUT_VERSION>/   (OUTPUT_VERSION in room_geometry_config.py;
#                                                         override with $APP2_OUT)
#   coarse/{mesh,navier_stokes}  fine/{mesh,navier_stokes}  report/  logs/  env/
#   RUN_STATE  RUN_COMMANDS.txt  LAUNCH.txt  pipeline.pid
# Refuses to run if coarse/ or fine/ already exist there: earlier versions are
# never overwritten (bump OUTPUT_VERSION for a new geometry).
set -euo pipefail

APP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$APP/../.." && pwd)"
VERSION="$(sed -n 's/^OUTPUT_VERSION = "\(.*\)"$/\1/p' "$APP/room_geometry_config.py")"
[ -n "$VERSION" ] || { echo "OUTPUT_VERSION not found in room_geometry_config.py" >&2; exit 2; }
OUT="${APP2_OUT:-$APP/output/$VERSION}"
FX="${FX:-python3}"  # set FX to the FEniCSx-environment python
THREADS="${APP2_THREADS:-4}"
NS="$APP/Navier Stokes Solve/solve_application2_navier_stokes.py"

stamp() { date -Iseconds; }
state() { echo "$1 $(stamp)" > "$OUT/RUN_STATE"; echo "[$(stamp)] $1" >> "$OUT/logs/pipeline.log"; }

capture_env() {
  mkdir -p "$OUT/env"
  git -C "$REPO" rev-parse HEAD > "$OUT/env/git_commit.txt"
  git -C "$REPO" rev-parse --abbrev-ref HEAD > "$OUT/env/git_branch.txt"
  git -C "$REPO" status --porcelain > "$OUT/env/git_status.txt"
  git -C "$REPO" diff HEAD > "$OUT/env/git_diff_tracked.patch"
  "$FX" --version > "$OUT/env/python_fenicsx_env.txt" 2>&1
  "$FX" -m pip list --format=freeze > "$OUT/env/pip_list_fenicsx_env.txt" 2>&1 || true
  ls "$(dirname "$(dirname "$FX")")"/conda-meta/*.json 2>/dev/null | xargs -n1 basename | sed 's/\.json$//' \
      > "$OUT/env/conda_meta_fenicsx_env.txt" || true
  { hostname; uname -a; nproc; free -g | head -2; date -Iseconds; } > "$OUT/env/host.txt"
  ( cd "$APP" && sha256sum ./*.py ./*.sh ./room_layout_reference.* Room_mesh/*.py "Navier Stokes Solve"/*.py "test scripts"/*.py 2>/dev/null ) > "$OUT/env/sha256_application2_sources.txt"
}

run_pipeline() {
  mkdir -p "$OUT/logs"
  export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
  export MPLCONFIGDIR="$OUT/.mplcache"
  cd "$REPO"
  {
    echo "# Application2 setup pipeline ($(stamp)); version $VERSION; cwd $REPO; OMP_NUM_THREADS=$THREADS"
    echo "$FX Applications/Application2/Room_mesh/build_room_mesh.py --level all --out-root $OUT"
    echo "$FX \"Applications/Application2/Navier Stokes Solve/solve_application2_navier_stokes.py\" --level coarse --out-root $OUT"
    echo "$FX \"Applications/Application2/Navier Stokes Solve/solve_application2_navier_stokes.py\" --level fine --out-root $OUT"
    echo "$FX Applications/Application2/report_application2_flow.py --out-root $OUT"
  } > "$OUT/RUN_COMMANDS.txt"
  [ -d "$OUT/env" ] || capture_env

  state "MESH_RUNNING"
  "$FX" "$APP/Room_mesh/build_room_mesh.py" --level all --out-root "$OUT" > "$OUT/logs/mesh.log" 2>&1
  state "MESH_DONE"

  state "SOLVES_RUNNING"
  "$FX" "$NS" --level coarse --out-root "$OUT" > "$OUT/logs/ns_coarse.log" 2>&1 & pc=$!
  "$FX" "$NS" --level fine --out-root "$OUT" > "$OUT/logs/ns_fine.log" 2>&1 & pf=$!
  set +e
  wait $pc; ec=$?
  wait $pf; ef=$?
  set -e
  echo "[$(stamp)] ns_coarse exit=$ec ns_fine exit=$ef (0 converged, 2 not converged, other = error)" >> "$OUT/logs/pipeline.log"

  state "REPORT_RUNNING"
  "$FX" "$APP/report_application2_flow.py" --out-root "$OUT" > "$OUT/logs/report.log" 2>&1
  if [ "$ec" -eq 0 ] && [ "$ef" -eq 0 ]; then state "DONE_CONVERGED"; else state "DONE_WITH_ISSUES coarse=$ec fine=$ef"; fi
}

case "${1:-}" in
  run)
    if [ -d "$OUT/coarse" ] || [ -d "$OUT/fine" ]; then
      echo "refusing: $OUT/coarse or $OUT/fine exists (move it aside first)" >&2; exit 1
    fi
    mkdir -p "$OUT/logs"
    trap 'state "FAILED (line $LINENO)"' ERR
    run_pipeline
    ;;
  launch)
    if [ -d "$OUT/coarse" ] || [ -d "$OUT/fine" ]; then
      echo "refusing: $OUT/coarse or $OUT/fine exists (move it aside first)" >&2; exit 1
    fi
    mkdir -p "$OUT/logs"
    capture_env
    setsid nohup bash "$APP/run_application2_setup.sh" run < /dev/null > "$OUT/logs/pipeline_driver.log" 2>&1 &
    echo $! > "$OUT/pipeline.pid"
    { echo "launched $(stamp) pid $(cat "$OUT/pipeline.pid") (setsid session leader; pgid = pid)";
      echo "command: setsid nohup bash $APP/run_application2_setup.sh run < /dev/null > $OUT/logs/pipeline_driver.log 2>&1 &"; } \
      | tee "$OUT/LAUNCH.txt"
    ;;
  status)
    echo "output: $OUT"
    [ -f "$OUT/RUN_STATE" ] && echo "state : $(cat "$OUT/RUN_STATE")" || echo "state : (no run)"
    if [ -f "$OUT/pipeline.pid" ]; then
      pid=$(cat "$OUT/pipeline.pid")
      if ps -p "$pid" > /dev/null 2>&1; then echo "pid $pid: running"; else echo "pid $pid: finished"; fi
    fi
    for f in "$OUT"/logs/ns_coarse.log "$OUT"/logs/ns_fine.log; do
      if [ -f "$f" ]; then echo "--- $(basename "$f"): $(grep -E '^(Picard [0-9]+|status)' "$f" | tail -1)"; fi
    done
    if [ -f "$OUT/logs/pipeline.log" ]; then echo "--- pipeline.log"; tail -4 "$OUT/logs/pipeline.log"; fi
    ;;
  *)
    echo "usage: $0 {launch|status|run}" >&2; exit 2 ;;
esac
