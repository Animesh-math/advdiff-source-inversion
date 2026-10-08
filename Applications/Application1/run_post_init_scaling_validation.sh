#!/usr/bin/env bash
# End-to-end validation of --post-init-scaling: rerun saved production commands
# from scratch (same truth, sensors, initializer settings) with the new flags,
# through the same instrumentation wrappers.  Cases (sequential):
#   M2_rho400_gndiag    width_study_B_L2 rho400 command + --post-init-scaling gn-diag
#   M5_S4_gndiag        five_source_M5_S4sensors command + --post-init-scaling gn-diag
#   M5_S4_onstall       five_source_M5_S4sensors command + --post-init-scaling on-stall
# Baselines are the saved unscaled runs of the same commands.
#   setsid nohup ./run_post_init_scaling_validation.sh EXP < /dev/null > EXP/batch_driver.log 2>&1 &
set -uo pipefail
EXP="${1:?usage: run_post_init_scaling_validation.sh <abs experiment dir>}"
APP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$APP/../.." && pwd)"
PY="${PY:-python3}"  # override with PY=/path/to/python
W_CMDS="$APP/output/width_study_B_L2_20261003_222835/commands.sh"
F_CMD="$APP/output/five_source_M5_S4sensors_20261003_204754/command.sh"
cd "$APP" || exit 2
export PYTHONUNBUFFERED=1
mkdir -p "$EXP/scripts" "$EXP/mplcache" "$EXP/cases"
export MPLCONFIGDIR="$EXP/mplcache"
[[ -e "$EXP/BATCH_STATE" ]] && { echo "already launched"; exit 3; }
log() { echo "[$(date -Iseconds)] $*" | tee -a "$EXP/batch_progress.log"; }
trap 'log "received SIGTERM/SIGINT"; [[ -n "${child:-}" ]] && kill "$child" 2>/dev/null; echo killed > "$EXP/BATCH_STATE"; exit 143' TERM INT
log "batch start pid=$$ sid=$(ps -o sid= $$ | tr -d ' ') host=$(hostname)"
echo $$ > "$EXP/batch.pid"; echo running > "$EXP/BATCH_STATE"
cp "$0" "$EXP/scripts/"
git -C "$REPO" rev-parse HEAD > "$EXP/git_commit.txt"
git -C "$REPO" status --porcelain > "$EXP/git_status.txt"
git -C "$REPO" diff HEAD > "$EXP/git_diff_tracked.patch"
uptime > "$EXP/host_load_at_launch.txt"

# Build the three commands from the saved ones: new exp dir / case id / case name + flags.
"$PY" - "$EXP" "$W_CMDS" "$F_CMD" > "$EXP/commands.sh" <<'EOF'
import shlex, sys
exp, w_cmds, f_cmd = sys.argv[1:4]
def retarget(line, case, extra):
    a = shlex.split(line)
    a[a.index("--exp-dir") + 1] = exp
    a[a.index("--case-id") + 1] = case
    a[a.index("--case-name") + 1] = case
    return " ".join(shlex.quote(x) for x in a + extra)
w = [l for l in open(w_cmds).read().splitlines() if "--case-id rho400" in l][0]
f = [l for l in open(f_cmd).read().splitlines() if l.strip()][0]
print(retarget(w, "M2_rho400_gndiag", ["--post-init-scaling", "gn-diag"]))
print(retarget(f, "M5_S4_gndiag", ["--post-init-scaling", "gn-diag"]))
print(retarget(f, "M5_S4_onstall", ["--post-init-scaling", "on-stall"]))
EOF
mapfile -t CMDS < "$EXP/commands.sh"
CASES=(M2_rho400_gndiag M5_S4_gndiag M5_S4_onstall)
for i in "${!CASES[@]}"; do
    c="${CASES[$i]}"; mkdir -p "$EXP/cases/$c"; runlog="$EXP/cases/$c/run.log"
    start_s="$(date +%s)"; log "START $c"
    { echo "=== $c ==="; echo "start: $(date -Iseconds)"; echo "command: ${CMDS[$i]}"; echo "---"; } > "$runlog"
    eval "${CMDS[$i]}" >> "$runlog" 2>&1 < /dev/null &
    child=$!; wait "$child"; status=$?; child=""
    { echo "---"; echo "end: $(date -Iseconds)"; echo "exit_status: $status"; echo "wall_seconds: $(( $(date +%s) - start_s ))"; } >> "$runlog"
    log "DONE  $c exit=$status wall=$(( $(date +%s) - start_s ))s"
done
echo finished > "$EXP/BATCH_STATE"; log "ALL DONE"
