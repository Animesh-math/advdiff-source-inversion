#!/usr/bin/env bash
# Capture provenance, then start the study fully detached (survives SSH logout):
#   setsid nohup bash scripts/run_batch.sh "$EXP" 10 8  </dev/null  >logs/batch_driver.log 2>&1 &
set -euo pipefail
EXP="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$EXP/../../../.." && pwd)"  # EXP = Applications/Application1/output/<name>
PY="${PY:-python3}"; FX="${FX:-}"  # FX (FEniCSx python) is optional, provenance only
mkdir -p "$EXP/env" "$EXP/logs"
git -C "$REPO" rev-parse HEAD > "$EXP/env/git_commit.txt"
git -C "$REPO" rev-parse --abbrev-ref HEAD > "$EXP/env/git_branch.txt"
git -C "$REPO" status --porcelain > "$EXP/env/git_status.txt"
git -C "$REPO" diff HEAD > "$EXP/env/git_diff_tracked.patch"
"$PY" -m pip freeze > "$EXP/env/pip_freeze_projenv.txt" 2>&1 || true
"$PY" --version > "$EXP/env/python_projenv.txt" 2>&1
[ -n "$FX" ] && "$FX" -m pip list --format=freeze > "$EXP/env/pip_list_fenicsx_env.txt" 2>&1 || true
[ -n "$FX" ] && ls "$(dirname "$(dirname "$FX")")"/conda-meta/*.json 2>/dev/null | xargs -n1 basename | sed 's/\.json$//' > "$EXP/env/conda_meta_fenicsx_env.txt" || true
{ hostname; uname -a; nproc; free -g | head -2; date -Iseconds; } > "$EXP/env/host.txt"
( cd "$EXP" && sha256sum scripts/* config/* commands.tsv 2>/dev/null ) > "$EXP/env/sha256_scripts_config.txt"
sha256sum "$REPO"/Applications/Application1/*.py "$REPO"/Src/Solvers/*.py "$REPO"/Src/Optimizers/*.py "$REPO/Src/Objective/gradient_test.py" "$REPO/Applications/Application1/Navier Stokes Solve/solve_application1_navier_stokes.py" "$REPO/Applications/Application1/DC_mesh/dc_navy_yard_mesh.msh" "$REPO/Applications/Application1/output/navier_stokes/application1_navier_stokes_velocity_nodes.npz" > "$EXP/env/sha256_repo_inputs.txt"
cd "$EXP"
PY="$PY" setsid nohup bash scripts/run_batch.sh "$EXP" 10 8 < /dev/null > "$EXP/logs/batch_driver.log" 2>&1 &
echo $! > "$EXP/batch.pid"
{ echo "launched $(date -Iseconds) pid $(cat "$EXP/batch.pid")"; echo "command: PY="$PY" setsid nohup bash scripts/run_batch.sh "$EXP" 10 8 < /dev/null > logs/batch_driver.log 2>&1 &"; } | tee "$EXP/LAUNCH.txt"
