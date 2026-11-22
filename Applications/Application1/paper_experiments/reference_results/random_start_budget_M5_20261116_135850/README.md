# Five-source random-start benchmark at matched PDE-sweep budget

Reference (ours): `output/post_init_scaling_validation_20261004_125952/cases/M5_S4_gndiag`
(sensitivity init + GN-diag L-BFGS-B, converged). NOT the stalled unscaled run
`five_source_M5_S4sensors_20261003_204754` (iteration limit 500, J = 34.7).

Budget (verified from the reference `run_instrumentation.json`, see `config/budget.json`):
initialization 3069 fwd + 12 adj; local refinement 84 fwd + 82 adj
(1 objective at the initialized control + GN-diag 16/15 + L-BFGS-B 67/67) ->
**3153 fwd + 94 adj = 3247 sweeps**. Per start: 325 (r01..r07), 324 (r08..r10).

Layout
- `scripts/prepare.py`   freezes reference, budget, 10 starts (seeds 2026111601..10) -> `config/`, `commands.tsv`
- `scripts/run_case.py`  one budgeted start through the unmodified driver (PDE-level budget guard)
- `scripts/run_batch.sh` 10 starts in parallel, 8 threads each, resumable; partial report after each start
- `scripts/report.py`    `report/REPORT.md`, comparison/per-start CSV+JSON, figures
- `scripts/launch.sh`    provenance (`env/`) + `setsid nohup` launch
- `validation/`          pre-launch guard tests at tfinal 10 (budgets 41 and 400); not part of the results
- `cases/rNN/`           driver output `full/`, `evaluations.csv`, `iterates.csv`, `run_result.json`, `run_instrumentation.json`
- `logs/`, `batch_progress.log`, `BATCH_STATE`, `batch.pid`, `failures.log`

Inspect: `cat BATCH_STATE; tail batch_progress.log; tail -n 3 logs/r01.log`
Re-run report: `<inverse-env python> scripts/report.py --exp-dir $PWD`
Resume after a crash: `bash scripts/launch.sh` (finished starts are skipped).
