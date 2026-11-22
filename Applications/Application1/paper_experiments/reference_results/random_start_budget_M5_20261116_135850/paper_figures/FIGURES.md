# App1 five-source matched-budget benchmark: paper figures and tables

Figures are vector PDF plus 300 dpi PNG. Tables are in `tables/` as booktabs LaTeX and full-precision CSV; see `tables_preview.pdf`.

- `budget_reconstruction`: sensitivity-initialized run against the best random start by objective. The random start's path ends at
  its best feasible, fully evaluated point, which is what the driver reports on a budget stop.
- `budget_starts`: all ten budgeted starts.
- `budget_objective_vs_solves`: best objective so far with the starts concatenated in the order r01..r10 (light vertical lines
  separate starts). The sensitivity curve has only its two recorded points: after initialization and after L-BFGS-B.
- `budget_sensor_traces`: the four sensors where the best random start has the largest misfit (chosen to show the difference,
  not a typical sensor); post-hoc forward solves, not counted in the budget.

Scope: one five-source configuration, noiseless inverse-crime data, ten starts. A cost-matched comparison, not a statistical study.
