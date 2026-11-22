# App1 random-start baseline: paper figures and tables

Figures are vector PDF plus 300 dpi PNG. Tables are in `tables/` as booktabs LaTeX (needs `booktabs`; `table_all_runs` also needs
`longtable`) and full-precision CSV. `tables_preview.pdf` shows every table.

- `reconstruction_<case>`: sensitivity-initialized run against the random start with the lowest final objective. Colours identify
  the matched true source. The right column zooms on the sensitivity-initialized path to each source. Iterates may cross
  buildings because the optimizer runs with raw feasibility; only final centres are required to lie in the fluid.
- `starts_<case>`: all ten random starts of the case; a blue title marks a recovered start.
- `recovery_overview`: final errors and objectives of all random starts and of the sensitivity-initialized runs.
- `best_of_k`: best objective so far against cumulative solves, starts taken in the fixed order r01..r10.
- `solve_cost`: total forward + adjoint solves. The sensitivity total includes the initializer.
- `reference_convergence`: L-BFGS-B history of the sensitivity-initialized runs (iteration 0 = initialized control).
- `sensor_traces`: the four strongest sensors of each case. Traces are post-hoc forward solves, not counted in any solve total.

Scope: noiseless inverse-crime data, one truth per case, ten starts per case; the original (`p = 0`) velocity field.
