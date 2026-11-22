# Application2 inversion paper figures

Each figure is saved as a vector PDF and a 300 dpi PNG. Suggested captions:

- `problem_setup`: room geometry, supply vents, door and exhaust outlets, eight wall-mounted sensors, and the three release locations with amplitudes; shaded discs have radius sqrt(rho) = 0.20 m.
- `reconstruction_case_<c>`: recovery of the two sources. Panel (a) locates the zoom windows; the zooms show the sensitivity initialization, the L-BFGS-B iterates, the final estimate and the true centre for clean and noisy data. Zoom scales differ between panels; an inset magnifies the iterates when they are short compared with the distance to the true centre.
- `convergence_history`: objective relative to the zero-source objective, unscaled gradient norm and largest permutation-matched centre error at each accepted L-BFGS-B iterate (iteration 0 = initialization).
- `traces_case_<c>`: the four sensors with the largest clean-data RMS. Model traces are post-hoc coarse-mesh forward solves (not counted in the optimization solve totals); every 20th time sample is plotted.

Cases: 12, 13, 23. True controls are used only for scoring and plotting, after the runs.
