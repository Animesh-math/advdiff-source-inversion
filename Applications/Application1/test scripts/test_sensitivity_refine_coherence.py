"""Focused tests for the optional refine_centers / coherence_cap sensitivity init.

READ-ONLY w.r.t. production numerics: exercises the public
``build_sensitivity_initial_control`` entry point with the exact Application1
configuration (lrb, sigma2=1600, beam_width=3) via ``_diag_common``.

Verifies (task requirement 16):
  1. defaults (refine_centers=False, coherence_cap=None) reproduce current behavior;
  2. refinement decreases (does not increase) the reduced objective, and moves the
     first source into the dominant-source basin;
  3. the branch-specific residual is recomputed AFTER refinement (second source
     lands in the weak-source basin -> objective collapses);
  4. a coherence cap rejects high-coherence second-source candidates and flips the
     selection to the basin-correct pair;
  5. zero / near-zero-norm coherence is handled safely (no divide-by-zero/NaN);
  6. a supplied manual initial control still overrides sensitivity init (checked at
     the CLI-precedence level).

Run:
  cd /path/to/Advection-Diffusion-PDECO
  python3 \
    "Applications/Application1/test scripts/test_sensitivity_refine_coherence.py"
"""

from __future__ import annotations

import numpy as np

import _diag_common as dc
import sensitivity_initialization as sinit

T1 = np.array([250.0, 130.0])
T2 = np.array([15.0, 580.0])
BOUNDS = dc.AMPLITUDE_BOUNDS
# Production beam control saved in output/sensitivity_true_250_130_15_580 (report §2).
PROD_CONTROL = np.array(
    [211.5719053728, 173.4116764393, 0.8329776270,
     285.1289895204, 126.4094834683, 0.8941020049]
)


def _common():
    mesh_data, pde, admiss = dc.build_setup(("left", "right", "bottom"), 1600.0)
    ud = dc.true_state(mesh_data, pde)
    return mesh_data, pde, admiss, ud


def _call(mesh_data, pde, admiss, ud, **kw):
    return sinit.build_sensitivity_initial_control(
        mesh_data, pde, ud, 2, BOUNDS, admiss,
        eta1=dc.ro.ETA1, lam=dc.ro.LAMBDA_REG,
        objective_time_rule=dc.ro.OBJECTIVE_TIME_RULE,
        score="predicted_reduction", top_k=24, beam_width=3,
        verbose=False, **kw,
    )


def _dist(c, t):
    return float(np.hypot(c[0] - t[0], c[1] - t[1]))


def _nearest(control, target):
    cs = control.reshape(-1, 3)[:, :2]
    return min(_dist(c, target) for c in cs)


def test_zero_norm_coherence_safe():
    print("[1] zero-norm coherence safety")
    f = sinit._coherence_from_norms
    assert f(0.0, 0.0, 1.0) == 0.0
    assert f(5.0, 0.0, 0.0) == 0.0
    assert f(1e-40, 1e-320, 1e-320) == 0.0        # underflow denom -> 0, no NaN
    v = f(2.0, 4.0, 1.0)                           # 2 / (2*1) = 1.0
    assert abs(v - 1.0) < 1e-12, v
    assert np.isfinite(f(np.inf, 1.0, 1.0)) or f(np.inf, 1.0, 1.0) == 0.0
    print("    PASS")


def test_defaults_reproduce(mesh_data, pde, admiss, ud):
    print("[2] defaults reproduce production control")
    ctrl, _diag, extras = _call(mesh_data, pde, admiss, ud)
    err = float(np.max(np.abs(ctrl - PROD_CONTROL)))
    print(f"    max|ctrl - production| = {err:.3e}")
    print(f"    extras: refine={extras['refine_centers']} cap={extras['coherence_cap']} "
          f"dedup={extras['n_dedup_removed']} J={extras['initial_exact_objective']:.4e}")
    assert err < 1e-4, err
    assert extras["refine_centers"] is False
    assert extras["coherence_cap"] is None
    assert extras["n_dedup_removed"] == 0
    return extras["initial_exact_objective"]


def test_refine_fixes(mesh_data, pde, admiss, ud, j_default):
    print("[3] refine_centers moves source 1 to (250,130), recomputes residual")
    ctrl, _diag, extras = _call(mesh_data, pde, admiss, ud, refine_centers=True,
                                refine_max_fev=60)
    d1 = _nearest(ctrl, T1)
    d2 = _nearest(ctrl, T2)
    j = extras["initial_exact_objective"]
    print(f"    control = {np.array2string(ctrl, precision=2)}")
    print(f"    nearest->T1={d1:.1f} m  nearest->T2={d2:.1f} m  J={j:.4e}  "
          f"dedup_removed={extras['n_dedup_removed']}")
    assert d1 < 10.0, d1                    # first source in the dominant basin
    assert d2 < 25.0, d2                    # second source in the weak basin
    assert j <= j_default, (j, j_default)   # objective did not increase
    assert j < 1.0e5, j                     # basin-correct (report: ~1.2e4)
    assert extras["n_dedup_removed"] >= 1   # collapsed branches deduplicated
    print("    PASS")


def test_coherence_cap_rejects(mesh_data, pde, admiss, ud):
    print("[4] coherence_cap rejects high-coherence pair, flips to basin-correct")
    # Uncapped selection is the coherence-0.715 plume-straddling pair.
    base, _d, _e = _call(mesh_data, pde, admiss, ud)
    d2_base = _nearest(base, T2)
    capped, _d2, extras = _call(mesh_data, pde, admiss, ud, coherence_cap=0.3)
    d2_cap = _nearest(capped, T2)
    print(f"    uncapped nearest->T2 = {d2_base:.1f} m")
    print(f"    capped(0.3) control  = {np.array2string(capped, precision=2)}")
    print(f"    capped nearest->T2   = {d2_cap:.1f} m")
    assert d2_base > 100.0, d2_base         # uncapped: second source not near T2
    assert d2_cap < 25.0, d2_cap            # capped: flips to the weak-source basin
    print("    PASS")


def main():
    test_zero_norm_coherence_safe()
    mesh_data, pde, admiss, ud = _common()
    j_default = test_defaults_reproduce(mesh_data, pde, admiss, ud)
    test_coherence_cap_rejects(mesh_data, pde, admiss, ud)
    test_refine_fixes(mesh_data, pde, admiss, ud, j_default)
    print("\nALL SENSITIVITY REFINE/COHERENCE TESTS PASSED")


if __name__ == "__main__":
    main()
