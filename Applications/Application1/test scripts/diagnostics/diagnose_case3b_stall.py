"""Why did Case 3b's L-BFGS-B make zero progress? Probe J and grad at its init.

READ-ONLY. Reproduces the exact reduced objective/gradient the driver uses
(run_optimization.build_evaluator) at the Case 3b sensitivity init, at the first
continuation width sigma2=1600 where stage 1 did 0 iterations, and line-probes
along -grad with bound projection + admissibility, to tell apart:
  (a) not a descent direction,
  (b) bad scaling (only a microscopic step reduces J),
  (c) the non-fluid invalid-control penalty wall (trial steps leave the mesh).
"""
from __future__ import annotations
import numpy as np
import sys as _sys_dc
from pathlib import Path as _Path_dc
_sys_dc.path.insert(0, str(_Path_dc(__file__).resolve().parents[1]))  # _diag_common
import _diag_common as dc  # sets sys.path for App2 + App1
import run_optimization as ro
import gradient_test as gt
from mesh_admissibility import MeshAdmissibility

md = ro.configure_measurement_boundaries(ro.build_mesh_data(), ["left", "right", "bottom"])
vel, _ = ro.load_velocity_field_file(str(ro.DEFAULT_VELOCITY_FIELD_FILE), md)
ro.PDE["tfinal"] = 150.0
ro.PDE["sigma2"] = 1600.0
ro.PDE["velocity_nodes"] = vel
adm = MeshAdmissibility(md)

true = np.array([260.0, 300.0, 1.55, 60.0, 180.0, 1.10])
init = np.array([276.70079126628343, 299.23289027376404, 1.245466051878722,
                 107.9552177193691, 252.99682653768105, 0.8116989255223073])

coord = np.asarray(md["coord"])[:, :2]
xmin, xmax = coord[:, 0].min(), coord[:, 0].max()
ymin, ymax = coord[:, 1].min(), coord[:, 1].max()
lo = np.array([xmin, ymin, 0.1, xmin, ymin, 0.1])
hi = np.array([xmax, ymax, 2.0, xmax, ymax, 2.0])

target = gt.run_forward(md, ro.PDE, true)
_c, _h, obj, grad, _cb = ro.build_evaluator(
    md, target, "location_and_intensity", ro.MESH_DOMAIN_BOUND_CASE, None, init, adm)

J0 = obj(init)
g = grad(init)
print(f"J0(sigma=1600) = {J0:.6e}")
print(f"||g|| = {np.linalg.norm(g):.6e}")
np.set_printoptions(precision=4, suppress=False)
print(f"grad = {g}")
print(f"  |g| position comps  = {np.abs(g[[0,1,3,4]])}")
print(f"  |g| amplitude comps = {np.abs(g[[2,5]])}")
print(f"init sources admissible? "
      f"s1={adm.is_point_in_domain(init[0],init[1])} "
      f"s2={adm.is_point_in_domain(init[3],init[4])}")
# distance from each init source to the nearest non-fluid boundary (probe a ring)
for sid,(sx,sy) in enumerate([(init[0],init[1]),(init[3],init[4])],1):
    rmin=None
    for r in [2,4,6,8,10,15,20,30,40,60]:
        hit=any(not adm.is_point_in_domain(sx+r*np.cos(t), sy+r*np.sin(t))
                for t in np.linspace(0,2*np.pi,24,endpoint=False))
        if hit: rmin=r; break
    print(f"  source {sid} at ({sx:.1f},{sy:.1f}): nearest non-fluid within ~{rmin} m"
          if rmin else f"  source {sid}: fluid out to 60 m")

gn = g / np.linalg.norm(g)
print("\nprojected line probe along -grad (steepest-descent first step):")
print(f"{'t':>10} {'valid':>7} {'J':>14} {'J-J0':>14}")
for t in [1e-8,1e-6,1e-4,1e-3,1e-2,1e-1,1.0,10.0,50.0,100.0,300.0]:
    zt = np.clip(init - t*gn, lo, hi)
    valid = adm.validate_sources_in_domain(zt)
    Jt = obj(zt)
    print(f"{t:>10.0e} {('yes' if valid else 'INVALID'):>7} {Jt:>14.6e} {Jt-J0:>14.3e}")

# amplitude-only and position-only descent, to expose scale imbalance
print("\namplitude-only vs position-only steepest steps (unit-normalized blocks):")
for label, idx in [("amp", [2,5]), ("pos", [0,1,3,4])]:
    gb = np.zeros(6); gb[idx] = g[idx]
    nb = np.linalg.norm(gb)
    if nb == 0: continue
    gbn = gb/nb
    best=None
    for t in [1e-6,1e-4,1e-3,1e-2,1e-1,1,10,50,100]:
        zt=np.clip(init - t*gbn, lo, hi)
        if not adm.validate_sources_in_domain(zt): continue
        Jt=obj(zt)
        if best is None or Jt<best[1]: best=(t,Jt)
    if best: print(f"  {label}: best feasible J = {best[1]:.6e} at t={best[0]:.0e} "
                   f"(dJ={best[1]-J0:.3e})")
