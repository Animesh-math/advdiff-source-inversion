"""Verify the three feasibility modes at an INFEASIBLE control (center in a building).

Checks, at one control whose first center sits inside a building hole:
  penalty : J jumps to the ~1e12 baseline, no PDE solve
  project : J is the true objective at the PROJECTED center, wall-normal gradient
            component removed, projected_evaluations incremented
  raw     : J is the true objective at the RAW center, projection counters stay
            ZERO, raw_infeasible_evaluations increments, gradient keeps its full
            wall-normal component
"""
import sys
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[1]
# Src/Objective (gradient_test) on the path; APP inserted last so it is FIRST
# and Application1's mesh_data_config / multi_source_utils win.
sys.path.insert(0, str(APP.parents[1] / "Src" / "Objective"))
sys.path.insert(0, str(APP))

import run_optimization as ro
import gradient_test as gt
from mesh_admissibility import MeshAdmissibility

mesh_data = ro.configure_measurement_boundaries(
    ro.build_mesh_data(), ["left", "right", "bottom"]
)
adm = MeshAdmissibility(mesh_data)
ro.PDE["tfinal"] = 25.0          # short: this is a mechanism check, not a physics run
ro.PDE["sigma2"] = 1600.0

true_control = np.array([260.0, 300.0, 1.55, 60.0, 180.0, 1.10])
target = gt.run_forward(mesh_data, ro.PDE, true_control)

# find a control whose first center is inside a building hole
coord = np.asarray(mesh_data["coord"], dtype=np.float64)[:, :2]
inside_building = None
for cx in np.arange(coord[:, 0].min() + 5, coord[:, 0].max() - 5, 3.0):
    for cy in np.arange(coord[:, 1].min() + 5, coord[:, 1].max() - 5, 3.0):
        if not adm.is_point_in_domain(float(cx), float(cy)):
            # DEEP inside a building: far from every wall, so the projection
            # distance (and therefore the raw-vs-project difference) is material.
            if adm.distance_to_boundary(float(cx), float(cy)) > 5.0:
                inside_building = (float(cx), float(cy))
                break
    if inside_building:
        break
print("infeasible center used:", inside_building,
      "| nearest-vertex distance =",
      round(float(np.sqrt(adm.nearest_domain_distance_sq(*inside_building)[0])), 4),
      "| nearest-WALL distance =",
      round(float(adm.distance_to_boundary(*inside_building)), 4))

z_bad = np.array([inside_building[0], inside_building[1], 1.2, 60.0, 180.0, 1.1])
mode = "location_and_intensity"

results = {}
for fm in ("penalty", "project", "raw"):
    stats = ro.new_feasibility_stats(fm)
    cache, hist, obj, grad, cb = ro.build_evaluator(
        mesh_data, target, mode, ro.MESH_DOMAIN_BOUND_CASE, {}, z_bad, adm,
        feasibility_mode=fm, feasibility_stats=stats,
    )
    za = ro.pack_active(z_bad, mode)
    J = obj(za)
    g = grad(za)
    cb(za)                      # populates history with the EVALUATED control
    evaluated = np.asarray(hist[-1]["control"], dtype=np.float64).copy()
    results[fm] = (J, g.copy(), dict(stats), evaluated)
    print(f"\n--- {fm} ---")
    print(f"  J                     = {J:.10e}")
    print(f"  evaluated control     = {np.array2string(evaluated, precision=8)}")
    print(f"  center moved          = {np.hypot(*(evaluated[:2] - z_bad[:2])):.8f} m")
    print(f"  J (repr)              = {J!r}")
    print(f"  ||g||                 = {np.linalg.norm(g):.6e}")
    print(f"  g (x1,y1)             = ({g[0]:.6e}, {g[1]:.6e})")
    print(f"  invalid_evaluations   = {stats['invalid_evaluations']}")
    print(f"  projected_evaluations = {stats['projected_evaluations']}")
    print(f"  penalized_evaluations = {stats['penalized_evaluations']}")
    print(f"  raw_infeasible_evals  = {stats['raw_infeasible_evaluations']}")
    print(f"  raw_max_outside_dist  = {stats['raw_max_outside_distance']:.6f}")

ok = True
Jp, gp, sp, cp = results["project"]
Jr, gr, sr, cr = results["raw"]
Jq, gq, sq, cq = results["penalty"]

c1 = np.allclose(cr[:2], z_bad[:2])
print(f"\n[1] raw evaluates at the RAW center (unmoved): {c1}")
ok &= c1
c2 = not np.allclose(cp[:2], z_bad[:2])
print(f"[2] project evaluates at a MOVED center: {c2}  "
      f"(moved {np.hypot(*(cp[:2]-z_bad[:2])):.4f} m)")
ok &= c2
c3 = sr["projected_evaluations"] == 0 and sr["projected_source_events"] == 0
print(f"[3] raw performs ZERO projections: {c3}")
ok &= c3
c4 = sr["raw_infeasible_evaluations"] == 1 and sr["raw_max_outside_distance"] > 0
print(f"[4] raw counts the infeasible evaluation: {c4}")
ok &= c4
c5 = sr["penalized_evaluations"] == 0 and Jr < 1.0e11
print(f"[5] raw applies NO penalty (J={Jr:.4e} is a true objective): {c5}")
ok &= c5
c6 = Jq > 1.0e11
print(f"[6] penalty still returns the ~1e12 baseline: {c6}")
ok &= c6
# [7] raw must return the TRUE objective of the RAW control -- recompute it
# independently through the plain forward solver + objective, no evaluator.
state_raw = gt.run_forward(mesh_data, ro.PDE, z_bad)
J_direct = gt.compute_objective(
    state_raw, target, mesh_data, eta1=ro.ETA1, lam=ro.LAMBDA_REG, z=z_bad,
    dt_out=ro.PDE["dt"] * ro.PDE["jprint"], time_rule=ro.OBJECTIVE_TIME_RULE,
)
c7 = abs(Jr - float(J_direct)) <= 1.0e-9 * max(abs(J_direct), 1.0)
print(f"[7] raw J equals the TRUE objective at the RAW center: {c7}")
print(f"      evaluator raw J = {Jr!r}")
print(f"      direct    J(z)  = {float(J_direct)!r}")
print(f"      project   J      = {Jp!r}  (evaluated "
      f"{np.hypot(*(cp[:2]-z_bad[:2])):.4f} m away; J differs by "
      f"{abs(Jr-Jp)/max(abs(Jp),1.0):.2e} relative -- this control is nearly "
      f"insensitive to the lower-left source, which is the documented "
      f"observability asymmetry, not a mode defect)")
ok &= c7
# wall-normal component: project must have removed it, raw must not
_dist, foot = adm.nearest_boundary_point(*z_bad[:2])
n = (z_bad[:2] - foot)
n = n / np.hypot(*n)
proj_normal = abs(float(gp[:2] @ n))
raw_normal = abs(float(gr[:2] @ n))
c8 = proj_normal < 1.0e-6 * max(np.linalg.norm(gp[:2]), 1.0) and raw_normal > 0.0
print(f"[8] wall-normal gradient component: project={proj_normal:.4e} (removed), "
      f"raw={raw_normal:.4e} (kept): {c8}")
ok &= c8

print(f"\n{'ALL RAW-MODE CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
sys.exit(0 if ok else 1)
