"""Small end-to-end check of the final feasibility repair/certification step.

Starts from a deliberately INFEASIBLE two-source control (the first source
center placed deep inside a building hole, well outside the fluid mesh) and
runs ``final_feasibility_polish`` with a small iteration cap, using the same
"raw"-mode evaluator construction the ``--final-feasibility-polish`` CLI flag
uses in ``run_optimization.py``. Short tfinal/sigma2 (as in
``test_feasibility_modes_raw.py``) so this runs in well under a minute -- it
is a mechanism check, not a physics/accuracy run.

Reports initial/repaired signed clearances, final feasible status,
initial/final objective, and the repaired control, then asserts the repaired
control's centers are all inside the fluid mesh.
"""
import sys
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[1]
# APP must end up FIRST: Test Application2 also ships a mesh_data_config.py
sys.path.insert(0, str(APP.parents[1] / "Applications" / "Test Application2"))
sys.path.insert(0, str(APP))

import run_optimization as ro
import gradient_test as gt
from mesh_admissibility import MeshAdmissibility

mesh_data = ro.configure_measurement_boundaries(
    ro.build_mesh_data(), ["left", "right", "bottom"]
)
adm = MeshAdmissibility(mesh_data)
ro.PDE["tfinal"] = 25.0
ro.PDE["sigma2"] = 1600.0

true_control = np.array([260.0, 300.0, 1.55, 60.0, 180.0, 1.10])
true_state = gt.run_forward(mesh_data, ro.PDE, true_control)

# Find a point deep inside a building hole (far from every wall) so the
# repair has real work to do, reusing the same search as
# test_feasibility_modes_raw.py.
coord = np.asarray(mesh_data["coord"], dtype=np.float64)[:, :2]
inside_building = None
for cx in np.arange(coord[:, 0].min() + 5, coord[:, 0].max() - 5, 3.0):
    for cy in np.arange(coord[:, 1].min() + 5, coord[:, 1].max() - 5, 3.0):
        if not adm.is_point_in_domain(float(cx), float(cy)):
            if adm.distance_to_boundary(float(cx), float(cy)) > 5.0:
                inside_building = (float(cx), float(cy))
                break
    if inside_building:
        break
if inside_building is None:
    raise RuntimeError("could not find a test point deep inside a building hole")
print("infeasible starting center:", inside_building)

initial_control = np.array(
    [inside_building[0], inside_building[1], 1.2, 60.0, 180.0, 1.1]
)
mode = "location_and_intensity"
num_sources = 2

bounds = ro.mesh_domain_active_bounds_for_mode(
    mode, adm.bounding_box, num_sources=num_sources
)
bounds = ro.apply_amplitude_floor(bounds, mode, None, num_sources=num_sources)

# feasibility_mode="raw": objective/gradient stay well-defined at every
# intermediate SLSQP iterate, including centers outside the fluid mesh; the
# repair itself enforces admissibility through explicit clearance
# constraints. Matches the construction in run_optimization.py's
# --final-feasibility-polish integration.
_, _, obj, grad, _ = ro.build_evaluator(
    mesh_data, true_state, mode, ro.MESH_DOMAIN_BOUND_CASE, {}, initial_control,
    adm, feasibility_mode="raw", feasibility_stats=None,
)

clearance_before = ro.signed_clearances(initial_control, adm)
objective_before = float(obj(initial_control))

print("\n=== Before repair ===")
print(f"control:            {np.array2string(initial_control, precision=6)}")
print(f"signed clearances:  {np.array2string(clearance_before, precision=6)}")
print(f"objective:          {objective_before:.10e}")

MAX_ITER = 15
MARGIN = 0.5
result = ro.final_feasibility_polish(
    initial_control, adm, bounds, obj, grad,
    clearance_margin=MARGIN, max_iter=MAX_ITER, method="slsqp",
)

repaired = np.asarray(result["repaired_control"], dtype=np.float64)
clearance_after = ro.signed_clearances(repaired, adm)

print("\n=== After repair ===")
print(f"repaired control:   {np.array2string(repaired, precision=6)}")
print(f"signed clearances:  {np.array2string(clearance_after, precision=6)}")
print(f"objective:          {result['objective_after']:.10e}")
print(f"feasible:           {result['feasible']}")
print(f"was_projected:      {result['was_projected']}")
print(f"iterations:         {result['iterations']}")
print(f"message:            {result['message']}")

print("\n=== Summary ===")
print(f"objective before -> after: {result['objective_before']:.10e} -> "
      f"{result['objective_after']:.10e}")
print(f"min clearance before -> after: {result['clearance_before_min']:.6f} -> "
      f"{result['clearance_after_min']:.6f} m  (margin={MARGIN} m)")

ok = True
c1 = bool(clearance_before[0] < 0.0)
print(f"[1] first source starts OUTSIDE the fluid mesh: {c1}")
ok &= c1
c2 = all(
    adm.is_point_in_domain(float(x), float(y))
    for x, y, _a in ro.reshape_control(repaired)
)
print(f"[2] every repaired source center is inside the fluid mesh: {c2}")
ok &= c2
c3 = bool(result["feasible"])
print(f"[3] repair certifies feasible at margin={MARGIN} m: {c3}")
ok &= c3
c4 = bool(result["was_projected"])
print(f"[4] the infeasible starting center was projected first: {c4}")
ok &= c4
c5 = result["iterations"] <= MAX_ITER
print(f"[5] repair respected the max_iter cap ({result['iterations']} <= {MAX_ITER}): {c5}")
ok &= c5

print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
