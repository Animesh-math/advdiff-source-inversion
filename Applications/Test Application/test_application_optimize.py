"""First optimization workflow for the Test Application.

Builds mesh_data, defines objective/gradient callbacks, runs scipy optimization
through a reusable optimizer class, and saves diagnostics.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "Src"
SOLVER_DIR = SRC_DIR / "Solvers"
OPTIMIZER_DIR = SRC_DIR / "Optimizers"

for path in (APP_DIR,):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

from mesh_data_config import build_mesh_data


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_OPT = _load_module_from_file("optimizer_mod", OPTIMIZER_DIR / "optimizer.py")
_FWD = _load_module_from_file("adv_diff_forward_mod", SOLVER_DIR / "adv_diff_forward.py")
_ADJ = _load_module_from_file("adv_diffusion_adjoint_mod", SOLVER_DIR / "adv_diffusion_adjoint.py")
_LIN = _load_module_from_file("linear_fe_data_mod", SOLVER_DIR / "linear_fe_data.py")
_QUAD = _load_module_from_file("quadrature_data_mod", SOLVER_DIR / "quadrature_data.py")

Optimizer = _OPT.Optimizer
adv_diff_dyn = _FWD.adv_diff_dyn
adv_diffusion_adjoint = _ADJ.adv_diffusion_adjoint
jacobian = _LIN.jacobian
jacobian_det = _LIN.jacobian_det
shape_functions = _LIN.shape_functions
interval_quadrature_0_to_1 = _QUAD.interval_quadrature_0_to_1
triangle_quadrature = _QUAD.triangle_quadrature


def source_from_control(z: np.ndarray, sigma2: float = 0.005, amplitude: float = 1.0):
    xc = float(z[0])
    yc = float(z[1])

    def source(x: float, y: float) -> float:
        r2 = (x - xc) ** 2 + (y - yc) ** 2
        return float(amplitude * np.exp(-r2 / sigma2))

    return source


def source_derivatives_from_control(
    z: np.ndarray, sigma2: float = 0.005, amplitude: float = 1.0
):
    xc = float(z[0])
    yc = float(z[1])

    def dsdx0(x: float, y: float) -> float:
        r2 = (x - xc) ** 2 + (y - yc) ** 2
        base = amplitude * np.exp(-r2 / sigma2)
        return float(base * (2.0 * (x - xc) / sigma2))

    def dsdy0(x: float, y: float) -> float:
        r2 = (x - xc) ** 2 + (y - yc) ** 2
        base = amplitude * np.exp(-r2 / sigma2)
        return float(base * (2.0 * (y - yc) / sigma2))

    return dsdx0, dsdy0


def run_forward(mesh_data, pde: dict, z: np.ndarray) -> np.ndarray:
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)
    connections = np.asarray(mesh_data["connections"], dtype=int)
    return adv_diff_dyn(
        npoin=coord.shape[0],
        nelem=connections.shape[0],
        ngauss=pde["ngauss"],
        lx=pde["lx"],
        ly=pde["ly"],
        k=pde["k"],
        vx=pde["vx"],
        vy=pde["vy"],
        dt=pde["dt"],
        tfinal=pde["tfinal"],
        jprint=pde["jprint"],
        coord=coord,
        connections=connections,
        source_func=source_from_control(z, sigma2=pde["sigma2"], amplitude=pde["amp"]),
        mesh_data=mesh_data,
        dirichlet_tags=["left"],
    )


def assemble_load_vector(npoin, nelem, coord, connections, ngauss, scalar_func):
    weight, xgauss, ygauss = triangle_quadrature(ngauss)
    b = np.zeros(npoin, dtype=np.float64)

    for ielem in range(nelem):
        elem = connections[ielem]
        xcoord = coord[elem, 0]
        ycoord = coord[elem, 1]
        jac = jacobian(xcoord, ycoord)
        det_jac = jacobian_det(jac)

        blocal = np.zeros(3, dtype=np.float64)
        for j in range(3):
            for g in range(ngauss):
                nsf = shape_functions(xgauss[g], ygauss[g])
                x = float(np.dot(nsf, xcoord))
                y = float(np.dot(nsf, ycoord))
                blocal[j] += scalar_func(x, y) * nsf[j] * weight[g] * det_jac

        for j in range(3):
            b[elem[j]] += blocal[j]

    return b


def assemble_lumped_mass(npoin, nelem, coord, connections, ngauss):
    weight, xgauss, ygauss = triangle_quadrature(ngauss)
    mass_lumped = np.zeros(npoin, dtype=np.float64)

    for ielem in range(nelem):
        elem = connections[ielem]
        xcoord = coord[elem, 0]
        ycoord = coord[elem, 1]
        jac = jacobian(xcoord, ycoord)
        det_jac = jacobian_det(jac)

        mlocal = np.zeros((3, 3), dtype=np.float64)
        for j in range(3):
            for i in range(3):
                for g in range(ngauss):
                    nsf = shape_functions(xgauss[g], ygauss[g])
                    mlocal[j, i] += nsf[j] * nsf[i] * weight[g] * det_jac

        for j in range(3):
            jglob = elem[j]
            for i in range(3):
                mass_lumped[jglob] += mlocal[j, i]

    return mass_lumped


def compute_objective(
    state_out: np.ndarray,
    ud_out: np.ndarray,
    mesh_data,
    eta1: float,
    lam: float,
    z: np.ndarray,
    dt_out: float,
) -> float:
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)
    edges = np.asarray(
        mesh_data["boundary_edges_by_name"][mesh_data.get("measurement_boundary", "right")],
        dtype=int,
    )
    st_weight, st_xgauss = interval_quadrature_0_to_1(3)

    track = 0.0
    for n in range(1, state_out.shape[0]):
        for edge in edges:
            a, b = int(edge[0]), int(edge[1])
            xa, ya = coord[a, 0], coord[a, 1]
            xb, yb = coord[b, 0], coord[b, 1]
            edge_len = float(np.hypot(xb - xa, yb - ya))

            for g, s in enumerate(st_xgauss):
                n1 = 1.0 - s
                n2 = s
                u_q = float(n1 * state_out[n, a] + n2 * state_out[n, b])
                ud_q = float(n1 * ud_out[n, a] + n2 * ud_out[n, b])
                mismatch = u_q - ud_q
                track += 0.5 * eta1 * (mismatch * mismatch) * st_weight[g] * edge_len

    j_tracking = dt_out * track
    j_reg = 0.5 * lam * float(np.dot(z, z))
    return float(j_tracking + j_reg)


def compute_gradient(
    z: np.ndarray,
    state_out: np.ndarray,
    ud_out: np.ndarray,
    mesh_data,
    pde: dict,
    eta1: float,
    lam: float,
) -> np.ndarray:
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)
    connections = np.asarray(mesh_data["connections"], dtype=int)

    p_out = adv_diffusion_adjoint(
        npoin=coord.shape[0],
        nelem=connections.shape[0],
        ngauss=pde["ngauss"],
        lx=pde["lx"],
        ly=pde["ly"],
        k=pde["k"],
        vx=pde["vx"],
        vy=pde["vy"],
        dt=pde["dt"],
        tfinal=pde["tfinal"],
        jprint=pde["jprint"],
        coord=coord,
        connections=connections,
        state_out=state_out,
        ud=ud_out,
        eta1=eta1,
        st_ngauss=3,
        mesh_data=mesh_data,
        dirichlet_tags=["left"],
        measurement_boundary=mesh_data.get("measurement_boundary", "right"),
    )

    dsdx0, dsdy0 = source_derivatives_from_control(
        z, sigma2=pde["sigma2"], amplitude=pde["amp"]
    )
    dbdz0 = assemble_load_vector(
        npoin=coord.shape[0],
        nelem=connections.shape[0],
        coord=coord,
        connections=connections,
        ngauss=pde["ngauss"],
        scalar_func=dsdx0,
    )
    dbdz1 = assemble_load_vector(
        npoin=coord.shape[0],
        nelem=connections.shape[0],
        coord=coord,
        connections=connections,
        ngauss=pde["ngauss"],
        scalar_func=dsdy0,
    )

    mass_lumped = assemble_lumped_mass(
        npoin=coord.shape[0],
        nelem=connections.shape[0],
        coord=coord,
        connections=connections,
        ngauss=pde["ngauss"],
    )
    dirichlet_nodes = np.asarray(
        mesh_data["boundary_nodes_by_name"].get("left", np.array([], dtype=int)),
        dtype=int,
    )
    mass_lumped[dirichlet_nodes] = 1.0
    inv_mass = 1.0 / mass_lumped

    dt_out = pde["dt"] * pde["jprint"]
    s0 = np.einsum("ni,i->n", p_out, inv_mass * dbdz0)
    s1 = np.einsum("ni,i->n", p_out, inv_mass * dbdz1)

    # Keep the exact discrete convention validated by gradient diagnostics.
    g_adj = np.array(
        [dt_out * float(np.sum(s0[1:])), dt_out * float(np.sum(s1[1:]))],
        dtype=np.float64,
    )
    g_reg = lam * np.asarray(z, dtype=np.float64)
    return g_reg + g_adj


def compute_gradient_components(
    z: np.ndarray,
    state_out: np.ndarray,
    ud_out: np.ndarray,
    mesh_data,
    pde: dict,
    eta1: float,
    lam: float,
):
    g_total = compute_gradient(
        z=z,
        state_out=state_out,
        ud_out=ud_out,
        mesh_data=mesh_data,
        pde=pde,
        eta1=eta1,
        lam=lam,
    )
    g_reg = lam * np.asarray(z, dtype=np.float64)
    g_adj = g_total - g_reg
    return g_reg, g_adj, g_total


def main():
    # Conservative recovery-oriented tuning for manufactured-data inversion.
    # Keep the same model structure; only strengthen right-boundary information
    # modestly so the optimizer can recover z_ref from a distinct z0.
    pde = {
        "lx": 1.0,
        "ly": 0.4,
        "k": 0.1,
        "vx": 0.2,  
        "vy": 0.0,
        "dt": 1.0e-4,
        "tfinal": 0.3,
        "jprint": 1,
        "ngauss": 3,
        "sigma2": 0.005,
        "amp": 1.0,
    }
    eta1 = 1.0e3
    lam = 1.0e-7

    z_ref = np.array([0.73, 0.32], dtype=np.float64)
    z0 = np.array([0.6, 0.4], dtype=np.float64)
    z_zero = np.array([0.0, 0.0], dtype=np.float64)
    margin = 0.0
    bounds = [
        (margin, pde["lx"] - margin),
        (margin, pde["ly"] - margin),
    ]

    def project_to_bounds(z: np.ndarray, bounds_, name: str) -> np.ndarray:
        z_in = np.asarray(z, dtype=np.float64).reshape(-1)
        z_out = z_in.copy()
        for i, (lower, upper) in enumerate(bounds_):
            z_out[i] = np.clip(z_out[i], lower, upper)
        if not np.allclose(z_in, z_out):
            print(f"[setup] {name} projected into bounds: {z_in} -> {z_out}")
        return z_out

    z_ref = project_to_bounds(z_ref, bounds, "z_ref")
    z0 = project_to_bounds(z0, bounds, "z0")
    z_zero = project_to_bounds(z_zero, bounds, "z_zero")

    mesh_data = build_mesh_data()
    dt_out = pde["dt"] * pde["jprint"]

    print("Building desired boundary data from reference control...")
    ud_out = run_forward(mesh_data, pde, z_ref)

    problem_data = {
        "pde": pde,
        "eta1": eta1,
        "lam": lam,
        "ud_out": ud_out,
        "dt_out": dt_out,
    }

    def objective_function(control, mesh_data_, problem_data_):
        state_out = run_forward(mesh_data_, problem_data_["pde"], np.asarray(control, dtype=np.float64))
        return compute_objective(
            state_out=state_out,
            ud_out=problem_data_["ud_out"],
            mesh_data=mesh_data_,
            eta1=problem_data_["eta1"],
            lam=problem_data_["lam"],
            z=np.asarray(control, dtype=np.float64),
            dt_out=problem_data_["dt_out"],
        )

    def gradient_function(control, mesh_data_, problem_data_):
        z = np.asarray(control, dtype=np.float64)
        state_out = run_forward(mesh_data_, problem_data_["pde"], z)
        return compute_gradient(
            z=z,
            state_out=state_out,
            ud_out=problem_data_["ud_out"],
            mesh_data=mesh_data_,
            pde=problem_data_["pde"],
            eta1=problem_data_["eta1"],
            lam=problem_data_["lam"],
        )

    # Application-level iteration diagnostics.
    history_iter = []
    history_control = []
    history_objective = []
    history_grad_norm = []

    def history_callback(xk, *args):
        try:
            if hasattr(xk, "x"):
                control = np.asarray(xk.x, dtype=np.float64).reshape(-1)
            else:
                control = np.asarray(xk, dtype=np.float64).reshape(-1)
        except Exception as exc:
            print(f"[callback] Skipping update: unable to parse callback argument ({exc})")
            return

        jval = objective_function(control, mesh_data, problem_data)
        grad = gradient_function(control, mesh_data, problem_data)
        gnorm = float(np.linalg.norm(grad))

        k = len(history_iter) + 1
        history_iter.append(k)
        history_control.append(control.copy())
        history_objective.append(float(jval))
        history_grad_norm.append(gnorm)
        print(f"[iter {k:03d}] J={jval:.12e}  ||g||={gnorm:.6e}  z={control}")

    j_ref = objective_function(z_ref, mesh_data, problem_data)
    j0 = objective_function(z0, mesh_data, problem_data)
    j_zero = objective_function(z_zero, mesh_data, problem_data)
    g0 = gradient_function(z0, mesh_data, problem_data)
    state0 = run_forward(mesh_data, pde, z0)
    g_reg0, g_adj0, g_total0 = compute_gradient_components(
        z=z0,
        state_out=state0,
        ud_out=ud_out,
        mesh_data=mesh_data,
        pde=pde,
        eta1=eta1,
        lam=lam,
    )
    dist0_ref = float(np.linalg.norm(z0 - z_ref))

    print("\n=== Test Application Optimization Setup ===")
    print("method: L-BFGS-B")
    print(f"z_ref: {z_ref}")
    print(f"z0: {z0}")
    print(f"z_zero: {z_zero}")
    print(f"bounds: {bounds}")
    print(f"J(z_ref): {j_ref:.12e}")
    print(f"initial objective J(z0): {j0:.12e}")
    print(f"J(z_zero): {j_zero:.12e}")
    print(f"initial gradient norm ||g(z0)||: {np.linalg.norm(g0):.6e}")
    print(f"g_reg(z0): {g_reg0}  ||g_reg||={np.linalg.norm(g_reg0):.6e}")
    print(f"g_adj(z0): {g_adj0}  ||g_adj||={np.linalg.norm(g_adj0):.6e}")
    print(f"g_total(z0): {g_total0}  ||g_total||={np.linalg.norm(g_total0):.6e}")
    print(f"||z0 - z_ref||: {dist0_ref:.6e}")

    optimizer = Optimizer(
        objective_function=objective_function,
        gradient_function=gradient_function,
        optimization_method="L-BFGS-B",
        max_iter=60,
        tol=1.0e-8,
        bounds=bounds,
        callback=history_callback,
        options={"disp": True, "maxls": 100, "ftol": 1.0e-20, "gtol": 1.0e-12},
    )

    result = optimizer.optimize(
        initial_guess=z0,
        mesh_data=mesh_data,
        problem_data=problem_data,
    )

    z_opt = np.asarray(result.x, dtype=np.float64).reshape(-1)
    j_opt = float(result.fun)
    final_state = run_forward(mesh_data, pde, z_opt)
    dist_opt_ref = float(np.linalg.norm(z_opt - z_ref))
    closer_than_start = dist_opt_ref < dist0_ref

    print("\n=== Optimization Summary ===")
    print(f"method: {result.get('method', 'L-BFGS-B') if isinstance(result, dict) else 'L-BFGS-B'}")
    print(f"initial guess z0: {z0}")
    print(f"reference control z_ref: {z_ref}")
    print(f"bounds: {bounds}")
    print(f"initial objective: {j0:.12e}")
    print(f"final control: {z_opt}")
    print(f"final objective: {j_opt:.12e}")
    print(f"J(z_ref): {j_ref:.12e}")
    print(f"J(z0): {j0:.12e}")
    print(f"J(z_zero): {j_zero:.12e}")
    print(f"||z0 - z_ref||: {dist0_ref:.6e}")
    print(f"||z_opt - z_ref||: {dist_opt_ref:.6e}")
    print(f"closer_to_z_ref_than_start: {closer_than_start}")
    print(f"success: {result.success}")
    print(f"iterations: {result.nit}")
    print(f"message: {result.message}")

    if history_iter:
        print("\n=== Recorded Iteration History ===")
        for i, zhist, jhist, ghist in zip(
            history_iter, history_control, history_objective, history_grad_norm
        ):
            print(f"k={i:03d}  J={jhist:.12e}  ||g||={ghist:.6e}  z={zhist}")

    summary_path = APP_DIR / "optimization_summary.txt"
    history_path = APP_DIR / "optimization_history.npz"
    state_path = APP_DIR / "optimized_state.npy"

    summary_lines = [
        "Test Application Optimization Summary",
        f"method: L-BFGS-B",
        f"z0: {z0.tolist()}",
        f"z_ref: {z_ref.tolist()}",
        f"bounds: {bounds}",
        f"j_z_ref: {j_ref:.16e}",
        f"initial_objective: {j0:.16e}",
        f"j_z_zero: {j_zero:.16e}",
        f"final_control: {z_opt.tolist()}",
        f"final_objective: {j_opt:.16e}",
        f"dist_z0_zref: {dist0_ref:.16e}",
        f"dist_zopt_zref: {dist_opt_ref:.16e}",
        f"closer_to_z_ref_than_start: {closer_than_start}",
        f"success: {bool(result.success)}",
        f"iterations: {int(result.nit)}",
        f"message: {str(result.message)}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    np.savez(
        history_path,
        z0=z0,
        z_ref=z_ref,
        z_opt=z_opt,
        initial_objective=np.array([j0], dtype=np.float64),
        final_objective=np.array([j_opt], dtype=np.float64),
        iter=np.asarray(history_iter, dtype=np.int64),
        control_history=np.asarray(history_control, dtype=np.float64),
        objective_history=np.asarray(history_objective, dtype=np.float64),
        grad_norm_history=np.asarray(history_grad_norm, dtype=np.float64),
    )
    np.save(state_path, final_state)

    print(f"Saved summary: {summary_path}")
    print(f"Saved history: {history_path}")
    print(f"Saved final state: {state_path}")


if __name__ == "__main__":
    main()
