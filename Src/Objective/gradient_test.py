"""Shared reduced objective, forward solve and discrete-adjoint gradient.

Used by both production applications (imported as ``import gradient_test as gt``):
``run_forward``, ``compute_objective(_parts)``, ``compute_gradient(_parts)`` and
the RK4 discrete adjoint re-exported from ``Src/Solvers``.  The module name is
historical (it began as a Taylor-test script) and is kept so that the saved
experiment drivers and the solve counters, which patch ``gt.run_forward`` /
``gt.rk4_discrete_source_adjoint`` on this module object, work unchanged.

``mesh_data_config`` and ``multi_source_utils`` are imported by bare name and
resolve to the CALLING application's modules: Application1's
``run_optimization`` puts its own directory first on ``sys.path`` and then
appends ``Src/Objective``; Application2 loads Application1 the same way
(``transport_model._App1``).  Import this module only through an application.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    plt = None


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]
SOLVER_DIR = PROJECT_ROOT / "Src" / "Solvers"

from mesh_data_config import build_mesh_data
from multi_source_utils import (
    NUM_SOURCES,
    build_initial_control,
    build_reference_control,
    component_names,
    full_bounds_array,
    source_derivatives_from_control,
    source_from_control,
)


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_FWD = _load_module_from_file("adv_diff_forward", SOLVER_DIR / "adv_diff_forward.py")
_ADJ = _load_module_from_file("adv_diffusion_adjoint", SOLVER_DIR / "adv_diffusion_adjoint.py")
_LIN = _load_module_from_file("linear_fe_data", SOLVER_DIR / "linear_fe_data.py")
_QUAD = _load_module_from_file("quadrature_data", SOLVER_DIR / "quadrature_data.py")

adv_diff_dyn = _FWD.adv_diff_dyn
adv_diffusion_adjoint = _ADJ.adv_diffusion_adjoint
rk4_discrete_source_adjoint = _ADJ.rk4_discrete_source_adjoint
jacobian = _LIN.jacobian
jacobian_det = _LIN.jacobian_det
shape_functions = _LIN.shape_functions
interval_quadrature_0_to_1 = _QUAD.interval_quadrature_0_to_1
triangle_quadrature = _QUAD.triangle_quadrature


def run_forward(mesh_data, pde, z: np.ndarray, source_func=None) -> np.ndarray:
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)
    connections = np.asarray(mesh_data["connections"], dtype=int)
    if source_func is None:
        source_func = source_from_control(z, r2=pde["sigma2"])
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
        source_func=source_func,
        mesh_data=mesh_data,
        dirichlet_tags=["left"],
        velocity_nodes=pde.get("velocity_nodes"),
    )


def assemble_load_vector(npoin, nelem, coord, connections, ngauss, scalar_func):
    weight, xgauss, ygauss = triangle_quadrature(ngauss)
    b = np.zeros(npoin, dtype=np.float64)

    for ielem in range(nelem):
        elem = connections[ielem]
        xcoord = coord[elem, 0]
        ycoord = coord[elem, 1]
        J = jacobian(xcoord, ycoord)
        detJ = jacobian_det(J)

        blocal = np.zeros(3, dtype=np.float64)
        for j in range(3):
            for g in range(ngauss):
                nsf = shape_functions(xgauss[g], ygauss[g])
                x = float(np.dot(nsf, xcoord))
                y = float(np.dot(nsf, ycoord))
                blocal[j] += scalar_func(x, y) * nsf[j] * weight[g] * detJ

        for j in range(3):
            b[elem[j]] += blocal[j]

    return b


def assemble_lumped_mass(npoin, nelem, coord, connections, ngauss):
    weight, xgauss, ygauss = triangle_quadrature(ngauss)
    m_lumped = np.zeros(npoin, dtype=np.float64)

    for ielem in range(nelem):
        elem = connections[ielem]
        xcoord = coord[elem, 0]
        ycoord = coord[elem, 1]
        J = jacobian(xcoord, ycoord)
        detJ = jacobian_det(J)

        mlocal = np.zeros((3, 3), dtype=np.float64)
        for j in range(3):
            for i in range(3):
                for g in range(ngauss):
                    nsf = shape_functions(xgauss[g], ygauss[g])
                    mlocal[j, i] += nsf[j] * nsf[i] * weight[g] * detJ

        for j in range(3):
            jglob = elem[j]
            for i in range(3):
                m_lumped[jglob] += mlocal[j, i]

    return m_lumped


def _time_reduce(series: np.ndarray, dt_out: float, rule: str) -> float:
    if rule == "all":
        return dt_out * float(np.sum(series))
    if rule == "exclude_first":
        return dt_out * float(np.sum(series[1:]))
    if rule == "exclude_last":
        return dt_out * float(np.sum(series[:-1]))
    if rule == "trapezoid":
        return dt_out * float(0.5 * series[0] + np.sum(series[1:-1]) + 0.5 * series[-1])
    raise ValueError(f"Unknown time reduction rule: {rule}")


def observation_operator(mesh_data):
    """Return the non-boundary observation operator wired into ``mesh_data``.

    ``None`` means the historical boundary observation, whose implementation is
    left exactly as it was.  A node-sensor operator (see
    ``Src/Solvers/observation_operator.py``) is threaded through
    ``mesh_data["observation"]`` so the objective, the RK4 discrete adjoint and
    the initializer all consume the same operator without new call signatures.
    """
    if not isinstance(mesh_data, dict):
        return None
    return mesh_data.get("observation")


def compute_objective_parts(
    state_out,
    ud_out,
    mesh_data,
    eta1: float,
    lam: float,
    z: np.ndarray,
    dt_out: float,
    time_rule: str,
):
    observation = observation_operator(mesh_data)
    if observation is not None:
        # J = 0.5 * eta1 * T_reduce_n( || C U^n - d^n ||_W^2 ), i.e. the SAME
        # temporal quadrature convention as the boundary branch below.
        tracking_series = observation.objective_series(state_out, ud_out)
        j_tracking = 0.5 * eta1 * _time_reduce(tracking_series, dt_out, time_rule)
        j_reg = 0.5 * lam * float(np.dot(z, z))
        return j_tracking, j_reg, j_tracking + j_reg

    coord = np.asarray(mesh_data["coord"], dtype=np.float64)
    edges = np.asarray(
        mesh_data["boundary_edges_by_name"][mesh_data.get("measurement_boundary", "right")],
        dtype=int,
    )
    st_weight, st_xgauss = interval_quadrature_0_to_1(3)

    tracking_series = np.zeros(state_out.shape[0], dtype=np.float64)
    for n in range(state_out.shape[0]):
        track_n = 0.0
        for edge in edges:
            a, b = int(edge[0]), int(edge[1])
            xa, ya = coord[a, 0], coord[a, 1]
            xb, yb = coord[b, 0], coord[b, 1]
            elen = float(np.hypot(xb - xa, yb - ya))

            for g, s in enumerate(st_xgauss):
                n1 = 1.0 - s
                n2 = s
                u_q = float(n1 * state_out[n, a] + n2 * state_out[n, b])
                ud_q = float(n1 * ud_out[n, a] + n2 * ud_out[n, b])
                mis = u_q - ud_q
                track_n += (mis * mis) * st_weight[g] * elen
        tracking_series[n] = track_n

    j_tracking = 0.5 * eta1 * _time_reduce(tracking_series, dt_out, time_rule)
    j_reg = 0.5 * lam * float(np.dot(z, z))
    return j_tracking, j_reg, j_tracking + j_reg


def compute_objective(
    state_out,
    ud_out,
    mesh_data,
    eta1: float,
    lam: float,
    z: np.ndarray,
    dt_out: float,
    time_rule: str,
):
    _, _, j_total = compute_objective_parts(
        state_out, ud_out, mesh_data, eta1, lam, z, dt_out, time_rule
    )
    return j_total


def compute_gradient_parts(
    z,
    state_out,
    ud_out,
    mesh_data,
    pde,
    eta1: float,
    lam: float,
    gradient_time_rule: str,
):
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)
    connections = np.asarray(mesh_data["connections"], dtype=int)

    source_adjoint = rk4_discrete_source_adjoint(
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
        velocity_nodes=pde.get("velocity_nodes"),
        objective_time_rule=pde.get("objective_time_rule", "exclude_first"),
    )

    derivative_specs = source_derivatives_from_control(z, r2=pde["sigma2"])
    load_vectors = [
        assemble_load_vector(
            npoin=coord.shape[0],
            nelem=connections.shape[0],
            coord=coord,
            connections=connections,
            ngauss=pde["ngauss"],
            scalar_func=spec["func"],
        )
        for spec in derivative_specs
    ]

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
    g_adj_eff = []
    g_adj_raw = []

    for db in load_vectors:
        db_eff = inv_mass * db
        g_adj_eff.append(float(np.dot(source_adjoint, db_eff)))
        g_adj_raw.append(float(np.dot(source_adjoint, db)))

    g_reg = lam * np.asarray(z, dtype=np.float64)
    g_adj_eff = np.asarray(g_adj_eff, dtype=np.float64)
    g_adj_raw = np.asarray(g_adj_raw, dtype=np.float64)
    g_total = g_reg + g_adj_eff
    return g_reg, g_adj_eff, g_total, g_adj_raw


def compute_gradient(
    z,
    state_out,
    ud_out,
    mesh_data,
    pde,
    eta1: float,
    lam: float,
    gradient_time_rule: str,
):
    _, _, g_total, _ = compute_gradient_parts(
        z, state_out, ud_out, mesh_data, pde, eta1, lam, gradient_time_rule
    )
    return g_total


def component_fd_check(mesh_data, pde, eta1, lam, z0, ud_out, grad0, dt_out, objective_time_rule):
    eps_list = [1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7]
    print("\n=== Componentwise Central FD Check ===")
    for j, name in enumerate(component_names()):
        print(f"component: {name} (index {j})")
        for eps in eps_list:
            ej = np.zeros_like(z0)
            ej[j] = 1.0
            zp = z0 + eps * ej
            zm = z0 - eps * ej
            jp = compute_objective(
                run_forward(mesh_data, pde, zp),
                ud_out,
                mesh_data,
                eta1=eta1,
                lam=lam,
                z=zp,
                dt_out=dt_out,
                time_rule=objective_time_rule,
            )
            jm = compute_objective(
                run_forward(mesh_data, pde, zm),
                ud_out,
                mesh_data,
                eta1=eta1,
                lam=lam,
                z=zm,
                dt_out=dt_out,
                time_rule=objective_time_rule,
            )
            fd = (jp - jm) / (2.0 * eps)
            adj = float(grad0[j])
            diff = adj - fd
            rel = abs(diff) / max(1.0e-14, abs(fd), abs(adj))
            print(
                f"  eps={eps:.1e} adj={adj:.16e} fd={fd:.16e} "
                f"abs_diff={abs(diff):.16e} rel_diff={rel:.16e}"
            )


def directional_slope_check(
    mesh_data,
    pde,
    eta1,
    lam,
    z0,
    d,
    hs,
    j0,
    gdotd,
    dt_out,
    ud_out,
    objective_time_rule,
):
    print("\n=== One-Sided Directional Slope Check ===")
    print(f"grad dot d = {gdotd:.16e}")
    for h in hs:
        zh = z0 + h * d
        jh = compute_objective(
            run_forward(mesh_data, pde, zh),
            ud_out,
            mesh_data,
            eta1=eta1,
            lam=lam,
            z=zh,
            dt_out=dt_out,
            time_rule=objective_time_rule,
        )
        slope = (jh - j0) / h
        print(f"  h={h:.1e} slope={slope:.16e} grad_dot_d={gdotd:.16e} diff={slope-gdotd:.16e}")


def _max_admissible_step(z0: np.ndarray, d: np.ndarray, bounds: np.ndarray) -> float:
    h_max = np.inf
    for j in range(z0.size):
        if d[j] > 0.0:
            h_max = min(h_max, (bounds[j, 1] - z0[j]) / d[j])
        elif d[j] < 0.0:
            h_max = min(h_max, (bounds[j, 0] - z0[j]) / d[j])
    return float(h_max)


def _taylor_steps(z0: np.ndarray, d: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    h_max = _max_admissible_step(z0, d, bounds)
    h0 = min(1.0e-1, 0.5 * h_max)
    if h0 <= 0.0:
        raise ValueError("No admissible positive step available for Taylor test direction.")
    return np.array([h0 * (10.0 ** (-k)) for k in range(6)], dtype=np.float64)


def taylor_gradient_test(
    mesh_data,
    pde,
    eta1: float,
    lam: float,
    objective_time_rule: str,
    gradient_time_rule: str,
):
    bounds = full_bounds_array()
    z_ref = build_reference_control()
    z0 = build_initial_control()

    rng = np.random.default_rng(7)
    d = rng.normal(size=z0.size)
    d /= np.linalg.norm(d)
    hs = _taylor_steps(z0, d, bounds)

    dt_out = pde["dt"] * pde["jprint"]

    ud_out = run_forward(mesh_data, pde, z_ref)
    u0 = run_forward(mesh_data, pde, z0)

    j_track0, j_reg0, j0 = compute_objective_parts(
        u0,
        ud_out,
        mesh_data,
        eta1=eta1,
        lam=lam,
        z=z0,
        dt_out=dt_out,
        time_rule=objective_time_rule,
    )
    g_reg0, g_adj0, grad0, g_adj_raw0 = compute_gradient_parts(
        z0, u0, ud_out, mesh_data, pde, eta1=eta1, lam=lam, gradient_time_rule=gradient_time_rule
    )
    gdotd = float(np.dot(grad0, d))
    gdotd_raw = float(np.dot(g_adj_raw0, d))

    h_slope = min(1.0e-6, 0.1 * hs[-2])
    z_slope = z0 + h_slope * d
    u_slope = run_forward(mesh_data, pde, z_slope)
    j_slope = compute_objective(
        u_slope,
        ud_out,
        mesh_data,
        eta1=eta1,
        lam=lam,
        z=z_slope,
        dt_out=dt_out,
        time_rule=objective_time_rule,
    )
    fd_slope = (j_slope - j0) / h_slope

    print(f"=== Multi-Source Gradient Test (NUM_SOURCES={NUM_SOURCES}) ===")
    print(f"z_ref: {np.array2string(z_ref, precision=16)}")
    print(f"z0: {np.array2string(z0, precision=16)}")
    print(f"objective_time_rule: {objective_time_rule}")
    print(f"gradient_time_rule:  {gradient_time_rule}")
    print(f"component_names: {component_names()}")
    print(f"bounds: {bounds.tolist()}")
    print(f"direction d (normalized): {np.array2string(d, precision=16)}")
    print(f"Taylor step sizes: {np.array2string(hs, precision=6)}")
    print(f"J_tracking(z0): {j_track0:.16e}")
    print(f"J_reg(z0):      {j_reg0:.16e}")
    print(f"J_total(z0):    {j0:.16e}")
    print(f"grad_reg(z0):   {np.array2string(g_reg0, precision=16)}")
    print(f"grad_adj(z0):   {np.array2string(g_adj0, precision=16)}")
    print(f"grad_adj_raw(z0): {np.array2string(g_adj_raw0, precision=16)}")
    print(f"grad_total(z0): {np.array2string(grad0, precision=16)}")
    print(f"||grad(z0)||_2: {np.linalg.norm(grad0):.16e}")
    print(f"grad_adj dot d: {float(np.dot(g_adj0, d)):.16e}")
    print(f"grad_adj_raw dot d: {gdotd_raw:.16e}")
    print(f"FD slope h={h_slope:.1e}: {fd_slope:.16e}")

    component_fd_check(mesh_data, pde, eta1, lam, z0, ud_out, g_adj0, dt_out, objective_time_rule)
    print("\n--- Componentwise FD Check (raw pairing) ---")
    component_fd_check(mesh_data, pde, eta1, lam, z0, ud_out, g_adj_raw0, dt_out, objective_time_rule)
    directional_slope_check(
        mesh_data, pde, eta1, lam, z0, d, hs[-4:], j0, gdotd, dt_out, ud_out, objective_time_rule
    )
    print("\n--- Directional Slope Check (raw pairing reference) ---")
    directional_slope_check(
        mesh_data, pde, eta1, lam, z0, d, hs[-4:], j0, gdotd_raw, dt_out, ud_out, objective_time_rule
    )

    print("\n=== Taylor Remainder Table ===")
    print(f"{'h':>10} {'J(z0+h d)':>22} {'R(h)':>18} {'rate':>12}")

    prev_h = None
    prev_r = None
    rates = []
    remainders = []
    for h in hs:
        zh = z0 + h * d
        inside = np.all((zh >= bounds[:, 0]) & (zh <= bounds[:, 1]))
        if not inside:
            raise ValueError(f"z0 + h*d left bounds at h={h}: {zh}")

        uh = run_forward(mesh_data, pde, zh)
        jh = compute_objective(
            uh,
            ud_out,
            mesh_data,
            eta1=eta1,
            lam=lam,
            z=zh,
            dt_out=dt_out,
            time_rule=objective_time_rule,
        )
        rem = abs(jh - j0 - h * gdotd)
        remainders.append(rem)

        if prev_h is None:
            rate_str = "-"
        else:
            rate = np.log(prev_r / rem) / np.log(prev_h / h)
            rates.append(rate)
            rate_str = f"{rate:.6f}"

        print(f"{h:10.1e} {jh:22.14e} {rem:18.10e} {rate_str:>12}")
        prev_h = h
        prev_r = rem

    mean_rate = float(np.mean(rates)) if rates else float("nan")
    print(f"\nmean observed rate: {mean_rate:.6f}")
    best_window_mean = -1.0
    best_window = None
    if len(rates) >= 3:
        for i in range(len(rates) - 2):
            wmean = float(np.mean(rates[i:i + 3]))
            if wmean > best_window_mean:
                best_window_mean = wmean
                best_window = (i, i + 2)
    if best_window is not None:
        print(f"best 3-rate window mean: {best_window_mean:.6f} (rate[{best_window[0]}]..rate[{best_window[1]}])")

    if best_window_mean >= 1.8:
        print("Gradient check verdict: PASS (quadratic regime observed before floor)")
    elif mean_rate > 1.5:
        print("Gradient check verdict: PASS WITH FLOOR (near-quadratic with small-h floor)")
    else:
        print("Gradient check verdict: FAIL")

    remainders = np.asarray(remainders, dtype=np.float64)
    if plt is None:
        print("saved plot: skipped (matplotlib not installed)")
    else:
        plot_path = Path.cwd() / "gradient_test_remainder_loglog.png"
        plt.figure(figsize=(6.0, 4.2))
        plt.loglog(hs, remainders, "o-", label="Taylor remainder")
        ref = remainders[0] * (hs / hs[0]) ** 2
        plt.loglog(hs, ref, "--", label="O(h^2) reference")
        plt.xlabel("h")
        plt.ylabel("|J(z+h d) - J(z) - h grad(z)·d|")
        plt.title(f"{NUM_SOURCES}-Source Gradient Taylor Test")
        plt.grid(True, which="both", ls=":")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)
        plt.close()
        print(f"saved plot: {plot_path}")

