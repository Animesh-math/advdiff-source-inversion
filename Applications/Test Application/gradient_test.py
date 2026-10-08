"""Standalone adjoint-gradient diagnostics and Taylor test for Test Application.

This script validates the reduced gradient before any optimizer is built.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
SOLVER_DIR = PROJECT_ROOT / "Src" / "Solvers"

if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from mesh_data_config import build_mesh_data


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


def source_derivatives_from_control(z: np.ndarray, sigma2: float = 0.005, amplitude: float = 1.0):
    """For S = A exp(-((x-xc)^2+(y-yc)^2)/sigma2):
    dS/dxc = S * 2(x-xc)/sigma2, dS/dyc = S * 2(y-yc)/sigma2.
    """
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


def run_forward(mesh_data, pde, z: np.ndarray) -> np.ndarray:
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
    """Assemble lumped mass used by forward dynamics u_t = M^{-1}(b-Ku)."""
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


def compute_objective(state_out, ud_out, mesh_data, eta1: float, lam: float, z: np.ndarray, dt_out: float):
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)
    edges = np.asarray(
        mesh_data["boundary_edges_by_name"][mesh_data.get("measurement_boundary", "right")],
        dtype=int,
    )
    st_weight, st_xgauss = interval_quadrature_0_to_1(3)

    # Match adjoint forcing time-grid usage: n = 1..N (rhs_all[n] in solver).
    track = 0.0
    nframes = state_out.shape[0]
    for n in range(1, nframes):
        for e in edges:
            a, b = int(e[0]), int(e[1])
            xa, ya = coord[a, 0], coord[a, 1]
            xb, yb = coord[b, 0], coord[b, 1]
            elen = float(np.hypot(xb - xa, yb - ya))

            for g, s in enumerate(st_xgauss):
                n1 = 1.0 - s
                n2 = s
                u_q = float(n1 * state_out[n, a] + n2 * state_out[n, b])
                ud_q = float(n1 * ud_out[n, a] + n2 * ud_out[n, b])
                mis = u_q - ud_q
                track += 0.5 * eta1 * (mis * mis) * st_weight[g] * elen

    j_tracking = dt_out * track
    j_reg = 0.5 * lam * float(np.dot(z, z))
    return j_tracking + j_reg


def _time_reduce(series: np.ndarray, dt_out: float, rule: str) -> float:
    """Time reduction helper for sensitivity diagnostics."""
    if rule == "all":
        return dt_out * float(np.sum(series))
    if rule == "exclude_first":
        return dt_out * float(np.sum(series[1:]))
    if rule == "exclude_last":
        return dt_out * float(np.sum(series[:-1]))
    if rule == "trapezoid":
        return dt_out * float(0.5 * series[0] + np.sum(series[1:-1]) + 0.5 * series[-1])
    raise ValueError(f"Unknown time reduction rule: {rule}")


def compute_gradient_parts(z, state_out, ud_out, mesh_data, pde, eta1: float, lam: float):
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

    dsdx0, dsdy0 = source_derivatives_from_control(z, sigma2=pde["sigma2"], amplitude=pde["amp"])
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
    dirichlet_nodes = np.asarray(mesh_data["boundary_nodes_by_name"].get("left", np.array([], dtype=int)), dtype=int)
    mass_lumped[dirichlet_nodes] = 1.0
    inv_mass = 1.0 / mass_lumped

    dt_out = pde["dt"] * pde["jprint"]
    dbdz0_eff = inv_mass * dbdz0
    dbdz1_eff = inv_mass * dbdz1
    s0 = np.einsum("ni,i->n", p_out, dbdz0_eff)
    s1 = np.einsum("ni,i->n", p_out, dbdz1_eff)

    g_reg = lam * np.asarray(z, dtype=np.float64)

    # With the current discrete adjoint convention in this codebase,
    # the reduced gradient matches FD checks as g = lambda z + int p * db/dz dt.
    g_adj_all = np.array([_time_reduce(s0, dt_out, "all"), _time_reduce(s1, dt_out, "all")])
    g_adj_excl_first = np.array([
        _time_reduce(s0, dt_out, "exclude_first"),
        _time_reduce(s1, dt_out, "exclude_first"),
    ])
    g_adj_excl_last = np.array([
        _time_reduce(s0, dt_out, "exclude_last"),
        _time_reduce(s1, dt_out, "exclude_last"),
    ])
    g_adj_trap = np.array([_time_reduce(s0, dt_out, "trapezoid"), _time_reduce(s1, dt_out, "trapezoid")])

    g_total = g_reg + g_adj_excl_first

    return {
        "p_out": p_out,
        "dbdz0": dbdz0,
        "dbdz1": dbdz1,
        "dbdz0_eff": dbdz0_eff,
        "dbdz1_eff": dbdz1_eff,
        "s0": s0,
        "s1": s1,
        "g_reg": g_reg,
        "g_adj_all": g_adj_all,
        "g_adj_excl_first": g_adj_excl_first,
        "g_adj_excl_last": g_adj_excl_last,
        "g_adj_trap": g_adj_trap,
        "g_total": g_total,
    }


def diagnose_source_derivatives(z: np.ndarray, pde):
    dsdx0, dsdy0 = source_derivatives_from_control(z, sigma2=pde["sigma2"], amplitude=pde["amp"])
    s = source_from_control(z, sigma2=pde["sigma2"], amplitude=pde["amp"])
    h = 1.0e-7

    pts = [(0.2, 0.2), (0.7, 0.2), (0.9, 0.3)]
    print("\n=== Source Derivative Check (pointwise FD) ===")
    for (x, y) in pts:
        zp = np.array([z[0] + h, z[1]])
        zm = np.array([z[0] - h, z[1]])
        fd_x = (source_from_control(zp, pde["sigma2"], pde["amp"])(x, y) - source_from_control(zm, pde["sigma2"], pde["amp"])(x, y)) / (2.0 * h)
        an_x = dsdx0(x, y)

        zp = np.array([z[0], z[1] + h])
        zm = np.array([z[0], z[1] - h])
        fd_y = (source_from_control(zp, pde["sigma2"], pde["amp"])(x, y) - source_from_control(zm, pde["sigma2"], pde["amp"])(x, y)) / (2.0 * h)
        an_y = dsdy0(x, y)

        err_x = abs(an_x - fd_x) / (abs(fd_x) + 1.0e-16)
        err_y = abs(an_y - fd_y) / (abs(fd_y) + 1.0e-16)
        print(f"(x,y)=({x:.3f},{y:.3f}) S={s(x,y):.4e} dS/dx0 an={an_x:.4e} fd={fd_x:.4e} relerr={err_x:.3e}")
        print(f"                      dS/dy0 an={an_y:.4e} fd={fd_y:.4e} relerr={err_y:.3e}")


def fd_directional_derivative(mesh_data, pde, eta1, lam, z, d, ud_out, h_fd=1.0e-4):
    zp = z + h_fd * d
    zm = z - h_fd * d
    jp = compute_objective(run_forward(mesh_data, pde, zp), ud_out, mesh_data, eta1, lam, zp, pde["dt"] * pde["jprint"])
    jm = compute_objective(run_forward(mesh_data, pde, zm), ud_out, mesh_data, eta1, lam, zm, pde["dt"] * pde["jprint"])
    return (jp - jm) / (2.0 * h_fd)


def run_taylor_test(mesh_data, pde, eta1: float, lam: float):
    z_ref = np.array([0.70, 0.20], dtype=np.float64)
    z = np.array([0.90, 0.20], dtype=np.float64)

    d = np.array([1.0, -0.35], dtype=np.float64)
    d /= np.linalg.norm(d)

    ud_out = run_forward(mesh_data, pde, z_ref)
    state_out = run_forward(mesh_data, pde, z)

    dt_out = pde["dt"] * pde["jprint"]
    j0 = compute_objective(state_out, ud_out, mesh_data, eta1=eta1, lam=lam, z=z, dt_out=dt_out)
    parts = compute_gradient_parts(z, state_out, ud_out, mesh_data, pde, eta1=eta1, lam=lam)

    g_reg = parts["g_reg"]
    g_adj = parts["g_adj_excl_first"]
    g_total = parts["g_total"]

    dir_reg = float(np.dot(g_reg, d))
    dir_adj = float(np.dot(g_adj, d))
    dir_total = float(np.dot(g_total, d))

    # Independent FD directional derivative for sign/scaling consistency.
    dir_fd = fd_directional_derivative(mesh_data, pde, eta1, lam, z, d, ud_out, h_fd=1.0e-4)
    dir_plus = float(np.dot(g_reg + g_adj, d))

    print("\n=== Gradient Test Setup ===")
    print(f"Forward solver file: {(SOLVER_DIR / 'adv_diff_forward.py')}")
    print(f"Adjoint solver file: {(SOLVER_DIR / 'adv_diffusion_adjoint.py')}")
    print(f"Base control z: {z}")
    print(f"Reference control z_ref (for u_d): {z_ref}")
    print(f"Direction d (normalized): {d}")
    print(f"J(z): {j0:.16e}")

    print("\n=== Gradient Decomposition ===")
    print(f"g_reg                = {np.array2string(g_reg, precision=16)}")
    print(f"g_adj (all)          = {np.array2string(parts['g_adj_all'], precision=16)}")
    print(f"g_adj (exclude_first)= {np.array2string(parts['g_adj_excl_first'], precision=16)}")
    print(f"g_adj (exclude_last) = {np.array2string(parts['g_adj_excl_last'], precision=16)}")
    print(f"g_adj (trapezoid)    = {np.array2string(parts['g_adj_trap'], precision=16)}")
    print(f"g_total (used)       = {np.array2string(g_total, precision=16)}")

    print(f"dir(g_reg)                = {dir_reg:.16e}")
    print(f"dir(g_adj used)           = {dir_adj:.16e}")
    print(f"dir(g_total used)         = {dir_total:.16e}")
    print(f"dir(g_reg + g_adj used)   = {dir_plus:.16e}")
    print(f"dir(FD central, h=1e-4)   = {dir_fd:.16e}")

    print("\n=== Adjoint/Assembly Diagnostics ===")
    p_out = parts["p_out"]
    mid = p_out.shape[0] // 2
    print(f"p_out shape: {p_out.shape}")
    print(f"||p(t0)||2={np.linalg.norm(p_out[0]):.6e}, ||p(tmid)||2={np.linalg.norm(p_out[mid]):.6e}, ||p(tT)||2={np.linalg.norm(p_out[-1]):.6e}")
    print(f"max|p| over all frames = {np.max(np.abs(p_out)):.6e}")
    print(f"||dbdz0||2={np.linalg.norm(parts['dbdz0']):.6e}, ||dbdz1||2={np.linalg.norm(parts['dbdz1']):.6e}")
    print(f"||M^-1 dbdz0||2={np.linalg.norm(parts['dbdz0_eff']):.6e}, ||M^-1 dbdz1||2={np.linalg.norm(parts['dbdz1_eff']):.6e}")
    print(f"max|dbdz0|={np.max(np.abs(parts['dbdz0'])):.6e}, max|dbdz1|={np.max(np.abs(parts['dbdz1'])):.6e}")
    print(f"||p·dbdz0 over time||2={np.linalg.norm(parts['s0']):.6e}, ||p·dbdz1 over time||2={np.linalg.norm(parts['s1']):.6e}")

    hs = np.array([2e-2, 1e-2, 5e-3, 2.5e-3, 1.25e-3, 6.25e-4, 3.125e-4, 1.5625e-4], dtype=np.float64)

    rows = []
    remainders = []
    for h in hs:
        zh = z + h * d
        uh = run_forward(mesh_data, pde, zh)
        jh = compute_objective(uh, ud_out, mesh_data, eta1=eta1, lam=lam, z=zh, dt_out=dt_out)
        rem = abs(jh - j0 - h * dir_total)
        rows.append((h, jh, rem, rem / h, rem / (h * h)))
        remainders.append(rem)

    remainders = np.asarray(remainders)
    rates = np.log(remainders[:-1] / remainders[1:]) / np.log(hs[:-1] / hs[1:])

    print("\n=== Taylor Remainder Table ===")
    print(f"{'h':>12} {'J(z+h d)':>22} {'remainder':>18} {'remainder/h':>18} {'remainder/h^2':>18}")
    for h, jh, rem, rem_h, rem_h2 in rows:
        print(f"{h:12.4e} {jh:22.14e} {rem:18.10e} {rem_h:18.10e} {rem_h2:18.10e}")

    print("\n=== Observed Rates ===")
    for k, rate in enumerate(rates):
        band = "large->moderate" if k < 3 else "moderate->small"
        print(f"rate[{k}] ({band}) between h={hs[k]:.4e} and h={hs[k+1]:.4e}: {rate:.6f}")

    # Simple window-based assessment.
    finite_rates = rates[np.isfinite(rates)]
    best_mean = -1.0
    best_window = None
    if finite_rates.size >= 3:
        for i in range(0, len(rates) - 2):
            w = rates[i:i+3]
            m = float(np.mean(w))
            if m > best_mean:
                best_mean = m
                best_window = (i, i + 2)

    if best_mean >= 1.8:
        assessment = "PASS"
    elif best_mean >= 1.5:
        assessment = "PASS WITH FLOOR"
    else:
        assessment = "FAIL"

    print("\n=== Assessment ===")
    if best_window is not None:
        i0, i1 = best_window
        print(f"Best 3-rate window: rate[{i0}]..rate[{i1}], mean={best_mean:.6f}")
    print(f"Gradient test assessment: {assessment}")
    print("Smallest h values can be dominated by roundoff/discretization floor.")

    # Descent-direction sanity check.
    print("\n=== Descent Check ===")
    for eps in [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]:
        z_new = z - eps * g_total
        j_new = compute_objective(run_forward(mesh_data, pde, z_new), ud_out, mesh_data, eta1, lam, z_new, dt_out)
        print(f"eps={eps:.1e}  J(z-eps*g)={j_new:.16e}  delta={j_new - j0:.6e}")

    plot_path = APP_DIR / "gradient_test_remainder_loglog.png"
    plt.figure(figsize=(6.0, 4.2))
    plt.loglog(hs, remainders, "o-", label="Taylor remainder")
    ref = remainders[0] * (hs / hs[0]) ** 2
    plt.loglog(hs, ref, "--", label="O(h^2) reference")
    plt.xlabel("h")
    plt.ylabel("|J(z+hd)-J(z)-h g·d|")
    plt.title("Adjoint Gradient Taylor Test")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    plot_path2 = APP_DIR / "gradient_test_remainder_over_h2.png"
    plt.figure(figsize=(6.0, 4.2))
    plt.loglog(hs, remainders / (hs * hs), "o-")
    plt.xlabel("h")
    plt.ylabel("remainder / h^2")
    plt.title("Taylor Remainder Scaled by h^2")
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(plot_path2, dpi=160)
    plt.close()

    print(f"Saved plot: {plot_path}")
    print(f"Saved plot: {plot_path2}")


if __name__ == "__main__":
    pde = {
        "lx": 1.0,
        "ly": 0.4,
        "k": 0.1,
        "vx": 0.1,
        "vy": 0.0,
        "dt": 1.0e-4,
        "tfinal": 0.02,
        "jprint": 1,
        "ngauss": 3,
        "sigma2": 0.005,
        "amp": 1.0,
    }

    # Chosen so adjoint contribution is visible in diagnostics.
    eta1 = 1.0e3
    lam = 1.0e-6

    mesh_data = build_mesh_data()
    diagnose_source_derivatives(np.array([0.90, 0.20]), pde)
    run_taylor_test(mesh_data, pde=pde, eta1=eta1, lam=lam)
