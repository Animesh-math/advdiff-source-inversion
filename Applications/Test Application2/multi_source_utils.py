"""Application-local utilities for multi-source Gaussian controls."""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


NUM_SOURCES = 2

X_BOUNDS = (0.0, 1.0)
Y_BOUNDS = (0.0, 0.4)
A_BOUNDS = (0.0, 2.0)

_BASE_BOUNDS = np.array(
    [
        [X_BOUNDS[0], X_BOUNDS[1]],
        [Y_BOUNDS[0], Y_BOUNDS[1]],
        [A_BOUNDS[0], A_BOUNDS[1]],
    ],
    dtype=np.float64,
)


def control_dim(num_sources: int = NUM_SOURCES) -> int:
    return 3 * int(num_sources)


def reshape_control(z: np.ndarray, num_sources: int = NUM_SOURCES) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    expected = control_dim(num_sources)
    if z.size != expected:
        raise ValueError(f"Expected control of length {expected}, got {z.size}")
    return z.reshape(int(num_sources), 3)


def flatten_control(Z: np.ndarray, num_sources: int = NUM_SOURCES) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float64)
    if Z.shape != (int(num_sources), 3):
        raise ValueError(f"Expected control array of shape ({num_sources}, 3), got {Z.shape}")
    return Z.reshape(-1).copy()


def component_names(num_sources: int = NUM_SOURCES) -> List[str]:
    names: List[str] = []
    for i in range(int(num_sources)):
        names.extend([f"x{i + 1}", f"y{i + 1}", f"A{i + 1}"])
    return names


def full_bounds_array(num_sources: int = NUM_SOURCES) -> np.ndarray:
    return np.vstack([_BASE_BOUNDS.copy() for _ in range(int(num_sources))])


def source_from_control(z: np.ndarray, r2: float):
    Z = reshape_control(z)

    def source(x: float, y: float) -> float:
        total = 0.0
        for xc, yc, amp in Z:
            g = np.exp(-(((x - xc) ** 2 + (y - yc) ** 2) / r2))
            total += float(amp * g)
        return float(total)

    return source


def source_derivatives_from_control(z: np.ndarray, r2: float):
    Z = reshape_control(z)
    derivatives = []

    for i, (xc, yc, amp) in enumerate(Z):
        def dsdx(x: float, y: float, xc=xc, yc=yc, amp=amp) -> float:
            g = np.exp(-(((x - xc) ** 2 + (y - yc) ** 2) / r2))
            return float((2.0 * amp * (x - xc) / r2) * g)

        def dsdy(x: float, y: float, xc=xc, yc=yc, amp=amp) -> float:
            g = np.exp(-(((x - xc) ** 2 + (y - yc) ** 2) / r2))
            return float((2.0 * amp * (y - yc) / r2) * g)

        def dsdA(x: float, y: float, xc=xc, yc=yc) -> float:
            g = np.exp(-(((x - xc) ** 2 + (y - yc) ** 2) / r2))
            return float(g)

        derivatives.extend(
            [
                {"name": f"x{i + 1}", "func": dsdx},
                {"name": f"y{i + 1}", "func": dsdy},
                {"name": f"A{i + 1}", "func": dsdA},
            ]
        )

    return derivatives


def _reference_positions(num_sources: int) -> np.ndarray:
    n = int(num_sources)
    ncols = max(2, math.ceil(math.sqrt(n)))
    nrows = math.ceil(n / ncols)

    xs = np.linspace(0.18, 0.82, ncols)
    ys = np.linspace(0.10, 0.30, nrows)

    points = []
    for row, y in enumerate(ys):
        row_xs = xs if row % 2 == 0 else xs[::-1]
        for x in row_xs:
            points.append((float(x), float(y)))

    return np.asarray(points[:n], dtype=np.float64)


def build_reference_control(num_sources: int = NUM_SOURCES) -> np.ndarray:
    positions = _reference_positions(num_sources)
    amps = np.array([1.4 - 0.18 * (i % 4) for i in range(int(num_sources))], dtype=np.float64)

    Z = np.zeros((int(num_sources), 3), dtype=np.float64)
    Z[:, 0:2] = positions
    Z[:, 2] = np.clip(amps, 0.7, 1.5)
    return flatten_control(Z, num_sources=num_sources)


def build_initial_control(num_sources: int = NUM_SOURCES) -> np.ndarray:
    ref = reshape_control(build_reference_control(num_sources), num_sources=num_sources)
    shifted = np.roll(ref[:, 0:2], shift=max(1, int(num_sources) // 2), axis=0)

    Z = np.zeros((int(num_sources), 3), dtype=np.float64)
    for i in range(int(num_sources)):
        dx = 0.10 if i % 2 == 0 else -0.08
        dy = -0.06 if i % 2 == 0 else 0.05
        Z[i, 0] = np.clip(shifted[i, 0] + dx, 0.12, 0.88)
        Z[i, 1] = np.clip(shifted[i, 1] + dy, 0.06, 0.34)
        Z[i, 2] = np.clip(0.55 + 0.22 * ((i + 1) % 5), 0.4, 1.6)

    return flatten_control(Z, num_sources=num_sources)


def pack_active(z_full: np.ndarray, mode: str, num_sources: int = NUM_SOURCES) -> np.ndarray:
    Z = reshape_control(z_full, num_sources=num_sources)
    if mode == "location_only":
        return Z[:, 0:2].reshape(-1).copy()
    if mode == "intensity_only":
        return Z[:, 2].copy()
    if mode == "location_and_intensity":
        return flatten_control(Z, num_sources=num_sources)
    raise ValueError(f"Unknown CONTROL_MODE: {mode}")


def unpack_active(
    z_active: np.ndarray,
    z_template: np.ndarray,
    mode: str,
    fixed_values: Dict[str, np.ndarray],
    num_sources: int = NUM_SOURCES,
) -> np.ndarray:
    Z = reshape_control(z_template, num_sources=num_sources).copy()
    za = np.asarray(z_active, dtype=np.float64)

    if mode == "location_only":
        Z[:, 0:2] = za.reshape(int(num_sources), 2)
        Z[:, 2] = np.asarray(fixed_values["A_fixed"], dtype=np.float64)
        return flatten_control(Z, num_sources=num_sources)
    if mode == "intensity_only":
        Z[:, 0:2] = np.asarray(fixed_values["centers_fixed"], dtype=np.float64)
        Z[:, 2] = za
        return flatten_control(Z, num_sources=num_sources)
    if mode == "location_and_intensity":
        return reshape_control(za, num_sources=num_sources).reshape(-1).copy()
    raise ValueError(f"Unknown CONTROL_MODE: {mode}")


def restrict_gradient(g_full: np.ndarray, mode: str, num_sources: int = NUM_SOURCES) -> np.ndarray:
    G = reshape_control(g_full, num_sources=num_sources)
    if mode == "location_only":
        return G[:, 0:2].reshape(-1).copy()
    if mode == "intensity_only":
        return G[:, 2].copy()
    if mode == "location_and_intensity":
        return flatten_control(G, num_sources=num_sources)
    raise ValueError(f"Unknown CONTROL_MODE: {mode}")


def active_bounds_for_mode(mode: str, num_sources: int = NUM_SOURCES):
    full_bounds = full_bounds_array(num_sources)
    if mode == "location_only":
        Z = full_bounds.reshape(int(num_sources), 3, 2)
        return [tuple(bounds) for bounds in Z[:, 0:2, :].reshape(-1, 2)]
    if mode == "intensity_only":
        Z = full_bounds.reshape(int(num_sources), 3, 2)
        return [tuple(bounds) for bounds in Z[:, 2, :]]
    if mode == "location_and_intensity":
        return [tuple(bounds) for bounds in full_bounds]
    raise ValueError(f"Unknown CONTROL_MODE: {mode}")


def mode_fixed_values(mode: str, z_ref_full: np.ndarray, num_sources: int = NUM_SOURCES) -> Dict[str, np.ndarray]:
    Zref = reshape_control(z_ref_full, num_sources=num_sources)
    if mode == "location_only":
        return {"A_fixed": Zref[:, 2].copy()}
    if mode == "intensity_only":
        return {"centers_fixed": Zref[:, 0:2].copy()}
    if mode == "location_and_intensity":
        return {}
    raise ValueError(f"Unknown CONTROL_MODE: {mode}")
