"""Application-local utilities for multi-source Gaussian controls."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from mesh_data_config import (
    CIRCLE_CENTER_X,
    CIRCLE_CENTER_Y,
    CIRCLE_RADIUS,
    RECTANGLE_HEIGHT,
    RECTANGLE_WIDTH,
)


NUM_SOURCES = 2
MESH_DOMAIN_BOUND_CASE = "mesh_domain"

TRUE_CONTROL = np.array(
    [
        0.62,
        0.10,
        1.20,
        0.72,
        0.30,
        0.90,
    ],
    dtype=np.float64,
)
INITIAL_CONTROL = np.array(
    [
        0.38,
        0.16,
        0.45,
        0.60,
        0.34,
        0.40,
    ],
    dtype=np.float64,
)

# Deprecated legacy source-label boxes retained for regression compatibility.
BOUND_CASE_LOCATION_BOUNDS = {
    "broad": (
        ((0.08, 0.92), (0.05, 0.35)),
        ((0.08, 0.92), (0.05, 0.35)),
    ),
    "both_left": (
        ((0.08, 0.42), (0.05, 0.35)),
        ((0.08, 0.42), (0.05, 0.35)),
    ),
    "left_right": (
        ((0.08, 0.42), (0.05, 0.35)),
        ((0.58, 0.92), (0.05, 0.35)),
    ),
    "right_left": (
        ((0.58, 0.92), (0.05, 0.35)),
        ((0.08, 0.42), (0.05, 0.35)),
    ),
    "both_right": (
        ((0.58, 0.92), (0.05, 0.35)),
        ((0.58, 0.92), (0.05, 0.35)),
    ),
    "lower_upper": (
        ((0.08, 0.92), (0.05, 0.12)),
        ((0.08, 0.92), (0.28, 0.35)),
    ),
    "upper_lower": (
        ((0.08, 0.92), (0.28, 0.35)),
        ((0.08, 0.92), (0.05, 0.12)),
    ),
}

# Backward-compatible alias for callers that still expect the broad boxes.
SOURCE_LOCATION_BOUNDS = BOUND_CASE_LOCATION_BOUNDS["broad"]
AMPLITUDE_BOUNDS = (0.0, 2.0)
OBSTACLE_TOLERANCE = 1.0e-4

MODE_OUTPUT_NAMES = {
    "intensity_only": "amplitude_only",
    "location_only": "location_only",
    "location_and_intensity": "full",
}


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
    expected_shape = (int(num_sources), 3)
    if Z.shape != expected_shape:
        raise ValueError(f"Expected control array of shape {expected_shape}, got {Z.shape}")
    return Z.reshape(-1).copy()


def component_names(num_sources: int = NUM_SOURCES) -> List[str]:
    names: List[str] = []
    for source_id in range(int(num_sources)):
        names.extend([f"x{source_id + 1}", f"y{source_id + 1}", f"A{source_id + 1}"])
    return names


def bound_case_names() -> tuple[str, ...]:
    return (MESH_DOMAIN_BOUND_CASE, *tuple(BOUND_CASE_LOCATION_BOUNDS))


def legacy_bound_case_names() -> tuple[str, ...]:
    return tuple(BOUND_CASE_LOCATION_BOUNDS)


def source_location_bounds_for_case(
    bound_case: str, num_sources: int = NUM_SOURCES
):
    try:
        location_bounds = BOUND_CASE_LOCATION_BOUNDS[bound_case]
    except KeyError as exc:
        raise ValueError(
            f"Unknown bound case {bound_case!r}; expected one of "
            f"{legacy_bound_case_names()}"
        ) from exc
    if int(num_sources) != len(location_bounds):
        raise ValueError(
            f"Bound case {bound_case!r} must define one location box per source"
        )
    return location_bounds


def full_bounds_array(
    num_sources: int = NUM_SOURCES, bound_case: str = "broad"
) -> np.ndarray:
    bounds = []
    for x_bounds, y_bounds in source_location_bounds_for_case(
        bound_case, num_sources=num_sources
    ):
        bounds.extend([x_bounds, y_bounds, AMPLITUDE_BOUNDS])
    return np.asarray(bounds, dtype=np.float64)


def build_reference_control(num_sources: int = NUM_SOURCES) -> np.ndarray:
    if int(num_sources) != NUM_SOURCES:
        raise ValueError(f"App3 defines exactly {NUM_SOURCES} reference sources")
    return TRUE_CONTROL.copy()


def build_initial_control(num_sources: int = NUM_SOURCES) -> np.ndarray:
    if int(num_sources) != NUM_SOURCES:
        raise ValueError(f"App3 defines exactly {NUM_SOURCES} initial sources")
    return INITIAL_CONTROL.copy()


def source_from_control(z: np.ndarray, r2: float):
    Z = reshape_control(z)

    def source(x: float, y: float) -> float:
        total = 0.0
        for xc, yc, amplitude in Z:
            gaussian = np.exp(-(((x - xc) ** 2 + (y - yc) ** 2) / r2))
            total += float(amplitude * gaussian)
        return float(total)

    return source


def source_derivatives_from_control(z: np.ndarray, r2: float):
    Z = reshape_control(z)
    derivatives = []

    for source_id, (xc, yc, amplitude) in enumerate(Z):
        def dsdx(x: float, y: float, xc=xc, yc=yc, amplitude=amplitude) -> float:
            gaussian = np.exp(-(((x - xc) ** 2 + (y - yc) ** 2) / r2))
            return float((2.0 * amplitude * (x - xc) / r2) * gaussian)

        def dsdy(x: float, y: float, xc=xc, yc=yc, amplitude=amplitude) -> float:
            gaussian = np.exp(-(((x - xc) ** 2 + (y - yc) ** 2) / r2))
            return float((2.0 * amplitude * (y - yc) / r2) * gaussian)

        def dsdA(x: float, y: float, xc=xc, yc=yc) -> float:
            gaussian = np.exp(-(((x - xc) ** 2 + (y - yc) ** 2) / r2))
            return float(gaussian)

        derivatives.extend(
            [
                {"name": f"x{source_id + 1}", "func": dsdx},
                {"name": f"y{source_id + 1}", "func": dsdy},
                {"name": f"A{source_id + 1}", "func": dsdA},
            ]
        )

    return derivatives


def pack_active(
    z_full: np.ndarray, mode: str, num_sources: int = NUM_SOURCES
) -> np.ndarray:
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


def restrict_gradient(
    g_full: np.ndarray, mode: str, num_sources: int = NUM_SOURCES
) -> np.ndarray:
    G = reshape_control(g_full, num_sources=num_sources)
    if mode == "location_only":
        return G[:, 0:2].reshape(-1).copy()
    if mode == "intensity_only":
        return G[:, 2].copy()
    if mode == "location_and_intensity":
        return flatten_control(G, num_sources=num_sources)
    raise ValueError(f"Unknown CONTROL_MODE: {mode}")


def active_bounds_for_mode(
    mode: str, num_sources: int = NUM_SOURCES, bound_case: str = "broad"
):
    """Return legacy source-region bounds for backward compatibility."""
    bounds = []
    for x_bounds, y_bounds in source_location_bounds_for_case(
        bound_case, num_sources=num_sources
    ):
        if mode in ("location_only", "location_and_intensity"):
            bounds.extend([x_bounds, y_bounds])
        if mode in ("intensity_only", "location_and_intensity"):
            bounds.append(AMPLITUDE_BOUNDS)
    if mode not in MODE_OUTPUT_NAMES:
        raise ValueError(f"Unknown CONTROL_MODE: {mode}")
    return bounds


def mesh_domain_active_bounds_for_mode(
    mode: str, bounding_box, num_sources: int = NUM_SOURCES
):
    """Return full mesh bounding-box location bounds for active variables."""
    xmin, xmax, ymin, ymax = (float(value) for value in bounding_box)
    if not np.all(np.isfinite([xmin, xmax, ymin, ymax])):
        raise ValueError("Mesh bounding box must contain only finite values")
    if xmin > xmax or ymin > ymax:
        raise ValueError(f"Invalid mesh bounding box: {bounding_box}")
    if mode not in MODE_OUTPUT_NAMES:
        raise ValueError(f"Unknown CONTROL_MODE: {mode}")

    bounds = []
    for _source_id in range(int(num_sources)):
        if mode in ("location_only", "location_and_intensity"):
            bounds.extend([(xmin, xmax), (ymin, ymax)])
        if mode in ("intensity_only", "location_and_intensity"):
            bounds.append(AMPLITUDE_BOUNDS)
    return bounds


def mode_fixed_values(
    mode: str, z_ref_full: np.ndarray, num_sources: int = NUM_SOURCES
) -> Dict[str, np.ndarray]:
    Zref = reshape_control(z_ref_full, num_sources=num_sources)
    if mode == "location_only":
        return {"A_fixed": Zref[:, 2].copy()}
    if mode == "intensity_only":
        return {"centers_fixed": Zref[:, 0:2].copy()}
    if mode == "location_and_intensity":
        return {}
    raise ValueError(f"Unknown CONTROL_MODE: {mode}")


def is_inside_obstacle(x: float, y: float) -> bool:
    distance = float(np.hypot(x - CIRCLE_CENTER_X, y - CIRCLE_CENTER_Y))
    return distance <= CIRCLE_RADIUS + OBSTACLE_TOLERANCE


def validate_control_within_bounds(
    control: np.ndarray, bound_case: str, label: str
) -> None:
    location_bounds = source_location_bounds_for_case(bound_case)
    for source_id, ((x, y, _amplitude), (x_bounds, y_bounds)) in enumerate(
        zip(reshape_control(control), location_bounds), start=1
    ):
        if not (x_bounds[0] <= x <= x_bounds[1]):
            raise ValueError(
                f"{label} source {source_id} x={x} is outside the selected "
                f"{bound_case!r} bounds {x_bounds}"
            )
        if not (y_bounds[0] <= y <= y_bounds[1]):
            raise ValueError(
                f"{label} source {source_id} y={y} is outside the selected "
                f"{bound_case!r} bounds {y_bounds}"
            )


def validate_sources_outside_obstacle(control: np.ndarray, label: str) -> None:
    control = np.asarray(control, dtype=np.float64)
    if not np.all(np.isfinite(control)):
        raise ValueError(f"{label} control must contain only finite values")

    for source_id, (x, y, amplitude) in enumerate(
        reshape_control(control), start=1
    ):
        if not (0.0 < x < RECTANGLE_WIDTH and 0.0 < y < RECTANGLE_HEIGHT):
            raise ValueError(
                f"{label} source {source_id} center ({x}, {y}) "
                "lies outside the rectangle"
            )
        if is_inside_obstacle(float(x), float(y)):
            raise ValueError(
                f"{label} source {source_id} center ({x}, {y}) lies inside or "
                "too close to the circular obstacle"
            )
        if not (AMPLITUDE_BOUNDS[0] <= amplitude <= AMPLITUDE_BOUNDS[1]):
            raise ValueError(
                f"{label} source {source_id} amplitude {amplitude} is outside "
                f"the allowed bounds {AMPLITUDE_BOUNDS}"
            )
