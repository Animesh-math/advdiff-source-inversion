"""Application-local utilities for Application1 multi-source Gaussian controls.

This mirrors ``Test Application4/multi_source_utils.py`` but is adapted to the
DC Navy Yard domain, whose coordinates are in LOCAL METERS with the south-west
corner of the outer domain at the origin (x in [0, ~302], y in [0, ~700]).  All
geometry-dependent defaults here are therefore expressed in meters / square
meters, not in the normalized [0, 1.4] x [0, 1.0] units of Test Application4.
(The mesh was translated once from projected GIS coordinates; global = local +
origin, with the origin recorded in DC_mesh/dc_navy_yard_coordinate_origin.json.)

The Gaussian source model, control ordering ``[x1, y1, A1, x2, y2, A2, ...]``,
packing/unpacking, and mode helpers are identical to Test Application4; only the
numeric geometry constants differ.  Source-center admissibility against the 21
DC building holes is handled by mesh point-location in ``mesh_admissibility.py``;
the analytic building-exclusion helpers used for Test Application4's *curved*
buildings are not needed here because the DC buildings are exact polygonal mesh
holes.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from mesh_data_config import (
    DOMAIN_X_MAX,
    DOMAIN_X_MIN,
    DOMAIN_Y_MAX,
    DOMAIN_Y_MIN,
    OUTER_HEIGHT,
    OUTER_WIDTH,
)


# Default source count.  The Application1 driver accepts ``--num-sources M`` and
# threads the selected M through the helpers below; ``NUM_SOURCES`` remains the
# default so every existing two-source call site is unchanged.
NUM_SOURCES = 2
MESH_DOMAIN_BOUND_CASE = "mesh_domain"

# Two admissible source centers in the DC fluid/street region (LOCAL meters),
# with amplitudes analogous to Test Application4 (dimensionless emission
# strength).  Source 1 sits upstream in the western street corridor so its plume
# advects east/south toward the outflow (right/bottom) observation boundaries;
# source 2 sits in the south-eastern corridor.  Both centers were verified
# admissible against the DC fluid mesh (not inside any of the 21 building holes).
# These are the same physical points as the original GIS controls
# ([400070, 134300], [400300, 134120]) minus the coordinate origin.
TRUE_CONTROL = np.array(
    [
        37.0278,
        381.9172,
        1.10,
        267.0278,
        201.9172,
        0.85,
    ],
    dtype=np.float64,
)
INITIAL_CONTROL = np.array(
    [
        147.0278,
        381.9172,
        0.60,
        57.0278,
        561.9172,
        0.50,
    ],
    dtype=np.float64,
)

# Deprecated legacy source-label boxes retained only for CLI compatibility with
# the Test Application4 driver.  Unlike App4 (normalized units), these are
# derived from the DC fluid-domain bounding box in meters.  New Application1
# experiments should use the default mesh_domain workflow, which uses the full
# mesh bounding box plus mesh admissibility.
_XMIN, _XMAX = DOMAIN_X_MIN, DOMAIN_X_MAX
_YMIN, _YMAX = DOMAIN_Y_MIN, DOMAIN_Y_MAX
_XMID = 0.5 * (_XMIN + _XMAX)
_YMID = 0.5 * (_YMIN + _YMAX)
BOUND_CASE_LOCATION_BOUNDS = {
    "broad": (
        ((_XMIN, _XMAX), (_YMIN, _YMAX)),
        ((_XMIN, _XMAX), (_YMIN, _YMAX)),
    ),
    "both_left": (
        ((_XMIN, _XMID), (_YMIN, _YMAX)),
        ((_XMIN, _XMID), (_YMIN, _YMAX)),
    ),
    "left_right": (
        ((_XMIN, _XMID), (_YMIN, _YMAX)),
        ((_XMID, _XMAX), (_YMIN, _YMAX)),
    ),
    "right_left": (
        ((_XMID, _XMAX), (_YMIN, _YMAX)),
        ((_XMIN, _XMID), (_YMIN, _YMAX)),
    ),
    "both_right": (
        ((_XMID, _XMAX), (_YMIN, _YMAX)),
        ((_XMID, _XMAX), (_YMIN, _YMAX)),
    ),
    "lower_upper": (
        ((_XMIN, _XMAX), (_YMIN, _YMID)),
        ((_XMIN, _XMAX), (_YMID, _YMAX)),
    ),
    "upper_lower": (
        ((_XMIN, _XMAX), (_YMID, _YMAX)),
        ((_XMIN, _XMAX), (_YMIN, _YMID)),
    ),
}

# Backward-compatible alias for callers that still expect the broad boxes.
SOURCE_LOCATION_BOUNDS = BOUND_CASE_LOCATION_BOUNDS["broad"]
AMPLITUDE_BOUNDS = (0.0, 2.0)
# Geometry tolerance in meters (DC coordinates span hundreds of meters).
GEOMETRY_TOLERANCE = 1.0

MODE_OUTPUT_NAMES = {
    "intensity_only": "amplitude_only",
    "location_only": "location_only",
    "location_and_intensity": "full",
}


def control_dim(num_sources: int = NUM_SOURCES) -> int:
    return 3 * int(num_sources)


def infer_num_sources(z) -> int:
    """Number of sources implied by a flat control of length ``3 * M``.

    The control layout is ``[x1, y1, A1, ..., xM, yM, AM]``, so any admissible
    control length is a positive multiple of three and determines M uniquely.
    Used so the shape helpers below work for arbitrary M without every caller
    having to thread the source count; a caller that knows M can still pass it
    explicitly and get the stricter length check.
    """
    values = np.asarray(z, dtype=np.float64).reshape(-1)
    if values.size == 0 or values.size % 3 != 0:
        raise ValueError(
            f"Control of length {values.size} does not describe a whole number "
            "of sources; expected a positive multiple of 3 "
            "([x, y, A] per source)"
        )
    return int(values.size // 3)


def reshape_control(
    z: np.ndarray, num_sources: Optional[int] = None
) -> np.ndarray:
    """Reshape a flat control to ``(M, 3)``.

    With ``num_sources=None`` (the default) M is inferred from the control
    length, which is what makes the shared helpers M-agnostic.  Passing
    ``num_sources`` explicitly keeps the original strict length validation and
    is what the CLI uses to reject a control of the wrong size.
    """
    z = np.asarray(z, dtype=np.float64)
    if num_sources is None:
        num_sources = infer_num_sources(z)
    expected = control_dim(num_sources)
    if z.size != expected:
        raise ValueError(f"Expected control of length {expected}, got {z.size}")
    return z.reshape(int(num_sources), 3)


def flatten_control(
    Z: np.ndarray, num_sources: Optional[int] = None
) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float64)
    if num_sources is None:
        if Z.ndim != 2 or Z.shape[1] != 3:
            raise ValueError(
                f"Expected control array of shape (M, 3), got {Z.shape}"
            )
        num_sources = int(Z.shape[0])
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
        raise ValueError(f"Application1 defines exactly {NUM_SOURCES} reference sources")
    return TRUE_CONTROL.copy()


def build_initial_control(num_sources: int = NUM_SOURCES) -> np.ndarray:
    if int(num_sources) != NUM_SOURCES:
        raise ValueError(f"Application1 defines exactly {NUM_SOURCES} initial sources")
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


def source_directional_derivative_from_control(
    z: np.ndarray, direction: np.ndarray, r2: float
):
    """Return the pointwise first directional derivative S_z(z)[direction]."""
    direction = np.asarray(direction, dtype=np.float64)
    expected = control_dim(infer_num_sources(z))
    if direction.size != expected:
        raise ValueError(f"Expected direction of length {expected}, got {direction.size}")

    derivative_specs = source_derivatives_from_control(z, r2=r2)

    def directional(x: float, y: float) -> float:
        total = 0.0
        for spec, weight in zip(derivative_specs, direction):
            total += float(weight) * spec["func"](x, y)
        return float(total)

    return directional


def source_hessian_matrix_from_control(z: np.ndarray, r2: float):
    """Return callable pointwise Hessian D^2_zz S(x; z) for all sources."""
    Z = reshape_control(z)
    dim = control_dim(num_sources=Z.shape[0])
    sigma_s2 = 0.5 * float(r2)
    inv_sigma_s2 = 1.0 / sigma_s2

    def hessian(x: float, y: float) -> np.ndarray:
        H = np.zeros((dim, dim), dtype=np.float64)
        for source_id, (xc, yc, amplitude) in enumerate(Z):
            offset = 3 * source_id
            gaussian = np.exp(-(((x - xc) ** 2 + (y - yc) ** 2) / r2))
            ax = (x - xc) * inv_sigma_s2
            ay = (y - yc) * inv_sigma_s2
            block = gaussian * np.array(
                [
                    [amplitude * (ax * ax - inv_sigma_s2), amplitude * ax * ay, ax],
                    [amplitude * ax * ay, amplitude * (ay * ay - inv_sigma_s2), ay],
                    [ax, ay, 0.0],
                ],
                dtype=np.float64,
            )
            H[offset : offset + 3, offset : offset + 3] = block
        return H

    return hessian


def source_hessian_directional_from_control(
    z: np.ndarray, direction: np.ndarray, r2: float
):
    """Return component functions for D^2 S(z)[direction, e_j]."""
    direction = np.asarray(direction, dtype=np.float64)
    num_sources = infer_num_sources(z)
    expected = control_dim(num_sources)
    if direction.size != expected:
        raise ValueError(f"Expected direction of length {expected}, got {direction.size}")

    hessian = source_hessian_matrix_from_control(z, r2=r2)
    names = component_names(num_sources)
    specs = []
    for component_index in range(expected):
        def d2s(x: float, y: float, component_index=component_index) -> float:
            return float(hessian(x, y)[component_index, :] @ direction)

        specs.append({"name": names[component_index], "func": d2s})
    return specs


def pack_active(
    z_full: np.ndarray, mode: str, num_sources: Optional[int] = None
) -> np.ndarray:
    Z = reshape_control(z_full, num_sources=num_sources)
    num_sources = int(Z.shape[0])
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
    num_sources: Optional[int] = None,
) -> np.ndarray:
    Z = reshape_control(z_template, num_sources=num_sources).copy()
    num_sources = int(Z.shape[0])
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
    g_full: np.ndarray, mode: str, num_sources: Optional[int] = None
) -> np.ndarray:
    G = reshape_control(g_full, num_sources=num_sources)
    num_sources = int(G.shape[0])
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
    mode: str, z_ref_full: np.ndarray, num_sources: Optional[int] = None
) -> Dict[str, np.ndarray]:
    Zref = reshape_control(z_ref_full, num_sources=num_sources)
    if mode == "location_only":
        return {"A_fixed": Zref[:, 2].copy()}
    if mode == "intensity_only":
        return {"centers_fixed": Zref[:, 0:2].copy()}
    if mode == "location_and_intensity":
        return {}
    raise ValueError(f"Unknown CONTROL_MODE: {mode}")


def is_inside_excluded_geometry(x: float, y: float) -> bool:
    """Coarse bounding-box exclusion for the DC domain (meters).

    Unlike Test Application4, Application1 has no analytic building footprints in
    this module: the 21 DC building holes are represented exactly by the mesh and
    excluded via mesh point-location in ``mesh_admissibility.py``.  This helper
    only rejects points outside the DC fluid-domain bounding box and is used by
    the deprecated legacy ``--bound-case`` validation path.  The authoritative
    building/hole admissibility is the mesh locator.
    """
    tol = GEOMETRY_TOLERANCE
    if x < DOMAIN_X_MIN - tol or x > DOMAIN_X_MAX + tol:
        return True
    if y < DOMAIN_Y_MIN - tol or y > DOMAIN_Y_MAX + tol:
        return True
    return False


def validate_control_within_bounds(
    control: np.ndarray, bound_case: str, label: str
) -> None:
    sources = reshape_control(control)
    # Raises when a legacy per-label box case does not define one box per
    # source, so an M-source control can never be silently truncated here.
    location_bounds = source_location_bounds_for_case(
        bound_case, num_sources=sources.shape[0]
    )
    for source_id, ((x, y, _amplitude), (x_bounds, y_bounds)) in enumerate(
        zip(sources, location_bounds), start=1
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
        if is_inside_excluded_geometry(float(x), float(y)):
            raise ValueError(
                f"{label} source {source_id} center ({x}, {y}) lies outside "
                "the DC fluid-domain bounding box"
            )
        if not (AMPLITUDE_BOUNDS[0] <= amplitude <= AMPLITUDE_BOUNDS[1]):
            raise ValueError(
                f"{label} source {source_id} amplitude {amplitude} is outside "
                f"the allowed bounds {AMPLITUDE_BOUNDS}"
            )
