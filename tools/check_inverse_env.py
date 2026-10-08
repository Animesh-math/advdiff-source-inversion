#!/usr/bin/env python3
"""Check whether the local environment can run the inverse-optimization drivers.

Companion to ``tools/check_fenicsx_env.py``, which covers the separate FEniCSx
velocity-field environment.  This one checks the inverse-solver interpreter used by
every ``Applications/*/run_optimization.py`` run.

Packages are split into REQUIRED (the driver cannot run without them) and
OPTIONAL (a specific selectable option needs them).  A missing optional package
is reported with the option it gates and is NOT an error; a missing required
package exits non-zero.

Usage:
    python3 \
        tools/check_inverse_env.py
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, Dict, Optional, Tuple


REQUIRED_PACKAGES: Tuple[str, ...] = (
    "numpy",
    "scipy",
    "matplotlib",
    "meshio",
)

#: package -> what selecting it enables.
OPTIONAL_PACKAGES: Dict[str, str] = {
    "osqp": (
        "Application1 --amplitude-solver osqp (default joint amplitude solver; "
        "--amplitude-solver enumeration needs no extra package)"
    ),
}

#: Minimum version for optional packages whose API this repository pins.
OPTIONAL_MIN_MAJOR: Dict[str, int] = {
    "osqp": 1,
}


def version_for(module: Any) -> str:
    version = getattr(module, "__version__", None)
    return str(version) if version else "unknown"


def import_package(name: str) -> Optional[Any]:
    try:
        return importlib.import_module(name)
    except Exception as exc:
        print(f"{name}: FAILED ({type(exc).__name__}: {exc})")
        return None


def check_required() -> bool:
    ok = True
    for name in REQUIRED_PACKAGES:
        module = import_package(name)
        if module is None:
            ok = False
            continue
        print(f"{name}: OK version={version_for(module)}")
    return ok


def check_optional() -> None:
    for name, enables in OPTIONAL_PACKAGES.items():
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            print(
                f"{name}: MISSING ({type(exc).__name__}: {exc})\n"
                f"    gates: {enables}\n"
                f"    install: {sys.executable} -m pip install '{name}>="
                f"{OPTIONAL_MIN_MAJOR.get(name, 0)}.0'"
            )
            continue
        version = version_for(module)
        print(f"{name}: OK version={version}  (gates: {enables})")
        minimum = OPTIONAL_MIN_MAJOR.get(name)
        if minimum is None:
            continue
        try:
            major = int(version.split(".", 1)[0])
        except ValueError:
            print(f"    WARNING: cannot parse major version from {version!r}")
            continue
        if major < minimum:
            print(
                f"    WARNING: {name} {version} is older than the expected "
                f"{minimum}.x API"
            )


def run_amplitude_qp_check() -> bool:
    """Solve one tiny box-constrained QP with OSQP, as the amplitude fit does."""
    try:
        import numpy as np
        import osqp
        import scipy.sparse as sp
    except Exception as exc:
        print(f"osqp amplitude-QP check: SKIPPED ({type(exc).__name__}: {exc})")
        return False

    try:
        # min 0.5 a^T H a - a^T h over [0.1, 2.0]^2; interior minimizer
        # a = H^{-1} h = [0.5, 0.5].
        H = np.array([[2.0, 1.0], [1.0, 2.0]])
        h = np.array([1.5, 1.5])
        problem = osqp.OSQP()
        settings = {"verbose": False, "eps_abs": 1e-12, "eps_rel": 1e-12}
        settings["polishing" if int(osqp.__version__[0]) >= 1 else "polish"] = True
        problem.setup(
            P=sp.triu(sp.csc_matrix(H), format="csc"),
            q=-h,
            A=sp.identity(2, format="csc"),
            l=np.full(2, 0.1),
            u=np.full(2, 2.0),
            **settings,
        )
        result = problem.solve()
        status = str(result.info.status)
        error = float(np.max(np.abs(np.asarray(result.x) - 0.5)))
    except Exception as exc:
        print(f"osqp amplitude-QP check: FAILED ({type(exc).__name__}: {exc})")
        return False

    if status != "solved" or error > 1.0e-9:
        print(f"osqp amplitude-QP check: FAILED (status={status} err={error:.3e})")
        return False
    print(f"osqp amplitude-QP check: OK status={status} max_error={error:.3e}")
    return True


def main() -> int:
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")

    print("\n-- required --")
    required_ok = check_required()

    print("\n-- optional --")
    check_optional()
    run_amplitude_qp_check()

    if not required_ok:
        print("\nRESULT: required package(s) missing")
        return 1
    print("\nRESULT: required packages present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
