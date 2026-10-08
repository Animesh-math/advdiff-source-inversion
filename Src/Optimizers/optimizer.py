"""Reusable optimization wrapper around scipy.optimize.minimize."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize


ObjectiveFunction = Callable[[np.ndarray, Any, Any], float]
GradientFunction = Callable[[np.ndarray, Any, Any], np.ndarray]


class Optimizer:
    """Generic optimizer that delegates PDE/problem specifics to callbacks."""

    def __init__(
        self,
        objective_function: ObjectiveFunction,
        gradient_function: GradientFunction,
        optimization_method: str = "L-BFGS-B",
        max_iter: int = 100,
        tol: float = 1.0e-8,
        bounds=None,
        callback=None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.objective_function = objective_function
        self.gradient_function = gradient_function
        self.optimization_method = optimization_method
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.bounds = bounds
        self.callback = callback
        self.options = dict(options) if options is not None else {}

    def optimize(self, initial_guess, mesh_data=None, problem_data=None):
        """Run scipy optimization and return OptimizeResult."""
        x0 = np.asarray(initial_guess, dtype=np.float64).reshape(-1)

        def f(control):
            value = self.objective_function(
                np.asarray(control, dtype=np.float64), mesh_data, problem_data
            )
            return float(value)

        def jac(control):
            grad = self.gradient_function(
                np.asarray(control, dtype=np.float64), mesh_data, problem_data
            )
            grad = np.asarray(grad, dtype=np.float64).reshape(-1)
            if grad.shape != x0.shape:
                raise ValueError(
                    "Gradient shape mismatch: "
                    f"expected {x0.shape}, got {grad.shape}."
                )
            if not np.all(np.isfinite(grad)):
                raise ValueError("Gradient contains non-finite values.")
            return grad

        scipy_options = dict(self.options)
        scipy_options.setdefault("maxiter", self.max_iter)

        print(
            f"[Optimizer] method={self.optimization_method}, "
            f"max_iter={self.max_iter}, tol={self.tol:.1e}"
        )

        return minimize(
            fun=f,
            x0=x0,
            method=self.optimization_method,
            jac=jac,
            bounds=self.bounds,
            tol=self.tol,
            callback=self.callback,
            options=scipy_options,
        )


__all__ = ["Optimizer"]
