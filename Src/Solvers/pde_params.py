"""Shared PDE parameter containers for solver calls."""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


@dataclass(frozen=True)
class PDEParams:
    k: float
    velocity: Tuple[float, float]
    dt: float
    tfinal: float
    ngauss: int
    st_ngauss: int = 3
    dirichlet_tags: Optional[List[str]] = None
    velocity_nodes: Optional[Any] = None

    @property
    def vx(self) -> float:
        return self.velocity[0]

    @property
    def vy(self) -> float:
        return self.velocity[1]
