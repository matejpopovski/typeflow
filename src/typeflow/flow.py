"""Conditional flow matching on the straight path from noise to glyph, and ODE samplers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

Velocity = Callable[[Tensor, Tensor], Tensor]


def sample_time(n: int, generator: torch.Generator, uniform_share: float = 0.1) -> Tensor:
    """Draw training times, mostly from a logit-normal centred on the middle of the path.

    Near t=0 the target is almost the dataset mean and near t=1 almost the data itself, so the
    middle is where the network has the most left to learn. A uniform share keeps the ends covered,
    which matters because every sampler starts its first step exactly at t=0.
    """
    t = torch.sigmoid(torch.randn(n, generator=generator))
    uniform = torch.rand(n, generator=generator)
    pick = torch.rand(n, generator=generator) < uniform_share
    return torch.where(pick, uniform, t)


def flow_pair(x1: Tensor, noise: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """Return the point on the path at time ``t`` and the velocity that the network regresses."""
    shaped = t.reshape(-1, *([1] * (x1.dim() - 1)))
    return (1 - shaped) * noise + shaped * x1, x1 - noise


@dataclass(frozen=True)
class Tableau:
    """An explicit Runge-Kutta method."""

    name: str
    order: int
    a: tuple[tuple[float, ...], ...]
    b: tuple[float, ...]
    c: tuple[float, ...]

    @property
    def stages(self) -> int:
        return len(self.b)


TABLEAUS = {
    t.name: t
    for t in (
        Tableau("euler", 1, ((),), (1.0,), (0.0,)),
        Tableau("midpoint", 2, ((), (0.5,)), (0.0, 1.0), (0.0, 0.5)),
        Tableau("heun", 2, ((), (1.0,)), (0.5, 0.5), (0.0, 1.0)),
        Tableau(
            "rk4",
            4,
            ((), (0.5,), (0.0, 0.5), (0.0, 0.0, 1.0)),
            (1 / 6, 1 / 3, 1 / 3, 1 / 6),
            (0.0, 0.5, 0.5, 1.0),
        ),
    )
}


def touches_endpoint(tableau: Tableau) -> bool:
    return tableau.c[-1] == 1.0


def velocity_calls(method: str, steps: int, skip_endpoint: bool = True) -> int:
    tableau = TABLEAUS[method]
    if skip_endpoint and touches_endpoint(tableau):
        return (steps - 1) * tableau.stages + 1
    return steps * tableau.stages


def integrate(
    velocity: Velocity, x0: Tensor, steps: int, method: str = "heun", skip_endpoint: bool = False
) -> Tensor:
    """Integrate from t=0 to t=1 on a uniform grid.

    With ``skip_endpoint`` the last step falls back to Euler for methods that would otherwise
    evaluate the field at t=1. A flow's input carries no noise there, so a trained network cannot
    know which trajectory it is on, and its velocity is wrong by the size of the noise. Used in a
    final corrector stage, that error stays in the output as grain of amplitude h/2.
    """
    tableau = TABLEAUS[method]
    h = 1.0 / steps
    x = x0
    for step in range(steps):
        active = tableau
        if skip_endpoint and step == steps - 1 and touches_endpoint(tableau):
            active = TABLEAUS["euler"]
        t0 = step * h
        slopes: list[Tensor] = []
        for coefficients, c in zip(active.a, active.c):
            stage = x
            for weight, slope in zip(coefficients, slopes):
                if weight:
                    stage = stage + h * weight * slope
            slopes.append(velocity(stage, x.new_full((x.shape[0],), t0 + c * h)))
        for weight, slope in zip(active.b, slopes):
            if weight:
                x = x + h * weight * slope
    return x


def guided(conditional: Velocity, unconditional: Velocity, scale: float) -> Velocity:
    """Classifier-free guidance. A scale of 1 is the plain conditional field."""
    if scale == 1.0:
        return conditional

    def velocity(x: Tensor, t: Tensor) -> Tensor:
        base = unconditional(x, t)
        return base + scale * (conditional(x, t) - base)

    return velocity
