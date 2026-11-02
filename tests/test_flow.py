import pytest
import torch

from typeflow.flow import (
    TABLEAUS,
    flow_pair,
    guided,
    integrate,
    sample_time,
    touches_endpoint,
    velocity_calls,
)


def test_path_starts_at_noise_and_ends_at_data() -> None:
    x1, noise = torch.ones(2, 1, 4, 4), torch.zeros(2, 1, 4, 4)
    start, _ = flow_pair(x1, noise, torch.zeros(2))
    end, _ = flow_pair(x1, noise, torch.ones(2))
    assert torch.equal(start, noise) and torch.equal(end, x1)


def test_target_is_the_derivative_of_the_path() -> None:
    x1, noise = torch.randn(2, 1, 4, 4), torch.randn(2, 1, 4, 4)
    t = torch.tensor([0.3, 0.7])
    h = 1e-3
    a, target = flow_pair(x1, noise, t)
    b, _ = flow_pair(x1, noise, t + h)
    assert torch.allclose((b - a) / h, target, atol=1e-2)


def test_time_stays_on_the_unit_interval() -> None:
    t = sample_time(10_000, torch.Generator().manual_seed(0))
    assert t.min() >= 0 and t.max() <= 1
    assert abs(t.mean() - 0.5) < 0.02


def test_time_sampling_covers_the_ends() -> None:
    t = sample_time(100_000, torch.Generator().manual_seed(0))
    assert (t < 0.02).float().mean() > 0.0015


@pytest.mark.parametrize("name", TABLEAUS)
def test_tableaus_are_consistent(name: str) -> None:
    tableau = TABLEAUS[name]
    assert sum(tableau.b) == pytest.approx(1)
    for row, c in zip(tableau.a, tableau.c):
        assert sum(row) == pytest.approx(c)


@pytest.mark.parametrize("name", TABLEAUS)
def test_a_straight_path_is_exact_in_one_step(name: str) -> None:
    x0 = torch.tensor([[2.0]])
    result = integrate(lambda x, t: torch.full_like(x, 3.0), x0, 1, name)
    assert result.item() == pytest.approx(5.0)


@pytest.mark.parametrize("name", TABLEAUS)
def test_observed_order_of_convergence(name: str) -> None:
    """On dx/dt = x, halving the step divides the error by 2^order."""
    x0 = torch.ones(1, dtype=torch.float64)
    exact = torch.e

    def error(steps: int) -> float:
        return abs(integrate(lambda x, t: x, x0, steps, name).item() - exact)

    observed = torch.log2(torch.tensor(error(8) / error(16))).item()
    assert observed == pytest.approx(TABLEAUS[name].order, abs=0.2)


def test_time_dependent_field_uses_stage_times() -> None:
    """dx/dt = 3t^2 has the exact solution x(1) = 1, which RK4 hits on any grid."""
    x0 = torch.zeros(1, dtype=torch.float64)
    result = integrate(lambda x, t: 3 * t**2, x0, 2, "rk4")
    assert result.item() == pytest.approx(1.0, abs=1e-9)


def test_the_optimal_field_for_one_data_point_reaches_it() -> None:
    """For a single target the marginal field is (x1 - x) / (1 - t)."""
    target = torch.tensor([0.25, -0.5])

    def velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (target - x) / (1 - t[:, None]).clamp_min(1e-3)

    out = integrate(velocity, torch.randn(1, 2), 32, "euler")
    assert torch.allclose(out, target[None], atol=1e-5)


def noise_blind_field(noise: torch.Tensor, target: torch.Tensor):
    """A field that follows the true path but cannot see the noise at t=1, like a trained flow."""

    def velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return target if t[0].item() >= 1.0 else target - noise

    return velocity


@pytest.mark.parametrize("name", ["heun", "rk4"])
def test_a_field_that_is_blind_at_the_endpoint_leaves_grain_unless_it_is_skipped(name: str) -> None:
    noise, target = torch.randn(1, 64), torch.full((1, 64), -1.0)
    field = noise_blind_field(noise, target)
    steps = 8
    grainy = integrate(field, noise, steps, name)
    clean = integrate(field, noise, steps, name, skip_endpoint=True)
    assert (grainy - target).abs().max() > 0.01
    assert torch.allclose(clean, target, atol=1e-5)


@pytest.mark.parametrize("name", TABLEAUS)
def test_skipping_the_endpoint_never_evaluates_t_equal_one(name: str) -> None:
    seen: list[float] = []

    def velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        seen.append(t[0].item())
        return x

    integrate(velocity, torch.ones(1), 4, name, skip_endpoint=True)
    assert max(seen) < 1.0
    assert len(seen) == velocity_calls(name, 4)


def test_only_methods_with_a_final_stage_at_one_are_changed() -> None:
    assert {n for n, t in TABLEAUS.items() if touches_endpoint(t)} == {"heun", "rk4"}
    assert velocity_calls("heun", 8) == 15
    assert velocity_calls("heun", 8, skip_endpoint=False) == 16
    assert velocity_calls("midpoint", 8) == 16
    assert velocity_calls("euler", 8) == 8


def test_guidance_of_one_is_the_conditional_field() -> None:
    conditional = lambda x, t: x + 1  # noqa: E731
    assert guided(conditional, lambda x, t: x * 0, 1.0) is conditional


def test_guidance_extrapolates_from_unconditional_to_conditional() -> None:
    velocity = guided(lambda x, t: x + 2, lambda x, t: x, 3.0)
    assert velocity(torch.zeros(1), torch.zeros(1)).item() == 6.0
