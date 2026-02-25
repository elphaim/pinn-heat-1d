"""Tests for models/heat_pinn.py"""

import pytest
import torch
import numpy as np
from models.heat_pinn import HeatPINN, analytical_solution


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def forward_model():
    return HeatPINN(layers=[2, 20, 20, 1], alpha_true=0.01, inverse=False)


@pytest.fixture
def inverse_model():
    return HeatPINN(layers=[2, 20, 20, 1], inverse=True, alpha_init=0.02)


@pytest.fixture
def sample_points():
    """10 collocation points with gradients enabled."""
    x = torch.linspace(0, 1, 10).reshape(-1, 1).requires_grad_(True)
    t = torch.linspace(0, 1, 10).reshape(-1, 1).requires_grad_(True)
    return x, t


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

class TestForwardPass:
    def test_output_shape(self, forward_model, sample_points):
        x, t = sample_points
        u = forward_model(x, t)
        assert u.shape == (10, 1)

    def test_output_is_finite(self, forward_model, sample_points):
        x, t = sample_points
        u = forward_model(x, t)
        assert torch.isfinite(u).all()

    def test_batch_size_independence(self, forward_model):
        """Output at a given point should not depend on batch size."""
        x1 = torch.tensor([[0.5]]).requires_grad_(False)
        t1 = torch.tensor([[0.3]]).requires_grad_(False)
        u_single = forward_model(x1, t1)

        x_batch = torch.tensor([[0.5], [0.2], [0.8]])
        t_batch = torch.tensor([[0.3], [0.1], [0.9]])
        u_batch = forward_model(x_batch, t_batch)

        assert torch.allclose(u_single, u_batch[0:1], atol=1e-12)


# ---------------------------------------------------------------------------
# Residual
# ---------------------------------------------------------------------------

class TestResidual:
    def test_residual_shape(self, forward_model, sample_points):
        x, t = sample_points
        r = forward_model.residual(x, t)
        assert r.shape == (10, 1)

    def test_residual_is_finite(self, forward_model, sample_points):
        x, t = sample_points
        r = forward_model.residual(x, t)
        assert torch.isfinite(r).all()

    def test_residual_gradient_flows_to_params(self, forward_model, sample_points):
        x, t = sample_points
        r = forward_model.residual(x, t)
        loss = (r ** 2).mean()
        loss.backward()
        grads = [p.grad for p in forward_model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients reached model parameters"
        assert all(torch.isfinite(g).all() for g in grads)


# ---------------------------------------------------------------------------
# Inverse problem
# ---------------------------------------------------------------------------

class TestInverseProblem:
    def test_alpha_is_learnable(self, inverse_model):
        assert inverse_model.alpha.requires_grad is True

    def test_alpha_initial_value(self, inverse_model):
        assert abs(inverse_model.get_alpha() - 0.02) < 1e-8

    def test_alpha_receives_gradient(self, inverse_model, sample_points):
        x, t = sample_points
        r = inverse_model.residual(x, t)
        loss = (r ** 2).mean()
        loss.backward()
        assert inverse_model.alpha.grad is not None
        assert torch.isfinite(inverse_model.alpha.grad).all()

    def test_forward_model_alpha_is_fixed(self, forward_model):
        assert not forward_model.alpha.requires_grad


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class TestLossFunction:
    def test_loss_returns_scalar_and_dict(self, forward_model):
        N = 10
        x_f = torch.rand(N, 1, requires_grad=True)
        t_f = torch.rand(N, 1, requires_grad=True)
        x_bc = torch.rand(N, 1)
        t_bc = torch.rand(N, 1)
        u_bc = torch.zeros(N, 1)
        x_ic = torch.rand(N, 1)
        t_ic = torch.zeros(N, 1)
        u_ic = torch.sin(np.pi * x_ic)

        total, losses = forward_model.loss_function(
            x_f, t_f, x_bc, t_bc, u_bc, x_ic, t_ic, u_ic
        )
        assert total.dim() == 0, "Total loss should be a scalar"
        assert isinstance(losses, dict)
        for key in ("total", "residual", "boundary", "initial"):
            assert key in losses

    def test_loss_is_non_negative(self, forward_model):
        N = 10
        x_f = torch.rand(N, 1, requires_grad=True)
        t_f = torch.rand(N, 1, requires_grad=True)
        x_bc = torch.rand(N, 1)
        t_bc = torch.rand(N, 1)
        u_bc = torch.zeros(N, 1)
        x_ic = torch.rand(N, 1)
        t_ic = torch.zeros(N, 1)
        u_ic = torch.sin(np.pi * x_ic)

        total, _ = forward_model.loss_function(
            x_f, t_f, x_bc, t_bc, u_bc, x_ic, t_ic, u_ic
        )
        assert total.item() >= 0


# ---------------------------------------------------------------------------
# Analytical solution
# ---------------------------------------------------------------------------

class TestAnalyticalSolution:
    def test_initial_condition(self):
        """At t=0, solution should equal sin(pi*x)."""
        x = np.linspace(0, 1, 50)
        t = np.zeros_like(x)
        u = analytical_solution(x, t, alpha=0.01)
        np.testing.assert_allclose(u, np.sin(np.pi * x), atol=1e-12)

    def test_boundary_conditions(self):
        """At x=0 and x=1, solution should be 0 for all t."""
        t = np.linspace(0, 1, 50)
        u_left = analytical_solution(np.zeros_like(t), t, alpha=0.01)
        u_right = analytical_solution(np.ones_like(t), t, alpha=0.01)
        np.testing.assert_allclose(u_left, 0.0, atol=1e-12)
        np.testing.assert_allclose(u_right, 0.0, atol=1e-12)

    def test_decay_in_time(self):
        """Solution should decay monotonically at the midpoint."""
        x = 0.5 * np.ones(100)
        t = np.linspace(0, 1, 100)
        u = analytical_solution(x, t, alpha=0.01)
        assert np.all(np.diff(u) <= 0), "Solution should decay in time"

    def test_predict_matches_forward(self, forward_model):
        """predict() should return the same values as forward() but as numpy."""
        x = torch.rand(5, 1)
        t = torch.rand(5, 1)
        u_forward = forward_model(x, t).detach().numpy()
        u_predict = forward_model.predict(x, t)
        np.testing.assert_allclose(u_forward, u_predict, atol=1e-12)