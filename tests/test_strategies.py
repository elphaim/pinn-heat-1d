"""Tests for models/heat_pinn_strategy.py"""

import pytest
import torch
import numpy as np
from models.heat_pinn_strategy import (
    StrategicPINN,
    StrongFormLoss,
    WeakFormLoss,
    MultiFidelityLoss,
)
from utils.test_functions import generate_compact_gaussians


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    return StrategicPINN(layers=[2, 20, 20, 1], alpha_true=0.01, inverse=False)


@pytest.fixture
def base_data():
    """Minimal dataset for testing loss computation."""
    N = 20
    x_f = torch.rand(N, 1)
    t_f = torch.rand(N, 1)
    x_bc = torch.cat([torch.zeros(N, 1), torch.ones(N, 1)], dim=0)
    t_bc = torch.rand(2 * N, 1)
    u_bc = torch.zeros(2 * N, 1)
    x_ic = torch.rand(N, 1)
    t_ic = torch.zeros(N, 1)
    u_ic = torch.sin(np.pi * x_ic)
    return {
        "x_f": x_f, "t_f": t_f,
        "x_bc": x_bc, "t_bc": t_bc, "u_bc": u_bc,
        "x_ic": x_ic, "t_ic": t_ic, "u_ic": u_ic,
    }


@pytest.fixture
def mf_data(base_data):
    """Extend base data with multi-fidelity fields."""
    N_hf, N_lf = 10, 30
    data = dict(base_data)
    data["x_hf"] = torch.rand(N_hf, 1)
    data["t_hf"] = torch.rand(N_hf, 1)
    data["u_hf"] = torch.sin(np.pi * data["x_hf"])  # approximate IC values
    data["x_lf"] = torch.rand(N_lf, 1)
    data["t_lf"] = torch.rand(N_lf, 1)
    data["u_lf"] = torch.sin(np.pi * data["x_lf"]) + 0.05 * torch.randn(N_lf, 1)
    data["sigma_hf"] = 0.01
    data["sigma_lf"] = 0.1
    return data


# ---------------------------------------------------------------------------
# StrongFormLoss
# ---------------------------------------------------------------------------

class TestStrongFormLoss:
    def test_returns_scalar_and_dict(self, model, base_data):
        model.set_loss_strategy(StrongFormLoss())
        total, losses = model.compute_loss(base_data)
        assert total.dim() == 0
        assert isinstance(losses, dict)
        for key in ("total", "residual", "boundary", "initial"):
            assert key in losses

    def test_loss_is_positive(self, model, base_data):
        model.set_loss_strategy(StrongFormLoss())
        total, _ = model.compute_loss(base_data)
        assert total.item() > 0

    def test_gradient_flows(self, model, base_data):
        model.set_loss_strategy(StrongFormLoss())
        total, _ = model.compute_loss(base_data)
        total.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)


# ---------------------------------------------------------------------------
# WeakFormLoss
# ---------------------------------------------------------------------------

class TestWeakFormLoss:
    def test_returns_scalar_and_dict(self, model, base_data):
        model.set_loss_strategy(
            WeakFormLoss(integration_method="gauss_legendre", n_integration_points=5)
        )
        test_funcs, test_doms = generate_compact_gaussians(
            n_funcs=4, support_radius=0.3
        )
        data = dict(base_data)
        data["test_funcs"] = test_funcs
        data["test_doms"] = test_doms
        total, losses = model.compute_loss(data)
        assert total.dim() == 0
        assert "weak_res_nonzero" in losses

    def test_gradient_flows(self, model, base_data):
        model.set_loss_strategy(
            WeakFormLoss(integration_method="gauss_legendre", n_integration_points=5)
        )
        test_funcs, test_doms = generate_compact_gaussians(
            n_funcs=4, support_radius=0.3
        )
        data = dict(base_data)
        data["test_funcs"] = test_funcs
        data["test_doms"] = test_doms
        total, _ = model.compute_loss(data)
        total.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0


# ---------------------------------------------------------------------------
# MultiFidelityLoss
# ---------------------------------------------------------------------------

class TestMultiFidelityLoss:
    def test_uncertainty_weighting(self, model, mf_data):
        model.set_loss_strategy(MultiFidelityLoss(weighting="uncertainty"))
        total, losses = model.compute_loss(mf_data)
        assert total.dim() == 0
        assert "measurement_hf" in losses
        assert "measurement_lf" in losses

    def test_fixed_weighting(self, model, mf_data):
        model.set_loss_strategy(
            MultiFidelityLoss(weighting="fixed", lambda_hf=1.0, lambda_lf=0.1)
        )
        total, losses = model.compute_loss(mf_data)
        assert total.item() > 0
        assert losses["effective_lambda_hf"] == 1.0
        assert losses["effective_lambda_lf"] == 0.1

    def test_gradient_flows(self, model, mf_data):
        model.set_loss_strategy(MultiFidelityLoss(weighting="uncertainty"))
        total, _ = model.compute_loss(mf_data)
        total.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0


# ---------------------------------------------------------------------------
# Strategy switching
# ---------------------------------------------------------------------------

class TestStrategySwitching:
    def test_switch_does_not_crash(self, model, base_data):
        model.set_loss_strategy(StrongFormLoss())
        total_1, _ = model.compute_loss(base_data)
        assert torch.isfinite(total_1)

        # Switch back to strong form (the default)
        model.set_loss_strategy(StrongFormLoss())
        total_2, _ = model.compute_loss(base_data)
        assert torch.isfinite(total_2)

    def test_default_strategy_is_strong_form(self, model):
        assert isinstance(model.loss_strategy, StrongFormLoss)