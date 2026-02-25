"""Tests for utils/integrator.py"""

import pytest
import torch
import numpy as np
from utils.integrator import (
    IntegratorFactory,
    GaussLegendreIntegrator,
    SimpsonIntegrator,
    MonteCarloIntegrator,
)


DOMAIN = [[0.0, 1.0], [0.0, 1.0]]


# ---------------------------------------------------------------------------
# Known integrals
# ---------------------------------------------------------------------------

def constant_one(x, t):
    """∫∫ 1 dx dt = 1.0 over [0,1]²."""
    return torch.ones_like(x)


def polynomial(x, t):
    """∫∫ (x² + t²) dx dt = 2/3 over [0,1]²."""
    return x ** 2 + t ** 2


def separable_trig(x, t):
    """∫∫ sin(πx)·cos(πt) dx dt = 0 over [0,1]²."""
    return torch.sin(np.pi * x) * torch.cos(np.pi * t)


# ---------------------------------------------------------------------------
# Gauss–Legendre
# ---------------------------------------------------------------------------

class TestGaussLegendre:
    def test_constant(self):
        gl = GaussLegendreIntegrator(n_points=5)
        result = gl.integrate(constant_one, DOMAIN)
        assert abs(result.item() - 1.0) < 1e-12

    def test_polynomial(self):
        gl = GaussLegendreIntegrator(n_points=10)
        result = gl.integrate(polynomial, DOMAIN)
        assert abs(result.item() - 2.0 / 3.0) < 1e-10

    def test_separable_trig(self):
        gl = GaussLegendreIntegrator(n_points=15)
        result = gl.integrate(separable_trig, DOMAIN)
        assert abs(result.item()) < 1e-10


# ---------------------------------------------------------------------------
# Simpson
# ---------------------------------------------------------------------------

class TestSimpson:
    def test_constant(self):
        s = SimpsonIntegrator(n_points=5)
        result = s.integrate(constant_one, DOMAIN)
        assert abs(result.item() - 1.0) < 1e-10

    def test_polynomial(self):
        s = SimpsonIntegrator(n_points=21)
        result = s.integrate(polynomial, DOMAIN)
        assert abs(result.item() - 2.0 / 3.0) < 1e-6

    def test_forces_odd_n_points(self):
        s = SimpsonIntegrator(n_points=10)
        assert s.n_points % 2 == 1


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

class TestMonteCarlo:
    def test_polynomial_rough_accuracy(self):
        """MC should get within ~1% of 2/3 with enough samples."""
        mc = MonteCarloIntegrator(n_samples=100_000)
        result = mc.integrate(polynomial, DOMAIN, seed=42)
        assert abs(result.item() - 2.0 / 3.0) < 0.01


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:
    @pytest.mark.parametrize("method", ["gauss_legendre", "simpson", "monte_carlo"])
    def test_create_returns_integrator(self, method):
        integrator = IntegratorFactory.create(method, n_points=10)
        assert hasattr(integrator, "integrate")

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            IntegratorFactory.create("nonexistent")


# ---------------------------------------------------------------------------
# Gradient preservation
# ---------------------------------------------------------------------------

class TestGradientPreservation:
    @pytest.mark.parametrize("method", ["gauss_legendre", "simpson"])
    def test_gradient_through_integral(self, method):
        """Integral of α·x·t over [0,1]² = α/4. Gradient w.r.t. α should be 0.25."""
        alpha = torch.tensor(2.0, requires_grad=True)

        def func(x, t):
            return alpha * x * t

        integrator = IntegratorFactory.create(method, n_points=10)
        result = integrator.integrate(func, DOMAIN)
        result.backward()

        assert alpha.grad is not None
        assert abs(alpha.grad.item() - 0.25) < 1e-6