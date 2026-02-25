"""Tests for data/heat_data.py"""

import pytest
import torch
import numpy as np
from data.heat_data import HeatEquationData, prepare_multi_fidelity_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def data_gen():
    return HeatEquationData(
        L=1.0, T=1.0, alpha=0.01,
        N_f=100, N_bc=20, N_ic=20,
        N_sensors=5, N_time_measurements=5,
        noise_level=0.01,
        random_seed=42,
    )


# ---------------------------------------------------------------------------
# Collocation points
# ---------------------------------------------------------------------------

class TestCollocationPoints:
    def test_shape(self, data_gen):
        x_f, t_f = data_gen.generate_collocation_points(method="uniform")
        assert x_f.shape == (100, 1)
        assert t_f.shape == (100, 1)

    def test_domain_bounds(self, data_gen):
        x_f, t_f = data_gen.generate_collocation_points()
        assert x_f.min() >= 0.0
        assert x_f.max() <= 1.0
        assert t_f.min() >= 0.0
        assert t_f.max() <= 1.0

    def test_lhs_method(self, data_gen):
        x_f, t_f = data_gen.generate_collocation_points(method="lhs")
        assert x_f.shape == (100, 1)


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

class TestBoundaryConditions:
    def test_shape(self, data_gen):
        x_bc, t_bc, u_bc = data_gen.generate_boundary_conditions()
        assert x_bc.shape[0] == 2 * data_gen.N_bc  # left + right
        assert u_bc.shape == x_bc.shape

    def test_values_are_zero(self, data_gen):
        _, _, u_bc = data_gen.generate_boundary_conditions()
        assert torch.allclose(u_bc, torch.zeros_like(u_bc))

    def test_locations(self, data_gen):
        x_bc, _, _ = data_gen.generate_boundary_conditions()
        # All points should be at x=0 or x=1
        at_boundary = (x_bc.abs() < 1e-12) | ((x_bc - 1.0).abs() < 1e-12)
        assert at_boundary.all()


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

class TestInitialConditions:
    def test_shape(self, data_gen):
        x_ic, t_ic, u_ic = data_gen.generate_initial_conditions()
        assert x_ic.shape == (20, 1)
        assert t_ic.shape == (20, 1)

    def test_time_is_zero(self, data_gen):
        _, t_ic, _ = data_gen.generate_initial_conditions()
        assert torch.allclose(t_ic, torch.zeros_like(t_ic))

    def test_values_match_sin(self, data_gen):
        x_ic, _, u_ic = data_gen.generate_initial_conditions()
        expected = torch.sin(np.pi * x_ic)
        assert torch.allclose(u_ic, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

class TestMeasurements:
    def test_shape(self, data_gen):
        x_m, t_m, u_m, info = data_gen.generate_measurements()
        expected_n = data_gen.N_sensors * data_gen.N_time_measurements
        assert x_m.shape == (expected_n, 1)

    def test_noiseless_matches_analytical(self, data_gen):
        x_m, t_m, u_m, info = data_gen.generate_measurements(add_noise=False)
        u_exact = data_gen.analytical_solution(
            x_m.numpy().flatten(), t_m.numpy().flatten()
        )
        np.testing.assert_allclose(
            u_m.numpy().flatten(), u_exact, atol=1e-12
        )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_data(self):
        gen1 = HeatEquationData(N_f=50, N_bc=10, N_ic=10, random_seed=123)
        gen2 = HeatEquationData(N_f=50, N_bc=10, N_ic=10, random_seed=123)

        x1, t1 = gen1.generate_collocation_points()
        x2, t2 = gen2.generate_collocation_points()
        assert torch.allclose(x1, x2)
        assert torch.allclose(t1, t2)

    def test_different_seed_different_data(self):
        gen1 = HeatEquationData(N_f=50, random_seed=1)
        gen2 = HeatEquationData(N_f=50, random_seed=2)

        x1, _ = gen1.generate_collocation_points()
        x2, _ = gen2.generate_collocation_points()
        assert not torch.allclose(x1, x2)


# ---------------------------------------------------------------------------
# Finite difference solver
# ---------------------------------------------------------------------------

class TestFiniteDifference:
    def test_output_shapes(self, data_gen):
        x, t, u = data_gen.finite_difference_solution(nx=11, nt=21)
        assert x.shape == (11,)
        assert t.shape == (21,)
        assert u.shape == (11, 21)

    def test_boundary_conditions_satisfied(self, data_gen):
        x, t, u = data_gen.finite_difference_solution(nx=21, nt=101)
        np.testing.assert_allclose(u[0, :], 0.0, atol=1e-12)
        np.testing.assert_allclose(u[-1, :], 0.0, atol=1e-12)

    def test_initial_condition(self, data_gen):
        x, t, u = data_gen.finite_difference_solution(nx=21, nt=101)
        expected = np.sin(np.pi * x)
        np.testing.assert_allclose(u[:, 0], expected, atol=1e-12)

    def test_reasonable_accuracy(self, data_gen):
        """FD with fine grid and correct alpha should approximate analytical well."""
        x, t, u = data_gen.finite_difference_solution(nx=101, nt=10001)
        X, T = np.meshgrid(x, t, indexing="ij")
        u_exact = data_gen.analytical_solution(X, T)
        rmse = np.sqrt(np.mean((u - u_exact) ** 2))
        assert rmse < 1e-3


# ---------------------------------------------------------------------------
# Full dataset
# ---------------------------------------------------------------------------

class TestFullDataset:
    def test_contains_all_keys(self, data_gen):
        data = data_gen.generate_full_dataset()
        required = {"x_f", "t_f", "x_bc", "t_bc", "u_bc",
                     "x_ic", "t_ic", "u_ic", "x_m", "t_m", "u_m"}
        assert required.issubset(data.keys())


# ---------------------------------------------------------------------------
# Multi-fidelity data preparation
# ---------------------------------------------------------------------------

class TestMultiFidelityData:
    def test_contains_all_keys(self, data_gen):
        mf = prepare_multi_fidelity_data(data_gen, hf_sensors=3, hf_times=3,
                                          lf_nx=6, lf_nt=11)
        required = {"x_f", "t_f", "x_bc", "t_bc", "u_bc",
                     "x_ic", "t_ic", "u_ic",
                     "x_hf", "t_hf", "u_hf",
                     "x_lf", "t_lf", "u_lf",
                     "sigma_hf", "sigma_lf"}
        assert required.issubset(mf.keys())

    def test_hf_has_fewer_points_than_lf(self, data_gen):
        mf = prepare_multi_fidelity_data(data_gen, hf_sensors=3, hf_times=3,
                                          lf_nx=11, lf_nt=31)
        assert mf["x_hf"].shape[0] < mf["x_lf"].shape[0]

    def test_sigma_hf_less_than_sigma_lf(self, data_gen):
        mf = prepare_multi_fidelity_data(data_gen, hf_sensors=3, hf_times=3,
                                          lf_nx=6, lf_nt=11)
        assert mf["sigma_hf"] < mf["sigma_lf"]