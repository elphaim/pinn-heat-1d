"""
Physics-Informed Neural Network for 1D Heat Equation

This module implements a PINN for solving both forward and inverse problems
of the 1D heat equation: 
    u_t = alpha * u_xx
using the weak form of the problem, see Wang et al. (2025)

Author: elphaim
Date: January 26, 2026
"""

import torch
import torch.nn as nn
import numpy as np
from torchquad import MonteCarlo, GaussLegendre, Simpson, Boole, set_up_backend
from typing import Tuple, Optional, Callable

# Set default torch dtype to float64 for integrals
torch.set_default_dtype(torch.float64)
# Set up torch backend for torchquad
set_up_backend(backend='torch', data_type='float64')


class HeatPINNWeak(nn.Module):
    """
    Physics-Informed Neural Network for the 1D heat equation.
    
    The network approximates u(x,t) where:
        u_t - alpha * u_xx = 0
    using a weak form, e.g. integrated against a set of test function
        
    For inverse problems, alpha is learned as a trainable parameter.
    For forward problems, alpha is fixed.
    
    Args:
        layers: List of layer sizes. Default: [2, 50, 50, 50, 50, 1]
                Input layer has size 2 (x, t), output has size 1 (u)
        alpha_true: True value of thermal diffusivity (for forward problem)
        inverse: If True, treat alpha as learnable parameter
        alpha_init: Initial guess for alpha (used in inverse problem)
    """

    def __init__(
            self,
            layers: list = [2, 50, 50, 50, 50, 1],
            alpha_true: Optional[float] = None,
            inverse: bool = False,
            alpha_init: float = 0.02,
    ):
        super(HeatPINNWeak, self).__init__()

        self.inverse = inverse
        
        # Build the neural network layers
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

        # Initialize weights using Xavier initialization
        self._initialize_weights()

        # Handle the thermal diffusivity parameter
        if inverse:
            # For inverse problem: alpha is learnable
            self.alpha = nn.Parameter(torch.tensor([alpha_init], dtype=torch.float64))
            print(f"Inverse problem mode: alpha initialized to {alpha_init}")
        else:
            # For forward problem: alpha is fixed
            if alpha_true is None:
                raise ValueError("Must provide alpha_true for forward problem")
            self.register_buffer('alpha', torch.tensor([alpha_true], dtype=torch.float64))
            print(f"Forward problem mode: alpha fixed to {alpha_true}")


    def _initialize_weights(self):
        """
        Initialize network weights using Xavier Normal initialization.
        """
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Spatial coordinates, shape (N, 1)
            t: Temporal coordinates, shape (N, 1)
            
        Returns:
            u: Temperature field, shape (N, 1)
        """
        # Concatenate inputs
        inputs = torch.cat([x, t], dim=1)  # Shape: (N, 2)
        
        # Pass through hidden layers with tanh activation
        out = inputs
        for i, layer in enumerate(self.layers[:-1]):
            out = torch.tanh(layer(out))
        
        # Output layer (no activation)
        u = self.layers[-1](out)
        
        return u
    

    def integrand(
            self, 
            x: torch.Tensor, 
            t: torch.Tensor, 
            test_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Uses automatic differentiation to compute derivatives
        
        Args:
            x: Spatial coordinates, shape (N, 1)
            t: Temporal coordinates, shape (N, 1)
            test_func: Test function ϕ(x, t) against which we integrate the PDE residual, shape (N, 1)
            
        Returns:
            integrand: PDE residual integrand after integration by parts, shape (N, 1)
        """
        # Test function and its derivatives
        phi = test_func(x, t)

        phi_x = torch.autograd.grad(
            outputs=phi,
            inputs=x,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
            retain_graph=True
        )[0]

        phi_t = torch.autograd.grad(
            outputs=phi,
            inputs=t,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
            retain_graph=True
        )[0]

        # Solution and its derivative
        u = self.forward(x, t)
        
        u_x = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        # PDE weak residual integrand
        integrand = u * phi_t - self.alpha * u_x * phi_x

        return integrand
    

    def compute_weak_residuals(
            self, 
            test_funcs: list, 
            test_doms: list,
            method: str, 
            n_points: int,
    ) -> torch.Tensor:
        """
        Compute weak residuals for all test functions.
    
        Args:
            test_funcs: List of test functions
            test_doms: List of integration domains
            method: Integration method
            n_points: Number of integration sample points, method-dependent
        
        Returns:
            residuals: Tensor of shape (n_test_funcs,)
        """
        # Validate inputs
        if len(test_funcs) != len(test_doms):
            raise ValueError(f"Mismatch: {len(test_funcs)} funcs, {len(test_doms)} doms")

        # Initialize results
        residuals = []

        # Choice of integrator
        if method == 'Monte Carlo':
            integrator = MonteCarlo()
        elif method == 'Simpson':
            integrator = Simpson()
        elif method == 'Boole':
            integrator = Boole()
        elif method == 'Gauss Legendre':
            integrator = GaussLegendre()
        else:
            print(f"Warning: unknown integration method {method}")
            return torch.zeros(len(test_funcs),)

        # Shrink the domains so the integrator samples points that are strictly inside the compact support
        # eps depends on support_radius in the test function generator (harmonize later, for now fixed at 2.5% shrinkage)
        eps = 0.01
        shrunk_doms = [[[x0 + eps, x1 - eps], [t0 + eps, t1 - eps]] for [[x0, x1], [t0, t1]] in test_doms]

        for i, (phi_func, dom) in enumerate(zip(test_funcs, shrunk_doms)):

            # Wrapper for torchquad that requires a Callable of a single variable
            def integrand_2d(inputs: torch.Tensor) -> torch.Tensor:
                """inputs shape: (N, 2) where [:, 0] is x, [:, 1] is t"""
                x = inputs[:, 0:1].clone().requires_grad_(True)
                t = inputs[:, 1:2].clone().requires_grad_(True)
                return self.integrand(x, t, phi_func)
            
            # Integrate
            try:
                weak_res = integrator.integrate(
                    integrand_2d,
                    dim=2,
                    N=n_points,
                    integration_domain=dom,
                    backend='torch'
                )

                # Check for numerical issues
                if torch.isnan(weak_res) or torch.isinf(weak_res):
                    print(f"Warning: NaN/Inf for test function {i}, setting to 0")
                    weak_res = torch.tensor(0.0)

                residuals.append(weak_res)
            
            except Exception as e:
                print(f"Warning: Integration failed for test function {i}: {e}")
                print(f"  Domain: {dom}")
                residuals.append(torch.tensor(0.0))

        residuals_tensor = torch.stack(residuals)

        # Report statistics (optional, uncomment if needed for diagnostics)
        #
        #nonzero_count = (residuals_tensor.abs() > 1e-8).sum().item()
        #print(f"Weak residuals: {nonzero_count}/{len(residuals)} non-zero "
        #  f"({100*nonzero_count/len(residuals):.1f}%)")
        #if nonzero_count < len(residuals) * 0.5:
        #    print(f"Warning: >50% of residuals are zero.")

        return residuals_tensor
    
    
    def loss_function(
        self,
        test_funcs: list,
        test_doms: list,
        method: str,
        n_points: int,
        x_bc: torch.Tensor,
        t_bc: torch.Tensor,
        u_bc: torch.Tensor,
        x_ic: torch.Tensor,
        t_ic: torch.Tensor,
        u_ic: torch.Tensor,
        x_m: Optional[torch.Tensor] = None,
        t_m: Optional[torch.Tensor] = None,
        u_m: Optional[torch.Tensor] = None,
        lambda_f: float = 1.0,
        lambda_bc: float = 1.0,
        lambda_ic: float = 1.0,
        lambda_m: float = 1.0
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total loss (MSE) for PINN training.
        
        Args:
            test_funcs: List of test functions for weak PDE residual
            test_doms: List of support domains for test functions
            method: Integration method
            n_points: Number of integration sample points, method-dependent
            x_bc, t_bc, u_bc: Boundary condition points and values
            x_ic, t_ic, u_ic: Initial condition points and values
            x_m, t_m, u_m: Measurement points (for inverse problem)
            lambda_f, lambda_bc, lambda_ic, lambda_m: Loss weights
            
        Returns:
            total_loss: Weighted sum of all loss components
            losses: Dictionary with individual loss components
        """
        # 1. PDE Weak Residual Loss
        weak_residuals = self.compute_weak_residuals(test_funcs, test_doms, method, n_points)
        loss_f = torch.mean(weak_residuals ** 2)
        
        # 2. Boundary Condition Loss
        u_bc_pred = self.forward(x_bc, t_bc)
        loss_bc = torch.mean((u_bc_pred - u_bc) ** 2)
        
        # 3. Initial Condition Loss
        u_ic_pred = self.forward(x_ic, t_ic)
        loss_ic = torch.mean((u_ic_pred - u_ic) ** 2)
        
        # 4. Measurement Loss (for inverse problem)
        if x_m is not None and t_m is not None and u_m is not None:
            u_m_pred = self.forward(x_m, t_m)
            loss_m = torch.mean((u_m_pred - u_m) ** 2)
        else:
            loss_m = torch.tensor(0.0)
        
        # Total weighted loss
        total_loss = (
            lambda_f * loss_f +
            lambda_bc * loss_bc +
            lambda_ic * loss_ic +
            lambda_m * loss_m
        )
        
        # Store individual losses for monitoring
        losses = {
            'total': total_loss.item(),
            'residual': loss_f.item(),
            'boundary': loss_bc.item(),
            'initial': loss_ic.item(),
            'measurement': loss_m.item() if torch.is_tensor(loss_m) else 0.0,
        }
        
        return total_loss, losses
    

    def predict(self, x: torch.Tensor, t: torch.Tensor) -> np.ndarray:
        """
        Predict temperature field at given points.
        
        Args:
            x: Spatial coordinates
            t: Temporal coordinates
            
        Returns:
            u: Predicted temperature (numpy array)
        """
        self.eval()
        with torch.no_grad():
            u = self.forward(x, t)
        return u.cpu().numpy()
    
    
    def get_alpha(self) -> float:
        """
        Return current value of alpha parameter.
        """
        return self.alpha.item()
    
    
# Utility function for analytical solution (validation)
def analytical_solution(x: np.ndarray, t: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Analytical solution for u_t = alpha * u_xx with:
        - Initial condition: u(x, 0) = sin(pi * x)
        - Boundary conditions: u(0, t) = u(1, t) = 0
    
    Solution: u(x, t) = sin(pi * x) * exp(-alpha * pi^2 * t)
    
    Args:
        x: Spatial coordinates (0 to 1)
        t: Temporal coordinates (0 to T)
        alpha: Thermal diffusivity
        
    Returns:
        u: Temperature field
    """
    return np.sin(np.pi * x) * np.exp(-alpha * np.pi**2 * t)


# Example usage and testing
if __name__ == "__main__":
    print("Testing HeatPINN implementation...\n")
    
    # Test 1: Forward problem
    print("=" * 60)
    print("Test 1: Forward Problem")
    print("=" * 60)

    def test_function(x, t):
        return torch.exp(-((x-0.5)**2 + (t-0.5)**2) / 0.1)
    
    model_forward = HeatPINNWeak(
        layers=[2, 50, 50, 50, 50, 1],
        alpha_true=0.01,
        inverse=False
    )
    print(f"Model architecture: {model_forward}")
    print(f"Number of parameters: {sum(p.numel() for p in model_forward.parameters())}")
    
    # Test forward pass
    x_test = torch.linspace(0, 1, 10).reshape(-1, 1)
    t_test = torch.linspace(0, 1, 10).reshape(-1, 1)
    u_pred = model_forward(x_test, t_test)
    print(f"Output shape: {u_pred.shape}")
    
    # Test residual computation
    x_test = x_test.requires_grad_(True)
    t_test = t_test.requires_grad_(True)
    residual = model_forward.compute_weak_residuals([test_function], [[[0,1],[0,1]]], method='Simpson', n_points=11)
    print(f"Residual shape: {residual.shape}")
    print(f"Residual mean before training: {residual.mean().item():.6f}")
    
    # Test 2: Inverse problem
    print("\n" + "=" * 60)
    print("Test 2: Inverse Problem")
    print("=" * 60)
    
    model_inverse = HeatPINNWeak(
        layers=[2, 50, 50, 50, 50, 1],
        inverse=True,
        alpha_init=0.02
    )
    print(f"Initial alpha guess: {model_inverse.get_alpha():.4f}")
    print(f"Alpha is learnable: {model_inverse.alpha.requires_grad}")
    
    # Test 3: Analytical solution
    print("\n" + "=" * 60)
    print("Test 3: Analytical Solution")
    print("=" * 60)
    
    x_np = np.linspace(0, 1, 5)
    t_np = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    
    for t_val in t_np:
        u_analytical = analytical_solution(x_np, t_val * np.ones_like(x_np), alpha=0.01)
        print(f"t = {t_val:.2f}: u_max = {u_analytical.max():.4f}, u_mean = {u_analytical.mean():.4f}")
    
    print("\nAll tests passed. Ready to train.")