"""
Data generation module for 1D Heat Equation PINN

Generates:
1. Smooth, compactly-supported Gaussian test functions for PDE weak residual
2. Boundary and initial condition points
3. Synthetic measurements with noise (for inverse problem)
4. Ground truth analytical solutions

Author: elphaim
Date: January 26, 2026
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from typing import Tuple, Dict, Optional

# Set default dtype for torch
torch.set_default_dtype(torch.float64)

class HeatEquationDataWeak:
    """
    Data generator for the 1D heat equation problem.
    
    Domain: x ∈ [0, L], t ∈ [0, T]
    PDE: u_t = alpha * u_xx
    IC: u(x, 0) = sin(π*x)
    BC: u(0, t) = u(L, t) = 0
    
    Args:
        L: Length of spatial domain, must be odd (default: 1.0)
        T: Final time (default: 1.0)
        alpha: True thermal diffusivity (default: 0.01)
        N_f: Number of Gaussian test functions for weak residual (default: 30)
        smooth: Smooth out the Gaussian test functions' support (default: True)
        N_bc: Number of collocation points for boundary conditions (default: 100)
        N_ic: Number of collocation points for initial conditions (default: 100)
        N_sensors: Number of spatial sensor locations (default: 10)
        N_time_measurements: Number of time measurements per sensor (default: 10)
        noise_level: Measurement noise std as fraction of signal (default: 0.01 for SNR~40dB)
        device: torch device (default: cpu)
    """
    
    def __init__(
        self,
        L: float = 1.0,
        T: float = 1.0,
        alpha: float = 0.01,
        N_f: int = 30,
        smooth: bool = True,
        N_bc: int = 100,
        N_ic: int = 100,
        N_sensors: int = 10,
        N_time_measurements: int = 10,
        noise_level: float = 0.01,
        device: str = 'cpu',
        random_seed: int = 42
    ):
        self.L = L
        self.T = T
        self.alpha = alpha
        self.N_f = N_f
        self.smooth = smooth
        self.N_bc = N_bc
        self.N_ic = N_ic
        self.N_sensors = N_sensors
        self.N_time_measurements = N_time_measurements
        self.noise_level = noise_level
        self.device = device
        
        # Set random seed for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        print(f"Data Generator initialized:")
        print(f"  Domain: x ∈ [0, {L}], t ∈ [0, {T}]")
        print(f"  True alpha: {alpha}")
        print(f"  Gaussian test functions: {N_f}")
        print(f"  Boundary points: {N_bc}")
        print(f"  Initial condition points: {N_ic}")
        print(f"  Measurements: {N_sensors} sensors × {N_time_measurements} times = {N_sensors * N_time_measurements} total")
        print(f"  Noise level: {noise_level:.1%} (SNR ≈ {-20*np.log10(noise_level):.0f} dB)")

    
    def analytical_solution(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Analytical solution for u_t = alpha * u_xx with:
            - Initial condition: u(x, 0) = sin(pi * x)
            - Boundary conditions: u(0, t) = u(1, t) = 0
    
        Solution: u(x, t) = sin(pi * x) * exp(-alpha * pi^2 * t)
    
        Args:
            x: Spatial coordinates (0 to L)
            t: Temporal coordinates (0 to T)
            alpha: Thermal diffusivity
        
        Returns:
            u: Temperature field
        """
        return np.sin(np.pi * x) * np.exp(-self.alpha * np.pi**2 * t)
    

    def generate_smooth_compact_gaussians(
        self,
        n_funcs: Optional[int] = None,
        support_radius: Optional[float] = None, 
        shrink: Optional[float] = None,
        min_separation: Optional[float] = None,
        smooth: Optional[bool] = None,
    ) -> Tuple[list, list]:
        """
        Generates a list of 2D compactly supported Gaussian test functions
    
        Args:
            n_funcs: Number of test functions to be generated (default: self.N_f)
            support_radius: Radius outside of which the function vanishes (default: min(self.L/5, self.T/5))
            shrink: offset for the x and t domains (default: support_radius so functions vanish on domain boundary for valid IBP)
            min_separation: Minimum distance between centers to prevent clustering (default: min(self.L/10, self.T/10))
            smooth: Smoothen Heaviside(R - r) with steep sigmoid
        
        Returns:
            test_funcs: List of callables, each accepting (x, t) tensors
            test_doms: List of support domain [[x_center - support_radius, x_center + support_radius], [t_center - support_radius, t_center + support_radius]]
        """
        # Global PDE domain
        domain = [[0, self.L], [0, self.T]]
        
        # Parameters
        if n_funcs is None:
            n_funcs = self.N_f
        
        if support_radius is None:
            support_radius = min(self.L/5, self.T/5)
        
        if shrink is None:
            shrink = support_radius
        # Validate the shrink
        x_range = domain[0][1] - domain[0][0]
        t_range = domain[1][1] - domain[1][0]
        if 2 * shrink >= min(x_range, t_range):
            raise ValueError(f"shrink ({shrink}) too large for domain")
        
        if min_separation is None:
            min_separation = min(self.L/10, self.T/10)

        if smooth is None:
            smooth = self.smooth

        # Initialize results
        test_funcs = []
        test_doms = []
        centers = []

        # Attempt to spread centers according to min_separation
        attempts = 0
        max_attempts = n_funcs * 100

        while len(test_funcs) < n_funcs and attempts < max_attempts:
            attempts += 1
            # Random center within shrunk domain
            x_center = np.random.uniform(domain[0][0] + shrink, domain[0][1] - shrink)
            t_center = np.random.uniform(domain[1][0] + shrink, domain[1][1] - shrink)
            # Check separation from existing centers
            if min_separation > 0:
                too_close = False
                for prev_center in centers:
                    dist = np.sqrt((x_center - prev_center[0])**2 + 
                                (t_center - prev_center[1])**2)
                    if dist < min_separation:
                        too_close = True
                        break
                if too_close:
                    continue
    
            # Create function with captured parameters
            def compact_gaussian(x, t, xc=x_center, tc=t_center, R=support_radius, smooth=self.smooth):
                """
                Compact support Gaussian: 
                φ(x,t) = Heaviside(R - r) * exp(-r²/(R² - r²))
                where r² = (x-xc)² + (t-tc)²

                Args:
                    smooth: Heaviside smoothed out using steep sigmoid
                """
                r_squared = (x - xc)**2 + (t - tc)**2
                R_squared = R**2

                if smooth:
                    # k is the smoothing factor: higher = steeper step
                    k = 1e8
                    # eps is a regularization factor to avoid the numerical issues at r = R
                    eps = 1e-8
                    # Smooth version
                    phi = torch.sigmoid(2*k*(R_squared - r_squared)) * torch.exp(-r_squared / (R_squared - r_squared + eps))
                else:
                    # Non-smooth version with Heaviside function
                    inside = r_squared < R_squared
                    phi = torch.where(inside, torch.exp(-r_squared / (R_squared - r_squared)), torch.zeros_like(x))

                return phi
        
            # Compute support domain clipped to global domain
            dom_xmin = max(x_center - support_radius, domain[0][0])
            dom_xmax = min(x_center + support_radius, domain[0][1])
            dom_tmin = max(t_center - support_radius, domain[1][0])
            dom_tmax = min(t_center + support_radius, domain[1][1])
        
            test_funcs.append(compact_gaussian)
            test_doms.append([[dom_xmin, dom_xmax], [dom_tmin, dom_tmax]])
            centers.append((x_center, t_center))

        if len(test_funcs) < n_funcs:
            print(f"Warning: Only generated {len(test_funcs)}/{n_funcs} test functions")
    
        return test_funcs, test_doms
    
    
    def generate_boundary_conditions(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate boundary condition points: u(0, t) = u(L, t) = 0
            
        Returns:
            x_bc: Spatial coordinates (at x=0 and x=L)
            t_bc: Temporal coordinates
            u_bc: Boundary values (all zeros)
        """
        # Left boundary (x=0)
        t_left = torch.rand(self.N_bc, 1) * self.T
        x_left = torch.zeros(self.N_bc, 1)
        
        # Right boundary (x=L)
        t_right = torch.rand(self.N_bc, 1) * self.T
        x_right = torch.ones(self.N_bc, 1) * self.L
        
        # Combine
        x_bc = torch.cat([x_left, x_right], dim=0)
        t_bc = torch.cat([t_left, t_right], dim=0)
        u_bc = torch.zeros_like(x_bc)  # Dirichlet BC: u = 0
        
        return x_bc.to(self.device), t_bc.to(self.device), u_bc.to(self.device)
    
    
    def generate_initial_conditions(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate initial condition points: u(x, 0) = sin(π*x)
            
        Returns:
            x_ic: Spatial coordinates
            t_ic: Temporal coordinates (all zeros)
            u_ic: Initial values
        """
        x_ic = torch.rand(self.N_ic, 1) * self.L
        t_ic = torch.zeros(self.N_ic, 1)
        u_ic = torch.sin(np.pi * x_ic)
        
        return x_ic.to(self.device), t_ic.to(self.device), u_ic.to(self.device)
    
    
    def generate_measurements(
        self, 
        add_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Generate synthetic sensor measurements with noise.
        
        Simulates sensors placed at regular intervals that measure temperature
        at regular time intervals.
        
        Args:
            add_noise: Whether to add Gaussian noise to measurements
            
        Returns:
            x_m: Measurement spatial coordinates
            t_m: Measurement temporal coordinates
            u_m: Measured temperature (with noise if add_noise=True)
            info: Dictionary with additional information (true values, SNR, etc.)
        """
        # Sensor locations (evenly spaced, excluding boundaries)
        x_sensors = np.linspace(self.L / (self.N_sensors + 1), 
                               self.L * self.N_sensors / (self.N_sensors + 1), 
                               self.N_sensors)
        
        # Measurement times (evenly spaced)
        t_measurements = np.linspace(0, self.T, self.N_time_measurements)
        
        # Create meshgrid
        X_mesh, T_mesh = np.meshgrid(x_sensors, t_measurements)
        x_m_np = X_mesh.flatten()
        t_m_np = T_mesh.flatten()
        
        # Compute true values using analytical solution
        u_true = self.analytical_solution(x_m_np, t_m_np)
        
        # Add noise
        if add_noise:
            noise = np.random.normal(0, self.noise_level * np.abs(u_true))
            u_measured = u_true + noise
            
            # Compute actual SNR
            signal_power = np.mean(u_true ** 2)
            noise_power = np.mean(noise ** 2)
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            u_measured = u_true
            snr_db = np.inf
        
        # Convert to torch tensors
        x_m = torch.tensor(x_m_np.reshape(-1, 1), dtype=torch.float64)
        t_m = torch.tensor(t_m_np.reshape(-1, 1), dtype=torch.float64)
        u_m = torch.tensor(u_measured.reshape(-1, 1), dtype=torch.float64)
        
        # Store metadata
        info = {
            'x_sensors': x_sensors,
            't_measurements': t_measurements,
            'u_true': u_true,
            'u_measured': u_measured,
            'snr_db': snr_db,
            'noise_std': self.noise_level * np.abs(u_true).mean()
        }
        
        print(f"\nMeasurements generated:")
        print(f"  Sensor locations: {x_sensors}")
        print(f"  Time points: {t_measurements}")
        print(f"  Total measurements: {len(x_m)}")
        if add_noise:
            print(f"  Actual SNR: {snr_db:.1f} dB")
        
        return x_m.to(self.device), t_m.to(self.device), u_m.to(self.device), info
    

    def generate_full_dataset(
        self, 
    ) -> Tuple[Dict[str, list], Dict[str, torch.Tensor]]:
        """
        Generate complete dataset for training.
        
        Returns:
            data_funcs: Dictionary containing the test function data

            data_pts: Dictionary containing the points data:
                - 'x_bc', 't_bc', 'u_bc': Boundary conditions
                - 'x_ic', 't_ic', 'u_ic': Initial conditions
                - 'x_m', 't_m', 'u_m': Measurements
                - 'measurement_info': Additional measurement metadata
        """
        print("\n" + "=" * 60)
        print("Generating complete dataset...")
        print("=" * 60)
        
        # Generate all data
        test_funcs, test_doms = self.generate_smooth_compact_gaussians()
        x_bc, t_bc, u_bc = self.generate_boundary_conditions()
        x_ic, t_ic, u_ic = self.generate_initial_conditions()
        x_m, t_m, u_m, info = self.generate_measurements()
        
        data_funcs = {
            'test_funcs': test_funcs, 'test_doms': test_doms,
        }
        
        data_pts = {
            'x_bc': x_bc, 't_bc': t_bc, 'u_bc': u_bc,
            'x_ic': x_ic, 't_ic': t_ic, 'u_ic': u_ic,
            'x_m': x_m, 't_m': t_m, 'u_m': u_m,
            'measurement_info': info
        }
        
        print("\nDataset generation complete.")
        return data_funcs, data_pts


    def visualize_data(self, data_funcs: Dict[str, list], data_pts: Dict[str, torch.Tensor], save_path: Optional[str] = None):
        """
        Visualize the generated dataset.
        
        Args:
            data_funcs, data_pts: Dictionaries from generate_full_dataset()
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Get test function data
        test_funcs = data_funcs['test_funcs']
        test_doms = data_funcs['test_doms']
        domain = [[0, self.L], [0, self.T]]
 
        # Convert points data to numpy for plotting
        x_bc = data_pts['x_bc'].cpu().numpy()
        t_bc = data_pts['t_bc'].cpu().numpy()
        x_ic = data_pts['x_ic'].cpu().numpy()
        t_ic = data_pts['t_ic'].cpu().numpy()
        x_m = data_pts['x_m'].cpu().numpy()
        t_m = data_pts['t_m'].cpu().numpy()
        u_m = data_pts['u_m'].cpu().numpy()
        
        # Plot 1: Test functions
        assert len(test_funcs) == len(test_doms)
        # Grid
        x = torch.linspace(domain[0][0], domain[0][1], 200)
        t = torch.linspace(domain[1][0], domain[1][1], 200)
        X, T = torch.meshgrid(x, t, indexing="ij")
        # Accumulate all Gaussians
        Z_total = torch.zeros_like(X)
        with torch.no_grad():
            for func in test_funcs:
                Z_total += func(X, T)
        # Plot combined field
        axes[0, 0].imshow(
            Z_total.numpy(),
            extent=(domain[0][0], domain[0][1],
                    domain[1][0], domain[1][1]),
            origin="lower",
            cmap="viridis",
            alpha=0.8,
            aspect="auto"
        )
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('t')
        axes[0, 0].set_title(f'Test Gaussians (N={self.N_f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Boundary and Initial Conditions
        axes[0, 1].scatter(x_bc, t_bc, s=10, c='red', label='BC', alpha=0.5)
        axes[0, 1].scatter(x_ic, t_ic, s=10, c='green', label='IC', alpha=0.5)
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('t')
        axes[0, 1].set_title(f'Boundary (N={len(x_bc)}) & Initial Conditions (N={len(x_ic)})')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Measurement locations
        axes[1, 0].scatter(x_m, t_m, s=50, c='orange', marker='x', linewidths=2)
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('t')
        axes[1, 0].set_title(f'Measurement Locations (N={len(x_m)})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Measured temperature values
        scatter = axes[1, 1].scatter(x_m, t_m, s=100, c=u_m, cmap='coolwarm', marker='o')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('t')
        axes[1, 1].set_title('Measured Temperature')
        plt.colorbar(scatter, ax=axes[1, 1], label='u (temperature)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    print("Testing HeatEquationData class...\n")
    
    # Create data generator
    data_gen = HeatEquationDataWeak(
        L=1.0,
        T=1.0,
        alpha=0.01,
        N_f=30,
        smooth=True,
        N_bc=100,
        N_ic=100,
        N_sensors=10,
        N_time_measurements=10,
        noise_level=0.05,
        device='cpu'
    )
    
    # Generate full dataset
    data_funcs, data_pts = data_gen.generate_full_dataset()
    
    # Print dataset info
    print("\nDataset summary:")
    for key, value in data_pts.items():
        if key != 'measurement_info':
            print(f"  {key}: shape {value.shape}")
    
    # Visualize
    data_gen.visualize_data(data_funcs, data_pts)
    
    # Test analytical solution at a few points
    print("\nTesting analytical solution:")
    x_test = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    t_test = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    u_test = data_gen.analytical_solution(x_test, t_test)
    print(f"u(x, t=0) = sin(πx): {u_test}")
    
    print("\nAll data generation tests passed.")