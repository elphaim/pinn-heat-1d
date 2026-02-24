# PINN Optimization Benchmark: 1D Heat Equation

A modular PyTorch framework for benchmarking Physics-Informed Neural Network (PINN) training strategies on the 1D heat equation. The project compares **strong-form**, **weak-form**, and **multi-fidelity** loss formulations under a unified architecture, with support for both forward and inverse problems.

## Motivation

PINNs embed PDE constraints directly into the neural network loss function, but their practical performance is highly sensitive to the choice of loss formulation, optimizer scheduling, and loss weighting. This project provides a controlled testbed — the 1D heat equation with a known analytical solution — to isolate and compare these optimization strategies before applying them to more complex PDEs.

## Problem Setup

The benchmark solves:

$$u_t = \alpha \, u_{xx}, \quad x \in [0, 1],\; t \in [0, T]$$

with initial condition $u(x, 0) = \sin(\pi x)$ and homogeneous Dirichlet boundary conditions. The analytical solution $u(x,t) = \sin(\pi x)\,e^{-\alpha \pi^2 t}$ enables exact error quantification.

## Key Features

**Loss Formulations (Strategy Pattern)**
- **Strong-form:** Point-wise PDE residual via automatic differentiation.
- **Weak-form:** Integrated residual against compactly supported Gaussian test functions, with multiple quadrature backends (Gauss–Legendre, Simpson, Monte Carlo, adaptive Simpson).
- **Multi-fidelity:** Combines sparse high-fidelity measurements with dense low-fidelity data (from a finite-difference solver with deliberate model error), weighted by inverse variance.

**Training Infrastructure**
- Adam → L-BFGS optimizer switching based on configurable plateau/stagnation criteria.
- Adaptive loss weighting using a gradient-norm proxy inspired by the NTK-based scheme of Wang et al. (2021), with EMA smoothing.
- Per-component gradient norm tracking for loss landscape diagnostics.
- Full training history export (JSON) and checkpointing.

**Data Generation**
- Analytical ground truth, Latin Hypercube or uniform collocation sampling, and synthetic noisy sensor measurements.
- Finite-difference (FTCS) solver for generating low-fidelity data with controllable resolution and model error.

**Inverse Problem Support**
- Thermal diffusivity $\alpha$ can be treated as a learnable parameter, with measurement loss driving its recovery.

## Repository Structure

```
├── models/
│   ├── heat_pinn.py             # Core PINN model (forward/inverse)
│   └── heat_pinn_strategy.py    # Strategy pattern: loss formulations + StrategicPINN
├── training/
│   └── trainer_strategy.py      # Trainer with Adam→L-BFGS, adaptive weights, diagnostics
├── data/
│   └── heat_data.py             # Data generation, FD solver, multi-fidelity data prep
├── utils/
│   ├── integrator.py            # Quadrature methods (Gauss–Legendre, Simpson, MC, adaptive)
│   ├── test_functions.py        # Compact Gaussian test functions with placement strategies
│   └── plotter.py               # Visualization utilities
├── experiments/
│   └── multi_fidelity_comparison.py  # HF-only vs LF-only vs multi-fidelity benchmark
├── notebooks/
│   ├── heat_equation_PINN.ipynb           # Basic PINN demo
│   ├── heat_equation_modular.ipynb        # Strong vs weak form comparison
│   ├── heat_equation_Hard_vs_Soft.ipynb   # Hard vs soft BC enforcement
│   └── heat_equation_strategic.ipynb      # Test function placement strategies
├── 1d_heat.pdf                  # Summary of results and findings
├── pyproject.toml
└── requirements.txt
```

## Getting Started

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0

```bash
git clone https://github.com/elphaim/pinn-heat-1d.git
cd pinn-heat-1d
pip install -e .
```

**Run the multi-fidelity comparison experiment:**

```bash
python -m experiments.multi_fidelity_comparison
```

**Explore interactively** via the notebooks in `notebooks/`.

## Design Decisions

- **Strategy pattern for losses:** Swapping between strong-form, weak-form, and multi-fidelity loss requires no changes to the model or trainer — only the strategy object is replaced. This keeps experiments comparable and the codebase extensible.
- **Autograd-compatible quadrature:** All integration methods preserve the PyTorch computational graph, so weak-form residuals are fully differentiable and compatible with standard optimizers.
- **Finite-difference as low-fidelity source:** The FTCS solver intentionally uses a biased $\alpha$ and coarse grid, producing data that is cheap but systematically inaccurate — a realistic proxy for low-fidelity simulations in multi-fidelity workflows.

## Tech Stack

PyTorch · NumPy · SciPy · Matplotlib

## Status

This is an active personal project. The 1D heat equation serves as a validation baseline; the intent is to extend these optimization techniques to more complex PDEs. See `TODO.md` for planned improvements.

## References

- Raissi, M., Perdikaris, P., Karniadakis, G.E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686–707.
- Wang, S., Teng, Y., Perdikaris, P. (2021). *Understanding and mitigating gradient flow pathologies in physics-informed neural networks.* SIAM Journal on Scientific Computing, 43(5), A3055–A3081.
