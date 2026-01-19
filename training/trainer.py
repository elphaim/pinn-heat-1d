"""
Training module for Physics-Informed Neural Networks

Implements training loop with:
- Adaptive loss weighting
- Progress monitoring and visualization
- Checkpointing
- Early stopping

Author: elphaim
Date: January 19, 2026
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Callable
import time
from pathlib import Path
import json


class PINNTrainer:
    """
    Trainer class for Physics-Informed Neural Networks.
    
    Args:
        model: PINN model to train
        data: Dictionary containing all training data
        device: Device for computation ('cpu' or 'cuda')
        learning_rate: Initial learning rate (default: 1e-3)
        adaptive_weights: Use adaptive loss weighting (default: True)
        weight_update_freq: How often to update weights (default: 100 epochs)
    """
    
    def __init__(
        self,
        model,
        data: Dict[str, torch.Tensor],
        device: str = 'cpu',
        learning_rate: float = 1e-3,
        adaptive_weights: bool = True,
        weight_update_freq: int = 100
    ):
        self.model = model.to(device)
        self.data = data
        self.device = device
        self.adaptive_weights = adaptive_weights
        self.weight_update_freq = weight_update_freq
        
        # Initialize optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Initialize loss weights
        self.lambda_f = 1.0
        self.lambda_bc = 1.0
        self.lambda_ic = 1.0
        self.lambda_m = 1.0 if model.inverse else 0.0
        
        # History tracking
        self.history = {
            'epoch': [],
            'total_loss': [],
            'residual_loss': [],
            'boundary_loss': [],
            'initial_loss': [],
            'measurement_loss': [],
            'lambda_f': [],
            'lambda_bc': [],
            'lambda_ic': [],
            'lambda_m': []
        }
        
        if model.inverse:
            self.history['alpha'] = []
        
        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Adaptive weights: {adaptive_weights}")
        print(f"  Problem type: {'Inverse' if model.inverse else 'Forward'}")
    
    def compute_ntk_weights(self) -> Dict[str, float]:
        """
        Compute adaptive loss weights using Neural Tangent Kernel approach.
        
        Following Wang et al. (2021): λ = Tr[K] / Tr[K_component]
        
        Returns:
            weights: Dictionary with updated lambda values
        """
        # We compute loss magnitudes to balance training
        # Temporarily enable gradients on collocation points for residual computation
        self.model.eval()
        
        # Create temporary copies with gradients enabled
        x_f = self.data['x_f'].clone().detach().requires_grad_(True)
        t_f = self.data['t_f'].clone().detach().requires_grad_(True)
        
        with torch.no_grad():
            # For non-residual losses, no gradients needed on inputs
            u_bc_pred = self.model.forward(self.data['x_bc'], self.data['t_bc'])
            loss_bc = torch.mean((u_bc_pred - self.data['u_bc']) ** 2)
            
            u_ic_pred = self.model.forward(self.data['x_ic'], self.data['t_ic'])
            loss_ic = torch.mean((u_ic_pred - self.data['u_ic']) ** 2)
        
        # For residual, we need a separate computation with gradients enabled on x,t
        # but not tracking gradients for weight updates
        residual = self.model.residual(x_f, t_f)
        loss_f = torch.mean(residual ** 2).detach()
        
        # Compute weights inversely proportional to loss magnitude
        # This ensures all components decrease at similar rates
        losses = torch.tensor([loss_f.item(), loss_bc.item(), loss_ic.item()])
        
        # Avoid division by zero
        losses = torch.clamp(losses, min=1e-10)
        
        # Mean loss for normalization
        mean_loss = losses.mean()
        
        # Compute weights
        lambda_f = (mean_loss / loss_f.item())
        lambda_bc = (mean_loss / loss_bc.item())
        lambda_ic = (mean_loss / loss_ic.item())
        
        # For measurement loss (inverse problem)
        if self.model.inverse and 'x_m' in self.data:
            with torch.no_grad():
                u_m_pred = self.model.forward(self.data['x_m'], self.data['t_m'])
                loss_m = torch.mean((u_m_pred - self.data['u_m']) ** 2)
            lambda_m = (mean_loss / loss_m.item()) if loss_m.item() > 1e-10 else 1.0
        else:
            lambda_m = 0.0
        
        self.model.train()
        
        return {
            'lambda_f': float(lambda_f),
            'lambda_bc': float(lambda_bc),
            'lambda_ic': float(lambda_ic),
            'lambda_m': float(lambda_m)
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Execute one training epoch.
        
        Returns:
            losses: Dictionary with loss values
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Prepare data for measurement loss
        x_m = self.data.get('x_m', None)
        t_m = self.data.get('t_m', None)
        u_m = self.data.get('u_m', None)
        
        # Enable gradients on collocation points for PDE residual computation
        # This is needed because residual() uses autograd to compute derivatives
        x_f = self.data['x_f'].requires_grad_(True)
        t_f = self.data['t_f'].requires_grad_(True)
        
        # Compute loss
        total_loss, losses = self.model.loss_function(
            x_f=x_f,
            t_f=t_f,
            x_bc=self.data['x_bc'],
            t_bc=self.data['t_bc'],
            u_bc=self.data['u_bc'],
            x_ic=self.data['x_ic'],
            t_ic=self.data['t_ic'],
            u_ic=self.data['u_ic'],
            x_m=x_m,
            t_m=t_m,
            u_m=u_m,
            lambda_f=self.lambda_f,
            lambda_bc=self.lambda_bc,
            lambda_ic=self.lambda_ic,
            lambda_m=self.lambda_m
        )
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return losses
    
    def train(
        self,
        epochs: int = 10000,
        print_every: int = 1000,
        plot_every: int = 2000,
        save_path: Optional[str] = None
    ):
        """
        Main training loop.
        
        Args:
            epochs: Number of training epochs
            print_every: Print progress every N epochs
            plot_every: Plot progress every N epochs
            save_path: Path to save model checkpoints
        """
        print("\n" + "="*60)
        print("Starting training...")
        print("="*60)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Update adaptive weights periodically
            if self.adaptive_weights and epoch > 0 and epoch % self.weight_update_freq == 0:
                weights = self.compute_ntk_weights()
                self.lambda_f = weights['lambda_f']
                self.lambda_bc = weights['lambda_bc']
                self.lambda_ic = weights['lambda_ic']
                self.lambda_m = weights['lambda_m']
            
            # Train one epoch
            losses = self.train_epoch()
            
            # Record history
            self.history['epoch'].append(epoch)
            self.history['total_loss'].append(losses['total'])
            self.history['residual_loss'].append(losses['residual'])
            self.history['boundary_loss'].append(losses['boundary'])
            self.history['initial_loss'].append(losses['initial'])
            self.history['measurement_loss'].append(losses['measurement'])
            self.history['lambda_f'].append(self.lambda_f)
            self.history['lambda_bc'].append(self.lambda_bc)
            self.history['lambda_ic'].append(self.lambda_ic)
            self.history['lambda_m'].append(self.lambda_m)
            
            if self.model.inverse:
                self.history['alpha'].append(self.model.get_alpha())
            
            # Print progress
            if epoch % print_every == 0:
                elapsed = time.time() - start_time
                print(f"\nEpoch {epoch}/{epochs} ({elapsed:.1f}s)")
                print(f"  Total Loss: {losses['total']:.6e}")
                print(f"  Residual: {losses['residual']:.6e} (λ={self.lambda_f:.2f})")
                print(f"  Boundary: {losses['boundary']:.6e} (λ={self.lambda_bc:.2f})")
                print(f"  Initial: {losses['initial']:.6e} (λ={self.lambda_ic:.2f})")
                if self.model.inverse:
                    print(f"  Measurement: {losses['measurement']:.6e} (λ={self.lambda_m:.2f})")
                    print(f"  Alpha: {self.model.get_alpha():.6f} (true: 0.01)")
            
            # Plot progress
            if epoch % plot_every == 0 and epoch > 0:
                self.plot_progress()
        
        total_time = time.time() - start_time
        print(f"\nTraining complete. Total time: {total_time:.1f}s")
        
        # Save final model
        if save_path:
            self.save_checkpoint(save_path)
    
    def plot_progress(self, save_path: Optional[str] = None):
        """
        Plot training progress.
        
        Args:
            save_path: Optional path to save figure
        """
        n_plots = 3 if self.model.inverse else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 4))
        
        epochs = self.history['epoch']
        
        # Plot 1: Loss components
        ax = axes[0]
        ax.semilogy(epochs, self.history['total_loss'], 'k-', label='Total', linewidth=2)
        ax.semilogy(epochs, self.history['residual_loss'], 'b-', label='Residual', alpha=0.7)
        ax.semilogy(epochs, self.history['boundary_loss'], 'r-', label='Boundary', alpha=0.7)
        ax.semilogy(epochs, self.history['initial_loss'], 'g-', label='Initial', alpha=0.7)
        if self.model.inverse:
            ax.semilogy(epochs, self.history['measurement_loss'], 'm-', label='Measurement', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Adaptive weights
        ax = axes[1]
        ax.plot(epochs, self.history['lambda_f'], 'b-', label='λ_f (residual)')
        ax.plot(epochs, self.history['lambda_bc'], 'r-', label='λ_bc (boundary)')
        ax.plot(epochs, self.history['lambda_ic'], 'g-', label='λ_ic (initial)')
        if self.model.inverse:
            ax.plot(epochs, self.history['lambda_m'], 'm-', label='λ_m (measurement)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weight')
        ax.set_title('Adaptive Loss Weights')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Alpha convergence (inverse problem only)
        if self.model.inverse:
            ax = axes[2]
            ax.plot(epochs, self.history['alpha'], 'b-', linewidth=2, label='Predicted α')
            ax.axhline(y=0.01, color='r', linestyle='--', linewidth=2, label='True α = 0.01')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('α (thermal diffusivity)')
            ax.set_title('Parameter Recovery')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Show error percentage
            final_alpha = self.history['alpha'][-1]
            error_pct = abs(final_alpha - 0.01) / 0.01 * 100
            ax.text(0.05, 0.95, f'Error: {error_pct:.2f}%', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint and training history."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'final_alpha': self.model.get_alpha() if self.model.inverse else None
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
        
        # Also save history as JSON
        history_path = path.replace('.pt', '_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)