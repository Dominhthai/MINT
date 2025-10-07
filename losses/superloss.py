import numpy as np
from scipy.special import lambertw
import torch
import torch.nn as nn


class Superloss(nn.Module):
    """
    Superloss for curriculum learning.
    Implements difficulty-based sample weighting using Lambert W function.
    """
    def __init__(self, tau=0.0, lam=1.0, fac=0.9):
        super(Superloss, self).__init__()
        self.tau = tau  # Difficulty threshold (adaptive)
        self.lam = lam  # Regularization strength
        self.fac = fac  # Factor for exponential moving average

    def forward(self, loss):
        """
        Apply curriculum weighting to loss.
        
        Args:
            loss: Per-instance losses (shape: [batch_size])
            
        Returns:
            Curriculum-weighted loss (scalar)
        """
        origin_loss = loss.detach().cpu().numpy()
        self.loss_mean = origin_loss.mean()
        
        # Initialize or update difficulty threshold
        if self.tau == 0.0: 
            self.tau = self.loss_mean
        if self.fac > 0.0: 
            self.tau = self.fac * self.tau + (1.0 - self.fac) * self.loss_mean

        # Compute curriculum weights using Lambert W function
        beta = (origin_loss - self.tau) / self.lam
        gamma = -2.0 / np.exp(1.0) + 1e-12
        sigma = np.exp(-lambertw(0.5 * np.maximum(beta, gamma)).real)
        self.sigma = torch.from_numpy(sigma).to(loss.device)
        
        # Apply curriculum weighting 
        super_loss = (loss - self.tau) * self.sigma + self.lam * (torch.log(self.sigma) ** 2)
        return torch.mean(super_loss)

    def clone(self, loss):
        """Use previously computed sigma weights"""
        super_loss = (loss - self.tau) * self.sigma + self.lam * (torch.log(self.sigma) ** 2)
        return torch.mean(super_loss)
