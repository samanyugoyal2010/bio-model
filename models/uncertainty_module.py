"""
Uncertainty quantification module using Bayesian neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with learnable weight and bias distributions.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
    ):
        """
        Initialize Bayesian linear layer.
        
        Args:
            in_features: Input features
            out_features: Output features
            prior_mu: Prior mean
            prior_sigma: Prior standard deviation
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable parameters for weight distribution
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_log_sigma = nn.Parameter(torch.randn(out_features, in_features) * -1.0)
        
        # Learnable parameters for bias distribution
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_log_sigma = nn.Parameter(torch.randn(out_features) * -1.0)
        
        # Prior parameters
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Forward pass with weight sampling.
        
        Args:
            x: Input tensor (B, in_features)
            sample: Whether to sample weights (True) or use mean (False)
        
        Returns:
            Output tensor (B, out_features)
        """
        if sample and self.training:
            # Sample weights from learned distribution
            weight_sigma = torch.exp(self.weight_log_sigma)
            weight_epsilon = torch.randn_like(self.weight_mu)
            weight = self.weight_mu + weight_sigma * weight_epsilon
            
            bias_sigma = torch.exp(self.bias_log_sigma)
            bias_epsilon = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + bias_sigma * bias_epsilon
        else:
            # Use mean (deterministic)
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Compute KL divergence between learned and prior distributions.
        
        Returns:
            KL divergence value
        """
        # Weight KL
        weight_kl = self._kl_divergence(
            self.weight_mu,
            torch.exp(self.weight_log_sigma),
            self.prior_mu,
            self.prior_sigma,
        )
        
        # Bias KL
        bias_kl = self._kl_divergence(
            self.bias_mu,
            torch.exp(self.bias_log_sigma),
            self.prior_mu,
            self.prior_sigma,
        )
        
        return weight_kl + bias_kl
    
    def _kl_divergence(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        prior_mu: float,
        prior_sigma: float,
    ) -> torch.Tensor:
        """Compute KL divergence for a parameter."""
        kl = (
            0.5 * (
                torch.log(prior_sigma ** 2 / (sigma ** 2 + 1e-8)) +
                (sigma ** 2 + (mu - prior_mu) ** 2) / (prior_sigma ** 2) -
                1.0
            )
        )
        return kl.sum()


class UncertaintyModule(nn.Module):
    """
    Uncertainty quantification module.
    Estimates both epistemic (model) and aleatoric (data) uncertainty.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 1,
        num_samples: int = 10,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
    ):
        """
        Initialize uncertainty module.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_samples: Number of Monte Carlo samples
            prior_mu: Prior mean
            prior_sigma: Prior standard deviation
        """
        super().__init__()
        
        self.num_samples = num_samples
        
        # Bayesian layers
        self.bayesian_layers = nn.Sequential(
            BayesianLinear(input_dim, hidden_dim, prior_mu, prior_sigma),
            nn.ReLU(),
            BayesianLinear(hidden_dim, hidden_dim, prior_mu, prior_sigma),
            nn.ReLU(),
            BayesianLinear(hidden_dim, output_dim, prior_mu, prior_sigma),
        )
        
        # Aleatoric uncertainty head (predicts data noise)
        self.aleatoric_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus(),  # Ensure positive
        )
    
    def forward(
        self,
        features: torch.Tensor,
        num_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.
        
        Args:
            features: Input features (B, input_dim)
            num_samples: Number of Monte Carlo samples (default: self.num_samples)
        
        Returns:
            Tuple of:
                - mean: (B, output_dim) - Mean prediction
                - epistemic_uncertainty: (B, output_dim) - Epistemic uncertainty
                - aleatoric_uncertainty: (B, output_dim) - Aleatoric uncertainty
        """
        if num_samples is None:
            num_samples = self.num_samples if self.training else 1
        
        # Monte Carlo sampling for epistemic uncertainty
        samples = []
        for _ in range(num_samples):
            sample = self.bayesian_layers(features)
            samples.append(sample)
        
        samples = torch.stack(samples, dim=0)  # (num_samples, B, output_dim)
        
        # Mean prediction
        mean = samples.mean(dim=0)  # (B, output_dim)
        
        # Epistemic uncertainty (variance across samples)
        epistemic_uncertainty = samples.var(dim=0)  # (B, output_dim)
        
        # Aleatoric uncertainty (predicted data noise)
        aleatoric_uncertainty = self.aleatoric_head(features)  # (B, output_dim)
        
        return mean, epistemic_uncertainty, aleatoric_uncertainty
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Compute total KL divergence for all Bayesian layers.
        
        Returns:
            Total KL divergence
        """
        total_kl = 0.0
        for layer in self.bayesian_layers:
            if isinstance(layer, BayesianLinear):
                total_kl += layer.kl_divergence()
        return total_kl

