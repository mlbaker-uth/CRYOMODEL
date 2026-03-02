# crymodel/ml/model.py
"""PyTorch MLP model for ion/water classification."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class IonWaterMLP(nn.Module):
    """MLP for classifying water candidates as specific ion types or water."""

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.cls = nn.Linear(128, n_classes)
        self.aux = nn.Linear(128, 2)  # water vs ion
        self.log_temp = nn.Parameter(torch.zeros(1))  # for post-hoc calibration

    def forward(self, x: torch.Tensor, calibrate: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input features [B, in_dim]
            calibrate: If True, apply temperature scaling
            
        Returns:
            logits: Class logits [B, n_classes]
            aux: Water vs ion logits [B, 2]
        """
        h = self.net(x)
        logits = self.cls(h)
        aux = self.aux(h)
        if calibrate:
            T = torch.exp(self.log_temp).clamp(0.5, 10.0)
            logits = logits / T
        return logits, aux


def focal_ce(input: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean") -> torch.Tensor:
    """Focal cross-entropy loss.
    
    Args:
        input: Logits [B, C]
        target: Class indices [B]
        weight: Class weights [C] or None
        gamma: Focal loss gamma parameter
        reduction: 'mean' or 'sum'
        
    Returns:
        Loss scalar or tensor
    """
    logp = F.log_softmax(input, dim=-1)
    p = torch.exp(logp)
    logp_t = logp[torch.arange(input.size(0)), target]
    p_t = p[torch.arange(input.size(0)), target]
    loss = -(1 - p_t) ** gamma * logp_t
    if weight is not None:
        w = weight[target]
        loss = loss * w
    return loss.mean() if reduction == "mean" else loss.sum()

