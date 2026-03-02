import torch, torch.nn as nn, torch.nn.functional as F

class IonWaterMLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 128), nn.SiLU(), nn.Dropout(dropout),
        )
        self.cls = nn.Linear(128, n_classes)
        self.aux = nn.Linear(128, 2)  # water vs ion
        self.log_temp = nn.Parameter(torch.zeros(1))  # for post-hoc calibration

    def forward(self, x, calibrate=False):
        h = self.net(x)
        logits = self.cls(h)
        aux = self.aux(h)
        if calibrate:
            T = torch.exp(self.log_temp).clamp(0.5, 10.0)
            logits = logits / T
        return logits, aux

def focal_ce(input, target, weight=None, gamma=2.0, reduction='mean'):
    # input: [B,C] logits; target: [B] long
    logp = F.log_softmax(input, dim=-1)
    p = torch.exp(logp)
    logp_t = logp[torch.arange(input.size(0)), target]
    p_t = p[torch.arange(input.size(0)), target]
    loss = -(1 - p_t)**gamma * logp_t
    if weight is not None:
        w = weight[target]
        loss = loss * w
    return loss.mean() if reduction=='mean' else loss.sum()
