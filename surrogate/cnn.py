import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialDropout2d(nn.Module):
    """Drops entire channels — proper MC Dropout for conv layers."""
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x
        mask = torch.bernoulli(
            torch.full((x.shape[0], x.shape[1], 1, 1), 1 - self.p, device=x.device)
        ) / (1 - self.p)
        return x * mask


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class CNNRegressor(nn.Module):
    """
    Input:  (B, 1, 256, 256)  — Vp channel only
    Output: (B, 1)            — predicted misfit in [0, 1] (MinMax scaled)

    Design rationale:
    - Features are large (layer boundaries, plume edges), so big kernels
      early and aggressive pooling are fine
    - GroupNorm instead of InstanceNorm to preserve Vp magnitudes
    - SpatialDropout2d in backbone for MC Dropout uncertainty
    - Kept lean: 8→16→32→64 channels is plenty for these images
    - Sigmoid output kept for MinMax-scaled [0,1] labels
    """
    def __init__(self, base_channels=8, dropout=0.15):
        super().__init__()

        bc = base_channels

        # 256×256 → 128×128
        # Large 7×7 kernel captures broad Vp layer boundaries
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, bc, 7, padding=3, bias=False),
            nn.GroupNorm(min(8, bc), bc),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
        )
        self.drop1 = SpatialDropout2d(dropout)

        # 128×128 → 64×64
        self.layer2 = nn.Sequential(
            nn.Conv2d(bc, bc * 2, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, bc * 2), bc * 2),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(bc * 2),
            nn.MaxPool2d(2),
        )
        self.drop2 = SpatialDropout2d(dropout)

        # 64×64 → 32×32
        self.layer3 = nn.Sequential(
            nn.Conv2d(bc * 2, bc * 4, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, bc * 4), bc * 4),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(bc * 4),
            nn.MaxPool2d(2),
        )
        self.drop3 = SpatialDropout2d(dropout)

        # 32×32 → 16×16
        self.layer4 = nn.Sequential(
            nn.Conv2d(bc * 4, bc * 8, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, bc * 8), bc * 8),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(bc * 8),
            nn.MaxPool2d(2),
        )
        self.drop4 = SpatialDropout2d(dropout)

        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Regressor head (NO dropout — Stable Output Layer) ──────
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bc * 8, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.drop1(self.layer1(x))
        x = self.drop2(self.layer2(x))
        x = self.drop3(self.layer3(x))
        x = self.drop4(self.layer4(x))
        x = self.gap(x)
        return self.regressor(x)

    def enable_mc_dropout(self):
        """Activate backbone SpatialDropout2d for MC uncertainty passes (SOL)."""
        for m in self.modules():
            if isinstance(m, SpatialDropout2d):
                m.training = True

    def disable_mc_dropout(self):
        """Return to fully deterministic inference."""
        for m in self.modules():
            if isinstance(m, SpatialDropout2d):
                m.training = False

    def freeze_for_finetuning(self, phase=1):
        """Progressive unfreezing for adaptive regime.

        phase 1: head only  (regressor)
        phase 2: head + last two conv stages
        phase 3: everything
        """
        for p in self.parameters():
            p.requires_grad = False

        if phase >= 1:
            for p in self.regressor.parameters():
                p.requires_grad = True
        if phase >= 2:
            for p in self.layer3.parameters():
                p.requires_grad = True
            for p in self.layer4.parameters():
                p.requires_grad = True
            for p in self.drop3.parameters():
                p.requires_grad = True
            for p in self.drop4.parameters():
                p.requires_grad = True
        if phase >= 3:
            for p in self.parameters():
                p.requires_grad = True