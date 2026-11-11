"""Arquitectura SNN Spike-HAR++ (sin modificaciones respecto al original)."""
from typing import Optional

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate, utils

from config import Config


def get_surrogate(cfg: Config):
    surrogates = {
        "atan": surrogate.atan,
        "fast_sigmoid": surrogate.fast_sigmoid,
        "sigmoid": surrogate.sigmoid,
    }
    if cfg.surrogate_type not in surrogates:
        raise ValueError(f"Surrogate desconocido: {cfg.surrogate_type}")
    return surrogates[cfg.surrogate_type]()


def get_group_norm(channels: int, max_groups: int = 4) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(num_groups=groups, num_channels=channels)


class ConvSimplifiedSpikingAttention(nn.Module):
    """Bloque CB-S3A: atención spiking Hadamard punto a punto (Q ⊙ K ⊙ V)."""

    def __init__(self, dim: int, cfg: Config):
        super().__init__()
        sg = get_surrogate(cfg)
        b, t, g = cfg.beta_lif, cfg.threshold_lif, cfg.max_groups_gn

        self.conv_q = nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        self.gn_q = get_group_norm(dim, g)
        self.lif_q = snn.Leaky(beta=b, threshold=t, spike_grad=sg, init_hidden=True)

        self.conv_k = nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        self.gn_k = get_group_norm(dim, g)
        self.lif_k = snn.Leaky(beta=b, threshold=t, spike_grad=sg, init_hidden=True)

        self.lif_out = snn.Leaky(beta=b, threshold=t, spike_grad=sg, init_hidden=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.lif_q(self.gn_q(self.conv_q(x)))
        k = self.lif_k(self.gn_k(self.conv_k(x)))
        return self.lif_out(q * k * x + x)


class SpikingMLPBranch(nn.Module):
    """Rama Spiking Feed-Forward modulada por máscara continua."""

    def __init__(self, dim: int, cfg: Config):
        super().__init__()
        sg = get_surrogate(cfg)
        b, t, g = cfg.beta_lif, cfg.threshold_lif, cfg.max_groups_gn
        hidden = dim * cfg.expansion_mlp

        self.conv1 = nn.Conv2d(dim, hidden, 3, padding=1, bias=False)
        self.gn1 = get_group_norm(hidden, g)
        self.lif1 = snn.Leaky(beta=b, threshold=t, spike_grad=sg, init_hidden=True)

        self.conv2 = nn.Conv2d(hidden, dim, 3, padding=1, bias=False)
        self.gn2 = get_group_norm(dim, g)
        self.lif2 = snn.Leaky(beta=b, threshold=t, spike_grad=sg, init_hidden=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.lif1(self.gn1(self.conv1(x)))
        if mask is not None:
            h = h * (1.0 + mask)
        return self.lif2(self.gn2(self.conv2(h)))


class SpikeHAPlusPlusHybrid(nn.Module):
    """Red Neuronal Pulsante Híbrida Spike-HAR++ (Intacta)."""

    def __init__(self, cfg: Config):
        super().__init__()
        sg = get_surrogate(cfg)
        b, t, g = cfg.beta_lif, cfg.threshold_lif, cfg.max_groups_gn
        in_ch, ed = cfg.sensor_size[2], cfg.embed_dim

        # Patch Embedding
        self.pe_conv1 = nn.Conv2d(in_ch, ed // 2, 3, padding=1, bias=False)
        self.pe_gn1 = get_group_norm(ed // 2, g)
        self.pe_lif1 = snn.Leaky(beta=b, threshold=t, spike_grad=sg, init_hidden=True)
        self.pe_pool1 = nn.MaxPool2d(2, 2)

        self.pe_conv2 = nn.Conv2d(ed // 2, ed, 3, padding=1, bias=False)
        self.pe_gn2 = get_group_norm(ed, g)
        self.pe_lif2 = snn.Leaky(beta=b, threshold=t, spike_grad=sg, init_hidden=True)
        self.pe_pool2 = nn.MaxPool2d(2, 2)
        self.drop_pe = nn.Dropout2d(cfg.dropout_pe)

        # Ramas paralelas
        self.cb_s3a = ConvSimplifiedSpikingAttention(ed, cfg)
        self.spk_mlp = SpikingMLPBranch(ed, cfg)
        self.att_branch = nn.Sequential(
            nn.Conv2d(ed, ed // 4, 1),
            get_group_norm(ed // 4, g),
            nn.ReLU(),
            nn.Conv2d(ed // 4, 1, 1),
            nn.Sigmoid(),
        )

        # Readout Head
        ph, pw = cfg.pool_size_head
        self.global_pool = nn.AdaptiveAvgPool2d(cfg.pool_size_head)
        self.flatten = nn.Flatten()
        self.drop_out = nn.Dropout(cfg.dropout_head)

        self.fc1 = nn.Linear(ed * ph * pw, cfg.fc_hidden)
        self.lif_fc1 = snn.Leaky(beta=b, threshold=t, spike_grad=sg, init_hidden=True)
        self.fc2 = nn.Linear(cfg.fc_hidden, cfg.num_classes)
        self.lif_out = snn.Leaky(beta=b, threshold=t, spike_grad=sg,
                                 init_hidden=True, output=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        utils.reset(self)
        spk_rec = []
        for step in range(x.shape[1]):
            x_t = x[:, step]
            s1 = self.pe_pool1(self.pe_lif1(self.pe_gn1(self.pe_conv1(x_t))))
            s2 = self.pe_pool2(self.pe_lif2(self.pe_gn2(self.pe_conv2(s1))))
            s2 = self.drop_pe(s2)

            mask = self.att_branch(s2)
            s_out = s2 + self.cb_s3a(s2) + self.spk_mlp(s2, mask=mask)

            pooled = self.global_pool(s_out)
            f1 = self.drop_out(self.lif_fc1(self.fc1(self.flatten(pooled))))
            spk_class, _ = self.lif_out(self.fc2(f1))
            spk_rec.append(spk_class)

        return torch.stack(spk_rec, dim=0)