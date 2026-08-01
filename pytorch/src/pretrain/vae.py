"""
SNP-VAE 模型定义与损失函数
================================

提供 SNP-VAE 变分自编码器的网络结构和训练损失。

网络结构:
  Encoder: Linear(N, 256) → ELU → Linear(256, 128) → ELU
           → 分叉为 μ(128→D_snp) 和 logσ²(128→D_snp)
  Decoder: Linear(D_snp, 128) → ELU → Linear(128, 256) → ELU → Linear(256, N)

损失函数:
  L = L_recon + β · L_KL
  其中 L_recon 为 MSE 重建损失，L_KL 为 KL 散度正则化项

作者: Xiang Yang
邮箱: Georicl@outlook.com
创建时间: 2026-07-11
"""

import torch
import torch.nn as nn


class SNPVAE(nn.Module):
    def __init__(self, num_samples: int, d_snp: int = 64, hidden_dim: int = 256):
        """
        num_samples: 样本数 (SNP向量的长度)
        d_snp: snp 嵌入维度(D_snp 隐空间维度), 默认为64
        hidden_dim: 编码器/解码器隐藏层宽度, 默认256
        """
        super().__init__()

        # 中间层维度：hidden_dim 和 hidden_dim//2
        mid_dim = hidden_dim // 2

        # --编码器--
        self.encoder = nn.Sequential(
            nn.Linear(num_samples, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, mid_dim),
            nn.ELU(),
        )

        # 输出均值与方差, 以进行VAE
        self.fc_mu = nn.Linear(mid_dim, d_snp)
        self.fc_logvar = nn.Linear(mid_dim, d_snp)

        # --解码器--
        self.decoder = nn.Sequential(
            nn.Linear(d_snp, mid_dim),
            nn.ELU(),
            nn.Linear(mid_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, num_samples)
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)  # (B, N) -> (B, 128)
        mu = self.fc_mu(h)  # 计算均值 (B, 128) -> (B, D_snp)
        logvar = self.fc_logvar(h)  # 计算均值 (B, 128) -> (B, D_snp)
        return mu, logvar

    # 由于在, N(μ, σ²) 中采样是不可导的, 所以需要将其转换为可导形式

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # 数学原理: z = μ + σ · ε, 此时 ε ~ N(0, 1)
        # 经过该转换后, 函数z 对μ 与σ 都可导, 即可梯度反向传播
        # 标准差
        std = torch.exp(0.5 * logvar)
        # ε ~ N(0, I)，与 z 同形状
        eps = torch.randn_like(std)

        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # 解码器, 重建基因型向量
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# 计算损失函数


def vae_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction="mean")

    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )

    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss
