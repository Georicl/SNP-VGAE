"""
M1: SNP-VAE 模型定义与损失函数
================================

提供 SNP-VAE 变分自编码器的网络结构和训练损失。

网络结构:
  Encoder: Linear(N, 256) → ELU → Linear(256, 128) → ELU
           → 分叉为 μ(128→D_snp) 和 logσ²(128→D_snp)
  Decoder: Linear(D_snp, 128) → ELU → Linear(128, 256) → ELU → Linear(256, N)

可扩展性:
  - 网络深度/宽度可通过修改 encoder/decoder 的 Sequential 层数调整
  - 隐空间维度 d_snp 通过参数控制，默认为 64
  - β-VAE 的 β 参数在 vae_loss 中可调，控制重建质量与隐空间正则化的平衡
"""

import torch
import torch.nn as nn


class SNPVAE(nn.Module):
    def __init__(self, num_samples: int, d_snp: int = 64):
        """
        num_samples: 样本数 (SNP向量的长度)
        d_snp: snp 嵌入维度(D_snp 隐空间维度), 默认为64
        """
        super().__init__()

        # --编码器--
        self.encoder = nn.Sequential(
            # 使用线性模型作为全链接层, 输入维度: N(样本数), 输出维度: 256.
            nn.Linear(num_samples, 256),
            # 使用 ELU 做激活函数: f(x) = x if x>0 else α*(exp(x)-1)
            # 相对于ReLU更平滑, 在负区间有梯度
            nn.ELU(),
            # 第二层连接层: 将维度256降维为128,
            nn.Linear(256, 128),
            nn.ELU(),
        )

        # 输出均值与方差, 以进行VAE
        self.fc_mu = nn.Linear(128, d_snp)
        self.fc_logvar = nn.Linear(128, d_snp)

        # --解码器--
        self.decoder = nn.Sequential(
            nn.Linear(d_snp, 128),
            nn.ELU(),
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Linear(256, num_samples)
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
