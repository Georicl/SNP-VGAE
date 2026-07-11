"""
M1: SNP 嵌入提取
================

从训练好的 SNP-VAE 编码器中提取所有 SNP 的嵌入向量。

输入: 预处理后的基因型张量 (M, N)
输出: SNP 嵌入矩阵 (M, D_snp)

原理:
  对每个 SNP 跑编码器取 μ（均值），不使用重参数化采样，保证确定性输出。

可扩展性:
  - 后续可支持提取 logvar 用于不确定性分析
  - 可加入降维（PCA/t-SNE）用于可视化
"""

import torch
import numpy as np
from pretrain.vae import SNPVAE


@torch.no_grad()
def extract_embeddings(
    model: SNPVAE,
    x: torch.Tensor,
    batch_size: int = 1024,
    device: str = "cpu",
) -> np.ndarray:
    model.eval()

    M = x.shape[0]
    embeddings = []

    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        x_batch = x[start:end].to(device)

        # 只跑编码器，取 μ 作为嵌入（不用重参数化采样，保证确定性输出）
        mu, _ = model.encode(x_batch)
        embeddings.append(mu.cpu())

    # 拼接所有 batch 的结果 → (M, D_snp)
    return torch.cat(embeddings, dim=0).numpy()
