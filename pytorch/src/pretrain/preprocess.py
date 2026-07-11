"""
M1: 基因型数据预处理
====================

将 PLINK 读取的原始基因型矩阵 (0/1/2/-9) 转换为 VAE 可用的浮点张量。

处理流程:
  int 矩阵 → float32 → -9 替换为 NaN → 行均值填充 → SNP 行标准化 → torch.Tensor

标准化说明:
  对每个 SNP（行）做 z-score 标准化: x_std = (x - mean) / std
  这使所有 SNP 在同一尺度上训练，避免高方差 SNP 主导 MSE 梯度。
  返回 (mean, std) 用于后续逆变换回原始尺度。

可扩展性:
  - 后续可加入 MAF 筛选等额外预处理步骤
  - 全 NA SNP 的处理策略可配置（当前默认为 0）
"""

import numpy as np
import torch


def preprocess_genotype(
    geno_matrix: np.ndarray,
    standardize: bool = True,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    预处理基因型矩阵。

    Args:
        geno_matrix: 原始基因型矩阵 (M, N)，值 0/1/2/-9
        standardize: 是否对每个 SNP 做 z-score 标准化（默认 True）

    Returns:
        (X, snp_mean, snp_std)
        X: 预处理后的张量 (M, N)
        snp_mean: 每个 SNP 的均值 (M,)，用于逆变换
        snp_std: 每个 SNP 的标准差 (M,)，用于逆变换
    """
    data = geno_matrix.astype(np.float32)

    # 将缺失值转换为 NaN
    data[data == -9.0] = np.nan
    # 对 SNP（行）计算均值
    row_means = np.nanmean(data, axis=1)
    row_means[np.isnan(row_means)] = 0.0

    for i in range(data.shape[0]):
        # 逐行对 nan 位置进行均值填充
        mask = np.isnan(data[i])
        data[i, mask] = row_means[i]

    if standardize:
        # SNP 行标准化: z-score = (x - mean) / std
        # 使每个 SNP 均值为 0、标准差为 1，消除尺度差异
        snp_mean = data.mean(axis=1)           # (M,)
        snp_std = data.std(axis=1)             # (M,)
        # 标准差为 0 的 SNP（常数 SNP）设为 1，避免除零
        snp_std[snp_std < 1e-8] = 1.0
        data = (data - snp_mean[:, np.newaxis]) / snp_std[:, np.newaxis]
    else:
        # 不标准化时，mean=0, std=1（恒等变换）
        snp_mean = np.zeros(data.shape[0], dtype=np.float32)
        snp_std = np.ones(data.shape[0], dtype=np.float32)

    return torch.tensor(data, dtype=torch.float32), snp_mean, snp_std


def inverse_standardize(
    x: torch.Tensor,
    snp_mean: np.ndarray,
    snp_std: np.ndarray,
) -> np.ndarray:
    """
    将标准化后的重建值逆变换回原始基因型尺度 (0/1/2)。

    Args:
        x: 标准化空间中的张量 (M, N)
        snp_mean: preprocess 返回的 SNP 均值 (M,)
        snp_std: preprocess 返回的 SNP 标准差 (M,)

    Returns:
        原始尺度下的 numpy 数组 (M, N)
    """
    x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    mean_col = snp_mean[:, np.newaxis]  # (M, 1) 用于广播
    std_col = snp_std[:, np.newaxis]
    return x_np * std_col + mean_col
