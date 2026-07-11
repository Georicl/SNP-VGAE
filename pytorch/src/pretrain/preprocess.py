"""
M1: 基因型数据预处理
====================

将 PLINK 读取的原始基因型矩阵 (0/1/2/-9) 转换为 VAE 可用的浮点张量。

处理流程:
  int 矩阵 → float32 → -9 替换为 NaN → 行均值填充 → torch.Tensor

可扩展性:
  - 后续可加入 MAF 筛选、标准化等预处理步骤
  - 全 NA SNP 的处理策略可配置（当前默认为 0）
"""

import numpy as np
import torch


def preprocess_genotype(geno_matrix: np.ndarray) -> torch.Tensor:
    # 进行数据的预处理, 将以SNP为行的矩阵导入为张量, 并将缺失值进行均值填充(以不影响方差估计)
    data = geno_matrix.astype(np.float32)

    # 将缺失值转换为NaN
    data[data == -9.0] = np.nan
    # 对SNP(行) 计算均值
    row_means = np.nanmean(data, axis=1)

    row_means[np.isnan(row_means)] = 0.0

    for i in range(data.shape[0]):
        # 逐行对nan位置进行均值填充
        mask = np.isnan(data[i])
        data[i, mask] = row_means[i]

    return torch.tensor(data, dtype=torch.float32)
