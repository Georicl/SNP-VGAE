"""
基因型数据共享工具模块
====================

提供跨模块复用的基因型预处理函数，确保各模块（图构建、归因分析等）
使用一致的缺失值插补策略。

核心功能:
  - 基因型矩阵转置与类型转换
  - 缺失值 (-9) 的 SNP 列均值插补
  - 全缺失 SNP 的安全处理（填充 0）

作者: Xiang Yang
邮箱: Georicl@outlook.com
创建时间: 2026-07-23
"""

import numpy as np


def impute_genotype(genotype: np.ndarray) -> np.ndarray:
    """
    基因型缺失值插补：将 -9 替换为对应 SNP 列的均值。

    插补策略与 build_node_features 保持一致，
    确保 X = G @ E 的线性关系在缺失位置仍然成立。

    参数:
        genotype: (M, N) 基因型矩阵，-9 为缺失值（项目约定格式）

    返回:
        G: (N, M) float32 矩阵，缺失值已填充为列均值
    """
    G = genotype.T.astype(np.float32)  # (N, M)
    missing_mask = G < -0.5

    if missing_mask.any():
        G_clean = G.copy()
        G_clean[missing_mask] = 0.0

        # 每列有效元素
        valid_count = G.shape[0] - missing_mask.sum(axis=0)  # (M,)
        col_sum = G_clean.sum(axis=0)  # (M,)

        # 计算列均值, 处理全缺失 SNP
        col_mean = np.zeros(G.shape[1], dtype=np.float32)
        nonzero_mask = valid_count > 0
        col_mean[nonzero_mask] = col_sum[nonzero_mask] / \
            valid_count[nonzero_mask]

        # 填补缺失
        G[missing_mask] = np.take(col_mean, np.where(missing_mask)[1])

    return G
