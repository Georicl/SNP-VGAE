"""
GCTA GRM 文件读取模块
======================

解析 GCTA 格式的基因组关系矩阵 (Genomic Relationship Matrix)，
包括二进制下三角矩阵文件 (.grm.bin) 和样本 ID 文件 (.grm.id)。

GRM 矩阵用于衡量样本间的遗传相似度，是构建 KNN 图的基础数据。

.grm.bin 文件格式:
  按逐行下三角顺序存储 float32 数据，
  即 G[0,0], G[1,0], G[1,1], G[2,0], ...，总元素数为 N*(N+1)/2

.grm.id 文件格式:
  每行包含 FID 和 IID，行号对应 GRM 矩阵的行列索引

作者: Xiang Yang
邮箱: Georicl@outlook.com
创建时间: 2026-07-06
"""

import numpy as np


def grm_reader(grm_bin_file: str, row_lens: int) -> np.ndarray:
    """
    读取 GCTA GRM 二进制文件（.grm.bin），重建完整的对称 GRM 矩阵。

    参数:
        grm_bin_file: .grm.bin 文件路径
        row_lens:     样本数量 N（矩阵维度）

    返回:
        N x N 的对称 GRM 矩阵（np.float32）

    异常:
        ValueError: 文件元素数量与 N*(N+1)/2 不匹配时抛出
    """
    data = np.fromfile(grm_bin_file, dtype=np.float32)

    # 验证下三角元素数量: N*(N+1)/2
    expected_size = row_lens * (row_lens + 1) // 2
    if data.size != expected_size:
        raise ValueError(
            f"GRM 二进制文件大小不匹配: 期望 {expected_size} 个元素 "
            f"(N={row_lens}), 实际读取 {data.size} 个元素"
        )

    # 初始化 N x N 零矩阵
    G = np.zeros((row_lens, row_lens), dtype=np.float32)

    # 将下三角数据填入矩阵
    idx = np.tril_indices(row_lens)
    G[idx] = data
    # 对称化：将严格下三角（k=-1）转置后加到上三角，对角线保持不变
    G += np.tril(G, k=-1).T
    return G


def grm_id_reader(grm_id_file: str) -> list:
    """
    读取 GCTA GRM 样本 ID 文件（.grm.id）。

    参数:
        grm_id_file: .grm.id 文件路径

    返回:
        样本 ID 列表，每个元素为 (FID, IID) 元组，
        列表顺序与 GRM 矩阵行列顺序一致。
    """
    grm_id_list: list[tuple[str, str]] = []
    with open(grm_id_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            grm_id_list.append((parts[0], parts[1]))

    return grm_id_list
