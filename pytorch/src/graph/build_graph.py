"""
样本图构建器
============

从 GRM 矩阵构建 KNN 稀疏邻接图，并执行 GCN 对称归一化，
同时将基因型与 SNP 嵌入聚合成节点特征。

核心算法:
  1. Top-K 近邻选择: 从 GRM 中为每个样本选取遗传相似度最高的 K 个邻居
  2. 对称化: element-wise max 确保图的无向性
  3. GCN 归一化: D^(-1/2) · A · D^(-1/2) 对称归一化
  4. 节点特征聚合: X = G_imputed @ E，将 SNP 嵌入线性映射到样本空间

作者: Xiang Yang
邮箱: Georicl@outlook.com
创建时间: 2026-07-15
"""

from data.grm import grm_reader
from data.genotype_utils import impute_genotype
import numpy as np
import torch


class GraphBuilder:
    def __init__(self) -> None:
        pass

    def knn_graph_builder(
        self,
        grm: np.ndarray,
        k: int,
        self_loops: bool = True,
        symmetric: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        从 GRM 矩阵构建 KNN 稀疏邻接矩阵（核心算法，不含文件 I/O）。

        参数:
            grm:        (N, N) GRM 对称矩阵 (np.float32)
            k:          每个节点的近邻数
            self_loops: 是否添加自环（权重=GRM对角线值 G[i,i]）
            symmetric:  是否对称化 (取 element-wise max)

        返回:
            adj_norm: (N, N) 稀疏张量, GCN 对称归一化 D^(-1/2)·A·D^(-1/2)
            adj_raw:  (N, N) 稀疏张量, 原始 KNN 邻接（含自环），用于边重建损失
        """
        N = grm.shape[0]

        if grm.shape[0] != grm.shape[1]:
            raise ValueError(f"GRM 必须是方阵, 得到 shape={grm.shape}")
        if k >= N:
            raise ValueError(f"K={k} 必须小于样本数 {N}")

        # --- Step 1: top-K 索引（排除对角线）---
        grm_work = grm.copy()
        np.fill_diagonal(grm_work, -np.inf)
        # argpartition: 第 -K 个位置左边都 ≤ 右边，取右边 K 个即为 top-K
        topk_indices = np.argpartition(grm_work, -k, axis=1)[:, -k:]  # (N, K)

        # --- Step 2: 在 numpy 稠密矩阵上构建 + 对称化 ---
        # torch.maximum 不支持 SparseCOO，因此在 numpy 层完成对称化
        adj_np = np.zeros((N, N), dtype=np.float32)
        row_idx = np.repeat(np.arange(N), k)
        col_idx = topk_indices.flatten()
        adj_np[row_idx, col_idx] = grm[row_idx, col_idx]

        # --- Step 3: 对称化 ---
        if symmetric:
            adj_np = np.maximum(adj_np, adj_np.T)  # element-wise max

        # --- Step 4: 添加自环（权重 = GRM 对角线 G[i,i]）---
        if self_loops:
            np.fill_diagonal(adj_np, np.diag(grm).astype(np.float32))

        adj_raw_dense = adj_np

        # --- Step 5: 转为稀疏张量 ---
        rows, cols = np.nonzero(adj_raw_dense)
        values = adj_raw_dense[rows, cols]
        indices = torch.from_numpy(np.stack([rows, cols])).long()

        adjacency = torch.sparse_coo_tensor(
            indices,
            torch.from_numpy(values).float(),
            size=(N, N),
        ).coalesce()

        adj_raw = adjacency

        # --- Step 6: GCN 对称归一化 D^(-1/2) · A · D^(-1/2) ---
        indices = adjacency.indices()
        values = adjacency.values()

        # 度 = 每行权重之和
        deg = torch.zeros(N, dtype=torch.float32)
        deg.scatter_add_(0, indices[0], values)

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0  # 孤立节点度=0 → 0

        # 每条边 (i,j) 乘 d_i^(-1/2) * d_j^(-1/2)
        norm_values = (
            deg_inv_sqrt[indices[0]] * values * deg_inv_sqrt[indices[1]]
        )

        adj_norm = torch.sparse_coo_tensor(
            indices, norm_values, size=(N, N)
        ).coalesce()

        return adj_norm, adj_raw

    @staticmethod
    def knn_graph_from_files(
        grm_bin: str,
        grm_id: str,
        k: int,
        self_loops: bool = True,
        symmetric: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        便捷封装: 从文件读取 GRM 后调用 knn_graph_builder。

        参数:
            grm_bin: .grm.bin 文件路径
            grm_id:  .grm.id 文件路径
            k: 近邻数
            self_loops: 是否添加自环
            symmetric:  是否对称化

        返回:
            adj_norm, adj_raw（同 knn_graph_builder）
        """
        # 统计样本数量
        with open(grm_id, "r") as f:
            row_lens = sum(1 for _ in f)

        grm_matrix = grm_reader(grm_bin_file=grm_bin, row_lens=row_lens)

        builder = GraphBuilder()
        return builder.knn_graph_builder(grm_matrix, k, self_loops, symmetric)

    def build_node_features(self,
                            genotype: np.ndarray,
                            snp_embeddings: np.ndarray,
                            ) -> torch.Tensor:
        """
        聚合 SNP 嵌入为节点特征: X = G_imputed @ E。

        通过插补后的基因型矩阵与 SNP 嵌入矩阵的线性组合，
        将每个样本表示为其 SNP 嵌入的加权和。

        参数:
            genotype:       基因型矩阵 (M, N)，-9 为缺失值
            snp_embeddings: SNP 嵌入矩阵 (M, D_snp)

        返回:
            node_features: 节点特征张量 (N, D_snp)
        """
        # --- 缺失值插补: -9 → SNP 列均值（共享逻辑） ---
        G = impute_genotype(genotype)  # (N, M)
        E = snp_embeddings.astype(np.float32)  # (M, D_snp)

        X = G @ E

        return torch.from_numpy(X)
