"""
VGAE 模型定义与联合损失函数
================================

变分图自编码器 (Variational Graph Auto-Encoder) 用于基因组预测。

架构:
  编码器: GCNConv1 → BN → ELU → Dropout → GCNConv2(μ) / GCNConv2(logσ)
  重参数化: Z = μ + exp(0.5·logσ) · ε
  边解码器: sigmoid(Z @ Z^T) → BCE 边重建损失
  表型预测头: MLP(Z) → MSE 表型预测损失

联合损失: L = L_pheno + α·L_edge + β·L_KL

作者: Xiang Yang
邮箱: Georicl@outlook.com
创建时间: 2026-07-15
"""

import torch
import torch.nn as nn
from torch import Tensor

from model.gcn import GCNConv


class VGAEModel(nn.Module):
    """
    VGAE 模型：GCN 编码器 + 边解码器 + 表型预测头。

    参数:
        d_snp:      输入特征维度（来自 M2b 的 D_snp）
        d_hidden:   GCN 隐藏层维度
        d_z:        隐空间维度
        dropout:    GCN 层后 dropout 概率
        mlp_hidden: 表型预测 MLP 隐藏层维度
    """

    def __init__(
        self,
        d_snp: int = 64,
        d_hidden: int = 128,
        d_z: int = 32,
        dropout: float = 0.5,
        mlp_hidden: int = 64,
    ) -> None:
        super().__init__()

        self.d_z = d_z

        # --- GCN 编码器 ---
        # 第一层: (N, D_snp) → (N, D_hidden)
        self.gcn1 = GCNConv(d_snp, d_hidden)
        self.bn1 = nn.BatchNorm1d(d_hidden)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

        # 第二层: (N, D_hidden) → (N, D_z)，分叉为 μ 和 logσ²
        self.gcn_mu = GCNConv(d_hidden, d_z)
        self.gcn_logvar = GCNConv(d_hidden, d_z)

        # --- 表型预测头 ---
        self.pheno_head = nn.Sequential(
            nn.Linear(d_z, mlp_hidden),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden, 1),
        )

    def encode(self, x: Tensor, adj_norm: Tensor) -> tuple[Tensor, Tensor]:
        """
        GCN 编码器。

        参数:
            x:        (N, D_snp) 节点特征
            adj_norm: (N, N) 稀疏归一化邻接矩阵

        返回:
            mu:     (N, D_z) 隐空间均值
            logvar: (N, D_z) 隐空间对数方差
        """
        # 第一层 GCN + BN + ELU + Dropout
        h1 = self.gcn1(x, adj_norm)
        h1 = self.bn1(h1)
        h1 = self.elu(h1)
        h1 = self.dropout(h1)

        # 第二层: 分叉为 μ 和 logσ²
        mu = self.gcn_mu(h1, adj_norm)
        logvar = self.gcn_logvar(h1, adj_norm)

        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        重参数化技巧: Z = μ + exp(0.5·logσ²) · ε, ε ~ N(0, I)

        参数:
            mu:     (N, D_z) 均值
            logvar: (N, D_z) 对数方差

        返回:
            z: (N, D_z) 采样的隐变量
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode_edges(self, z: Tensor, edge_index: Tensor) -> Tensor:
        """
        边重建: 仅计算指定边位置的概率（内存高效）。

        参数:
            z:          (N, D_z) 隐变量
            edge_index: (2, E) 边索引 (row, col)

        返回:
            edge_scores: (E,) 每条边的概率值
        """
        row, col = edge_index[0], edge_index[1]
        # 点积: z_i · z_j 对每条边
        scores = (z[row] * z[col]).sum(dim=1)
        return torch.sigmoid(scores)

    def predict_phenotype(self, z: Tensor) -> Tensor:
        """
        表型预测: MLP(Z) → 每个个体独立预测。

        参数:
            z: (N, D_z) 隐变量

        返回:
            y_pred: (N,) 表型预测值
        """
        return self.pheno_head(z).squeeze(-1)

    def forward(
        self, x: Tensor, adj_norm: Tensor, edge_index: Tensor | None = None
    ) -> dict[str, Tensor]:
        """
        VGAE 完整前向传播。

        参数:
            x:          (N, D_snp) 节点特征
            adj_norm:   (N, N) 稀疏归一化邻接矩阵
            edge_index: (2, E) 需要计算边分数的边索引。
                        若为 None，则使用 adj_norm 的非零位置。

        返回:
            dict:
              'y_pred':      (N,) 表型预测值
              'z':           (N, D_z) 隐变量
              'mu':          (N, D_z) 均值
              'logvar':      (N, D_z) 对数方差
              'edge_scores': (E,) 指定边的概率值
              'edge_index':  (2, E) 边索引
        """
        mu, logvar = self.encode(x, adj_norm)
        z = self.reparameterize(mu, logvar)

        # 确定需要计算边分数的位置
        if edge_index is None:
            edge_index = adj_norm.coalesce().indices()

        edge_scores = self.decode_edges(z, edge_index)
        y_pred = self.predict_phenotype(z)

        return {
            "y_pred": y_pred,
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "edge_scores": edge_scores,
            "edge_index": edge_index,
        }


def vgae_loss(
    y_pred: Tensor,
    y_true: Tensor,
    z: Tensor,
    edge_index: Tensor,
    adj_raw: Tensor,
    mu: Tensor,
    logvar: Tensor,
    alpha: float = 0.1,
    beta: float = 0.01,
    neg_sample_ratio: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    VGAE 联合损失函数。

    L = L_pheno + α·L_edge + β·L_KL

    参数:
        y_pred:           (N,) 表型预测值
        y_true:           (N,) 真实表型值
        z:                (N, D_z) 隐变量（用于计算负边分数）
        edge_index:       (2, E) 正边索引
        adj_raw:          (N, N) 稀疏原始邻接矩阵（含自环）
        mu:               (N, D_z) 隐空间均值
        logvar:           (N, D_z) 隐空间对数方差
        alpha:            边损失权重
        beta:             KL 散度权重
        neg_sample_ratio: 负采样数 / 正边数

    返回:
        (total_loss, pheno_loss, edge_loss, kl_loss)
    """
    N = y_pred.shape[0]

    # --- 1. 表型预测损失 (MSE) ---
    pheno_loss = nn.functional.mse_loss(y_pred, y_true)

    # --- 2. 边重建损失 (BCE + 负采样) ---
    edge_loss = edge_reconstruction_loss(
        z, edge_index, adj_raw, neg_sample_ratio
    )

    # --- 3. KL 散度 ---
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / N

    # --- 联合损失 ---
    total_loss = pheno_loss + alpha * edge_loss + beta * kl_loss

    return total_loss, pheno_loss, edge_loss, kl_loss


def edge_reconstruction_loss(
    z: Tensor,
    edge_index: Tensor,
    adj_raw: Tensor,
    neg_sample_ratio: float,
) -> Tensor:
    """
    边重建 BCE 损失（带负采样）。

    正边: adj_raw 中非零位置（由 edge_index 给出）
    负边: 随机采样 neg_sample_ratio * n_pos 条不在正边集合中的边
    所有边分数均从 z 按需计算，避免构建完整 N×N 矩阵。

    参数:
        z:                (N, D_z) 隐变量
        edge_index:       (2, E) 正边索引
        adj_raw:          (N, N) 稀疏原始邻接
        neg_sample_ratio: 负采样比例

    返回:
        edge_loss: 标量 BCE 损失
    """
    N = adj_raw.shape[0]
    pos_row, pos_col = edge_index[0], edge_index[1]
    n_pos = pos_row.shape[0]

    # 正边分数: sigmoid(z_i · z_j)
    pos_dots = (z[pos_row] * z[pos_col]).sum(dim=1)
    pos_scores = torch.sigmoid(pos_dots)
    pos_labels = torch.ones(n_pos, device=z.device)

    # 负采样: 直接在 (row, col) 空间采样，避免 N*N 展平空间
    n_neg = int(neg_sample_ratio * n_pos)

    # 随机生成候选负边 (row, col) 对
    neg_row = torch.randint(0, N, (n_neg,), device=z.device)
    neg_col = torch.randint(0, N, (n_neg,), device=z.device)

    # 过滤自环
    is_self_loop = neg_row == neg_col

    # 过滤正边: 检查 (neg_row, neg_col) 是否在正边集合中
    # 使用哈希方法: 将 (row, col) 编码为单个整数 row * N + col
    # 注意: 对于 N=2002, N*N=4M 在 int64 范围内，不会溢出
    pos_hash = pos_row.to(torch.int64) * N + pos_col.to(torch.int64)
    neg_hash = neg_row.to(torch.int64) * N + neg_col.to(torch.int64)
    is_pos = torch.isin(neg_hash, pos_hash)

    valid_mask = ~is_pos & ~is_self_loop
    neg_row_valid = neg_row[valid_mask]
    neg_col_valid = neg_col[valid_mask]

    if neg_row_valid.shape[0] == 0:
        pos_scores_clamped = torch.clamp(pos_scores, 1e-7, 1 - 1e-7)
        return nn.functional.binary_cross_entropy(pos_scores_clamped, pos_labels)

    # 负边分数: 同样从 z 按需计算
    neg_dots = (z[neg_row_valid] * z[neg_col_valid]).sum(dim=1)
    neg_scores = torch.sigmoid(neg_dots)
    neg_labels = torch.zeros(neg_row_valid.shape[0], device=z.device)

    # 合并正边和负边计算 BCE
    all_scores = torch.cat([pos_scores, neg_scores])
    all_labels = torch.cat([pos_labels, neg_labels])
    all_scores = torch.clamp(all_scores, 1e-7, 1 - 1e-7)

    edge_loss = nn.functional.binary_cross_entropy(all_scores, all_labels)

    return edge_loss
