"""
M4: SNP 归因模块
================

从训练好的 VGAE 模型反投影每个 SNP 对表型预测的贡献度。

核心原理（线性反投影）:
  节点特征 X = G @ E 是纯线性操作，
  因此梯度可精确分解到每个 SNP，无近似误差。

数学推导:
  设 s = Σ_i ŷ_i (全体预测值之和)，由链式法则:

    ∂s/∂G[i,j] = Σ_d (∂s/∂X[i,d]) · (∂X[i,d]/∂G[i,j])
               = Σ_d ∇X[i,d] · E[j,d]
               = ⟨∇X_i, E_j⟩

  其中 ∇X = ∂s/∂X 由 torch.autograd.grad 精确计算，
  穿透全部非线性层（2层 GCN + BN + ELU + MLP）。

  SNP j 的群体级贡献:
    score_j = Σ_i G[i,j] · ⟨∇X_i, E_j⟩

  矩阵形式（路径 A，内存最优）:
    W = G^T @ ∇X            → (M, D_snp)
    scores = (W * E).sum(1)  → (M,)
    中间张量仅 (M, D_snp) ≈ 622KB（M=9728, D_snp=64）

可扩展性:
  - 支持 per-individual 归因（个性化 SNP 分析）
  - 支持跨 CV fold 聚合（mean / median / max_abs）
  - 可扩展为 Integrated Gradients（解决深度网络梯度饱和问题）
  - 可按染色体分组输出（需传入 snp_chrom 信息）
"""

import numpy as np
import torch
from torch import Tensor
from typing import Optional

from model.vgae import VGAEModel
from data.genotype_utils import impute_genotype


def compute_snp_scores(
    model: VGAEModel,
    node_features: Tensor,
    adj_norm: Tensor,
    genotype: np.ndarray,
    snp_embeddings: np.ndarray,
    aggregate: str = "sum",
    device: str = "cpu",
) -> Tensor:
    """
    计算每个 SNP 对表型预测的群体级贡献分数。

    参数:
        model:          已训练的 VGAEModel（自动切 eval 模式）
        node_features:  (N, D_snp) 节点特征 X = G @ E
        adj_norm:       (N, N) 稀疏归一化邻接矩阵
        genotype:       (M, N) 基因型矩阵，-9 为缺失（项目约定格式）
        snp_embeddings: (M, D_snp) SNP 嵌入矩阵
        aggregate:      "sum"(默认) / "mean" / "abs_sum"
                        - "sum":    保留符号的群体总贡献
                        - "mean":   群体平均贡献（sum / N）
                        - "abs_sum": 绝对值之和（需逐个体计算）
        device:         计算设备

    返回:
        snp_scores: (M,) 每个 SNP 的贡献分数

    可扩展性:
      - "abs_sum" 需逐个体梯度，计算量 O(N) 次 autograd
      - 后续可支持按染色体分组的归因汇总
    """
    model.eval()

    # ---- Step 1: 插补基因型并转为 Tensor ----
    G_np = impute_genotype(genotype)  # (N, M)
    G = torch.from_numpy(G_np).to(device)       # (N, M)
    E = torch.from_numpy(snp_embeddings.astype(np.float32)).to(device)  # (M, D_snp)

    # ---- Step 2: 前向传播，获取 ∇X = ∂(Σŷ)/∂X ----
    X = node_features.clone().detach().to(device).requires_grad_(True)

    out = model(X, adj_norm.to(device))
    y_pred = out["y_pred"]  # (N,)
    s = y_pred.sum()

    grad_X = torch.autograd.grad(s, X, retain_graph=False)[0]  # (N, D_snp)

    # ---- Step 3: 路径 A 反投影到 SNP 空间 ----
    # W = G^T @ ∇X → (M, D_snp)
    #   W[j,d] = Σ_i G[i,j] · ∇X[i,d]
    W = G.T @ grad_X  # (M, D_snp)

    # scores_j = Σ_d W[j,d] · E[j,d] = ⟨W_j, E_j⟩
    snp_scores = (W * E).sum(dim=1)  # (M,)

    # ---- Step 4: 聚合方式 ----
    if aggregate == "mean":
        snp_scores = snp_scores / G.shape[0]
    elif aggregate == "abs_sum":
        # abs_sum 需要逐个体计算（见 compute_individual_scores）
        # 此处用群体级 |score_j| 作为近似
        snp_scores = snp_scores.abs()

    return snp_scores.detach().cpu()


def compute_individual_scores(
    model: VGAEModel,
    node_features: Tensor,
    adj_norm: Tensor,
    genotype: np.ndarray,
    snp_embeddings: np.ndarray,
    batch_individuals: int = 256,
    device: str = "cpu",
) -> tuple[Tensor, Tensor]:
    """
    计算逐个体的 SNP 贡献（个性化归因）。

    对每个个体 i，计算 ∂ŷ_i/∂X，再反投影到 SNP 空间。
    由于 GCN 中 ŷ_i 依赖 i 的多跳邻居，梯度 ∂ŷ_i/∂X 是
    全图耦合的（非对角），因此需要逐个体调用 autograd。

    参数:
        model:             已训练的 VGAEModel
        node_features:     (N, D_snp) 节点特征
        adj_norm:          (N, N) 稀疏归一化邻接
        genotype:          (M, N) 基因型矩阵，-9 为缺失
        snp_embeddings:    (M, D_snp) SNP 嵌入
        batch_individuals: 批量大小（仅用于进度提示，autograd 逐个体执行）
        device:            计算设备

    返回:
        ind_scores: (N, M) — 个体 i 在 SNP j 上的贡献
        pop_scores: (M,)   — 跨个体聚合（绝对值之和）

    可扩展性:
      - N=2002 时约 2002 次 autograd.grad 调用，单次 <5ms，总计 ~10s
      - 后续可用 torch.func.vmap 加速批量 jacobian 计算
    """
    model.eval()
    N = node_features.shape[0]

    # 插补基因型
    G_np = impute_genotype(genotype)  # (N, M)
    G = torch.from_numpy(G_np).to(device)
    E = torch.from_numpy(snp_embeddings.astype(np.float32)).to(device)  # (M, D_snp)

    # 前向传播（重用计算图）
    X = node_features.clone().detach().to(device).requires_grad_(True)
    out = model(X, adj_norm.to(device))
    y_pred = out["y_pred"]  # (N,)

    ind_scores = torch.zeros(N, G.shape[1], device="cpu")

    for i in range(N):
        # ∂ŷ_i/∂X ∈ (N, D_snp) — 注意：GCN 使得 ŷ_i 依赖全图 X
        grad_i = torch.autograd.grad(
            y_pred[i], X, retain_graph=True, create_graph=False
        )[0]  # (N, D_snp)

        # 个体 i 对 SNP j 的贡献:
        #   score_{i,j} = G[i,j] · ⟨∇X_i, E_j⟩
        #   = G[i,j] · Σ_d grad_i[i,d] · E[j,d]
        #
        # 注意：这里用 grad_i[i, :]（个体 i 自己行的梯度），
        #   因为 G[i,j] 是标量权重，∇X_i 是该个体节点特征的梯度。
        grad_i_self = grad_i[i, :]  # (D_snp,)

        # ⟨grad_i_self, E_j⟩ for all j → (M,)
        interaction = grad_i_self @ E.T  # (M,)

        # 加权基因型
        ind_scores[i] = (G[i, :] * interaction).cpu()

    pop_scores = ind_scores.abs().sum(dim=0)  # (M,)

    return ind_scores, pop_scores


def top_snps(
    scores: Tensor,
    top_k: int = 1000,
    snp_names: Optional[list[str]] = None,
) -> dict:
    """
    提取贡献度最高的 top-K SNP。

    参数:
        scores:    (M,) SNP 贡献分数
        top_k:     返回的 SNP 数量
        snp_names: 可选的 SNP 名称列表（长度 = M）

    返回:
        dict:
          'indices': (K,) int — SNP 索引
          'scores':  (K,) float — 贡献分数（保留符号）
          'names':   (K,) str | None — SNP 名称

    可扩展性:
      - 后续可按染色体分组返回 top-K
    """
    k = min(top_k, scores.shape[0])
    top_vals, top_idx = torch.topk(scores.abs(), k)

    result = {
        "indices": top_idx.numpy(),
        "scores": scores[top_idx].numpy(),
        "names": (
            [snp_names[i] for i in top_idx.numpy()]
            if snp_names is not None
            else None
        ),
    }
    return result


def aggregate_cv_scores(
    fold_scores: list[Tensor],
    aggregate: str = "mean",
) -> Tensor:
    """
    聚合多个 CV fold 的 SNP 分数。

    每个 fold 训练一个独立 VGAE，产生一组 SNP 分数。
    跨 fold 聚合可提高归因的稳定性和统计可靠性。

    参数:
        fold_scores: 每折 (M,) score 张量列表
        aggregate:   "mean"(默认) / "median" / "max_abs"

    返回:
        (M,) 聚合后的分数

    可扩展性:
      - "mean": 最稳定，推荐默认
      - "median": 对异常 fold 鲁棒
      - "max_abs": 保守估计，取最大效应
    """
    stacked = torch.stack(fold_scores, dim=0)  # (K_folds, M)

    if aggregate == "mean":
        return stacked.mean(dim=0)
    elif aggregate == "median":
        return stacked.median(dim=0).values
    elif aggregate == "max_abs":
        return stacked.abs().max(dim=0).values
    else:
        raise ValueError(f"未知聚合方式: {aggregate}")
