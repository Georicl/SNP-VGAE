"""
K-fold 交叉验证模块
====================

编排 VGAE 的 K-fold 交叉验证流程:
  1. SNP-VAE 预训练（CV 外层，所有 fold 共享嵌入）
  2. 图构建（GRM KNN + 特征聚合）
  3. K-fold 划分 → 每 fold 训练 VGAE → 评估
  4. 汇总 K-fold 结果（均值 ± 标准差）

数据流:
  genotype + snp_embeddings + grm + labels
  → GraphBuilder → K-fold masks → VGAETrainer × K
  → 汇总 metrics

作者: Xiang Yang
邮箱: Georicl@outlook.com
创建时间: 2026-07-23
"""

import os
import numpy as np
import torch
from typing import Any

from graph.build_graph import GraphBuilder
from model.vgae import VGAEModel
from train.trainer import VGAETrainer


def generate_kfold_masks(
    n_samples: int,
    k_folds: int = 5,
    seed: int = 42,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    生成 K-fold 的 train/val/test 掩码。

    每 fold: 1/K 测试, 1/K 验证, (K-2)/K 训练。

    参数:
        n_samples: 样本总数
        k_folds:   折数
        seed:      随机种子

    返回:
        [(train_mask, val_mask, test_mask), ...] × K
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    fold_size = n_samples // k_folds

    masks = []
    for fold in range(k_folds):
        # 测试集索引
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < k_folds - 1 else n_samples
        test_idx = indices[test_start:test_end]

        # 剩余索引中取验证集
        remaining = np.concatenate([indices[:test_start], indices[test_end:]])
        val_size = len(remaining) // (k_folds - 1)
        val_idx = remaining[:val_size]
        train_idx = remaining[val_size:]

        # 构建 bool 掩码
        train_mask = torch.zeros(n_samples, dtype=torch.bool)
        val_mask = torch.zeros(n_samples, dtype=torch.bool)
        test_mask = torch.zeros(n_samples, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        masks.append((train_mask, val_mask, test_mask))

    return masks


def run_kfold_cv(
    genotype: np.ndarray,
    snp_embeddings: np.ndarray,
    grm: np.ndarray,
    labels: np.ndarray,
    k_folds: int = 5,
    seed: int = 42,
    # 图构建参数
    k_neighbors: int = 30,
    # VGAE 模型参数
    d_snp: int = 64,
    d_hidden: int = 128,
    d_z: int = 32,
    dropout: float = 0.5,
    mlp_hidden: int = 64,
    # 训练参数
    lr: float = 1e-3,
    alpha: float = 0.1,
    beta_kl: float = 0.01,
    max_epochs: int = 500,
    patience: int = 50,
    weight_decay: float = 1e-5,
    kl_warmup_epochs: int = 0,
    # 环境参数
    device: str = "cpu",
    progress: bool = True,
    checkpoint_dir: str | None = None,
) -> dict[str, Any]:
    """
    执行完整的 K-fold 交叉验证。

    参数:
        genotype:      (M, N) 基因型矩阵（SNP × 样本），-9 为缺失
        snp_embeddings:(M, D_snp) SNP 嵌入矩阵
        grm:           (N, N) GRM 矩阵
        labels:        (N,) 表型值
        k_folds:       折数
        seed:          随机种子
        k_neighbors:   KNN 近邻数
        d_snp:         输入特征维度
        d_hidden:      GCN 隐藏层维度
        d_z:           隐空间维度
        dropout:       GCN dropout
        mlp_hidden:    预测头隐藏层
        lr:            学习率
        alpha:         边损失权重
        beta_kl:       KL 散度权重
        max_epochs:    最大 epoch
        patience:      早停耐心值
        weight_decay:  权重衰减
        kl_warmup_epochs: KL 退火轮数
        device:        计算设备
        progress:      显示进度条
        checkpoint_dir: checkpoint 目录（None 则不保存）

    返回:
        dict:
          'fold_metrics':  每 fold 的测试指标列表
          'mean_metrics':  各指标均值
          'std_metrics':   各指标标准差
          'fold_histories': 每 fold 的训练历史
          'models':        每 fold 训练好的模型列表
    """
    N = len(labels)
    masks = generate_kfold_masks(N, k_folds, seed)

    # 图构建（所有 fold 共享同一张图，因为 GRM 和 snp_embeddings 不变）
    if progress:
        print(f"\n===== M2: 图构建 (K={k_neighbors}) =====")
    builder = GraphBuilder()
    adj_norm, adj_raw = builder.knn_graph_builder(grm, k=k_neighbors)
    node_features = builder.build_node_features(genotype, snp_embeddings)

    # 确保 checkpoint 目录存在
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    labels_tensor = torch.from_numpy(labels).float()

    fold_metrics_list: list[dict[str, float]] = []
    fold_histories: list[dict] = []
    fold_models: list[VGAEModel] = []

    for fold_idx, (train_mask, val_mask, test_mask) in enumerate(masks):
        if progress:
            print(f"\n{'='*50}")
            print(f"Fold {fold_idx + 1}/{k_folds}")
            print(f"  训练: {train_mask.sum().item()}  "
                  f"验证: {val_mask.sum().item()}  "
                  f"测试: {test_mask.sum().item()}")
            print(f"{'='*50}")

        # 创建模型
        model = VGAEModel(
            d_snp=d_snp,
            d_hidden=d_hidden,
            d_z=d_z,
            dropout=dropout,
            mlp_hidden=mlp_hidden,
        )

        # checkpoint 路径
        ckpt_path = None
        if checkpoint_dir is not None:
            ckpt_path = f"{checkpoint_dir}/fold_{fold_idx + 1}.pt"

        # 训练
        trainer = VGAETrainer(
            model=model,
            adj_norm=adj_norm,
            adj_raw=adj_raw,
            node_features=node_features,
            labels=labels_tensor,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            lr=lr,
            alpha=alpha,
            beta=beta_kl,
            weight_decay=weight_decay,
            kl_warmup_epochs=kl_warmup_epochs,
            patience=patience,
            max_epochs=max_epochs,
            device=device,
            checkpoint_path=ckpt_path,
            progress=progress,
        )

        result = trainer.train()

        fold_metrics_list.append(result["test_metrics"])
        fold_histories.append(result["history"])
        fold_models.append(result["model"])

    # 汇总结果
    all_keys = fold_metrics_list[0].keys()
    mean_metrics = {}
    std_metrics = {}
    for key in all_keys:
        vals = [m[key] for m in fold_metrics_list]
        mean_metrics[key] = float(np.mean(vals))
        std_metrics[key] = float(np.std(vals))

    if progress:
        print(f"\n{'='*50}")
        print(f"K-fold 交叉验证结果 (K={k_folds})")
        print(f"{'='*50}")
        for key in all_keys:
            print(f"  {key.upper():>4s}: {mean_metrics[key]:.4f} ± {std_metrics[key]:.4f}")

    return {
        "fold_metrics": fold_metrics_list,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "fold_histories": fold_histories,
        "models": fold_models,
    }
