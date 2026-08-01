"""
置换检验模块
============

通过打乱表型标签来评估模型预测性能的统计显著性。

原理:
  1. 在真实数据上跑 K-fold CV，得到真实 R²（或 MSE）
  2. 重复 N 次: 随机打乱标签 → 跑 K-fold CV → 记录指标
  3. p-value = (比真实值好的置换次数 + 1) / (N + 1)

作者: Xiang Yang
邮箱: Georicl@outlook.com
创建时间: 2026-07-23
"""

import numpy as np
from typing import Any

from train.cross_validation import run_kfold_cv


def permutation_test(
    genotype: np.ndarray,
    snp_embeddings: np.ndarray,
    grm: np.ndarray,
    labels: np.ndarray,
    n_permutations: int = 100,
    metric_key: str = "r2",
    seed: int = 42,
    # 透传给 run_kfold_cv 的参数
    k_folds: int = 5,
    k_neighbors: int = 30,
    d_snp: int = 64,
    d_hidden: int = 128,
    d_z: int = 32,
    dropout: float = 0.5,
    mlp_hidden: int = 64,
    lr: float = 1e-3,
    alpha: float = 0.1,
    beta_kl: float = 0.01,
    max_epochs: int = 500,
    patience: int = 50,
    kl_warmup_epochs: int = 0,
    device: str = "cpu",
    progress: bool = True,
) -> dict[str, Any]:
    """
    执行置换检验。

    参数:
        genotype:       (M, N) 基因型矩阵
        snp_embeddings: (M, D_snp) SNP 嵌入
        grm:            (N, N) GRM 矩阵
        labels:         (N,) 表型值
        n_permutations: 置换次数
        metric_key:     用于计算 p-value 的指标名 ('r2', 'mse', 'r', 'mae')
        seed:           随机种子
        其他参数:       透传给 run_kfold_cv

    返回:
        dict:
          'real_score':         真实数据上的指标
          'permuted_scores':    置换得到的指标数组
          'p_value':            单侧 p-value
          'mean_permuted':      置换指标均值
          'std_permuted':       置换指标标准差
    """
    rng = np.random.RandomState(seed)

    # 1. 真实数据上的表现
    if progress:
        print("\n" + "=" * 60)
        print("[置换检验] Step 1: 真实数据 K-fold CV")
        print("=" * 60)

    real_result = run_kfold_cv(
        genotype=genotype,
        snp_embeddings=snp_embeddings,
        grm=grm,
        labels=labels,
        k_folds=k_folds,
        seed=seed,
        k_neighbors=k_neighbors,
        d_snp=d_snp,
        d_hidden=d_hidden,
        d_z=d_z,
        dropout=dropout,
        mlp_hidden=mlp_hidden,
        lr=lr,
        alpha=alpha,
        beta_kl=beta_kl,
        max_epochs=max_epochs,
        patience=patience,
        kl_warmup_epochs=kl_warmup_epochs,
        device=device,
        progress=progress,
    )
    real_score = real_result["mean_metrics"][metric_key]

    # 2. 置换检验
    permuted_scores = []

    # 对于 MSE/MAE，"更好"意味着更小，所以 p-value 计算方向不同
    lower_is_better = metric_key in ("mse", "mae")

    if progress:
        print(f"\n[置换检验] 真实 {metric_key} = {real_score:.4f}")
        print(f"[置换检验] 开始 {n_permutations} 次置换...")

    for i in range(n_permutations):
        if progress:
            print(f"  置换 {i + 1}/{n_permutations}", end="\r")

        # 打乱标签
        perm_idx = rng.permutation(len(labels))
        permuted_labels = labels[perm_idx]

        # 用打乱后的标签跑 K-fold CV
        perm_result = run_kfold_cv(
            genotype=genotype,
            snp_embeddings=snp_embeddings,
            grm=grm,
            labels=permuted_labels,
            k_folds=k_folds,
            seed=seed,
            k_neighbors=k_neighbors,
            d_snp=d_snp,
            d_hidden=d_hidden,
            d_z=d_z,
            dropout=dropout,
            mlp_hidden=mlp_hidden,
            lr=lr,
            alpha=alpha,
            beta_kl=beta_kl,
            max_epochs=max_epochs,
            patience=patience,
            kl_warmup_epochs=kl_warmup_epochs,
            device=device,
            progress=False,  # 置换时不显示详细进度
        )
        permuted_scores.append(perm_result["mean_metrics"][metric_key])

    permuted_scores = np.array(permuted_scores)

    # 3. 计算 p-value
    if lower_is_better:
        # MSE/MAE: p-value = 置换中比真实更小的比例
        n_extreme = np.sum(permuted_scores <= real_score)
    else:
        # R²/R: p-value = 置换中比真实更大的比例
        n_extreme = np.sum(permuted_scores >= real_score)

    p_value = (n_extreme + 1) / (n_permutations + 1)

    if progress:
        print(f"\n\n[置换检验] 结果:")
        print(f"  真实 {metric_key}:       {real_score:.4f}")
        print(f"  置换均值 ± 标准差: {permuted_scores.mean():.4f} ± {permuted_scores.std():.4f}")
        print(f"  p-value:           {p_value:.4f}")
        print(f"  {'显著' if p_value < 0.05 else '不显著'} (α=0.05)")

    return {
        "real_score": real_score,
        "permuted_scores": permuted_scores,
        "p_value": p_value,
        "mean_permuted": float(permuted_scores.mean()),
        "std_permuted": float(permuted_scores.std()),
        "metric_key": metric_key,
    }
