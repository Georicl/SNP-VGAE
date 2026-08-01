"""
训练可视化模块
==============

提供 VGAE 训练过程的可视化:
  - 训练/验证损失曲线
  - K-fold 指标对比图
  - 置换检验分布图

作者: Xiang Yang
邮箱: Georicl@outlook.com
创建时间: 2026-07-23
"""

import matplotlib
matplotlib.use("Agg")  # 非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    history: dict[str, list[float]],
    save_path: str | None = None,
    title: str = "VGAE 训练曲线",
) -> None:
    """
    绘制训练/验证损失曲线。

    参数:
        history:   训练历史 dict，包含 train_loss, val_loss 等列表
        save_path: 图片保存路径（None 则 plt.show()）
        title:     图标题
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: 总损失
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=1.5)
    axes[0].plot(epochs, history["val_loss"], label="Validation", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} — 总损失")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 右图: 表型损失 + 边损失
    if "train_pheno" in history:
        axes[1].plot(epochs, history["train_pheno"],
                     label="Train Pheno", linewidth=1.5)
        axes[1].plot(epochs, history["val_pheno"],
                     label="Val Pheno", linewidth=1.5)
    if "train_edge" in history:
        axes[1].plot(epochs, history["train_edge"],
                     label="Train Edge", linewidth=1.5, linestyle="--")
        axes[1].plot(epochs, history["val_edge"],
                     label="Val Edge", linewidth=1.5, linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title(f"{title} — 分项损失")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plots] 训练曲线已保存至 {save_path}")
    else:
        plt.show()
    plt.close()


def plot_kfold_results(
    fold_metrics: list[dict[str, float]],
    save_path: str | None = None,
    title: str = "K-fold 交叉验证结果",
) -> None:
    """
    绘制 K-fold 各指标对比柱状图。

    参数:
        fold_metrics: 每 fold 的指标 dict 列表
        save_path:    图片保存路径
        title:        图标题
    """
    metrics_keys = list(fold_metrics[0].keys())
    n_folds = len(fold_metrics)
    n_metrics = len(metrics_keys)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(n_folds)
    width = 0.8 / n_metrics

    for i, key in enumerate(metrics_keys):
        values = [m[key] for m in fold_metrics]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=key.upper(), alpha=0.8)

    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i+1}" for i in range(n_folds)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plots] K-fold 结果已保存至 {save_path}")
    else:
        plt.show()
    plt.close()


def plot_permutation_results(
    permuted_scores: np.ndarray,
    real_score: float,
    p_value: float,
    metric_key: str = "r2",
    save_path: str | None = None,
) -> None:
    """
    绘制置换检验结果分布图。

    参数:
        permuted_scores: 置换指标数组
        real_score:      真实数据指标
        p_value:         p-value
        metric_key:      指标名称
        save_path:       图片保存路径
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(permuted_scores, bins=30, alpha=0.7, color="steelblue",
            edgecolor="white", label="Permutation")
    ax.axvline(real_score, color="red", linewidth=2, linestyle="--",
               label=f"Real ({metric_key}={real_score:.4f})")

    ax.set_xlabel(metric_key.upper())
    ax.set_ylabel("Frequency")
    ax.set_title(f"置换检验 (p={p_value:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plots] 置换检验图已保存至 {save_path}")
    else:
        plt.show()
    plt.close()
