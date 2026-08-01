"""
VGAE 训练器
===========

封装 VGAE 模型的单 fold 训练循环，包含:
  - 联合损失优化（表型 MSE + 边重建 BCE + KL 散度）
  - 表型损失仅在 train_mask 上计算（防止数据泄漏）
  - 早停机制（基于验证集 loss）
  - Checkpoint 保存/恢复
  - KL 退火（warmup）
  - 训练历史记录与 tqdm 进度条

数据流:
  adj_norm, adj_raw, node_features, labels + train/val/test mask
  → VGAEModel 训练 → 最佳 checkpoint → 测试集评估

作者: Xiang Yang
邮箱: Georicl@outlook.com
创建时间: 2026-07-23
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Any

from model.vgae import VGAEModel, edge_reconstruction_loss
from train.metrics import evaluate_all


class VGAETrainer:
    """
    VGAE 单 fold 训练器。

    参数:
        model:            VGAEModel 实例
        adj_norm:         (N, N) 稀疏归一化邻接矩阵
        adj_raw:          (N, N) 稀疏原始邻接矩阵（含自环）
        node_features:    (N, D_snp) 节点特征
        labels:           (N,) 表型标签
        train_mask:       (N,) bool 训练集掩码
        val_mask:         (N,) bool 验证集掩码
        test_mask:        (N,) bool 测试集掩码
        lr:               学习率
        alpha:            边损失权重
        beta:             KL 散度权重
        neg_sample_ratio: 负采样比例
        weight_decay:     Adam 权重衰减
        patience:         早停耐心值
        max_epochs:       最大训练轮数
        kl_warmup_epochs: KL 退火轮数（β 从 0 线性增至目标值）
        device:           计算设备
        checkpoint_path:  checkpoint 保存路径（None 则不保存）
        progress:         是否显示 tqdm 进度条

    可扩展性:
      - 可支持边损失的动态负采样
      - 可支持多任务损失权重自动调节
    """

    def __init__(
        self,
        model: VGAEModel,
        adj_norm: torch.Tensor,
        adj_raw: torch.Tensor,
        node_features: torch.Tensor,
        labels: torch.Tensor,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        test_mask: torch.Tensor,
        lr: float = 1e-3,
        alpha: float = 0.1,
        beta: float = 0.01,
        neg_sample_ratio: float = 1.0,
        weight_decay: float = 1e-5,
        patience: int = 50,
        max_epochs: int = 500,
        kl_warmup_epochs: int = 0,
        device: str = "cpu",
        checkpoint_path: str | None = None,
        progress: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.adj_norm = adj_norm.to(device)
        self.adj_raw = adj_raw.to(device)
        self.node_features = node_features.to(device)
        self.labels = labels.to(device)
        self.train_mask = train_mask.to(device)
        self.val_mask = val_mask.to(device)
        self.test_mask = test_mask.to(device)

        self.alpha = alpha
        self.beta = beta
        self.neg_sample_ratio = neg_sample_ratio
        self.patience = patience
        self.max_epochs = max_epochs
        self.kl_warmup_epochs = kl_warmup_epochs
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.progress = progress

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # 训练历史
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_pheno": [],
            "val_pheno": [],
            "train_edge": [],
            "val_edge": [],
        }

    def _compute_loss(
        self, out: dict[str, torch.Tensor], mask: torch.Tensor, current_beta: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算联合损失，表型损失仅在 mask 样本上计算。

        L = L_pheno(mask) + α · L_edge(all) + β · L_KL(all)

        关键: 表型损失使用 mask 防止数据泄漏，
              边重建和 KL 散度在全部节点上计算（无监督），
              KL 归一化使用全量 N（非 mask 子集）。

        参数:
            out:           VGAEModel.forward() 的输出 dict
            mask:          (N,) bool 掩码，指定表型损失计算范围
            current_beta:  当前 KL 权重（含 warmup 退火）

        返回:
            (total_loss, pheno_loss, edge_loss, kl_loss)
        """
        N = out["y_pred"].shape[0]

        # --- 1. 表型预测损失 (MSE)，仅在 mask 上计算 ---
        y_pred_masked = out["y_pred"][mask]
        y_true_masked = self.labels[mask]
        pheno_loss = nn.functional.mse_loss(y_pred_masked, y_true_masked)

        # --- 2. 边重建损失 (BCE + 负采样)，全量无监督 ---
        edge_loss = edge_reconstruction_loss(
            z=out["z"],
            edge_index=out["edge_index"],
            adj_raw=self.adj_raw,
            neg_sample_ratio=self.neg_sample_ratio,
        )

        # --- 3. KL 散度，全量 N 归一化 ---
        kl_loss = -0.5 * torch.sum(
            1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp()
        ) / N

        # --- 联合损失 ---
        total_loss = pheno_loss + self.alpha * edge_loss + current_beta * kl_loss

        return total_loss, pheno_loss, edge_loss, kl_loss

    def _train_epoch(self, current_beta: float) -> dict[str, float]:
        """执行一个训练 epoch，返回各项损失。"""
        self.model.train()
        self.optimizer.zero_grad()

        # 前向传播
        out = self.model(self.node_features, self.adj_norm)

        # 联合损失（表型损失仅在 train_mask 上计算）
        total_loss, pheno_loss, edge_loss, kl_loss = self._compute_loss(
            out, self.train_mask, current_beta
        )

        total_loss.backward()
        self.optimizer.step()

        return {
            "total": total_loss.item(),
            "pheno": pheno_loss.item(),
            "edge": edge_loss.item(),
            "kl": kl_loss.item(),
        }

    @torch.no_grad()
    def _validate(self, current_beta: float) -> dict[str, float]:
        """验证模式，表型损失在 val_mask 上计算。"""
        self.model.eval()

        out = self.model(self.node_features, self.adj_norm)

        total_loss, pheno_loss, edge_loss, kl_loss = self._compute_loss(
            out, self.val_mask, current_beta
        )

        return {
            "total": total_loss.item(),
            "pheno": pheno_loss.item(),
            "edge": edge_loss.item(),
            "kl": kl_loss.item(),
        }

    @torch.no_grad()
    def _evaluate(self, mask: torch.Tensor) -> dict[str, float]:
        """在指定 mask 上评估表型预测指标。"""
        self.model.eval()
        out = self.model(self.node_features, self.adj_norm)
        y_pred = out["y_pred"][mask].cpu().numpy()
        y_true = self.labels[mask].cpu().numpy()
        return evaluate_all(y_pred, y_true)

    def train(self) -> dict[str, Any]:
        """
        执行完整训练流程。

        返回:
            dict:
              'best_epoch':    最佳 epoch 编号
              'best_val_loss': 最佳验证损失
              'test_metrics':  测试集评估指标 (mse, mae, r, r2)
              'history':       训练历史记录
              'model':         恢复最佳权重后的模型引用
        """
        best_val_loss = float("inf")
        best_epoch = 0
        patience_count = 0
        best_state: dict[str, torch.Tensor] | None = None

        epoch_iter = tqdm(
            range(1, self.max_epochs + 1),
            desc="VGAE 训练",
            disable=not self.progress,
        )

        for epoch in epoch_iter:
            # KL 退火: 前 warmup_epochs 轮 β 从 0 线性增至目标值
            if self.kl_warmup_epochs > 0:
                current_beta = self.beta * min(1.0, epoch / self.kl_warmup_epochs)
            else:
                current_beta = self.beta

            # 训练
            train_metrics = self._train_epoch(current_beta)

            # 验证
            val_metrics = self._validate(current_beta)

            # 记录历史
            self.history["train_loss"].append(train_metrics["total"])
            self.history["val_loss"].append(val_metrics["total"])
            self.history["train_pheno"].append(train_metrics["pheno"])
            self.history["val_pheno"].append(val_metrics["pheno"])
            self.history["train_edge"].append(train_metrics["edge"])
            self.history["val_edge"].append(val_metrics["edge"])

            # 更新进度条
            epoch_iter.set_postfix(
                t_loss=f"{train_metrics['total']:.4f}",
                v_loss=f"{val_metrics['total']:.4f}",
                pheno=f"{train_metrics['pheno']:.4f}",
                β=f"{current_beta:.4f}",
            )

            # 早停检查（退火期间不计入早停）
            if self.kl_warmup_epochs > 0 and epoch <= self.kl_warmup_epochs:
                # 退火阶段: 只更新最佳状态，不增加耐心计数
                if val_metrics["total"] < best_val_loss:
                    best_val_loss = val_metrics["total"]
                    best_epoch = epoch
                    best_state = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }
                    if self.checkpoint_path is not None:
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": best_state,
                                "val_loss": best_val_loss,
                                "optimizer_state_dict": self.optimizer.state_dict(),
                            },
                            self.checkpoint_path,
                        )
            elif val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                best_epoch = epoch
                patience_count = 0
                best_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }

                # 保存 checkpoint
                if self.checkpoint_path is not None:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": best_state,
                            "val_loss": best_val_loss,
                            "optimizer_state_dict": self.optimizer.state_dict(),
                        },
                        self.checkpoint_path,
                    )
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    if self.progress:
                        print(
                            f"\n[VGAE] 早停于 epoch {epoch}，"
                            f"最佳 epoch={best_epoch}，"
                            f"最佳验证损失={best_val_loss:.4f}"
                        )
                    break

        # 恢复最佳权重
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # 测试集评估
        test_metrics = self._evaluate(self.test_mask)

        if self.progress:
            print(
                f"[VGAE] 测试集: MSE={test_metrics['mse']:.4f}, "
                f"MAE={test_metrics['mae']:.4f}, "
                f"R={test_metrics['r']:.4f}, "
                f"R²={test_metrics['r2']:.4f}"
            )

        return {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "test_metrics": test_metrics,
            "history": self.history,
            "model": self.model,
        }
