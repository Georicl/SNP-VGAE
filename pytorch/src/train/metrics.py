"""
评估指标模块
============

提供基因组预测的标准评估指标: MSE, MAE, Pearson R, R²。

所有指标函数同时支持 numpy 数组和 PyTorch 张量输入，
便于在训练循环和离线评估中复用。

作者: Xiang Yang
邮箱: Georicl@outlook.com
创建时间: 2026-07-23
"""

import numpy as np
import torch
from torch import Tensor


def mse_loss(y_pred: np.ndarray | Tensor, y_true: np.ndarray | Tensor) -> float:
    """均方误差 (Mean Squared Error)。"""
    if isinstance(y_pred, Tensor):
        return torch.nn.functional.mse_loss(y_pred, y_true).item()
    return float(np.mean((y_pred - y_true) ** 2))


def mae_loss(y_pred: np.ndarray | Tensor, y_true: np.ndarray | Tensor) -> float:
    """平均绝对误差 (Mean Absolute Error)。"""
    if isinstance(y_pred, Tensor):
        return torch.nn.functional.l1_loss(y_pred, y_true).item()
    return float(np.mean(np.abs(y_pred - y_true)))


def pearson_r(y_pred: np.ndarray | Tensor, y_true: np.ndarray | Tensor) -> float:
    """Pearson 相关系数。"""
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
    if len(y_pred) < 2:
        return 0.0
    r = np.corrcoef(y_pred, y_true)[0, 1]
    return float(r) if not np.isnan(r) else 0.0


def r_squared(y_pred: np.ndarray | Tensor, y_true: np.ndarray | Tensor) -> float:
    """决定系数 R²（1 - SS_res / SS_tot）。"""
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def evaluate_all(
    y_pred: np.ndarray | Tensor, y_true: np.ndarray | Tensor
) -> dict[str, float]:
    """
    一次性计算所有评估指标。

    参数:
        y_pred: 预测值数组/张量
        y_true: 真实值数组/张量

    返回:
        {'mse': ..., 'mae': ..., 'r': ..., 'r2': ...}
    """
    return {
        "mse": mse_loss(y_pred, y_true),
        "mae": mae_loss(y_pred, y_true),
        "r": pearson_r(y_pred, y_true),
        "r2": r_squared(y_pred, y_true),
    }
