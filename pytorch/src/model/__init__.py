"""
VGAE 模型定义模块
==================

包含变分图自编码器 (VGAE) 的核心组件:
  - GCNConv: 基于稀疏矩阵乘法的图卷积层
  - VGAEModel: GCN 编码器 + 边解码器 + 表型预测头
  - 归因分析: 从训练好的模型反投影 SNP 对表型的贡献度

作者: Xiang Yang
邮箱: Georicl@outlook.com
创建时间: 2026-07-15
"""

from model.gcn import GCNConv
from model.vgae import VGAEModel, vgae_loss, edge_reconstruction_loss
from model.attribution import (
    compute_snp_scores,
    compute_individual_scores,
    top_snps,
    aggregate_cv_scores,
)

__all__ = [
    "GCNConv",
    "VGAEModel",
    "vgae_loss",
    "edge_reconstruction_loss",
    "compute_snp_scores",
    "compute_individual_scores",
    "top_snps",
    "aggregate_cv_scores",
]
