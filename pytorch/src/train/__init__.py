"""
VGAE 训练与评估模块
====================

包含 VGAE 模型的完整训练与评估流程:
  - trainer: 单 fold 训练器（含早停、checkpoint、KL 退火）
  - cross_validation: K-fold 交叉验证编排
  - metrics: 标准评估指标（MSE, MAE, Pearson R, R²）
  - permutation_test: 置换检验统计显著性评估
  - plots: 训练过程可视化

作者: Xiang Yang
邮箱: Georicl@outlook.com
创建时间: 2026-07-23
"""