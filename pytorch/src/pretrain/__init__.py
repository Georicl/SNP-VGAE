"""
SNP-VAE 预训练模块
==================

使用变分自编码器 (VAE) 对 SNP 基因型数据进行无监督预训练，
学习每个 SNP 的低维嵌入表示，为下游 VGAE 模型提供特征输入。

模块组成:
  - preprocess: 基因型数据预处理（缺失值填充、标准化）
  - vae: SNP-VAE 模型定义与损失函数
  - train: 完整训练流程（含早停、KL 退火）
  - extract_embeddings: 从训练好的模型提取 SNP 嵌入

作者: Xiang Yang
邮箱: Georicl@outlook.com
创建时间: 2026-07-11
"""
