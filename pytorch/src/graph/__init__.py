"""
样本图构建模块
==============

基于 GRM 矩阵构建 KNN 稀疏图，并将基因型与 SNP 嵌入
聚合为节点特征，为 VGAE 模型提供图结构输入。

作者: Xiang Yang
邮箱: Georicl@outlook.com
创建时间: 2026-07-15
"""

from graph.build_graph import GraphBuilder
