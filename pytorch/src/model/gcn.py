"""
M3: GCN 传播层
==============

实现纯 torch.sparse.mm 的图卷积传播，不依赖 PyTorch Geometric。
核心公式: H = A_norm @ X @ W + b

可扩展性:
  - 支持任意输入/输出特征维度
  - 可堆叠多层 GCNConv 构建更深的编码器
  - MPS 后端完全兼容（无 scatter/gather 操作）
"""

import torch
import torch.nn as nn
from torch import Tensor


class GCNConv(nn.Module):
    """
    单层 GCN 传播层。

    核心操作: H = sparse.mm(adj_norm, X) @ W + b
    其中 adj_norm 为 D^(-1/2) A D^(-1/2) 对称归一化稀疏邻接矩阵。

    参数:
        in_features:  输入特征维度（D_snp 或 D_hidden）
        out_features: 输出特征维度（D_hidden 或 D_z）

    输入:
        x:       (N, in_features) 节点特征张量
        adj_norm: (N, N) 稀疏归一化邻接矩阵

    输出:
        (N, out_features) 传播后的节点特征
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        # 权重矩阵与偏置
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        # 初始化: Xavier uniform 权重，零偏置
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, adj_norm: Tensor) -> Tensor:
        """
        GCN 前向传播。

        参数:
            x:        (N, in_features) 节点特征
            adj_norm: (N, N) 稀疏归一化邻接矩阵 (torch.sparse_coo_tensor)

        返回:
            (N, out_features) 传播后的特征
        """
        # 稀疏矩阵乘: (N, N) @ (N, in_features) -> (N, in_features)
        # 再乘权重: (N, in_features) @ (in_features, out_features) -> (N, out_features)
        support = torch.sparse.mm(adj_norm, x)
        output = support @ self.weight + self.bias
        return output
