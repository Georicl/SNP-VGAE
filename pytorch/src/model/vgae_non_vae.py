import torch
import torch.nn as nn
from torch import Tensor

from model.gcn import GCNConv
from model.vgae import VGAEModel


class VGAENoVAE(VGAEModel):

    def __init__(
        self,
        d_snp: int,
        d_hidden1: int = 512,
        d_hidden2: int = 128,
        d_z: int = 32,
        dropout: float = 0.7,
        mlp_hidden: int = 64
    ) -> None:

        super().__init__(d_snp=d_snp, d_hidden=64, d_z=d_z,
                         dropout=dropout, mlp_hidden=mlp_hidden)

        # 第一层压缩 从全量SNP -> 512
        self.gcn1 = GCNConv(d_snp, d_hidden1)
        self.bn1 = nn.BatchNorm1d(d_hidden1)
        # 第二层压缩 512 -> 128
        self.gcn2 = GCNConv(d_hidden1, d_hidden2)
        self.bn2 = nn.BatchNorm1d(d_hidden2)
        # 输出隐向量
        self.gcn_mu = GCNConv(d_hidden2, d_z)
        self.gcn_logvar = GCNConv(d_hidden2, d_z)

    def encode(self, x, adj_norm):
        # 第一层
        h1 = self.gcn1(x, adj_norm)
        h1 = self.bn1(h1)
        h1 = self.elu(h1)
        h1 = self.dropout(h1)

        # 第二层
        h2 = self.gcn2(h1, adj_norm)
        h2 = self.bn2(h2)
        h2 = self.elu(h2)
        h2 = self.dropout(h2)

        # 分叉
        mu = self.gcn_mu(h2, adj_norm)
        logvar = self.gcn_logvar(h2, adj_norm)
        return mu, logvar
