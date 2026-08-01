"""
VGAE 模型集成测试
==================

使用 test_data/ 下的真实 GRM + 基因型数据，
验证图构建输出 → VGAE 模型前向传播 + 损失计算的完整流程。

运行方式:
  PYTHONPATH=pytorch/src uv run python test/test_vgae_model.py

测试数据:
  test_data/total_autosome_grm.grm.bin  — GCTA GRM 矩阵
  test_data/total_autosome_grm.grm.id   — 样本 ID 文件
  test_data/hs_mice_genome_QCed.bed     — PLINK 基因型

作者: Xiang Yang
邮箱: Georicl@outlook.com
创建时间: 2026-07-15
"""

import os
import time

import numpy as np
import torch

from data.grm import grm_reader
from data.genotype import genotype_read
from graph.build_graph import GraphBuilder
from model.gcn import GCNConv
from model.vgae import VGAEModel, vgae_loss

# ============================================================================
# 数据路径配置
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_GRM_BIN = os.path.join(
    PROJECT_ROOT, "test_data", "total_autosome_grm.grm.bin")
DEFAULT_GRM_ID = os.path.join(
    PROJECT_ROOT, "test_data", "total_autosome_grm.grm.id")
DEFAULT_BED_FILE = os.path.join(
    PROJECT_ROOT, "test_data", "hs_mice_genome_QCed.bed")
DEFAULT_N_SAMPLES = 2002
DEFAULT_N_SNPS = 9728


def auto_detect_device() -> str:
    """自动检测设备"""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def test_gcn_conv():
    """测试 1: GCNConv 层基本功能"""
    print("\n" + "=" * 60)
    print("[测试 1] GCNConv 层基本功能")
    print("=" * 60)

    N, D_in, D_out = 100, 64, 128

    # 构造简单稀疏邻接（模拟归一化邻接）
    indices = torch.randint(0, N, (2, 300))
    values = torch.rand(300) * 0.1
    adj_norm = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()

    x = torch.randn(N, D_in)

    # 创建 GCNConv
    gcn = GCNConv(D_in, D_out)
    h = gcn(x, adj_norm)

    print(f"  输入: {x.shape}")
    print(f"  输出: {h.shape}")
    print(f"  权重形状: {gcn.weight.shape}")
    print(f"  输出值范围: [{h.min():.4f}, {h.max():.4f}]")
    print(f"  含 NaN: {torch.isnan(h).any().item()}")

    assert h.shape == (N, D_out), f"输出形状错误: {h.shape} != ({N}, {D_out})"
    assert not torch.isnan(h).any(), "输出含 NaN"
    assert gcn.weight.shape == (D_in, D_out), f"权重形状错误: {gcn.weight.shape}"

    # 验证梯度可传播
    loss = h.sum()
    loss.backward()
    assert gcn.weight.grad is not None, "权重梯度为 None"
    assert not torch.isnan(gcn.weight.grad).any(), "权重梯度含 NaN"
    print(f"  梯度传播: 成功")

    print(f"  ✅ GCNConv 通过")


def test_vgae_model(device: str = "cpu"):
    """测试 2: VGAEModel 前向传播"""
    print("\n" + "=" * 60)
    print(f"[测试 2] VGAEModel 前向传播 (device={device})")
    print("=" * 60)

    N, D_snp = 200, 64
    D_hidden, D_z = 128, 32

    # 构造模拟数据
    x = torch.randn(N, D_snp, device=device)

    # 构造稀疏邻接
    indices = torch.randint(0, N, (2, 600), device=device)
    values = torch.rand(600, device=device) * 0.1
    adj_norm = torch.sparse_coo_tensor(
        indices, values, (N, N)).coalesce()

    # 创建模型
    model = VGAEModel(d_snp=D_snp, d_hidden=D_hidden, d_z=D_z)
    model = model.to(device)

    # 前向传播
    start = time.time()
    out = model(x, adj_norm)
    elapsed = time.time() - start

    print(f"  输入: {x.shape}")
    print(f"  输出:")
    print(f"    y_pred:      {out['y_pred'].shape}")
    print(f"    z:           {out['z'].shape}")
    print(f"    mu:          {out['mu'].shape}")
    print(f"    logvar:      {out['logvar'].shape}")
    print(f"    edge_scores: {out['edge_scores'].shape}")
    print(f"    edge_index:  {out['edge_index'].shape}")
    print(f"  前向传播耗时: {elapsed*1000:.1f}ms")

    # 形状验证
    assert out['y_pred'].shape == (N,), f"y_pred 形状错误: {out['y_pred'].shape}"
    assert out['z'].shape == (N, D_z), f"z 形状错误: {out['z'].shape}"
    assert out['mu'].shape == (N, D_z), f"mu 形状错误: {out['mu'].shape}"
    assert out['logvar'].shape == (N, D_z), f"logvar 形状错误: {out['logvar'].shape}"
    # edge_scores 应为 (E,) 一维向量，E 为边数
    assert out['edge_scores'].dim() == 1, f"edge_scores 应为一维: {out['edge_scores'].shape}"
    assert out['edge_index'].shape[0] == 2, f"edge_index 应为 (2, E): {out['edge_index'].shape}"

    # 数值检查
    assert not torch.isnan(out['y_pred']).any(), "y_pred 含 NaN"
    assert not torch.isnan(out['z']).any(), "z 含 NaN"
    assert not torch.isinf(out['edge_scores']).any(), "edge_scores 含 Inf"

    # edge_scores 应在 [0, 1] 范围（sigmoid 输出）
    assert out['edge_scores'].min() >= 0, "edge_scores 最小值 < 0"
    assert out['edge_scores'].max() <= 1, "edge_scores 最大值 > 1"

    print(f"  edge_scores 范围: [{out['edge_scores'].min():.4f}, {out['edge_scores'].max():.4f}]")
    print(f"  边数: {out['edge_scores'].shape[0]}")
    print(f"  ✅ VGAEModel 前向传播通过")

    return model


def test_vgae_loss(device: str = "cpu"):
    """测试 3: VGAE 联合损失函数"""
    print("\n" + "=" * 60)
    print(f"[测试 3] VGAE 联合损失函数 (device={device})")
    print("=" * 60)

    N, D_z = 200, 32

    # 模拟模型输出（需要 requires_grad 以验证反向传播）
    z = torch.randn(N, D_z, device=device, requires_grad=True)
    y_pred = (z.sum(dim=1) * 0.1).requires_grad_(True)  # 从 z 派生以保持计算图
    y_true = torch.randn(N, device=device)
    mu = torch.randn(N, D_z, device=device, requires_grad=True)
    logvar = torch.randn(N, D_z, device=device, requires_grad=True) * 0.1

    # 模拟边索引和 adj_raw
    edge_index = torch.randint(0, N, (2, 600), device=device)
    adj_indices = torch.randint(0, N, (2, 600), device=device)
    adj_raw = torch.sparse_coo_tensor(
        adj_indices, torch.ones(600, device=device), (N, N)).coalesce()

    # 计算损失
    total, pheno, edge, kl = vgae_loss(
        y_pred, y_true, z, edge_index, adj_raw, mu, logvar
    )

    print(f"  total_loss: {total.item():.4f}")
    print(f"  pheno_loss: {pheno.item():.4f}")
    print(f"  edge_loss:  {edge.item():.4f}")
    print(f"  kl_loss:    {kl.item():.4f}")

    # 基本检查
    assert not torch.isnan(total), "total_loss 含 NaN"
    assert not torch.isinf(total), "total_loss 含 Inf"
    assert pheno.item() >= 0, "pheno_loss 应为非负"
    assert edge.item() >= 0, "edge_loss 应为非负"

    # 验证反向传播
    total.backward()
    print(f"  反向传播: 成功")

    print(f"  ✅ 损失函数通过")


def test_end_to_end_real_data(device: str):
    """测试 4: 端到端 — M2 输出 → M3 VGAE 模型（真实数据）"""
    print("\n" + "=" * 60)
    print(f"[测试 4] 端到端: M2 → M3 (真实数据, device={device})")
    print("=" * 60)

    N = DEFAULT_N_SAMPLES
    D_snp = 64
    K = 30

    # --- M2: 构建图 ---
    print("  [M2] 加载 GRM...")
    grm = grm_reader(DEFAULT_GRM_BIN, N)

    print("  [M2] 构建 KNN 图...")
    builder = GraphBuilder()
    start = time.time()
    adj_norm, adj_raw = builder.knn_graph_builder(grm, k=K)
    print(f"  adj_norm: {adj_norm.shape}, nnz={adj_norm._nnz()}")
    print(f"  adj_raw:  {adj_raw.shape}, nnz={adj_raw._nnz()}")
    print(f"  KNN 图构建耗时: {time.time() - start:.3f}s")

    # 将邻接矩阵移到目标设备
    adj_norm = adj_norm.to(device)
    adj_raw = adj_raw.to(device)

    # 模拟节点特征（实际由 M1 产生）
    print(f"  [M2] 生成模拟节点特征 (N={N}, D_snp={D_snp})...")
    x = torch.randn(N, D_snp, device=device) * 0.1

    # --- M3: VGAE 模型 ---
    print(f"  [M3] 创建 VGAEModel (d_snp={D_snp}, d_hidden=128, d_z=32)...")
    model = VGAEModel(d_snp=D_snp, d_hidden=128, d_z=32)
    model = model.to(device)

    # 统计参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {n_params:,}")

    # 前向传播
    print(f"  [M3] 前向传播...")
    start = time.time()
    out = model(x, adj_norm)
    forward_time = time.time() - start

    print(f"  y_pred:      {out['y_pred'].shape}, 范围 [{out['y_pred'].min():.4f}, {out['y_pred'].max():.4f}]")
    print(f"  z:           {out['z'].shape}")
    print(f"  mu:          {out['mu'].shape}, 范围 [{out['mu'].min():.4f}, {out['mu'].max():.4f}]")
    print(f"  logvar:      {out['logvar'].shape}, 范围 [{out['logvar'].min():.4f}, {out['logvar'].max():.4f}]")
    print(f"  edge_scores: {out['edge_scores'].shape}")
    print(f"  前向传播耗时: {forward_time*1000:.1f}ms")

    # 模拟表型标签
    y_true = torch.randn(N, device=device)

    # 计算损失
    print(f"  [M3] 计算联合损失...")
    total, pheno, edge, kl = vgae_loss(
        out['y_pred'], y_true,
        out['z'], out['edge_index'], adj_raw,
        out['mu'], out['logvar']
    )

    print(f"  total_loss: {total.item():.4f}")
    print(f"  pheno_loss: {pheno.item():.4f}")
    print(f"  edge_loss:  {edge.item():.4f}")
    print(f"  kl_loss:    {kl.item():.4f}")

    # 反向传播
    print(f"  [M3] 反向传播...")
    start = time.time()
    total.backward()
    backward_time = time.time() - start
    print(f"  反向传播耗时: {backward_time*1000:.1f}ms")

    # 验证梯度
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  警告: {name} 梯度为 None")
        else:
            grad_norm = param.grad.norm().item()
            has_nan = torch.isnan(param.grad).any().item()
            if has_nan:
                print(f"  错误: {name} 梯度含 NaN")
            assert not has_nan, f"{name} 梯度含 NaN"

    print(f"  所有参数梯度有效")

    # 多轮训练稳定性测试
    print(f"  [M3] 多轮训练稳定性测试 (10 epochs)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        optimizer.zero_grad()
        out = model(x, adj_norm)
        total, pheno, edge, kl = vgae_loss(
            out['y_pred'], y_true,
            out['z'], out['edge_index'], adj_raw,
            out['mu'], out['logvar']
        )
        total.backward()
        optimizer.step()

        if epoch % 3 == 0:
            print(f"    epoch {epoch}: total={total.item():.4f}, "
                  f"pheno={pheno.item():.4f}, edge={edge.item():.4f}, kl={kl.item():.4f}")

    print(f"  ✅ 端到端测试通过 (M2 → M3)")


def main():
    device = auto_detect_device()

    print("=" * 60)
    print("M3: VGAE 模型集成测试")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"GRM 文件: {DEFAULT_GRM_BIN}")
    print(f"基因型文件: {DEFAULT_BED_FILE}")
    print(f"样本数: {DEFAULT_N_SAMPLES}, SNP 数: {DEFAULT_N_SNPS}")

    total_start = time.time()

    # 测试 1: GCNConv 基本功能
    test_gcn_conv()

    # 测试 2: VGAEModel 前向传播
    test_vgae_model(device)

    # 测试 3: 损失函数
    test_vgae_loss(device)

    # 测试 4: 端到端（真实数据）
    test_end_to_end_real_data(device)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"全部测试通过! 总耗时: {total_elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
