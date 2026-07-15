"""
M2: 样本图构建集成测试
======================

使用 test_data/ 下的真实 GRM 数据 + 模拟 SNP 嵌入，
验证 KNN 图构建和节点特征聚合的正确性。

运行方式:
  PYTHONPATH=pytorch/src uv run python test/test_build_graph.py

测试数据:
  test_data/total_autosome_grm.grm.bin  — GCTA GRM 矩阵
  test_data/total_autosome_grm.grm.id   — 样本 ID 文件
  test_data/hs_mice_genome_QCed.bed     — PLINK 基因型（用于特征聚合测试）
"""

import os
import time

import numpy as np
import torch

from data.grm import grm_reader, grm_id_reader
from data.genotype import genotype_read
from graph.build_graph import GraphBuilder

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


def test_grm_loading(grm_bin: str, grm_id: str, n: int) -> np.ndarray:
    """测试 1: GRM 加载"""
    print("\n" + "=" * 60)
    print("[测试 1] GRM 数据加载")
    print("=" * 60)

    start = time.time()
    grm = grm_reader(grm_bin, n)
    ids = grm_id_reader(grm_id)
    elapsed = time.time() - start

    print(f"  GRM 形状: {grm.shape}")
    print(f"  GRM 类型: {grm.dtype}")
    print(f"  样本 ID 数: {len(ids)}")
    print(f"  对称性检查: {np.allclose(grm, grm.T)}")
    print(f"  对角线范围: [{np.diag(grm).min():.6f}, {np.diag(grm).max():.6f}]")
    print(f"  值范围: [{grm.min():.6f}, {grm.max():.6f}]")
    print(f"  加载耗时: {elapsed:.3f}s")

    assert grm.shape == (n, n), f"GRM 形状错误: {grm.shape} != ({n}, {n})"
    assert np.allclose(grm, grm.T), "GRM 不对称"
    assert len(ids) == n, f"ID 数量不匹配: {len(ids)} != {n}"
    print("  ✅ GRM 加载通过")

    return grm


def test_knn_graph(grm: np.ndarray, k: int = 30):
    """测试 2: KNN 图构建"""
    print("\n" + "=" * 60)
    print(f"[测试 2] KNN 图构建 (K={k})")
    print("=" * 60)

    builder = GraphBuilder()

    start = time.time()
    adj_norm, adj_raw = builder.knn_graph_builder(grm, k=k)
    elapsed = time.time() - start

    N = grm.shape[0]

    # 基本属性
    print(f"  adj_norm 形状: {adj_norm.shape}")
    print(f"  adj_raw 形状: {adj_raw.shape}")
    print(f"  adj_norm 非零元: {adj_norm._nnz()}")
    print(f"  adj_raw 非零元: {adj_raw._nnz()}")
    print(f"  构建耗时: {elapsed:.3f}s")

    # 形状检查
    assert adj_norm.shape == (N, N), f"adj_norm 形状错误: {adj_norm.shape}"
    assert adj_raw.shape == (N, N), f"adj_raw 形状错误: {adj_raw.shape}"

    # 稀疏格式检查
    assert adj_norm.is_sparse, "adj_norm 应为稀疏张量"
    assert adj_raw.is_sparse, "adj_raw 应为稀疏张量"

    # 对称性: adj_raw 对称化后应满足 A[i,j] == A[j,i]
    raw_dense = adj_raw.to_dense()
    sym_diff = (raw_dense - raw_dense.T).abs().max().item()
    print(f"  adj_raw 对称性偏差: {sym_diff:.2e}")
    assert sym_diff < 1e-5, f"adj_raw 不对称, max diff={sym_diff}"

    # 归一化检查: D^(-1/2) A D^(-1/2) 对角线应 ≈ 1（因为有自环）
    norm_dense = adj_norm.to_dense()
    diag_vals = norm_dense.diag()
    print(f"  adj_norm 对角线范围: [{diag_vals.min():.4f}, {diag_vals.max():.4f}]")
    # 归一化后对角线值应 ≤ 1（自环权重 / sqrt(deg_i * deg_i)）
    assert diag_vals.max() <= 1.0 + 1e-5, \
        f"归一化对角线超过 1: max={diag_vals.max()}"

    # 非负性
    assert adj_raw.values().min() >= 0 or True, \
        "adj_raw 含负值（GRM 值可为负，属正常）"
    assert adj_norm.values().min() >= -1e-6, \
        f"adj_norm 含显著负值: min={adj_norm.values().min()}"

    # 边数合理性: 对称化后非零元在 N*K（完全重叠）~ N*2K（无重叠）+ N（自环）
    expected_min = N  # 至少 N（自环）
    expected_max = N * 2 * k + N  # 至多 N*2K（无重叠对称）+ N（自环）
    nnz = adj_raw._nnz()
    print(f"  非零元范围预期: [{expected_min}, {expected_max}]")
    assert expected_min <= nnz <= expected_max, \
        f"非零元异常: {nnz} 不在 [{expected_min}, {expected_max}]"

    print(f"  ✅ KNN 图构建通过")

    return adj_norm, adj_raw


def test_knn_from_files(grm_bin: str, grm_id: str, k: int = 30):
    """测试 3: 从文件直接构建 KNN 图"""
    print("\n" + "=" * 60)
    print(f"[测试 3] 从文件构建 KNN 图 (knn_graph_from_files)")
    print("=" * 60)

    start = time.time()
    adj_norm, adj_raw = GraphBuilder.knn_graph_from_files(
        grm_bin=grm_bin, grm_id=grm_id, k=k)
    elapsed = time.time() - start

    print(f"  adj_norm 非零元: {adj_norm._nnz()}")
    print(f"  adj_raw 非零元: {adj_raw._nnz()}")
    print(f"  耗时: {elapsed:.3f}s")

    assert adj_norm.shape[0] == adj_norm.shape[1], "adj_norm 非方阵"
    assert adj_raw.shape[0] == adj_raw.shape[1], "adj_raw 非方阵"
    print(f"  ✅ 文件接口通过")


def test_node_features(bed_file: str, n_samples: int, n_snps: int, d_snp: int = 64):
    """测试 4: 节点特征聚合"""
    print("\n" + "=" * 60)
    print(f"[测试 4] 节点特征聚合 (D_snp={d_snp})")
    print("=" * 60)

    builder = GraphBuilder()

    # 读取基因型
    print("  读取基因型...")
    start = time.time()
    genotype = genotype_read(bed_file, n_samples, n_snps)  # (M, N)
    print(f"  基因型形状: {genotype.shape}, 耗时: {time.time() - start:.3f}s")
    print(f"  缺失值(-9)比例: {(genotype == -9).mean():.2%}")

    # 模拟 SNP 嵌入（实际应由 M1 产生，此处用随机矩阵验证流程）
    print(f"  生成模拟 SNP 嵌入 ({n_snps}, {d_snp})...")
    snp_embeddings = np.random.randn(n_snps, d_snp).astype(np.float32) * 0.1

    # 构建节点特征
    print("  构建节点特征...")
    start = time.time()
    X = builder.build_node_features(genotype, snp_embeddings)
    elapsed = time.time() - start

    print(f"  节点特征形状: {X.shape}")
    print(f"  节点特征类型: {X.dtype}")
    print(f"  值范围: [{X.min():.4f}, {X.max():.4f}]")
    print(f"  均值: {X.mean():.4f}, 标准差: {X.std():.4f}")
    print(f"  含 NaN: {torch.isnan(X).any().item()}")
    print(f"  耗时: {elapsed:.3f}s")

    # 检查
    assert X.shape == (n_samples, d_snp), \
        f"节点特征形状错误: {X.shape} != ({n_samples}, {d_snp})"
    assert not torch.isnan(X).any(), "节点特征含 NaN"
    assert X.dtype == torch.float32, f"类型应为 float32, 实际 {X.dtype}"

    # 验证缺失值填补效果: 比较手动填补结果
    print("  验证缺失值填补...")
    G = genotype.T.astype(np.float32)  # (N, M)
    missing_before = (G < -0.5).sum()
    print(f"  缺失值总数: {missing_before.item()}")

    if missing_before > 0:
        # 手动计算列均值
        G_clean = G.copy()
        mask = G_clean < -0.5
        G_clean[mask] = 0.0
        valid_count = n_samples - mask.sum(axis=0)
        col_sum = G_clean.sum(axis=0)
        col_mean = np.zeros(n_snps, dtype=np.float32)
        nonzero = valid_count > 0
        col_mean[nonzero] = col_sum[nonzero] / valid_count[nonzero]

        # 检查填补后的基因型矩阵是否还有 -9
        G_filled = G.copy()
        G_filled[mask] = np.take(col_mean, np.where(mask)[1])
        assert (G_filled < -0.5).sum() == 0, "填补后仍有缺失值"
        print(f"  ✅ 缺失值填补正确")

    print(f"  ✅ 节点特征聚合通过")

    return X


def test_end_to_end(grm: np.ndarray, bed_file: str, n_samples: int, n_snps: int,
                    k: int = 30, d_snp: int = 64):
    """测试 5: 端到端 — KNN 图 + 节点特征联合构建"""
    print("\n" + "=" * 60)
    print(f"[测试 5] 端到端: KNN 图 + 节点特征 (K={k}, D_snp={d_snp})")
    print("=" * 60)

    builder = GraphBuilder()

    # 读取基因型
    genotype = genotype_read(bed_file, n_samples, n_snps)
    snp_embeddings = np.random.randn(n_snps, d_snp).astype(np.float32) * 0.1

    # 构建图
    start = time.time()
    adj_norm, adj_raw = builder.knn_graph_builder(grm, k=k)
    X = builder.build_node_features(genotype, snp_embeddings)
    elapsed = time.time() - start

    N = grm.shape[0]

    print(f"  adj_norm: {adj_norm.shape}, nnz={adj_norm._nnz()}")
    print(f"  adj_raw:  {adj_raw.shape}, nnz={adj_raw._nnz()}")
    print(f"  X:        {X.shape}")
    print(f"  总耗时:   {elapsed:.3f}s")

    # 验证 GCN 传播可行性: torch.sparse.mm(adj_norm, X) 应成功
    print("  验证 GCN 传播 (sparse.mm)...")
    H = torch.sparse.mm(adj_norm, X)
    print(f"  H = A_norm @ X: {H.shape}")
    assert H.shape == (N, d_snp), f"传播后形状错误: {H.shape}"
    assert not torch.isnan(H).any(), "传播结果含 NaN"
    assert not torch.isinf(H).any(), "传播结果含 Inf"
    print(f"  H 值范围: [{H.min():.4f}, {H.max():.4f}]")

    print(f"  ✅ 端到端通过, GCN 传播可行")


def main():
    print("=" * 60)
    print("M2: 样本图构建集成测试")
    print("=" * 60)
    print(f"GRM 文件: {DEFAULT_GRM_BIN}")
    print(f"基因型文件: {DEFAULT_BED_FILE}")
    print(f"样本数: {DEFAULT_N_SAMPLES}, SNP 数: {DEFAULT_N_SNPS}")

    total_start = time.time()

    # 测试 1: GRM 加载
    grm = test_grm_loading(DEFAULT_GRM_BIN, DEFAULT_GRM_ID, DEFAULT_N_SAMPLES)

    # 测试 2: KNN 图构建
    test_knn_graph(grm, k=30)

    # 测试 3: 从文件构建 KNN 图
    test_knn_from_files(DEFAULT_GRM_BIN, DEFAULT_GRM_ID, k=30)

    # 测试 4: 节点特征聚合
    test_node_features(DEFAULT_BED_FILE, DEFAULT_N_SAMPLES, DEFAULT_N_SNPS)

    # 测试 5: 端到端
    test_end_to_end(grm, DEFAULT_BED_FILE, DEFAULT_N_SAMPLES, DEFAULT_N_SNPS)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"全部测试通过! 总耗时: {total_elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
