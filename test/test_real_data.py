"""
SNP-VAE 真实数据集成测试
========================

使用 test_data/ 下的真实基因型数据运行完整 SNP-VAE 训练流程。

运行方式:
  PYTHONPATH=pytorch/src uv run python test/test_real_data.py

数据要求:
  test_data/hs_mice_genome_QCed.bed  — PLINK 基因型文件
  样本数: 2002, SNP 数: 9728

注意:
  这是集成测试，运行时间较长（取决于设备和 epochs）。
  可通过命令行参数调整训练规模。

作者: Xiang Yang
邮箱: Georicl@outlook.com
创建时间: 2026-07-11
"""

import argparse
import os
import time

import numpy as np
import torch

from data.genotype import genotype_read
from pretrain.preprocess import preprocess_genotype, inverse_standardize
from pretrain.vae import SNPVAE, vae_loss
from pretrain.extract_embeddings import extract_embeddings
from pretrain.train import train_snp_vae

# ============================================================================
# 数据路径配置（通过命令行参数覆盖）
# ============================================================================

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 测试结果输出目录
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")

# 默认数据路径
DEFAULT_BED_FILE = os.path.join(
    PROJECT_ROOT, "test_data", "hs_mice_genome_QCed.bed")
DEFAULT_N_SAMPLES = 2002
DEFAULT_N_SNPS = 9728


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="SNP-VAE 真实数据集成测试")

    # 数据参数
    parser.add_argument("--bed-file", type=str, default=DEFAULT_BED_FILE,
                        help="PLINK .bed 文件路径")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES,
                        help="样本数")
    parser.add_argument("--n-snps", type=int, default=DEFAULT_N_SNPS,
                        help="SNP 数")

    # 训练参数
    parser.add_argument("--d-snp", type=int, default=64,
                        help="SNP 嵌入维度")
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮数（默认 50，完整训练建议 200+）")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="batch 大小")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="学习率")
    parser.add_argument("--beta", type=float, default=0.001,
                        help="KL 散度权重")
    parser.add_argument("--patience", type=int, default=50,
                        help="早停耐心值")
    parser.add_argument("--kl-warmup", type=int, default=150,
                        help="KL 退火预热轮数")
    parser.add_argument("--hidden-dim", type=int, default=1024,
                        help="隐藏层宽度")

    # 设备
    parser.add_argument("--device", type=str, default=None,
                        help="设备 (cpu/mps)，默认自动检测")

    # 输出
    parser.add_argument("--save-embeddings", type=str, default=None,
                        help="保存嵌入矩阵的路径 (.npy)")
    parser.add_argument("--save-model", type=str, default=None,
                        help="保存模型权重的路径 (.pt)")
    parser.add_argument("--no-eval", action="store_true",
                        help="跳过评价模块")

    return parser.parse_args()


def auto_detect_device() -> str:
    """自动检测设备"""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate_model(
    model: SNPVAE,
    genotype: np.ndarray,
    snp_embeddings: np.ndarray,
    device: str,
) -> dict:
    """
    评价 SNP-VAE 训练结果。

    评价维度:
      1. 标准化空间重建质量: MSE（标准化后）
      2. 原始尺度重建质量: 逆变换后 MSE、逐 SNP 相关性
      3. 嵌入空间: 均值/标准差、KL 散度、维度利用率
    """
    # --- 数据预处理 ---
    X, snp_mean, snp_std = preprocess_genotype(genotype)  # (M, N)
    M, N = X.shape

    model.eval()
    with torch.no_grad():
        # --- 重建 ---
        # 分批前向传播，避免内存溢出
        all_recon = []
        all_mu = []
        all_logvar = []
        batch_size = 1024

        for start in range(0, M, batch_size):
            end = min(start + batch_size, M)
            x_batch = X[start:end].to(device)
            x_recon, mu, logvar = model(x_batch)
            all_recon.append(x_recon.cpu())
            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())

        X_recon = torch.cat(all_recon, dim=0)  # (M, N)
        mu_all = torch.cat(all_mu, dim=0)      # (M, D_snp)
        logvar_all = torch.cat(all_logvar, dim=0)  # (M, D_snp)

    # ========================================================================
    # 1. 标准化空间重建质量
    # ========================================================================
    print("\n  [评价] 标准化空间重建质量")

    global_mse_std = torch.nn.functional.mse_loss(X_recon, X).item()
    print(f"    标准化空间 MSE: {global_mse_std:.6f}")

    # ========================================================================
    # 2. 原始尺度重建质量（逆变换后对比）
    # ========================================================================
    print("\n  [评价] 原始尺度重建质量")

    # 将标准化空间的重建值逆变换回原始尺度
    X_recon_orig = inverse_standardize(X_recon, snp_mean, snp_std)  # (M, N)
    X_orig = inverse_standardize(X, snp_mean, snp_std)              # (M, N)

    # 原始尺度 MSE
    global_mse_orig = np.mean((X_recon_orig - X_orig) ** 2)
    print(f"    原始尺度 MSE: {global_mse_orig:.6f}")

    # 逐 SNP 相关性（原始尺度）
    snp_correlations = []
    for i in range(M):
        orig = X_orig[i]
        recon = X_recon_orig[i]
        if orig.std() < 1e-8 or recon.std() < 1e-8:
            continue
        corr = np.corrcoef(orig, recon)[0, 1]
        snp_correlations.append(corr)

    snp_correlations = np.array(snp_correlations)
    print(f"    逐SNP相关性: mean={snp_correlations.mean():.4f}, "
          f"min={snp_correlations.min():.4f}, max={snp_correlations.max():.4f}")

    # 低重建质量 SNP 统计
    low_quality_mask = snp_correlations < 0.8
    n_low_quality = low_quality_mask.sum()
    print(f"    低重建质量SNP数(<0.8): {n_low_quality}/{len(snp_correlations)}")

    # 重建值分布
    print(f"    重建值范围: [{X_recon_orig.min():.4f}, {X_recon_orig.max():.4f}]")
    print(f"    重建值均值: {X_recon_orig.mean():.4f}, 标准差: {X_recon_orig.std():.4f}")

    # ========================================================================
    # 3. 嵌入空间质量
    # ========================================================================
    print("\n  [评价] 嵌入空间")

    # 嵌入统计
    emb_mean = snp_embeddings.mean()
    emb_std = snp_embeddings.std()
    print(f"    嵌入均值: {emb_mean:.4f}, 标准差: {emb_std:.4f}")

    # 嵌入值范围
    print(
        f"    嵌入范围: [{snp_embeddings.min():.4f}, {snp_embeddings.max():.4f}]")

    # KL 散度（全局）
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar_all - mu_all.pow(2) - logvar_all.exp(), dim=1)
    ).item()
    print(f"    KL散度: {kl_loss:.4f}")

    # 维度利用率（各维度的方差）
    dim_variance = snp_embeddings.var(axis=0)  # (D_snp,)
    active_dims = (dim_variance > 0.01).sum()
    print(f"    活跃维度数(方差>0.01): {active_dims}/{snp_embeddings.shape[1]}")
    print(f"    维度方差: mean={dim_variance.mean():.4f}, "
          f"min={dim_variance.min():.4f}, max={dim_variance.max():.4f}")

    # NaN 检查
    has_nan = np.isnan(snp_embeddings).any()
    print(f"    嵌入含NaN: {has_nan}")

    return {
        "global_mse_standardized": global_mse_std,
        "global_mse_original": global_mse_orig,
        "snp_correlation_mean": snp_correlations.mean(),
        "snp_correlation_min": snp_correlations.min(),
        "n_low_quality_snps": int(n_low_quality),
        "embedding_mean": emb_mean,
        "embedding_std": emb_std,
        "kl_divergence": kl_loss,
        "active_dims": int(active_dims),
        "has_nan": has_nan,
    }


def main():
    args = parse_args()

    # 设备选择
    device = args.device if args.device else auto_detect_device()
    print(f"=" * 60)
    print(f"SNP-VAE 真实数据集成测试")
    print(f"=" * 60)
    print(f"设备: {device}")
    print(f"数据文件: {args.bed_file}")
    print(f"数据规模: {args.n_snps} SNPs × {args.n_samples} samples")
    print(f"嵌入维度: {args.d_snp}")
    print(f"训练参数: epochs={args.epochs}, batch_size={args.batch_size}, "
          f"lr={args.lr}, β={args.beta}, patience={args.patience}")
    print(f"=" * 60)

    # 1. 读取基因型数据
    print("\n[1/3] 读取基因型数据...")
    start_time = time.time()
    genotype = genotype_read(args.bed_file, args.n_samples, args.n_snps)
    read_time = time.time() - start_time
    print(f"  基因型矩阵形状: {genotype.shape}")
    print(f"  值范围: [{genotype.min()}, {genotype.max()}]")
    print(f"  缺失值(-9)比例: {(genotype == -9).mean():.2%}")
    print(f"  读取耗时: {read_time:.1f}s")

    # 2. 训练 SNP-VAE
    print("\n[2/3] 训练 SNP-VAE...")
    start_time = time.time()
    model, snp_embeddings = train_snp_vae(
        genotype=genotype,
        d_snp=args.d_snp,
        beta=args.beta,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        device=device,
        progress=True,
        kl_warmup_epochs=args.kl_warmup,
        hidden_dim=args.hidden_dim,
    )
    train_time = time.time() - start_time
    print(f"  嵌入矩阵形状: {snp_embeddings.shape}")
    print(f"  嵌入值范围: [{snp_embeddings.min():.4f}, {snp_embeddings.max():.4f}]")
    print(
        f"  嵌入均值: {snp_embeddings.mean():.4f}, 标准差: {snp_embeddings.std():.4f}")
    print(f"  训练耗时: {train_time:.1f}s")

    # 3. 保存结果
    print("\n[3/3] 保存结果...")
    os.makedirs(RESULT_DIR, exist_ok=True)

    if args.save_embeddings:
        np.save(args.save_embeddings, snp_embeddings)
        print(f"  嵌入矩阵已保存: {args.save_embeddings}")
    else:
        save_path = os.path.join(RESULT_DIR, "snp_embeddings.npy")
        np.save(save_path, snp_embeddings)
        print(f"  嵌入矩阵已保存: {save_path}")

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        print(f"  模型权重已保存: {args.save_model}")

    # 4. 评价模块
    if not args.no_eval:
        print("\n[4/4] 模型评价...")
        evaluate_model(model, genotype, snp_embeddings, device)
    else:
        print("\n[4/4] 跳过评价模块")

    # 5. 结果摘要
    print("\n" + "=" * 60)
    print(f"测试完成!")
    print(f"  总耗时: {read_time + train_time:.1f}s")
    print(f"  嵌入矩阵: {snp_embeddings.shape}")
    print(f"  无 NaN: {not np.isnan(snp_embeddings).any()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
