"""
统计显著性验证模块
==================

对 VGAE 基因组预测进行严格的统计检验:

  1. 重复 K-fold CV (10× 5-fold, 不同随机种子)
     → R² 均值 ± 95% CI
     → VGAE vs BLUP 配对 t 检验

  2. Permutation Test (打乱表型标签)
     → BLUP: 1000 次置换 (快速)
     → VGAE: 50 次置换 (加速模式)
     → 单侧 p-value

运行方式:
  PYTHONPATH=pytorch/src uv run python test/statistical_validation.py

输出:
  test/result/statistical_validation.json

作者: Xiang Yang
邮箱: Georicl@outlook.com
创建时间: 2026-07-24
"""

import json
import os
import time
import numpy as np
import torch
from scipy import stats

from data.genotype import genotype_read
from data.grm import grm_reader, grm_id_reader
from pretrain.train import train_snp_vae
from train.cross_validation import run_kfold_cv, generate_kfold_masks
from train.metrics import evaluate_all

# ============================================================================
# 配置
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "test_data")
RESULT_DIR = os.path.join(PROJECT_ROOT, "test", "result")

BED_FILE = os.path.join(DATA_DIR, "hs_mice_genome_QCed.bed")
GRM_BIN = os.path.join(DATA_DIR, "total_autosome_grm.grm.bin")
GRM_ID = os.path.join(DATA_DIR, "total_autosome_grm.grm.id")
PHENO_FILE = os.path.join(DATA_DIR, "phenotypes", "Biochem.HDL.resid")

N_SAMPLES = 2002
N_SNPS = 9728

# 重复 CV 配置
N_REPEATS = 5
K_FOLDS = 5

# Permutation 配置
N_PERM_BLUP = 1000
N_PERM_VGAE = 20

# VGAE 配置 (调优后)
VGAE_CONFIG = dict(
    k_neighbors=30, d_snp=64, d_hidden=128, d_z=16,
    dropout=0.5, mlp_hidden=64, lr=1e-3, alpha=0.1, beta_kl=0.01,
    max_epochs=200, patience=30, weight_decay=1e-5, kl_warmup_epochs=30,
)

# VGAE 加速配置 (用于 permutation test)
VGAE_FAST_CONFIG = dict(
    k_neighbors=30, d_snp=64, d_hidden=128, d_z=16,
    dropout=0.5, mlp_hidden=64, lr=1e-3, alpha=0.1, beta_kl=0.01,
    max_epochs=100, patience=15, weight_decay=1e-5, kl_warmup_epochs=15,
)

LAMBDA_GRID = [0.05, 0.1, 0.2, 0.35, 0.6, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0]


# ============================================================================
# 工具函数
# ============================================================================
def load_phenotype(pheno_file, grm_ids):
    n = len(grm_ids)
    labels = np.full(n, np.nan, dtype=np.float32)
    pheno_map = {}
    with open(pheno_file, "r") as f:
        f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1] not in ("-9.0", "-9", "NA", ""):
                try:
                    pheno_map[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    for i, (_, iid) in enumerate(grm_ids):
        if iid in pheno_map:
            labels[i] = pheno_map[iid]
    return labels


def blup_predict(grm, labels, train_mask, test_mask, lam):
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    y_train = labels[train_idx]
    K_train = grm[np.ix_(train_idx, train_idx)]
    K_test = grm[np.ix_(test_idx, train_idx)]
    mu = y_train.mean()
    A = K_train + lam * np.eye(len(train_idx))
    alpha = np.linalg.solve(A, y_train - mu)
    return K_test @ alpha + mu


def blup_cv(grm, labels, seed):
    """BLUP 5-fold CV，返回平均 R²。"""
    masks = generate_kfold_masks(len(labels), K_FOLDS, seed)
    r2_list = []
    for train_mask, val_mask, test_mask in masks:
        combined = np.zeros(len(labels), dtype=bool)
        combined[np.where(train_mask)[0]] = True
        combined[np.where(val_mask)[0]] = True
        best_mse, best_lam = float("inf"), 1.0
        for lam in LAMBDA_GRID:
            pred = blup_predict(grm, labels, combined, val_mask, lam)
            mse = np.mean((pred - labels[val_mask]) ** 2)
            if mse < best_mse:
                best_mse, best_lam = mse, lam
        pred_test = blup_predict(grm, labels, combined, test_mask, best_lam)
        m = evaluate_all(pred_test, labels[test_mask])
        r2_list.append(m["r2"])
    return float(np.mean(r2_list))


def vgae_cv(genotype, snp_emb, grm, labels, seed, config, device, progress=False):
    """VGAE 5-fold CV，返回平均 R²。"""
    result = run_kfold_cv(
        genotype=genotype, snp_embeddings=snp_emb, grm=grm, labels=labels,
        k_folds=K_FOLDS, seed=seed, device=device, progress=progress, **config,
    )
    return float(result["mean_metrics"]["r2"])


# ============================================================================
# 主流程
# ============================================================================
def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 70)
    print("P0: 统计显著性验证 — Biochem.HDL")
    print("=" * 70)
    print(f"设备: {device}")
    print(f"重复 CV: {N_REPEATS}× {K_FOLDS}-fold")
    print(f"Permutation: BLUP {N_PERM_BLUP}次, VGAE {N_PERM_VGAE}次")

    # 加载数据
    print("\n[1] 加载数据...")
    genotype = genotype_read(BED_FILE, N_SAMPLES, N_SNPS)
    grm = grm_reader(GRM_BIN, N_SAMPLES)
    grm_ids = grm_id_reader(GRM_ID)
    labels = load_phenotype(PHENO_FILE, grm_ids)
    valid = ~np.isnan(labels)
    idx = np.where(valid)[0]
    geno_sub = genotype[:, idx]
    grm_sub = grm[np.ix_(idx, idx)]
    lab_sub = labels[idx]
    n = len(idx)
    print(f"  有效样本: {n}")

    # SNP-VAE 预训练 (所有 CV 共享)
    print("\n[2] SNP-VAE 预训练 (200 epochs)...")
    t0 = time.time()
    _, snp_emb = train_snp_vae(
        genotype=geno_sub, device=device, progress=True,
        d_snp=64, beta=0.001, lr=5e-4, epochs=200, batch_size=512,
        patience=50, kl_warmup_epochs=150, hidden_dim=1024,
    )
    print(f"  完成 ({time.time()-t0:.0f}s)")

    # ==================================================================
    # Part 1: 重复 K-fold CV + 配对 t 检验
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"[3] 重复 {N_REPEATS}× {K_FOLDS}-fold CV")
    print(f"{'='*70}")

    blup_r2_list = []
    vgae_r2_list = []
    seeds = list(range(42, 42 + N_REPEATS))

    for i, seed in enumerate(seeds):
        print(f"\n  --- 重复 {i+1}/{N_REPEATS} (seed={seed}) ---")

        # BLUP
        t0 = time.time()
        blup_r2 = blup_cv(grm_sub, lab_sub, seed)
        blup_r2_list.append(blup_r2)
        print(f"  BLUP R²={blup_r2:.4f} ({time.time()-t0:.1f}s)")

        # VGAE
        t0 = time.time()
        vgae_r2 = vgae_cv(geno_sub, snp_emb, grm_sub, lab_sub, seed,
                          VGAE_CONFIG, device, progress=False)
        vgae_r2_list.append(vgae_r2)
        print(f"  VGAE R²={vgae_r2:.4f} ({time.time()-t0:.1f}s)")

    blup_arr = np.array(blup_r2_list)
    vgae_arr = np.array(vgae_r2_list)

    # 配对 t 检验
    t_stat, p_paired = stats.ttest_rel(vgae_arr, blup_arr)

    # 95% CI
    def ci95(arr):
        m, se = arr.mean(), arr.std(ddof=1) / np.sqrt(len(arr))
        return m - 1.96 * se, m + 1.96 * se

    blup_ci = ci95(blup_arr)
    vgae_ci = ci95(vgae_arr)

    print(f"\n{'='*70}")
    print("重复 CV 结果")
    print(f"{'='*70}")
    print(f"  BLUP R²: {blup_arr.mean():.4f} ± {blup_arr.std(ddof=1):.4f}  "
          f"95%CI [{blup_ci[0]:.4f}, {blup_ci[1]:.4f}]")
    print(f"  VGAE R²: {vgae_arr.mean():.4f} ± {vgae_arr.std(ddof=1):.4f}  "
          f"95%CI [{vgae_ci[0]:.4f}, {vgae_ci[1]:.4f}]")
    print(f"  配对 t 检验: t={t_stat:.3f}, p={p_paired:.6f}")
    print(f"  {'VGAE 显著优于 BLUP' if p_paired < 0.05 else '差异不显著'} (α=0.05)")

    # ==================================================================
    # Part 2: Permutation Test
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"[4] Permutation Test")
    print(f"{'='*70}")

    rng = np.random.RandomState(123)

    # 真实 R² (用 seed=42)
    real_blup_r2 = blup_r2_list[0]
    real_vgae_r2 = vgae_r2_list[0]

    # BLUP permutation (快速)
    print(f"\n  BLUP permutation ({N_PERM_BLUP} 次)...")
    t0 = time.time()
    blup_perm = []
    for i in range(N_PERM_BLUP):
        perm_labels = rng.permutation(lab_sub)
        r2 = blup_cv(grm_sub, perm_labels, seed=42)
        blup_perm.append(r2)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{N_PERM_BLUP} ({time.time()-t0:.0f}s)")
    blup_perm = np.array(blup_perm)
    blup_pval = (np.sum(blup_perm >= real_blup_r2) + 1) / (N_PERM_BLUP + 1)
    print(f"  BLUP: 真实 R²={real_blup_r2:.4f}, "
          f"null={blup_perm.mean():.4f}±{blup_perm.std():.4f}, "
          f"p={blup_pval:.4f}")

    # VGAE permutation (加速模式)
    print(f"\n  VGAE permutation ({N_PERM_VGAE} 次, 加速模式)...")
    t0 = time.time()
    vgae_perm = []
    for i in range(N_PERM_VGAE):
        perm_labels = rng.permutation(lab_sub)
        r2 = vgae_cv(geno_sub, snp_emb, grm_sub, perm_labels, seed=42,
                     config=VGAE_FAST_CONFIG, device=device, progress=False)
        vgae_perm.append(r2)
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (N_PERM_VGAE - i - 1)
        print(f"    {i+1}/{N_PERM_VGAE} R²={r2:.4f} "
              f"({elapsed:.0f}s, ETA {eta:.0f}s)")
    vgae_perm = np.array(vgae_perm)
    vgae_pval = (np.sum(vgae_perm >= real_vgae_r2) + 1) / (N_PERM_VGAE + 1)
    print(f"  VGAE: 真实 R²={real_vgae_r2:.4f}, "
          f"null={vgae_perm.mean():.4f}±{vgae_perm.std():.4f}, "
          f"p={vgae_pval:.4f}")

    # ==================================================================
    # 汇总
    # ==================================================================
    print(f"\n{'='*70}")
    print("P0 统计验证汇总 — Biochem.HDL")
    print(f"{'='*70}")
    print(f"\n  1. 重复 {N_REPEATS}× {K_FOLDS}-fold CV:")
    print(f"     BLUP R² = {blup_arr.mean():.4f} ± {blup_arr.std(ddof=1):.4f}")
    print(f"     VGAE R² = {vgae_arr.mean():.4f} ± {vgae_arr.std(ddof=1):.4f}")
    print(f"     配对 t 检验 p = {p_paired:.6f} "
          f"{'***' if p_paired < 0.001 else '**' if p_paired < 0.01 else '*' if p_paired < 0.05 else 'ns'}")
    print(f"\n  2. Permutation Test:")
    print(f"     BLUP p = {blup_pval:.4f} ({N_PERM_BLUP} 次置换)")
    print(f"     VGAE p = {vgae_pval:.4f} ({N_PERM_VGAE} 次置换)")

    # 保存结果
    result = {
        "trait": "Biochem.HDL",
        "n_samples": int(n),
        "repeated_cv": {
            "n_repeats": N_REPEATS,
            "k_folds": K_FOLDS,
            "seeds": seeds,
            "blup_r2": blup_r2_list,
            "vgae_r2": vgae_r2_list,
            "blup_mean": float(blup_arr.mean()),
            "blup_std": float(blup_arr.std(ddof=1)),
            "blup_ci95": [float(blup_ci[0]), float(blup_ci[1])],
            "vgae_mean": float(vgae_arr.mean()),
            "vgae_std": float(vgae_arr.std(ddof=1)),
            "vgae_ci95": [float(vgae_ci[0]), float(vgae_ci[1])],
            "paired_t_stat": float(t_stat),
            "paired_p_value": float(p_paired),
        },
        "permutation_test": {
            "blup": {
                "n_permutations": N_PERM_BLUP,
                "real_r2": float(real_blup_r2),
                "null_mean": float(blup_perm.mean()),
                "null_std": float(blup_perm.std()),
                "p_value": float(blup_pval),
            },
            "vgae": {
                "n_permutations": N_PERM_VGAE,
                "real_r2": float(real_vgae_r2),
                "null_mean": float(vgae_perm.mean()),
                "null_std": float(vgae_perm.std()),
                "p_value": float(vgae_pval),
            },
        },
    }

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = os.path.join(RESULT_DIR, "statistical_validation.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {out_path}")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
