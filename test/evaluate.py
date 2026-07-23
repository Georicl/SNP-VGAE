"""
VGAE 基因组预测 — 统一评测框架
================================
对多个高遗传力性状执行标准化评测流程:
  1. SNP-VAE 预训练 (200 epochs, 默认参数)
  2. BLUP 基线 (GRM Ridge, 5-fold CV)
  3. VGAE 模型 (调优参数, 5-fold CV)
  4. 标准评测指标: MSE, MAE, Pearson R, R²
  5. 逐性状结果 + 总体汇总

运行方式:
  PYTHONPATH=pytorch/src uv run python test/evaluate.py

输出:
  test/result/<trait>/metrics.json   — 逐性状指标
  test/result/<trait>/snp_embeddings.npy — SNP 嵌入
  test/result/summary.json           — 总体汇总
"""

import json
import os
import time
import numpy as np
import torch

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

N_SAMPLES = 2002
N_SNPS = 9728

# 评测性状列表 (名称, h²估计, 文件路径)
TRAITS = [
    ("Biochem.HDL",   0.63, os.path.join(DATA_DIR, "phenotypes", "Biochem.HDL.resid")),
    ("Biochem.ALP",   0.55, os.path.join(DATA_DIR, "phenotypes", "Biochem.ALP.resid")),
    ("End.Weight",    0.42, os.path.join(DATA_DIR, "phenotypes", "End.Weight.resid")),
    ("Haem.MCV",      0.46, os.path.join(DATA_DIR, "phenotypes", "Haem.MCV.resid")),
]

# SNP-VAE 默认配置
SNP_VAE_CONFIG = dict(
    d_snp=64, beta=0.001, lr=5e-4,
    epochs=200, batch_size=512, patience=50,
    kl_warmup_epochs=150, hidden_dim=1024,
)

# VGAE 调优配置 (α=0.1, d_z=16 来自坐标下降搜索)
VGAE_CONFIG = dict(
    k_neighbors=30,
    d_snp=64, d_hidden=128, d_z=16,
    dropout=0.5, mlp_hidden=64,
    lr=1e-3, alpha=0.1, beta_kl=0.01,
    max_epochs=200, patience=30,
    weight_decay=1e-5, kl_warmup_epochs=30,
)

# BLUP λ 网格
LAMBDA_GRID = [0.05, 0.1, 0.2, 0.35, 0.6, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0]


# ============================================================================
# 工具函数
# ============================================================================
def load_phenotype(pheno_file: str, grm_ids: list) -> np.ndarray:
    """通过 IID 对齐表型，支持 .pheno 和 .resid 格式。"""
    n = len(grm_ids)
    labels = np.full(n, np.nan, dtype=np.float32)
    pheno_map = {}

    with open(pheno_file, "r") as f:
        first_line = f.readline().strip()
        is_resid = first_line.startswith("SUBJECT")
        if not is_resid:
            parts = first_line.split("\t")
            if len(parts) >= 3 and parts[2] not in ("-9.0", "-9", "NA", ""):
                pheno_map[parts[1]] = float(parts[2])
        for line in f:
            if is_resid:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[1] not in ("-9.0", "-9", "NA", ""):
                    try:
                        pheno_map[parts[0]] = float(parts[1])
                    except ValueError:
                        pass
            else:
                parts = line.strip().split("\t")
                if len(parts) >= 3 and parts[2] not in ("-9.0", "-9", "NA", ""):
                    pheno_map[parts[1]] = float(parts[2])

    matched = 0
    for i, (_, iid) in enumerate(grm_ids):
        if iid in pheno_map:
            labels[i] = pheno_map[iid]
            matched += 1
    return labels, matched


def blup_predict(grm, labels, train_mask, test_mask, lam):
    """BLUP 预测 (Ridge on GRM)。"""
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    y_train = labels[train_idx]
    K_train = grm[np.ix_(train_idx, train_idx)]
    K_test = grm[np.ix_(test_idx, train_idx)]
    mu = y_train.mean()
    y_c = y_train - mu
    A = K_train + lam * np.eye(len(train_idx))
    alpha = np.linalg.solve(A, y_c)
    return K_test @ alpha + mu


def blup_cv(grm, labels, masks):
    """BLUP 5-fold CV，内层选 λ。"""
    fold_metrics = []
    best_lambdas = []
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
        best_lambdas.append(best_lam)

        pred_test = blup_predict(grm, labels, combined, test_mask, best_lam)
        fold_metrics.append(evaluate_all(pred_test, labels[test_mask]))

    return fold_metrics, best_lambdas


def evaluate_trait(trait_name, h2_est, pheno_file, genotype, grm, grm_ids, device):
    """对单个性状执行完整评测流程。"""
    print(f"\n{'='*70}")
    print(f"  性状: {trait_name}  (HE h² ≈ {h2_est})")
    print(f"{'='*70}")

    # 1. 加载表型
    labels, matched = load_phenotype(pheno_file, grm_ids)
    valid = ~np.isnan(labels)
    idx = np.where(valid)[0]
    n = len(idx)
    print(f"  有效样本: {matched}/{N_SAMPLES} → {n}")

    geno_sub = genotype[:, idx]
    grm_sub = grm[np.ix_(idx, idx)]
    lab_sub = labels[idx]
    print(f"  表型: mean={lab_sub.mean():.4f}, std={lab_sub.std():.4f}, "
          f"range=[{lab_sub.min():.2f}, {lab_sub.max():.2f}]")

    masks = generate_kfold_masks(n, k_folds=5, seed=42)

    # 2. 均值基线
    mean_metrics = []
    for train_mask, _, test_mask in masks:
        y_pred = np.full(test_mask.sum().item(), labels[idx][np.where(train_mask)[0]].mean())
        mean_metrics.append(evaluate_all(y_pred, lab_sub[test_mask]))

    # 3. BLUP
    print(f"  [BLUP] 5-fold CV...")
    t0 = time.time()
    blup_metrics, blup_lams = blup_cv(grm_sub, lab_sub, masks)
    blup_time = time.time() - t0
    print(f"  [BLUP] 完成 ({blup_time:.1f}s), λ={blup_lams}")

    # 4. SNP-VAE
    print(f"  [SNP-VAE] 训练 (200 epochs)...")
    t0 = time.time()
    _, snp_emb = train_snp_vae(genotype=geno_sub, device=device, progress=True, **SNP_VAE_CONFIG)
    snp_time = time.time() - t0
    print(f"  [SNP-VAE] 完成 ({snp_time:.1f}s), 嵌入: {snp_emb.shape}")

    # 5. VGAE
    print(f"  [VGAE] 5-fold CV (α={VGAE_CONFIG['alpha']}, d_z={VGAE_CONFIG['d_z']})...")
    t0 = time.time()
    cv_results = run_kfold_cv(
        genotype=geno_sub, snp_embeddings=snp_emb, grm=grm_sub, labels=lab_sub,
        k_folds=5, seed=42, device=device, progress=True, **VGAE_CONFIG,
    )
    vgae_time = time.time() - t0
    print(f"  [VGAE] 完成 ({vgae_time:.1f}s)")

    # 6. 汇总
    def agg(metrics_list):
        return {k: {"mean": float(np.mean([m[k] for m in metrics_list])),
                     "std": float(np.std([m[k] for m in metrics_list]))}
                for k in ["mse", "mae", "r", "r2"]}

    result = {
        "trait": trait_name,
        "h2_estimate": h2_est,
        "n_samples": int(n),
        "pheno_mean": float(lab_sub.mean()),
        "pheno_std": float(lab_sub.std()),
        "mean_baseline": agg(mean_metrics),
        "blup": {**agg(blup_metrics), "best_lambdas": blup_lams, "time_s": blup_time},
        "vgae": {
            **{k: {"mean": float(cv_results["mean_metrics"][k]),
                    "std": float(cv_results["std_metrics"][k])}
               for k in ["mse", "mae", "r", "r2"]},
            "time_s": vgae_time,
            "config": {k: v for k, v in VGAE_CONFIG.items()},
        },
        "snp_vae_time_s": snp_time,
        "fold_r2": [float(cv_results["fold_metrics"][i]["r2"]) for i in range(5)],
    }

    # 打印
    print(f"\n  {'方法':<12s} {'MSE':>10s} {'MAE':>10s} {'R':>10s} {'R²':>10s}")
    print(f"  {'-'*56}")
    for name, m in [("均值", mean_metrics), ("BLUP", blup_metrics)]:
        a = agg(m)
        print(f"  {name:<12s} {a['mse']['mean']:>7.4f}±{a['mse']['std']:.3f}"
              f" {a['mae']['mean']:>7.4f}±{a['mae']['std']:.3f}"
              f" {a['r']['mean']:>7.4f}±{a['r']['std']:.3f}"
              f" {a['r2']['mean']:>7.4f}±{a['r2']['std']:.3f}")
    v = result["vgae"]
    print(f"  {'VGAE':<12s} {v['mse']['mean']:>7.4f}±{v['mse']['std']:.3f}"
          f" {v['mae']['mean']:>7.4f}±{v['mae']['std']:.3f}"
          f" {v['r']['mean']:>7.4f}±{v['r']['std']:.3f}"
          f" {v['r2']['mean']:>7.4f}±{v['r2']['std']:.3f}")

    return result, snp_emb


# ============================================================================
# 主流程
# ============================================================================
def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 70)
    print("VGAE 基因组预测 — 多性状统一评测")
    print("=" * 70)
    print(f"设备: {device}")
    print(f"性状数: {len(TRAITS)}")
    print(f"VGAE 配置: α={VGAE_CONFIG['alpha']}, d_z={VGAE_CONFIG['d_z']}, "
          f"lr={VGAE_CONFIG['lr']}, kl_warmup={VGAE_CONFIG['kl_warmup_epochs']}")

    # 加载共享数据
    print("\n[加载共享数据]")
    genotype = genotype_read(BED_FILE, N_SAMPLES, N_SNPS)
    grm = grm_reader(GRM_BIN, N_SAMPLES)
    grm_ids = grm_id_reader(GRM_ID)
    print(f"  基因型: {genotype.shape}, GRM: {grm.shape}")

    # 逐性状评测
    all_results = []
    os.makedirs(RESULT_DIR, exist_ok=True)

    for trait_name, h2_est, pheno_file in TRAITS:
        result, snp_emb = evaluate_trait(
            trait_name, h2_est, pheno_file, genotype, grm, grm_ids, device
        )
        all_results.append(result)

        # 保存逐性状结果
        trait_dir = os.path.join(RESULT_DIR, trait_name)
        os.makedirs(trait_dir, exist_ok=True)
        with open(os.path.join(trait_dir, "metrics.json"), "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        np.save(os.path.join(trait_dir, "snp_embeddings.npy"), snp_emb)
        print(f"  结果已保存: {trait_dir}/")

    # 总体汇总
    print(f"\n{'='*70}")
    print("总体评测汇总")
    print(f"{'='*70}")
    print(f"\n{'性状':<16s} {'h²':>6s} {'N':>6s} {'BLUP R²':>10s} {'VGAE R²':>10s} {'提升':>8s}")
    print("-" * 62)

    summary_rows = []
    for r in all_results:
        blup_r2 = r["blup"]["r2"]["mean"]
        vgae_r2 = r["vgae"]["r2"]["mean"]
        gain = vgae_r2 - blup_r2
        pct = (vgae_r2 / blup_r2 - 1) * 100 if blup_r2 > 0 else float("inf")
        print(f"  {r['trait']:<14s} {r['h2_estimate']:>5.2f} {r['n_samples']:>5d}"
              f" {blup_r2:>8.4f} {vgae_r2:>8.4f} {gain:>+7.4f} ({pct:+.0f}%)")
        summary_rows.append({
            "trait": r["trait"], "h2": r["h2_estimate"], "n": r["n_samples"],
            "blup_r2": blup_r2, "vgae_r2": vgae_r2,
            "gain": gain, "gain_pct": pct,
        })

    # 平均
    avg_blup = np.mean([s["blup_r2"] for s in summary_rows])
    avg_vgae = np.mean([s["vgae_r2"] for s in summary_rows])
    avg_gain = avg_vgae - avg_blup
    print(f"\n  {'平均':<14s} {'':>5s} {'':>5s}"
          f" {avg_blup:>8.4f} {avg_vgae:>8.4f} {avg_gain:>+7.4f}")

    # 保存汇总
    summary = {
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M"),
        "device": device,
        "vgae_config": VGAE_CONFIG,
        "snp_vae_config": SNP_VAE_CONFIG,
        "traits": summary_rows,
        "average": {
            "blup_r2": float(avg_blup),
            "vgae_r2": float(avg_vgae),
            "gain": float(avg_gain),
        },
        "per_trait_results": all_results,
    }
    summary_path = os.path.join(RESULT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  汇总已保存: {summary_path}")
    print(f"\n{'='*70}")
    print("评测完成!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
