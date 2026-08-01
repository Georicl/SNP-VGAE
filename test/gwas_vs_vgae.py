"""
VGAE 归因 vs PLINK GWAS 对比分析
==================================
在同一 HDL 数据上对比:
  - PLINK GWAS: 逐 SNP 线性回归 P-value (金标准)
  - VGAE 归因: 梯度反投影 SNP 贡献分数

输出:
  - 排名相关性 (Spearman)
  - Manhattan plot 对比
  - Top-K 重合分析
  - VGAE 独有 SNP (非线性信号)

运行: PYTHONPATH=pytorch/src uv run python test/gwas_vs_vgae.py
"""

import json
import os
import time
import numpy as np
import torch
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.genotype import genotype_read
from data.grm import grm_reader, grm_id_reader
from data.bim import bim_reader
from pretrain.train import train_snp_vae
from graph.build_graph import GraphBuilder
from model.vgae import VGAEModel
from model.attribution import compute_snp_scores, aggregate_cv_scores
from train.trainer import VGAETrainer
from train.cross_validation import generate_kfold_masks
from train.metrics import evaluate_all

# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "test_data")
RESULT_DIR = os.path.join(PROJECT_ROOT, "test", "result", "Biochem.HDL")
FIG_DIR = os.path.join(RESULT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

BED_FILE = os.path.join(DATA_DIR, "hs_mice_genome_QCed.bed")
BIM_FILE = os.path.join(DATA_DIR, "hs_mice_genome_QCed.bim")
GRM_BIN = os.path.join(DATA_DIR, "total_autosome_grm.grm.bin")
GRM_ID = os.path.join(DATA_DIR, "total_autosome_grm.grm.id")
PHENO_FILE = os.path.join(DATA_DIR, "phenotypes", "Biochem.HDL.resid")
GWAS_FILE = os.path.join(DATA_DIR, "HDL_gwas.assoc.linear")

N_SAMPLES = 2002
N_SNPS = 9728

VGAE_CONFIG = dict(
    k_neighbors=30, d_snp=64, d_hidden=128, d_z=16,
    dropout=0.5, mlp_hidden=64, lr=1e-3, alpha=0.1, beta_kl=0.01,
    max_epochs=200, patience=30, weight_decay=1e-5, kl_warmup_epochs=30,
)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
})


def load_gwas(path):
    """加载 PLINK GWAS 结果。"""
    snps, chrs, bps, betas, pvals = [], [], [], [], []
    with open(path) as f:
        f.readline()
        for line in f:
            parts = line.split()
            if len(parts) >= 9:
                chrs.append(int(parts[0]))
                snps.append(parts[1])
                bps.append(int(parts[2]))
                betas.append(float(parts[6]))
                pvals.append(float(parts[8]))
    return {
        "snp": snps, "chr": chrs, "bp": bps,
        "beta": np.array(betas), "p": np.array(pvals),
    }


def load_phenotype(pheno_file, grm_ids):
    n = len(grm_ids)
    labels = np.full(n, np.nan, dtype=np.float32)
    pheno_map = {}
    with open(pheno_file) as f:
        f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1] not in ("-9.0", "-9", "NA", ""):
                try: pheno_map[parts[0]] = float(parts[1])
                except ValueError: pass
    for i, (_, iid) in enumerate(grm_ids):
        if iid in pheno_map:
            labels[i] = pheno_map[iid]
    return labels


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 70)
    print("VGAE 归因 vs PLINK GWAS — Biochem.HDL")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1] 加载数据...")
    genotype = genotype_read(BED_FILE, N_SAMPLES, N_SNPS)
    grm = grm_reader(GRM_BIN, N_SAMPLES)
    grm_ids = grm_id_reader(GRM_ID)
    labels = load_phenotype(PHENO_FILE, grm_ids)
    gwas = load_gwas(GWAS_FILE)

    valid = ~np.isnan(labels)
    idx = np.where(valid)[0]
    geno_sub = genotype[:, idx]
    grm_sub = grm[np.ix_(idx, idx)]
    lab_sub = labels[idx]
    n = len(idx)
    M = geno_sub.shape[0]
    print(f"  样本: {n}, SNPs: {M}, GWAS SNPs: {len(gwas['snp'])}")

    # 2. SNP-VAE
    print("\n[2] SNP-VAE 预训练...")
    t0 = time.time()
    _, snp_emb = train_snp_vae(
        genotype=geno_sub, device=device, progress=True,
        d_snp=64, beta=0.001, lr=5e-4, epochs=200, batch_size=512,
        patience=50, kl_warmup_epochs=150, hidden_dim=1024,
    )
    print(f"  完成 ({time.time()-t0:.0f}s)")

    # 3. VGAE 5-fold + 归因
    print("\n[3] VGAE 5-fold 训练 + SNP 归因...")
    builder = GraphBuilder()
    adj_norm, adj_raw = builder.knn_graph_builder(grm_sub, k=30)
    node_features = builder.build_node_features(geno_sub, snp_emb)
    labels_t = torch.from_numpy(lab_sub).float()
    masks = generate_kfold_masks(n, k_folds=5, seed=42)

    fold_scores = []
    y_pred_all = np.zeros(n)

    for fi, (train_m, val_m, test_m) in enumerate(masks):
        print(f"\n  Fold {fi+1}/5")
        model = VGAEModel(d_snp=64, d_hidden=128, d_z=16, dropout=0.5, mlp_hidden=64)
        trainer = VGAETrainer(
            model=model, adj_norm=adj_norm, adj_raw=adj_raw,
            node_features=node_features, labels=labels_t,
            train_mask=train_m, val_mask=val_m, test_mask=test_m,
            lr=1e-3, alpha=0.1, beta=0.01, weight_decay=1e-5,
            kl_warmup_epochs=30, patience=30, max_epochs=200,
            device=device, progress=True,
        )
        result = trainer.train()

        # 预测
        model.eval()
        with torch.no_grad():
            out = model(node_features.to(device), adj_norm.to(device))
            yp = out["y_pred"].cpu().numpy()
        test_idx = np.where(test_m.numpy())[0]
        y_pred_all[test_idx] = yp[test_idx]

        # 归因
        scores = compute_snp_scores(
            model=model, node_features=node_features, adj_norm=adj_norm,
            genotype=geno_sub, snp_embeddings=snp_emb,
            aggregate="sum", device=device,
        )
        fold_scores.append(scores)
        print(f"  Top |score|: {scores.abs().max():.4f}")

    vgae_scores = aggregate_cv_scores(fold_scores, "mean").numpy()
    metrics = evaluate_all(y_pred_all, lab_sub)
    print(f"\n  VGAE 预测: R²={metrics['r2']:.4f}, R={metrics['r']:.4f}")

    # 4. 对比分析
    print(f"\n{'='*70}")
    print("[4] 对比分析")
    print(f"{'='*70}")

    gwas_p = gwas["p"]
    gwas_neg_log_p = -np.log10(np.maximum(gwas_p, 1e-300))
    vgae_abs = np.abs(vgae_scores)

    # 排名相关性
    gwas_rank = stats.rankdata(-gwas_neg_log_p)  # 越小越显著
    vgae_rank = stats.rankdata(-vgae_abs)
    spearman_r, spearman_p = stats.spearmanr(gwas_rank, vgae_rank)
    print(f"\n  Spearman 排名相关: ρ={spearman_r:.4f}, p={spearman_p:.2e}")

    # Top-K 重合
    for K in [50, 100, 500]:
        gwas_top = set(np.argsort(gwas_p)[:K])
        vgae_top = set(np.argsort(-vgae_abs)[:K])
        overlap = len(gwas_top & vgae_top)
        expected = K * K / M
        enrichment = overlap / expected if expected > 0 else 0
        print(f"  Top-{K}: 重合 {overlap}/{K} "
              f"(随机期望 {expected:.1f}, 富集 {enrichment:.1f}×)")

    # GWAS 显著 SNP (Bonferroni)
    bonf_thresh = 0.05 / M
    gwas_sig = np.where(gwas_p < bonf_thresh)[0]
    print(f"\n  GWAS 显著 SNP (Bonferroni α=0.05): {len(gwas_sig)}")
    if len(gwas_sig) > 0:
        for j in gwas_sig[:10]:
            print(f"    {gwas['snp'][j]:>14s}  chr{gwas['chr'][j]}:{gwas['bp'][j]}  "
                  f"P={gwas_p[j]:.2e}  VGAE|score|={vgae_abs[j]:.4f}")

    # 5. 图表
    print(f"\n[5] 生成图表...")

    # Fig A: Manhattan plot 对比
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    colors = plt.cm.tab10(np.array(gwas["chr"]) % 10)

    ax1.scatter(range(M), gwas_neg_log_p, c=colors, s=3, alpha=0.6, rasterized=True)
    ax1.axhline(y=-np.log10(bonf_thresh), color="red", linestyle="--",
                linewidth=1, label=f"Bonferroni (P={bonf_thresh:.1e})")
    ax1.set_ylabel("-log₁₀(P)")
    ax1.set_title("PLINK GWAS")
    ax1.legend(fontsize=8)

    ax2.scatter(range(M), vgae_abs, c=colors, s=3, alpha=0.6, rasterized=True)
    ax2.set_ylabel("|VGAE Score|")
    ax2.set_xlabel("SNP index (ordered by chromosome)")
    ax2.set_title("VGAE Attribution")

    # 染色体分隔线
    chrs = np.array(gwas["chr"])
    for c in range(1, 20):
        boundary = np.where(np.diff(chrs == c) != 0)[0]
        for b in boundary:
            ax1.axvline(x=b, color="#DDD", linewidth=0.5)
            ax2.axvline(x=b, color="#DDD", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "manhattan_comparison.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "manhattan_comparison.png"))
    plt.close(fig)
    print(f"  ✓ manhattan_comparison")

    # Fig B: 排名散点图
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(gwas_neg_log_p, vgae_abs, s=3, alpha=0.3, c="#555", rasterized=True)
    # 标注 Top SNP
    top_idx = np.argsort(gwas_p)[:20]
    ax.scatter(gwas_neg_log_p[top_idx], vgae_abs[top_idx],
               s=30, c="red", edgecolors="white", linewidth=0.5, zorder=5)
    ax.set_xlabel("GWAS -log₁₀(P)")
    ax.set_ylabel("VGAE |Score|")
    ax.set_title(f"Rank Correlation (Spearman ρ={spearman_r:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "rank_correlation.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "rank_correlation.png"))
    plt.close(fig)
    print(f"  ✓ rank_correlation")

    # 6. 保存结果
    result = {
        "spearman_rho": float(spearman_r),
        "spearman_p": float(spearman_p),
        "vgae_r2": float(metrics["r2"]),
        "vgae_r": float(metrics["r"]),
        "gwas_n_significant": int(len(gwas_sig)),
        "top_overlap": {
            f"top_{K}": int(len(set(np.argsort(gwas_p)[:K]) & set(np.argsort(-vgae_abs)[:K])))
            for K in [50, 100, 500]
        },
    }
    with open(os.path.join(RESULT_DIR, "gwas_vs_vgae.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  结果已保存: {RESULT_DIR}/gwas_vs_vgae.json")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
