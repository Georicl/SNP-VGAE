"""
SNP 显著性判定 — 跨折一致性 + GWAS 交叉验证
=============================================
方法 2: 跨折一致性 (5-fold 归因分数的均值/变异系数)
方法 3: GWAS 交叉验证 (PLINK P-value)

分类:
  Confirmed    — VGAE 一致高分 + GWAS 显著
  VGAE-unique  — VGAE 一致高分 + GWAS 不显著 (候选非线性)
  GWAS-unique  — GWAS 显著 + VGAE 低分
  Background   — 两者均不显著

运行: PYTHONPATH=pytorch/src uv run python test/snp_significance.py
"""

import json, os, time
import numpy as np
import torch
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.genotype import genotype_read
from data.grm import grm_reader, grm_id_reader
from pretrain.train import train_snp_vae
from graph.build_graph import GraphBuilder
from model.vgae import VGAEModel
from model.attribution import compute_snp_scores
from train.trainer import VGAETrainer
from train.cross_validation import generate_kfold_masks

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "test_data")
OUT_DIR = os.path.join(PROJECT_ROOT, "test", "result", "Biochem.HDL")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

BED = os.path.join(DATA_DIR, "hs_mice_genome_QCed.bed")
BIM = os.path.join(DATA_DIR, "hs_mice_genome_QCed.bim")
GRM_BIN = os.path.join(DATA_DIR, "total_autosome_grm.grm.bin")
GRM_ID = os.path.join(DATA_DIR, "total_autosome_grm.grm.id")
PHENO = os.path.join(DATA_DIR, "phenotypes", "Biochem.HDL.resid")
GWAS = os.path.join(DATA_DIR, "HDL_gwas.assoc.linear")

N_SAMPLES, N_SNPS = 2002, 9728
VGAE_CFG = dict(k_neighbors=30, d_snp=64, d_hidden=128, d_z=16,
    dropout=0.5, mlp_hidden=64, lr=1e-3, alpha=0.1, beta_kl=0.01,
    max_epochs=200, patience=30, weight_decay=1e-5, kl_warmup_epochs=30)

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False})


def load_pheno(path, grm_ids):
    n = len(grm_ids)
    lab = np.full(n, np.nan, dtype=np.float32)
    pm = {}
    with open(path) as f:
        f.readline()
        for l in f:
            p = l.split()
            if len(p) >= 2 and p[1] not in ("-9.0","-9","NA",""):
                try: pm[p[0]] = float(p[1])
                except: pass
    for i,(_,iid) in enumerate(grm_ids):
        if iid in pm: lab[i] = pm[iid]
    return lab

def load_gwas(path):
    snps,chrs,bps,pvals = [],[],[],[]
    with open(path) as f:
        f.readline()
        for l in f:
            p = l.split()
            if len(p) >= 9:
                chrs.append(int(p[0])); snps.append(p[1])
                bps.append(int(p[2])); pvals.append(float(p[8]))
    return snps, np.array(chrs), np.array(bps), np.array(pvals)

def load_bim(path):
    info = []
    with open(path) as f:
        for l in f:
            p = l.split()
            if len(p) >= 4:
                info.append({"chr": p[0], "name": p[1], "bp": int(p[3])})
    return info


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 70)
    print("SNP 显著性判定 — 跨折一致性 + GWAS 交叉验证")
    print("=" * 70)

    # 加载数据
    print("\n[1] 加载数据...")
    geno = genotype_read(BED, N_SAMPLES, N_SNPS)
    grm = grm_reader(GRM_BIN, N_SAMPLES)
    gids = grm_id_reader(GRM_ID)
    labels = load_pheno(PHENO, gids)
    snp_names, chrs, bps, gwas_p = load_gwas(GWAS)
    bim = load_bim(BIM)

    valid = ~np.isnan(labels)
    idx = np.where(valid)[0]
    geno_s = geno[:, idx]; grm_s = grm[np.ix_(idx, idx)]
    lab_s = labels[idx]; n = len(idx); M = geno_s.shape[0]
    print(f"  N={n}, M={M}")

    # SNP-VAE
    print("\n[2] SNP-VAE...")
    _, snp_emb = train_snp_vae(genotype=geno_s, device=device, progress=True,
        d_snp=64, beta=0.001, lr=5e-4, epochs=200, batch_size=512,
        patience=50, kl_warmup_epochs=150, hidden_dim=1024)

    # VGAE 5-fold + 逐折归因
    print("\n[3] VGAE 5-fold + 逐折归因...")
    builder = GraphBuilder()
    adj_norm, adj_raw = builder.knn_graph_builder(grm_s, k=30)
    nf = builder.build_node_features(geno_s, snp_emb)
    lab_t = torch.from_numpy(lab_s).float()
    masks = generate_kfold_masks(n, 5, 42)

    fold_scores = np.zeros((5, M))
    for fi, (tr, va, te) in enumerate(masks):
        print(f"\n  Fold {fi+1}/5")
        model = VGAEModel(d_snp=64, d_hidden=128, d_z=16, dropout=0.5, mlp_hidden=64)
        trainer = VGAETrainer(model=model, adj_norm=adj_norm, adj_raw=adj_raw,
            node_features=nf, labels=lab_t, train_mask=tr, val_mask=va, test_mask=te,
            lr=1e-3, alpha=0.1, beta=0.01, weight_decay=1e-5,
            kl_warmup_epochs=30, patience=30, max_epochs=200,
            device=device, progress=True)
        trainer.train()
        sc = compute_snp_scores(model=model, node_features=nf, adj_norm=adj_norm,
            genotype=geno_s, snp_embeddings=snp_emb, aggregate="sum", device=device)
        fold_scores[fi] = sc.numpy()

    # 跨折统计
    print("\n[4] 跨折一致性分析...")
    abs_scores = np.abs(fold_scores)
    mean_abs = abs_scores.mean(axis=0)       # 平均 |Score|
    std_abs = abs_scores.std(axis=0)          # 标准差
    cv = std_abs / (mean_abs + 1e-10)         # 变异系数
    sign_consistency = np.sign(fold_scores).mean(axis=0)  # 符号一致性 [-1, 1]

    # 显著性阈值
    score_thresh = np.percentile(mean_abs, 95)  # Top 5%
    cv_thresh = 0.5                              # CV < 0.5 = 一致
    gwas_thresh_nominal = 0.05
    gwas_thresh_bonf = 0.05 / M

    # 分类
    vgae_sig = (mean_abs >= score_thresh) & (cv < cv_thresh)
    gwas_sig_nom = gwas_p < gwas_thresh_nominal
    gwas_sig_bonf = gwas_p < gwas_thresh_bonf

    confirmed = vgae_sig & gwas_sig_nom
    vgae_unique = vgae_sig & ~gwas_sig_nom
    gwas_unique = ~vgae_sig & gwas_sig_nom
    background = ~vgae_sig & ~gwas_sig_nom

    print(f"\n  VGAE 显著 (Top5% + CV<0.5): {vgae_sig.sum()}")
    print(f"  GWAS 显著 (P<0.05):         {gwas_sig_nom.sum()}")
    print(f"  GWAS 显著 (Bonferroni):     {gwas_sig_bonf.sum()}")
    print(f"\n  === 分类结果 ===")
    print(f"  Confirmed (双确认):    {confirmed.sum()}")
    print(f"  VGAE-unique (非线性):  {vgae_unique.sum()}")
    print(f"  GWAS-unique (线性):    {gwas_unique.sum()}")
    print(f"  Background:            {background.sum()}")

    # 输出显著 SNP 表
    print(f"\n[5] 输出结果...")

    # 完整归因表
    out_path = os.path.join(OUT_DIR, "snp_significance.tsv")
    with open(out_path, "w") as f:
        f.write("IDX\tCHR\tSNP\tBP\tMean|Score|\tStd\tCV\tSignConsist\t"
                "GWAS_P\tGWAS_sig_nom\tGWAS_sig_bonf\tCategory\n")
        for j in range(M):
            info = bim[j] if j < len(bim) else {"chr":"?","name":f"SNP_{j}","bp":0}
            cat = ("Confirmed" if confirmed[j] else
                   "VGAE-unique" if vgae_unique[j] else
                   "GWAS-unique" if gwas_unique[j] else "Background")
            f.write(f"{j}\t{info['chr']}\t{info['name']}\t{info['bp']}\t"
                    f"{mean_abs[j]:.6f}\t{std_abs[j]:.6f}\t{cv[j]:.4f}\t"
                    f"{sign_consistency[j]:.3f}\t{gwas_p[j]:.6e}\t"
                    f"{'Y' if gwas_sig_nom[j] else 'N'}\t"
                    f"{'Y' if gwas_sig_bonf[j] else 'N'}\t{cat}\n")
    print(f"  ✓ {out_path}")

    # Top SNP 列表
    top_path = os.path.join(OUT_DIR, "significant_snps.tsv")
    sig_idx = np.where(vgae_sig)[0]
    sig_idx = sig_idx[np.argsort(-mean_abs[sig_idx])]
    with open(top_path, "w") as f:
        f.write("Rank\tIDX\tCHR\tSNP\tBP\tMean|Score|\tCV\tGWAS_P\tCategory\n")
        for rank, j in enumerate(sig_idx[:200], 1):
            info = bim[j] if j < len(bim) else {"chr":"?","name":f"SNP_{j}","bp":0}
            cat = "Confirmed" if confirmed[j] else "VGAE-unique"
            f.write(f"{rank}\t{j}\t{info['chr']}\t{info['name']}\t{info['bp']}\t"
                    f"{mean_abs[j]:.6f}\t{cv[j]:.4f}\t{gwas_p[j]:.6e}\t{cat}\n")
    print(f"  ✓ {top_path} (Top {min(200, len(sig_idx))})")

    # 图表
    print(f"\n[6] 生成图表...")

    # Fig: 分类 Manhattan plot
    fig, ax = plt.subplots(figsize=(12, 4.5))
    neg_log_p = -np.log10(np.maximum(gwas_p, 1e-300))

    colors = {"Background": "#CCCCCC", "GWAS-unique": "#4C72B0",
              "VGAE-unique": "#DD8452", "Confirmed": "#C44E52"}
    for cat in ["Background", "GWAS-unique", "VGAE-unique", "Confirmed"]:
        mask = {"Background": background, "GWAS-unique": gwas_unique,
                "VGAE-unique": vgae_unique, "Confirmed": confirmed}[cat]
        idx_cat = np.where(mask)[0]
        ax.scatter(idx_cat, mean_abs[idx_cat], c=colors[cat], s=4,
                   alpha=0.7, label=f"{cat} ({mask.sum()})", rasterized=True)

    ax.axhline(y=score_thresh, color="red", linestyle="--", linewidth=1,
               label=f"VGAE threshold (Top 5%)")
    ax.set_xlabel("SNP index (by chromosome)")
    ax.set_ylabel("Mean |VGAE Score| (5-fold)")
    ax.set_title("SNP Significance Classification")
    ax.legend(fontsize=8, loc="upper right", markerscale=3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "snp_classification.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "snp_classification.png"))
    plt.close(fig)
    print(f"  ✓ snp_classification")

    # Fig: 跨折一致性
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.hist(mean_abs, bins=50, color="#CCC", edgecolor="white", label="All")
    ax1.hist(mean_abs[vgae_sig], bins=20, color="#DD8452", edgecolor="white",
             alpha=0.8, label=f"VGAE sig ({vgae_sig.sum()})")
    ax1.axvline(x=score_thresh, color="red", linestyle="--", label=f"Threshold={score_thresh:.4f}")
    ax1.set_xlabel("Mean |Score|")
    ax1.set_ylabel("Count")
    ax1.set_title("(a) Score Distribution")
    ax1.legend(fontsize=8)

    ax2.scatter(mean_abs, cv, s=3, alpha=0.3, c="#999", rasterized=True)
    ax2.scatter(mean_abs[vgae_sig], cv[vgae_sig], s=10, c="#DD8452",
                alpha=0.8, label="VGAE sig")
    ax2.axhline(y=cv_thresh, color="red", linestyle="--", label=f"CV={cv_thresh}")
    ax2.axvline(x=score_thresh, color="blue", linestyle="--", label=f"Score thresh")
    ax2.set_xlabel("Mean |Score|")
    ax2.set_ylabel("CV (std/mean)")
    ax2.set_title("(b) Consistency")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 2)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fold_consistency.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fold_consistency.png"))
    plt.close(fig)
    print(f"  ✓ fold_consistency")

    # Fig: Venn-style 对比
    fig, ax = plt.subplots(figsize=(5, 4))
    categories = ["Confirmed", "VGAE-unique", "GWAS-unique"]
    counts = [confirmed.sum(), vgae_unique.sum(), gwas_unique.sum()]
    bars = ax.bar(categories, counts, color=[colors[c] for c in categories],
                  edgecolor="white", linewidth=1)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(cnt), ha="center", fontweight="bold")
    ax.set_ylabel("Number of SNPs")
    ax.set_title("SNP Classification Summary")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "classification_summary.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "classification_summary.png"))
    plt.close(fig)
    print(f"  ✓ classification_summary")

    # 保存 JSON
    summary = {
        "thresholds": {
            "vgae_score_top5pct": float(score_thresh),
            "cv_max": cv_thresh,
            "gwas_nominal": gwas_thresh_nominal,
            "gwas_bonferroni": float(gwas_thresh_bonf),
        },
        "counts": {
            "vgae_significant": int(vgae_sig.sum()),
            "gwas_nominal": int(gwas_sig_nom.sum()),
            "gwas_bonferroni": int(gwas_sig_bonf.sum()),
            "confirmed": int(confirmed.sum()),
            "vgae_unique": int(vgae_unique.sum()),
            "gwas_unique": int(gwas_unique.sum()),
            "background": int(background.sum()),
        },
    }
    with open(os.path.join(OUT_DIR, "snp_significance_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("完成!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
