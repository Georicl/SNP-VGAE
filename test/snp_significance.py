"""
稳定高归因 SNP 候选筛选 + nominal GWAS overlap
=============================================
Top 5% 平均绝对归因分数 + 跨折 CV < 0.5 是探索性候选筛选规则，
不是经过统计校准的显著性检验。GWAS P < 0.05 仅表示 nominal overlap，
不能作为独立重复验证，也不能据此推断候选 SNP 存在非线性效应。

GWAS 先按 SNP ID 与 BIM 严格一对一对齐；存在 TEST 列时仅选 ADD。
缺失/额外/重复 SNP、无效 P 值或坐标冲突会终止分析，不插补 P 值。
结果写入新的 candidate 文件名，保留既有 significance 结果文件。

运行: PYTHONPATH=pytorch/src uv run python test/snp_significance.py
"""

import json, os
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.genotype import genotype_read
from data.gwas import load_aligned_gwas
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


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 70)
    print("稳定高归因 SNP 候选筛选 + nominal GWAS overlap")
    print("=" * 70)

    # 加载数据
    print("\n[1] 加载数据...")
    geno = genotype_read(BED, N_SAMPLES, N_SNPS)
    grm = grm_reader(GRM_BIN, N_SAMPLES)
    gids = grm_id_reader(GRM_ID)
    labels = load_pheno(PHENO, gids)
    bim, gwas_p = load_aligned_gwas(GWAS, BIM, expected_n_snps=geno.shape[0])
    os.makedirs(FIG_DIR, exist_ok=True)
    print("  SNP-ID alignment: strict 1:1 match; TEST=ADD when present")

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

    # 探索性候选阈值（不是归因显著性检验）
    score_thresh = np.percentile(mean_abs, 95)  # Top 5%
    cv_thresh = 0.5                              # CV < 0.5 = 一致
    gwas_thresh_nominal = 0.05
    gwas_thresh_bonf = 0.05 / M

    # 分类
    stable_candidate = (mean_abs >= score_thresh) & (cv < cv_thresh)
    gwas_nominal_overlap = gwas_p < gwas_thresh_nominal
    gwas_bonferroni_screen = gwas_p < gwas_thresh_bonf

    candidate_with_nominal_overlap = stable_candidate & gwas_nominal_overlap
    candidate_without_nominal_overlap = stable_candidate & ~gwas_nominal_overlap
    nominal_gwas_only = ~stable_candidate & gwas_nominal_overlap
    neither_screen = ~stable_candidate & ~gwas_nominal_overlap

    print(f"\n  稳定高归因候选 (Top5% + CV<0.5): {stable_candidate.sum()}")
    print(f"  GWAS nominal (P<0.05):         {gwas_nominal_overlap.sum()}")
    print(f"  GWAS Bonferroni screen (0.05/M):     {gwas_bonferroni_screen.sum()}")
    print(f"\n  === 分类结果 ===")
    print(f"  候选 + nominal GWAS overlap:    {candidate_with_nominal_overlap.sum()}")
    print(f"  候选，无 nominal GWAS overlap:  {candidate_without_nominal_overlap.sum()}")
    print(f"  仅 nominal GWAS screen:    {nominal_gwas_only.sum()}")
    print(f"  未通过上述筛选:            {neither_screen.sum()}")

    # 输出候选 SNP 表
    print(f"\n[5] 输出结果...")

    # 完整归因表
    out_path = os.path.join(OUT_DIR, "snp_candidates.tsv")
    with open(out_path, "w") as f:
        f.write("IDX\tCHR\tSNP\tBP\tMean|Score|\tStd\tCV\tSignConsist\t"
                "GWAS_P\tGWAS_nominal_P_lt_0.05\tGWAS_P_lt_0.05_over_M\tCategory\n")
        for j in range(M):
            info = bim[j]
            cat = ("StableCandidate+NominalGWAS" if candidate_with_nominal_overlap[j] else
                   "StableCandidate_NoNominalOverlap" if candidate_without_nominal_overlap[j] else
                   "NominalGWASOnly" if nominal_gwas_only[j] else "NeitherScreen")
            f.write(f"{j}\t{info['chr']}\t{info['name']}\t{info['bp']}\t"
                    f"{mean_abs[j]:.6f}\t{std_abs[j]:.6f}\t{cv[j]:.4f}\t"
                    f"{sign_consistency[j]:.3f}\t{gwas_p[j]:.6e}\t"
                    f"{'Y' if gwas_nominal_overlap[j] else 'N'}\t"
                    f"{'Y' if gwas_bonferroni_screen[j] else 'N'}\t{cat}\n")
    print(f"  ✓ {out_path}")

    # Top SNP 列表
    top_path = os.path.join(OUT_DIR, "stable_snp_candidates.tsv")
    candidate_idx = np.where(stable_candidate)[0]
    candidate_idx = candidate_idx[np.argsort(-mean_abs[candidate_idx])]
    with open(top_path, "w") as f:
        f.write("Rank\tIDX\tCHR\tSNP\tBP\tMean|Score|\tCV\tGWAS_P\tCategory\n")
        for rank, j in enumerate(candidate_idx[:200], 1):
            info = bim[j]
            cat = "StableCandidate+NominalGWAS" if candidate_with_nominal_overlap[j] else "StableCandidate_NoNominalOverlap"
            f.write(f"{rank}\t{j}\t{info['chr']}\t{info['name']}\t{info['bp']}\t"
                    f"{mean_abs[j]:.6f}\t{cv[j]:.4f}\t{gwas_p[j]:.6e}\t{cat}\n")
    print(f"  ✓ {top_path} (Top {min(200, len(candidate_idx))})")

    # 图表
    print(f"\n[6] 生成图表...")

    # Fig: 分类 Manhattan plot
    fig, ax = plt.subplots(figsize=(12, 4.5))

    colors = {"NeitherScreen": "#CCCCCC", "NominalGWASOnly": "#4C72B0",
              "StableCandidate_NoNominalOverlap": "#DD8452", "StableCandidate+NominalGWAS": "#C44E52"}
    display_labels = {
        "NeitherScreen": "Neither screen",
        "NominalGWASOnly": "Nominal GWAS only",
        "StableCandidate_NoNominalOverlap": "Stable candidate (no nominal overlap)",
        "StableCandidate+NominalGWAS": "Stable candidate + nominal GWAS",
    }
    for cat in ["NeitherScreen", "NominalGWASOnly", "StableCandidate_NoNominalOverlap", "StableCandidate+NominalGWAS"]:
        mask = {"NeitherScreen": neither_screen, "NominalGWASOnly": nominal_gwas_only,
                "StableCandidate_NoNominalOverlap": candidate_without_nominal_overlap, "StableCandidate+NominalGWAS": candidate_with_nominal_overlap}[cat]
        idx_cat = np.where(mask)[0]
        ax.scatter(idx_cat, mean_abs[idx_cat], c=colors[cat], s=4,
                   alpha=0.7, label=f"{display_labels[cat]} ({mask.sum()})", rasterized=True)

    ax.axhline(y=score_thresh, color="red", linestyle="--", linewidth=1,
               label=f"Candidate score cutoff (Top 5%)")
    ax.set_xlabel("SNP index (BIM order)")
    ax.set_ylabel("Mean |VGAE Score| (5-fold)")
    ax.set_title("Stable High-attribution Candidates and Nominal GWAS Overlap")
    ax.legend(fontsize=8, loc="upper right", markerscale=3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "snp_candidate_classification.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "snp_candidate_classification.png"))
    plt.close(fig)
    print(f"  ✓ snp_candidate_classification")

    # Fig: 跨折一致性
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.hist(mean_abs, bins=50, color="#CCC", edgecolor="white", label="All")
    ax1.hist(mean_abs[stable_candidate], bins=20, color="#DD8452", edgecolor="white",
             alpha=0.8, label=f"Stable candidates ({stable_candidate.sum()})")
    ax1.axvline(x=score_thresh, color="red", linestyle="--", label=f"Threshold={score_thresh:.4f}")
    ax1.set_xlabel("Mean |Score|")
    ax1.set_ylabel("Count")
    ax1.set_title("(a) Score Distribution")
    ax1.legend(fontsize=8)

    ax2.scatter(mean_abs, cv, s=3, alpha=0.3, c="#999", rasterized=True)
    ax2.scatter(mean_abs[stable_candidate], cv[stable_candidate], s=10, c="#DD8452",
                alpha=0.8, label="Stable candidates")
    ax2.axhline(y=cv_thresh, color="red", linestyle="--", label=f"CV={cv_thresh}")
    ax2.axvline(x=score_thresh, color="blue", linestyle="--", label=f"Score thresh")
    ax2.set_xlabel("Mean |Score|")
    ax2.set_ylabel("CV (std/mean)")
    ax2.set_title("(b) Consistency")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 2)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "snp_candidate_consistency.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "snp_candidate_consistency.png"))
    plt.close(fig)
    print(f"  ✓ snp_candidate_consistency")

    # Fig: Venn-style 对比
    fig, ax = plt.subplots(figsize=(8, 4))
    categories = ["StableCandidate+NominalGWAS", "StableCandidate_NoNominalOverlap", "NominalGWASOnly"]
    counts = [candidate_with_nominal_overlap.sum(), candidate_without_nominal_overlap.sum(), nominal_gwas_only.sum()]
    bars = ax.bar(categories, counts, color=[colors[c] for c in categories],
                  edgecolor="white", linewidth=1)
    ax.set_xticks(range(3), ["Stable candidate +\nnominal GWAS",
                             "Stable candidate\n(no nominal overlap)",
                             "Nominal GWAS only"])
    ax.bar_label(bars, labels=[str(count) for count in counts], padding=3,
                 fontweight="bold")
    ax.set_ylim(0, max(1, max(counts)) * 1.2)
    ax.set_ylabel("Number of SNPs")
    ax.set_title("Candidate Screen Summary")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "snp_candidate_summary.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "snp_candidate_summary.png"))
    plt.close(fig)
    print(f"  ✓ snp_candidate_summary")

    # 保存 JSON
    summary = {
        "interpretation": {
            "attribution": "Stable high-attribution candidates are a heuristic top-5% "
                           "and CV<0.5 screen, not calibrated statistical significance.",
            "gwas_overlap": "P<0.05 is nominal overlap, not independent replication "
                            "or confirmation. Lack of overlap does not establish nonlinearity.",
            "bonferroni_family": "0.05/M applies to the M aligned selected GWAS tests; "
                                 "this script does not verify a larger original testing family.",
        },
        "alignment": {"key": "SNP ID", "order": "BIM", "unmatched_policy": "error",
                      "test_selection": "ADD if TEST column exists; otherwise supplied rows",
                      "coordinate_policy": "exact chromosome and positive base-pair match",
                      "n_snps": M},
        "thresholds": {
            "candidate_score_top5pct": float(score_thresh),
            "cv_max": cv_thresh,
            "gwas_nominal": gwas_thresh_nominal,
            "gwas_bonferroni": float(gwas_thresh_bonf),
        },
        "counts": {
            "stable_high_attribution_candidates": int(stable_candidate.sum()),
            "gwas_nominal": int(gwas_nominal_overlap.sum()),
            "gwas_bonferroni": int(gwas_bonferroni_screen.sum()),
            "candidate_with_nominal_overlap": int(candidate_with_nominal_overlap.sum()),
            "candidate_without_nominal_overlap": int(candidate_without_nominal_overlap.sum()),
            "nominal_gwas_only": int(nominal_gwas_only.sum()),
            "neither_screen": int(neither_screen.sum()),
        },
    }
    with open(os.path.join(OUT_DIR, "snp_candidates_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("完成!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
