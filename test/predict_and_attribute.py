"""
VGAE 基因组预测 — 推理 + SNP 归因 + 结果输出
=============================================
端到端流程:
  1. 训练 VGAE (5-fold CV, 保留每折模型)
  2. 表型预测 (跨折聚合)
  3. SNP 归因 (梯度反投影, 跨折聚合)
  4. 显著性评估 (置换 null 分布 + Bonferroni)
  5. 输出标准结果文件

输出文件 (test/result/<trait>/):
  predictions.tsv      — 个体预测值 (IID, y_true, y_pred, residual)
  snp_attribution.tsv  — SNP 归因表 (SNP_IDX, Score, |Score|, Rank, P_bonf)
  top_snps.tsv         — Top-K 显著 SNP
  model_summary.json   — 模型配置 + 性能汇总

运行:
  PYTHONPATH=pytorch/src uv run python test/predict_and_attribute.py
"""

import json
import os
import time
import numpy as np
import torch

from data.genotype import genotype_read
from data.grm import grm_reader, grm_id_reader
from data.bim import bim_reader
from pretrain.train import train_snp_vae
from graph.build_graph import GraphBuilder
from model.vgae import VGAEModel
from model.attribution import compute_snp_scores, top_snps, aggregate_cv_scores
from train.cross_validation import run_kfold_cv, generate_kfold_masks
from train.trainer import VGAETrainer
from train.metrics import evaluate_all

# ============================================================================
# 配置
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "test_data")
RESULT_DIR = os.path.join(PROJECT_ROOT, "test", "result")

BED_FILE = os.path.join(DATA_DIR, "hs_mice_genome_QCed.bed")
BIM_FILE = os.path.join(DATA_DIR, "hs_mice_genome_QCed.bim")
GRM_BIN = os.path.join(DATA_DIR, "total_autosome_grm.grm.bin")
GRM_ID = os.path.join(DATA_DIR, "total_autosome_grm.grm.id")

N_SAMPLES = 2002
N_SNPS = 9728

# 选择性状
TRAIT_NAME = "Biochem.HDL"
PHENO_FILE = os.path.join(DATA_DIR, "phenotypes", f"{TRAIT_NAME}.resid")

# 模型配置
SNP_VAE_CONFIG = dict(
    d_snp=64, beta=0.001, lr=5e-4, epochs=200, batch_size=512,
    patience=50, kl_warmup_epochs=150, hidden_dim=1024,
)
VGAE_CONFIG = dict(
    k_neighbors=30, d_snp=64, d_hidden=128, d_z=16,
    dropout=0.5, mlp_hidden=64, lr=1e-3, alpha=0.1, beta_kl=0.01,
    max_epochs=200, patience=30, weight_decay=1e-5, kl_warmup_epochs=30,
)

# 归因配置
TOP_K = 100                    # 输出 Top-K SNP
N_PERM_ATTRIB = 200            # 归因置换次数 (null 分布)
BONFERRONI_ALPHA = 0.05        # 显著性水平


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


def load_snp_info(bim_file):
    """从 BIM 文件加载 SNP 信息 (chr, bp, name)。"""
    snp_info = []
    with open(bim_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                snp_info.append({
                    "chr": parts[0],
                    "name": parts[1],
                    "bp": int(parts[3]),
                })
    return snp_info


def train_and_attribute(genotype, snp_emb, grm, labels, device):
    """训练 5-fold VGAE 并计算跨折 SNP 归因。"""
    N = len(labels)
    masks = generate_kfold_masks(N, k_folds=5, seed=42)

    builder = GraphBuilder()
    adj_norm, adj_raw = builder.knn_graph_builder(grm, k=VGAE_CONFIG["k_neighbors"])
    node_features = builder.build_node_features(genotype, snp_emb)
    labels_tensor = torch.from_numpy(labels).float()

    fold_models = []
    fold_scores = []
    all_y_pred = np.zeros(N)
    all_y_count = np.zeros(N)

    for fold_idx, (train_mask, val_mask, test_mask) in enumerate(masks):
        print(f"\n  Fold {fold_idx+1}/5")

        model = VGAEModel(
            d_snp=VGAE_CONFIG["d_snp"], d_hidden=VGAE_CONFIG["d_hidden"],
            d_z=VGAE_CONFIG["d_z"], dropout=VGAE_CONFIG["dropout"],
            mlp_hidden=VGAE_CONFIG["mlp_hidden"],
        )

        trainer = VGAETrainer(
            model=model, adj_norm=adj_norm, adj_raw=adj_raw,
            node_features=node_features, labels=labels_tensor,
            train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
            lr=VGAE_CONFIG["lr"], alpha=VGAE_CONFIG["alpha"],
            beta=VGAE_CONFIG["beta_kl"], weight_decay=VGAE_CONFIG["weight_decay"],
            kl_warmup_epochs=VGAE_CONFIG["kl_warmup_epochs"],
            patience=VGAE_CONFIG["patience"], max_epochs=VGAE_CONFIG["max_epochs"],
            device=device, progress=True,
        )

        result = trainer.train()
        fold_models.append(result["model"])

        # 测试集预测
        test_idx = np.where(test_mask.numpy())[0]
        model.eval()
        with torch.no_grad():
            out = model(node_features.to(device), adj_norm.to(device))
            y_pred = out["y_pred"].cpu().numpy()
        all_y_pred[test_idx] = y_pred[test_idx]
        all_y_count[test_idx] = 1

        # SNP 归因
        print(f"  计算 SNP 归因...")
        scores = compute_snp_scores(
            model=model, node_features=node_features, adj_norm=adj_norm,
            genotype=genotype, snp_embeddings=snp_emb,
            aggregate="sum", device=device,
        )
        fold_scores.append(scores)
        print(f"  Top SNP score: {scores.abs().max():.4f}")

    # 跨折聚合
    agg_scores = aggregate_cv_scores(fold_scores, aggregate="mean")

    return all_y_pred, agg_scores, fold_models


def compute_attribution_null(genotype, snp_emb, grm, labels, model,
                             adj_norm, node_features, n_perm, device):
    """通过置换表型计算归因 null 分布。"""
    rng = np.random.RandomState(99)
    N = len(labels)
    M = genotype.shape[0]

    # 记录每个 SNP 在置换中的最大 |score|
    null_max_scores = []

    for i in range(n_perm):
        perm_labels = rng.permutation(labels)
        perm_labels_t = torch.from_numpy(perm_labels).float()

        # 快速训练 (减少 epochs)
        perm_model = VGAEModel(
            d_snp=VGAE_CONFIG["d_snp"], d_hidden=VGAE_CONFIG["d_hidden"],
            d_z=VGAE_CONFIG["d_z"], dropout=VGAE_CONFIG["dropout"],
            mlp_hidden=VGAE_CONFIG["mlp_hidden"],
        )
        train_mask = torch.ones(N, dtype=torch.bool)
        val_mask = torch.zeros(N, dtype=torch.bool)
        val_mask[:N//5] = True
        train_mask[:N//5] = False

        trainer = VGAETrainer(
            model=perm_model, adj_norm=adj_norm, adj_raw=adj_norm,
            node_features=node_features, labels=perm_labels_t,
            train_mask=train_mask, val_mask=val_mask, test_mask=val_mask,
            lr=1e-3, alpha=0.1, beta=0.01, weight_decay=1e-5,
            kl_warmup_epochs=10, patience=10, max_epochs=50,
            device=device, progress=False,
        )
        trainer.train()

        scores = compute_snp_scores(
            model=perm_model, node_features=node_features, adj_norm=adj_norm,
            genotype=genotype, snp_embeddings=snp_emb,
            aggregate="sum", device=device,
        )
        null_max_scores.append(scores.abs().max().item())

        if (i + 1) % 20 == 0:
            print(f"    置换 {i+1}/{n_perm}")

    return np.array(null_max_scores)


# ============================================================================
# 主流程
# ============================================================================
def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 70)
    print(f"VGAE 推理 + SNP 归因 — {TRAIT_NAME}")
    print("=" * 70)

    # 加载数据
    print("\n[1] 加载数据...")
    genotype = genotype_read(BED_FILE, N_SAMPLES, N_SNPS)
    grm = grm_reader(GRM_BIN, N_SAMPLES)
    grm_ids = grm_id_reader(GRM_ID)
    snp_info = load_snp_info(BIM_FILE)
    labels = load_phenotype(PHENO_FILE, grm_ids)

    valid = ~np.isnan(labels)
    idx = np.where(valid)[0]
    geno_sub = genotype[:, idx]
    grm_sub = grm[np.ix_(idx, idx)]
    lab_sub = labels[idx]
    iids = [grm_ids[i][1] for i in idx]
    n = len(idx)
    print(f"  有效样本: {n}, SNPs: {geno_sub.shape[0]}")

    # SNP-VAE
    print("\n[2] SNP-VAE 预训练...")
    t0 = time.time()
    _, snp_emb = train_snp_vae(genotype=geno_sub, device=device,
                                progress=True, **SNP_VAE_CONFIG)
    print(f"  完成 ({time.time()-t0:.0f}s)")

    # 训练 + 归因
    print("\n[3] VGAE 5-fold 训练 + SNP 归因...")
    t0 = time.time()
    y_pred, snp_scores, fold_models = train_and_attribute(
        geno_sub, snp_emb, grm_sub, lab_sub, device
    )
    print(f"  完成 ({time.time()-t0:.0f}s)")

    # 预测性能
    metrics = evaluate_all(y_pred, lab_sub)
    print(f"\n  预测性能: R²={metrics['r2']:.4f}, R={metrics['r']:.4f}, "
          f"MSE={metrics['mse']:.4f}")

    # 显著性评估
    print(f"\n[4] 归因显著性 (置换 null, n={N_PERM_ATTRIB})...")
    builder = GraphBuilder()
    adj_norm, _ = builder.knn_graph_builder(grm_sub, k=VGAE_CONFIG["k_neighbors"])
    node_features = builder.build_node_features(geno_sub, snp_emb)

    t0 = time.time()
    null_max = compute_attribution_null(
        geno_sub, snp_emb, grm_sub, lab_sub, fold_models[0],
        adj_norm, node_features, N_PERM_ATTRIB, device
    )
    print(f"  完成 ({time.time()-t0:.0f}s)")
    print(f"  Null max|score|: {null_max.mean():.4f} ± {null_max.std():.4f}")

    # Bonferroni 校正阈值
    bonf_threshold = np.percentile(null_max, 100 * (1 - BONFERRONI_ALPHA / N_PERM_ATTRIB))
    print(f"  Bonferroni 阈值 (α={BONFERRONI_ALPHA}): {bonf_threshold:.4f}")

    # 输出结果
    print(f"\n[5] 输出结果...")
    out_dir = os.path.join(RESULT_DIR, TRAIT_NAME)
    os.makedirs(out_dir, exist_ok=True)

    # (a) 预测值
    pred_path = os.path.join(out_dir, "predictions.tsv")
    with open(pred_path, "w") as f:
        f.write("IID\ty_true\ty_pred\tresidual\n")
        for i in range(n):
            f.write(f"{iids[i]}\t{lab_sub[i]:.6f}\t{y_pred[i]:.6f}\t"
                    f"{lab_sub[i]-y_pred[i]:.6f}\n")
    print(f"  ✓ {pred_path}")

    # (b) SNP 归因表
    M = len(snp_scores)
    abs_scores = snp_scores.abs().numpy()
    ranks = np.argsort(-abs_scores) + 1  # 降序排名
    rank_arr = np.empty_like(ranks)
    rank_arr[np.argsort(-abs_scores)] = np.arange(1, M + 1)

    # p-value: 置换中 max|score| >= 当前 |score| 的比例
    p_values = np.array([
        (np.sum(null_max >= abs_scores[j]) + 1) / (N_PERM_ATTRIB + 1)
        for j in range(M)
    ])
    p_bonf = np.minimum(p_values * M, 1.0)  # Bonferroni

    attr_path = os.path.join(out_dir, "snp_attribution.tsv")
    with open(attr_path, "w") as f:
        f.write("SNP_IDX\tCHR\tSNP_NAME\tBP\tScore\tAbs_Score\tRank\tP_perm\tP_bonf\tSignificant\n")
        for j in range(M):
            info = snp_info[j] if j < len(snp_info) else {"chr": "?", "name": f"SNP_{j}", "bp": 0}
            sig = "YES" if abs_scores[j] >= bonf_threshold else ""
            f.write(f"{j}\t{info['chr']}\t{info['name']}\t{info['bp']}\t"
                    f"{snp_scores[j].item():.6f}\t{abs_scores[j]:.6f}\t"
                    f"{rank_arr[j]}\t{p_values[j]:.6f}\t{p_bonf[j]:.6f}\t{sig}\n")
    print(f"  ✓ {attr_path}")

    # (c) Top-K SNP
    top_path = os.path.join(out_dir, "top_snps.tsv")
    top_result = top_snps(snp_scores, top_k=TOP_K,
                          snp_names=[s["name"] for s in snp_info] if snp_info else None)
    with open(top_path, "w") as f:
        f.write("Rank\tSNP_IDX\tCHR\tSNP_NAME\tBP\tScore\tAbs_Score\tP_bonf\n")
        for rank_i, (idx_j, sc) in enumerate(
            zip(top_result["indices"], top_result["scores"]), 1
        ):
            info = snp_info[idx_j] if idx_j < len(snp_info) else {"chr": "?", "name": f"SNP_{idx_j}", "bp": 0}
            f.write(f"{rank_i}\t{idx_j}\t{info['chr']}\t{info['name']}\t"
                    f"{info['bp']}\t{sc:.6f}\t{abs(sc):.6f}\t{p_bonf[idx_j]:.6f}\n")
    print(f"  ✓ {top_path}")

    # (d) 汇总 JSON
    n_sig = int(np.sum(abs_scores >= bonf_threshold))
    summary = {
        "trait": TRAIT_NAME,
        "n_samples": n,
        "n_snps": M,
        "prediction": {k: float(v) for k, v in metrics.items()},
        "attribution": {
            "n_significant": n_sig,
            "bonferroni_threshold": float(bonf_threshold),
            "null_max_mean": float(null_max.mean()),
            "null_max_std": float(null_max.std()),
            "top_snp_score": float(abs_scores.max()),
            "top_snp_idx": int(np.argmax(abs_scores)),
            "top_snp_name": snp_info[int(np.argmax(abs_scores))]["name"] if snp_info else None,
        },
        "config": {"vgae": VGAE_CONFIG, "snp_vae": SNP_VAE_CONFIG},
    }
    summary_path = os.path.join(out_dir, "model_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {summary_path}")

    # 打印 Top-10
    print(f"\n{'='*70}")
    print(f"Top-10 SNP (by |Score|)")
    print(f"{'='*70}")
    print(f"{'Rank':>4s}  {'SNP':>14s}  {'CHR':>4s}  {'BP':>10s}  {'Score':>10s}  {'P_bonf':>10s}")
    print("-" * 60)
    for rank_i in range(min(10, TOP_K)):
        idx_j = top_result["indices"][rank_i]
        sc = top_result["scores"][rank_i]
        info = snp_info[idx_j] if idx_j < len(snp_info) else {"chr": "?", "name": f"SNP_{idx_j}", "bp": 0}
        print(f"{rank_i+1:>4d}  {info['name']:>14s}  {info['chr']:>4s}  "
              f"{info['bp']:>10d}  {sc:>+10.4f}  {p_bonf[idx_j]:>10.4f}")

    print(f"\n  显著 SNP 数 (Bonferroni α={BONFERRONI_ALPHA}): {n_sig}/{M}")
    print(f"\n{'='*70}")
    print("完成!")


if __name__ == "__main__":
    main()
