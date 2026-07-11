"""
M1: SNP-VAE 训练流程
====================

将预处理、模型创建、训练循环、早停、嵌入提取串联为完整训练流程。

数据流:
  genotype (M, N) → preprocess → 模型训练 → 早停恢复最佳权重 → 提取嵌入 (M, D_snp)

可扩展性:
  - 后续可加入学习率调度器（ReduceLROnPlateau）
  - 可支持从 checkpoint 恢复训练
  - 可加入 TensorBoard/WandB 日志记录
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from pretrain.preprocess import preprocess_genotype
from pretrain.vae import SNPVAE, vae_loss
from pretrain.extract_embeddings import extract_embeddings


def train_snp_vae(
    genotype: np.ndarray,
    d_snp: int = 64,
    beta: float = 0.01,
    lr: float = 1e-3,
    epochs: int = 200,
    batch_size: int = 512,
    patience: int = 20,
    device: str = "cpu",
    progress: bool = True,
) -> tuple[SNPVAE, np.ndarray]:

    # 1. 数据预处理
    print(f"=====开始[SNP-VAE]=====")
    X = preprocess_genotype(genotype)  # (M, N)
    M, N = X.shape  # shape获取行数和列数(SNPs和samples总数)
    print(f"[SNP-VAE] 数据形状 : {M} SNPs -- {N} samples")
    print(f"[SNP-VAE] SNP嵌入维度: [{d_snp}], β={beta}, lr={lr}")

    # 2. 创建模型
    model: SNPVAE = SNPVAE(num_samples=N, d_snp=d_snp).to(device)
    total_para = sum(p.numel() for p in model.parameters())
    # 打印模型参数总量
    print(f"[SNP-VAE] 模型参数量: {total_para:,}")

    # 3. Adam 自适应优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4.训练
    best_loss = float("inf")
    patience_count = 0  # 初始化计数器

    # 训练进度条（epoch 级别）
    epoch_iter = tqdm(range(epochs), desc="训练轮次", disable=not progress)
    for epoch in epoch_iter:
        # model.train() 启用 Dropout/BatchNorm 的训练模式行为
        model.train()

        indices = torch.randperm(M)
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        n_batches = 0

        for start in range(0, M, batch_size):
            # 取SNP的索引
            end = min(start + batch_size, M)
            batch_idx = indices[start:end]

            # 取出这批SNP的基因型向量 -> (B, N)
            x_batch = X[batch_idx].to(device)

            # 前向传播
            x_recon, mu, logvar = model(x_batch)

            # 损失计算
            loss, recon_l, kl_l = vae_loss(x_recon, x_batch, mu, logvar, beta)

            # 反向传播
            optimizer.zero_grad()  # 清除梯度
            loss.backward()
            optimizer.step()  # 更新参数

            # 累计损失
            epoch_loss += loss.item()
            epoch_recon += recon_l.item()
            epoch_kl += kl_l.item()
            n_batches += 1

        # 计算每个epoch的平均损失
        avg_loss: float = epoch_loss / n_batches
        avg_recon: float = epoch_recon / n_batches
        avg_kl: float = epoch_kl / n_batches

        # 更新进度条显示
        epoch_iter.set_postfix(
            loss=f"{avg_loss:.4f}", recon=f"{avg_recon:.4f}", kl=f"{avg_kl:.4f}")

        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_count = 0

            best_state = {k: v.clone() for k, v, in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"[SNP-VAE] 早停于 epoch {epoch+1}，最佳损失={best_loss:.4f}")
                break

    model.load_state_dict(best_state)

    # 提取SNP嵌入
    snp_embeddings = extract_embeddings(model, X, device=device)
    print(f"[SNP-VAE] 嵌入提取完成: {snp_embeddings.shape}")

    return model, snp_embeddings
