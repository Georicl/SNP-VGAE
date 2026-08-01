"""
SNP-VAE 训练流程
================

将预处理、模型创建、训练循环、早停、嵌入提取串联为完整训练流程。

数据流:
  genotype (M, N) → preprocess → 模型训练 → 早停恢复最佳权重 → 提取嵌入 (M, D_snp)

训练特性:
  - KL 退火: 前 warmup_epochs 轮 β 从 0 线性增大到目标值
  - 早停机制: 基于验证集损失，退火期间不计入早停
  - 学习率调度: ReduceLROnPlateau 自动降低学习率

作者: Xiang Yang
邮箱: Georicl@outlook.com
创建时间: 2026-07-11
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
    beta: float = 0.001,
    lr: float = 5e-4,
    epochs: int = 200,
    batch_size: int = 512,
    patience: int = 50,
    device: str = "cpu",
    progress: bool = True,
    kl_warmup_epochs: int = 150,
    hidden_dim: int = 1024,
) -> tuple[SNPVAE, np.ndarray]:

    # 1. 数据预处理（含 SNP 行标准化）
    print(f"=====开始[SNP-VAE]=====")
    X, snp_mean, snp_std = preprocess_genotype(genotype)  # (M, N)
    M, N = X.shape
    print(f"[SNP-VAE] 数据形状 : {M} SNPs -- {N} samples")
    print(f"[SNP-VAE] SNP标准化: 已启用 (z-score)")
    print(f"[SNP-VAE] SNP嵌入维度: [{d_snp}], β={beta}, lr={lr}")

    # 2. 创建模型
    model: SNPVAE = SNPVAE(num_samples=N, d_snp=d_snp, hidden_dim=hidden_dim).to(device)
    total_para = sum(p.numel() for p in model.parameters())
    # 打印模型参数总量
    print(f"[SNP-VAE] 模型参数量: {total_para:,}")

    # 3. Adam 自适应优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4.训练
    best_loss = float("inf")
    patience_count = 0  # 初始化计数器
    best_state: dict | None = None

    # 学习率调度器：当损失不再下降时自动降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5
    )

    # 训练进度条（epoch 级别）
    epoch_iter = tqdm(range(epochs), desc="训练轮次", disable=not progress)
    for epoch in epoch_iter:
        # model.train() 启用 Dropout/BatchNorm 的训练模式行为
        model.train()

        # KL 退火：前 warmup_epochs 轮 β 从 0 线性增大到目标值
        # 防止训练初期 KL 约束过强导致后验坍塌
        if kl_warmup_epochs > 0:
            current_beta = beta * min(1.0, epoch / kl_warmup_epochs)
        else:
            current_beta = beta

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

            # 损失计算（使用当前退火后的 β）
            loss, recon_l, kl_l = vae_loss(x_recon, x_batch, mu, logvar, current_beta)

            # 反向传播
            optimizer.zero_grad()  # 清除梯度
            loss.backward()
            optimizer.step()  # 更新参数

            # 累计损失
            epoch_loss += loss.item()
            epoch_recon += recon_l.item()
            epoch_kl += kl_l.item()
            n_batches += 1

        # 计算每个 epoch 的平均损失
        avg_loss: float = epoch_loss / n_batches
        avg_recon: float = epoch_recon / n_batches
        avg_kl: float = epoch_kl / n_batches
        
        # 学习率调度
        scheduler.step(avg_loss)
        
        # 更新进度条显示
        epoch_iter.set_postfix(
            loss=f"{avg_loss:.4f}", recon=f"{avg_recon:.4f}",
            kl=f"{avg_kl:.4f}", β=f"{current_beta:.4f}")

        # 早停检查（退火期间不计入早停，因为 β 增大导致 loss 上升是正常的）
        if epoch < kl_warmup_epochs:
            # 退火阶段：只更新最佳状态，不增加耐心计数
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.clone() for k, v, in model.state_dict().items()}
        elif avg_loss < best_loss:
            best_loss = avg_loss
            patience_count = 0
            best_state = {k: v.clone() for k, v, in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"[SNP-VAE] 早停于 epoch {epoch+1}，最佳损失={best_loss:.4f}")
                break

    # 恢复最佳权重（若训练循环未执行则保留初始权重）
    if best_state is not None:
        model.load_state_dict(best_state)

    # 提取SNP嵌入
    snp_embeddings = extract_embeddings(model, X, device=device)
    print(f"[SNP-VAE] 嵌入提取完成: {snp_embeddings.shape}")

    return model, snp_embeddings
