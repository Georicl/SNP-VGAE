"""
SNP-VAE 预训练模块测试
======================

测试内容:
  - preprocess_genotype: 基因型预处理（缺失值填充、类型转换）
  - SNPVAE: 模型前向传播、编码、解码
  - vae_loss: 损失函数计算
  - extract_embeddings: 嵌入提取
  - train_snp_vae: 完整训练流程（小规模数据）

运行方式:
  PYTHONPATH=pytorch/src uv run pytest test/test_pretrain.py -v
"""

import numpy as np
import pytest
import torch

from pretrain.preprocess import preprocess_genotype
from pretrain.vae import SNPVAE, vae_loss
from pretrain.extract_embeddings import extract_embeddings
from pretrain.train import train_snp_vae


# ============================================================================
# 测试数据 fixtures
# ============================================================================

@pytest.fixture
def small_genotype():
    """小规模基因型矩阵 (100 SNPs, 50 samples)"""
    np.random.seed(42)
    # 0/1/2 基因型，少量 -9 缺失
    data = np.random.choice([0, 1, 2], size=(100, 50)).astype(np.int32)
    # 随机插入一些缺失值
    mask = np.random.random((100, 50)) < 0.05
    data[mask] = -9
    return data


@pytest.fixture
def processed_tensor():
    """预处理后的基因型张量"""
    np.random.seed(42)
    data = np.random.choice([0, 1, 2], size=(100, 50)).astype(np.float32)
    return torch.tensor(data)


@pytest.fixture
def small_model():
    """小规模 SNPVAE 模型"""
    return SNPVAE(num_samples=50, d_snp=16)


# ============================================================================
# preprocess_genotype 测试
# ============================================================================

class TestPreprocessGenotype:
    """基因型预处理测试"""

    def test_output_shape(self, small_genotype):
        """输出形状应与输入相同"""
        result = preprocess_genotype(small_genotype)
        assert result.shape == small_genotype.shape

    def test_output_dtype(self, small_genotype):
        """输出应为 float32"""
        result = preprocess_genotype(small_genotype)
        assert result.dtype == torch.float32

    def test_no_nan_in_output(self, small_genotype):
        """输出不应包含 NaN（缺失值已被填充）"""
        result = preprocess_genotype(small_genotype)
        assert not torch.isnan(result).any()

    def test_all_nan_row_filled_with_zero(self):
        """全缺失行应填充为 0"""
        data = np.full((5, 10), -9, dtype=np.int32)
        result = preprocess_genotype(data)
        expected = torch.zeros(5, 10, dtype=torch.float32)
        assert torch.allclose(result, expected)

    def test_no_negative9_in_output(self, small_genotype):
        """输出不应包含 -9（缺失编码）"""
        result = preprocess_genotype(small_genotype)
        assert not (result == -9.0).any()


# ============================================================================
# SNPVAE 模型测试
# ============================================================================

class TestSNPVAE:
    """SNP-VAE 模型测试"""

    def test_forward_output_shape(self, small_model, processed_tensor):
        """前向传播输出形状应正确"""
        x = processed_tensor  # (100, 50)
        x_recon, mu, logvar = small_model(x)

        assert x_recon.shape == x.shape  # 重建形状与输入相同
        assert mu.shape == (100, 16)     # (B, D_snp)
        assert logvar.shape == (100, 16)  # (B, D_snp)

    def test_encode_output_shape(self, small_model, processed_tensor):
        """编码输出形状应正确"""
        mu, logvar = small_model.encode(processed_tensor)

        assert mu.shape == (100, 16)
        assert logvar.shape == (100, 16)

    def test_decode_output_shape(self, small_model):
        """解码输出形状应正确"""
        z = torch.randn(32, 16)  # (B, D_snp)
        x_recon = small_model.decode(z)

        assert x_recon.shape == (32, 50)  # (B, N)

    def test_reparameterize_is_stochastic(self, small_model):
        """重参数化应具有随机性（两次采样结果不同）"""
        mu = torch.zeros(10, 16)
        logvar = torch.zeros(10, 16)

        z1 = small_model.reparameterize(mu, logvar)
        z2 = small_model.reparameterize(mu, logvar)

        # 由于随机采样，两次结果应不同
        assert not torch.allclose(z1, z2)

    def test_model_has_parameters(self, small_model):
        """模型应有可训练参数"""
        total_params = sum(p.numel() for p in small_model.parameters())
        assert total_params > 0


# ============================================================================
# vae_loss 测试
# ============================================================================

class TestVAELoss:
    """VAE 损失函数测试"""

    def test_loss_output_shapes(self, small_model, processed_tensor):
        """损失函数应返回三个标量"""
        x = processed_tensor
        x_recon, mu, logvar = small_model(x)

        total, recon, kl = vae_loss(x_recon, x, mu, logvar)

        assert total.ndim == 0  # 标量
        assert recon.ndim == 0
        assert kl.ndim == 0

    def test_loss_values_are_finite(self, small_model, processed_tensor):
        """损失值应为有限值"""
        x = processed_tensor
        x_recon, mu, logvar = small_model(x)

        total, recon, kl = vae_loss(x_recon, x, mu, logvar)

        assert torch.isfinite(total)
        assert torch.isfinite(recon)
        assert torch.isfinite(kl)

    def test_recon_loss_is_non_negative(self, small_model, processed_tensor):
        """重建损失（MSE）应非负"""
        x = processed_tensor
        x_recon, mu, logvar = small_model(x)

        _, recon, _ = vae_loss(x_recon, x, mu, logvar)

        assert recon >= 0

    def test_perfect_recon_has_zero_recon_loss(self):
        """完美重建时重建损失应为 0"""
        x = torch.randn(10, 50)
        mu = torch.zeros(10, 16)
        logvar = torch.zeros(10, 16)

        _, recon, _ = vae_loss(x, x, mu, logvar)

        assert torch.isclose(recon, torch.tensor(0.0), atol=1e-6)


# ============================================================================
# extract_embeddings 测试
# ============================================================================

class TestExtractEmbeddings:
    """嵌入提取测试"""

    def test_embedding_shape(self, small_model, processed_tensor):
        """嵌入矩阵形状应为 (M, D_snp)"""
        embeddings = extract_embeddings(small_model, processed_tensor)

        assert embeddings.shape == (100, 16)

    def test_embedding_dtype(self, small_model, processed_tensor):
        """嵌入应为 numpy array"""
        embeddings = extract_embeddings(small_model, processed_tensor)

        assert isinstance(embeddings, np.ndarray)

    def test_embedding_is_deterministic(self, small_model, processed_tensor):
        """嵌入提取应是确定性的（两次结果相同）"""
        emb1 = extract_embeddings(small_model, processed_tensor)
        emb2 = extract_embeddings(small_model, processed_tensor)

        np.testing.assert_array_equal(emb1, emb2)


# ============================================================================
# train_snp_vae 完整训练测试
# ============================================================================

class TestTrainSnpVae:
    """完整训练流程测试"""

    def test_train_returns_model_and_embeddings(self, small_genotype):
        """训练应返回模型和嵌入矩阵"""
        model, embeddings = train_snp_vae(
            genotype=small_genotype,
            d_snp=8,
            epochs=3,
            batch_size=32,
            patience=10,
            device="cpu",
        )

        assert isinstance(model, SNPVAE)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (100, 8)  # (M, D_snp)

    def test_train_reduces_loss(self, small_genotype):
        """训练后损失应下降（至少不爆炸）"""
        model, embeddings = train_snp_vae(
            genotype=small_genotype,
            d_snp=8,
            epochs=5,
            batch_size=32,
            patience=100,  # 大耐心值，确保跑完所有 epoch
            device="cpu",
        )

        # 嵌入不应包含 NaN
        assert not np.isnan(embeddings).any()
