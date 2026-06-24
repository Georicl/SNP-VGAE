# GRM 实现验证流程

本文档描述如何验证 Eigen 实现的 Yang et al. (2010) GRM 算法与 GCTA 的一致性。

## 验证策略

使用相同的小规模测试数据,分别用两种方法计算 GRM,然后对比结果矩阵。

### 方法对比

| 方法 | 工具 | 平台 |
|------|------|------|
| **Eigen 实现** | 本项目 C++ 代码 | macOS (Apple Silicon) |
| **GCTA** | gcta64 --make-grm | Linux Xeon 服务器 |

---

## 步骤 1: 生成测试数据

在本地 Mac 上运行:

```bash
cd /Users/georicl/Documents/python_program/GAT

# 生成小规模测试数据 (50 样本 × 100 SNPs)
python scripts/generate_test_data.py \
    --output test_data/small_test \
    --samples 50 \
    --snps 100
```

输出文件:
- `test_data/small_test.bed` - PLINK 二进制基因型
- `test_data/small_test.bim` - SNP 信息
- `test_data/small_test.fam` - 样本信息
- `test_data/small_test_genotypes.txt` - 文本格式(用于验证)
- `test_data/small_test_samples.txt` - 样本 ID 列表

---

## 步骤 2: 在 Linux 服务器上运行 GCTA

将测试数据传输到 Linux 服务器:

```bash
# 从 Mac 上传到服务器
scp test_data/small_test.* user@linux-server:/path/to/work/

# SSH 登录服务器
ssh user@linux-server
cd /path/to/work/

# 运行 GCTA
gcta64 --bfile small_test --make-grm --out gcta_result

# 下载结果回 Mac
scp gcta_result.* user@mac:/Users/georicl/Documents/python_program/GAT/test_data/
```

GCTA 输出文件:
- `gcta_result.grm` - GRM 矩阵(三元组格式)
- `gcta_result.grm.id` - 样本 ID
- `gcta_result.grm.N.bin` - SNP 数量(二进制)

---

## 步骤 3: 用 Eigen 实现计算 GRM

### 选项 A: 使用单元测试数据(快速验证)

```bash
cd /Users/georicl/Documents/python_program/GAT/build
./tests/test_grm_export test_data/small_test test_data/eigen_result
```

这会生成:
- `test_data/eigen_result.json` - JSON 格式 GRM(便于调试)
- `test_data/eigen_result.grm.bin` - 自定义二进制格式

### 选项 B: 实现完整 PLINK 加载器(推荐长期方案)

需要在 `test_grm_export.cpp` 中添加 PLINK 文件读取功能:

```cpp
// TODO: 实现 read_plink_bed(), read_plink_bim(), read_plink_fam()
GenotypeData loadPlinkData(const std::string& prefix) {
    // 1. 读取 .fam 文件获取样本 ID
    // 2. 读取 .bim 文件获取 SNP 信息
    // 3. 读取 .bed 文件获取基因型矩阵
    // 返回 GenotypeData 结构体
}
```

---

## 步骤 4: 对比结果

```bash
cd /Users/georicl/Documents/python_program/GAT

python scripts/compare_grm.py \
    --eigen test_data/eigen_result.json \
    --gcta test_data/gcta_result
```

### 预期输出示例

```
============================================================
GRM Comparison: Eigen Implementation vs GCTA
============================================================
Matrix shape: 50 x 50
Total elements: 2500

--- Difference Statistics ---
Max absolute difference: 1.23e-14
Mean absolute difference: 3.45e-16
Median absolute difference: 2.10e-16
Std of differences: 8.90e-16

Symmetry check:
  Eigen Implementation max asymmetry: 1.11e-16
  GCTA max asymmetry: 2.22e-16

Pearson correlation: 1.0000000000

Elements within tolerance (1e-10):
  2500/2500 (100.00%)

✓ All elements within tolerance (1e-06)
```

---

## 验证标准

### ✅ 通过标准

- **最大绝对误差** < 1e-10 (浮点精度范围)
- **平均绝对误差** < 1e-14
- **Pearson 相关系数** > 0.9999999999
- **100% 元素**在 1e-10 容差内

### ⚠️ 需要调查的情况

- 如果有任何元素差异 > 1e-6
- 如果对角线元素不一致
- 如果对称性检查失败

### ❌ 失败的常见原因

1. **p 值计算方式不同**
   - Eigen: `p = sum(X) / (2 * valid_count)` (编码等位基因频率)
   - GCTA: 可能使用次等位基因频率?

2. **缺失值处理不同**
   - Eigen: NA → 2p (均值填充)
   - GCTA: 可能有不同的处理方式

3. **归一化分母不同**
   - Eigen: `sum(scale²) = sum(2*p*(1-p))` (Yang 标准)
   - VanRaden: `M` (SNP 数量)

---

## 调试技巧

### 检查中间变量

在 `GRMBuilder.cpp` 中添加调试输出:

```cpp
// 打印前 5 个 SNP 的 p 值
std::cout << "Allele frequencies (first 5 SNPs):" << std::endl;
for (int k = 0; k < std::min(5, M); ++k) {
    std::cout << "  SNP " << k << ": p=" << p(k)
              << ", scale=" << scale(k) << std::endl;
}

// 打印 Z 矩阵的部分内容
std::cout << "Z matrix (first 3x3):" << std::endl;
std::cout << Z.block(0, 0, 3, 3) << std::endl;
```

### 手动计算验证

对于超小数据集(3 样本 × 1 SNP),可以手工计算期望值:

```
基因型: [0, 1, 2]
p = (0+1+2)/(3*2) = 0.5
scale² = 2*0.5*0.5 = 0.5
Z = (X - 1) / sqrt(0.5) = [-1.414, 0, 1.414]
G = Z*Z^T / 0.5 = [[4, 0, -4], [0, 0, 0], [-4, 0, 4]]
```

---

## 常见问题

### Q1: 为什么会有微小差异?

**A**: 浮点运算顺序不同导致的舍入误差是正常的。只要差异 < 1e-10,就可以认为实现一致。

### Q2: GCTA 使用了多线程,会影响结果吗?

**A**: 理论上不会,但浮点加法的并行归约可能导致微小的数值差异。这仍然是可接受的。

### Q3: 如果差异很大怎么办?

**A**:
1. 检查是否使用了相同的算法(Yang vs VanRaden)
2. 确认缺失值处理方式一致
3. 验证等位基因频率计算是否正确
4. 检查矩阵转置和索引顺序

---

## 下一步

验证通过后:
1. ✅ 确认 Eigen 实现正确
2. ✅ 可以在生产环境中使用
3. ✅ 可以继续开发下游分析(LMM、GWAS 等)

如果发现错误:
1. 🔍 定位差异来源
2. 🐛 修复实现
3. 🔄 重新运行验证
