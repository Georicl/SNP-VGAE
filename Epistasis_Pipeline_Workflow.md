# 🧬 Epistasis Mining Pipeline (GNN + LMM) 工程标准流程指南

本文档总结了基于图神经网络 (GNN) 与线性混合模型 (LMM) 的高纯度上位效应挖掘流水线的完整工程规范、技术栈与执行流程。可作为新项目、新数据集上的复现与部署参考手册。

## 🛠 一、 核心技术栈 (Tech Stack)

*   **编程语言**: Python 3.x
*   **深度学习框架**: PyTorch (支持 CUDA / 苹果 MPS 硬件加速)
*   **图机器学习库**: PyTorch Geometric (PyG)
*   **数据处理与 I/O**: Pandas, NumPy, PyArrow (Parquet 引擎)
*   **算法加速实现**: 张量化 (Tensorized) Pearson 相关系数计算, 矩阵乘法加速 (LMM / GRM)
*   **存储格式**: Parquet (高压缩比、极速加载、强类型推断)

---

## 📂 二、 输入数据标准规范 (Input Data Spec)

所有数据必须经过质控清洗，并统一存放在 `src/preprocessed_data/` (或配置的数据目录) 下，采用 `.parquet` 格式。

### 1. `geno.parquet` (基因型剂量矩阵)
*   **格式**: 样本 x SNP 矩阵。
*   **列定义**: 
    *   第 1 列: `id` (样本唯一标识，必须为字符串格式，与表型严格对齐)。
    *   第 2~N 列: SNP 名称 (如 `SNP1`, `SNP2`)。
*   **值**: Float32 类型的填充剂量 (Imputed Dosages)，非离散值。

### 2. `pheno.parquet` (表型与协变量)
*   **列定义**:
    *   `id`: 样本唯一标识 (与 geno 对应)。
    *   `round`: 批次/群体分层特征 (用于图构建时的子群体物理切分依据)。
    *   `cageid` (等): 环境协变量。
    *   `bw0` (等): 连续性状目标表型。

### 3. `map.parquet` (SNP 物理映射表)
*   **列定义**:
    *   `id`: SNP 名称 (与 geno 列名对应)。
    *   `chr`: 染色体编号。
    *   `pos`: 物理位置坐标 (bp 级别)。

---

## 🚀 三、 端到端执行流程 (End-to-End Workflow)

整个流水线分为四个标准操作步骤，必须按顺序执行。

### 步骤 1: 全局去噪与表型残差化 (Phase 1)
**目标**: 使用 LMM 消除加性遗传背景与环境批次效应。

1.  **核心逻辑**: 
    *   调用 `src/GRM.py` (VanRaden 或 Yang 算法) 依托 GPU/MPS 计算 N x N Kinship 矩阵。
    *   调用 `src/REML.py` 拟合 LMM，分离随机效应 (加性多基因) 与固定效应 (环境协变量)。
2.  **执行命令**:
    ```bash
    python RunResiduals.py
    ```
3.  **输出文件**: `src/preprocessed_data/residuals.parquet` (包含剥离噪音后的纯净表型残差)。

### 步骤 2: 局部物理隔离与独立建图 (Phase 2)
**目标**: 构建无结构性伪阳性的图结构数据。

1.  **核心逻辑**:
    *   根据 `round` 特征切分群体 (`src/SplitePopulation.py`)。
    *   计算子群内 SNP 间的连锁不平衡 (LD, $R^2$) (`src/LD.py`)，受 `window_size_kb` 物理距离约束。
    *   转化为图结构: 节点特征(剂量向量), 边($R^2 \ge 0.2$), 全局图标签(表型残差)。
2.  **执行机制**: 
    *   该步骤集成在 `GraphConstructor.py` 中，通常在 `Train.py` 运行时自动在内存中构建，或者通过调用 `GraphConstructor` 预先缓存为 `pyg_dataset.pt`。

### 步骤 3: GNN 模型训练 (Phase 3)
**目标**: 提取高阶非线性互作信号。

1.  **核心逻辑**:
    *   加载 PyG 图数据。
    *   实例化 `src/Model.py` 中的 `GAT_GWAS` 模型 (Graph Attention Network)。
    *   执行带有 `ReduceLROnPlateau` 调度和 Early Stopping 的训练循环 (`src/Train.py`)。
2.  **执行命令**:
    ```bash
    python src/Train.py
    ```
3.  **输出文件**: `src/best_gat_model.pth` (收敛后的最优模型权重)。

### 步骤 4: 注意力权重解析与上位效应定位 (Phase 4)
**目标**: 解释黑盒模型，定位上位效应的 SNP 对。

1.  **核心逻辑**:
    *   加载训练好的模型并进入 `eval()` 模式。
    *   利用 `src/Interpret.py` 提取 GAT 层中的 Attention 权重 ($\alpha$)。
    *   计算每条边的平均注意力强度，映射回 SNP 物理位置与距离。
2.  **执行命令**:
    ```bash
    python src/Interpret.py
    ```
3.  **输出文件**: `src/attention_weights.parquet`。

---

## 📊 四、 结果分析与下游拓展 (Downstream Analysis)

最终产出的 `attention_weights.parquet` 包含以下核心字段：
`snp_a`, `snp_b`, `chr_a`, `pos_a`, `chr_b`, `pos_b`, `avg_attention`, `dist` (物理距离)。

**推荐的下游分析**:
1.  **过滤与筛选**: 提取 `dist == -1` (跨染色体互作) 或 `dist > 1MB` 且 `avg_attention` 极高 (Top 0.1%) 的 SNP 对。
2.  **可视化**: 绘制 2D 曼哈顿图或弦图 (Circos Plot) 展示全局上位互作网络。
3.  **功能注释**: 将核心 SNP 映射到基因组注释文件，进行 Pathway Enrichment 分析 (验证是否为同通路互作)。

---
**📝 注意事项 (Notes):**
*   **硬件显存**: `GRM` 和 `LD` 计算对显存有一定要求。若数据集激增，可调整 `src/LD.py` 中转化为 Float16 数据类型，或减小 Batch Size。
*   **超参数调节**: 切换到新表型时，请在 `Train.py` 中调整 Learning Rate (默认 0.001) 以及 `Model.py` 中的 `hidden_dim` (默认 16)。