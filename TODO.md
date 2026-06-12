# 🧬 GAT (Genotype-Attention-Trait) 上位效应挖掘流水线

## 一、 核心输入数据标准 (Standard Input Spec)

本项目直接处理原始分型数据，采用整数加性编码（Additive Encoding），统一存放在 `data/processed/`。

### 1. `geno.parquet` (基因型整数矩阵)
*   **结构**：样本(Row) x 位点(Column)。
*   **编码**：`int8` 类型。
    *   `0`: 纯合主等位 (Major Hom)
    *   `1`: 杂合 (Het)
    *   `2`: 纯合次等位 (Minor Hom)
    *   `-1`: 缺失数据 (Missing)
*   **对齐**：行索引必须与 `pheno.parquet` 的 `id` 严格一致。

### 2. `pheno.parquet` (表型与协变量)
*   **关键列**：
    *   `id`: 样本标识（FID_IID 组合）。
    *   `round`: 群体/批次划分标签（建图的关键依据）。
    *   `target`: 连续型表型残差或原始值。

### 3. `map.parquet` (位点元数据)
*   **关键列**：`id` (SNP名称), `chr`, `pos`。

---

## 二、 核心开发阶段 (Development Phases)

### 阶段 0: 数据解析与质控 (Pre-processing)
- [ ] **解析 .data 文件**：编写高性能解析器，将双列碱基转换为 0/1/2 编码。
- [ ] **缺失值预处理**：对 `-1` 进行简单填充（如位点众数填充）或保留待 LMM 处理。
- [ ] **坐标映射**：整合 `mapfile.txt` 生成标准的 `map.parquet`。

### 阶段 1: 线性降噪 (LMM Residualization)
*目标：通过计算 Kinship 矩阵排除加性遗传背景。*
- [ ] **GRM 构建**：利用 `geno` 矩阵计算 VanRaden 亲缘关系矩阵。
- [ ] **残差计算**：拟合混合线性模型，输出剥离加性效应后的 `residuals`。

### 阶段 2: 动态建图 (Graph Construction)
*目标：基于物理距离和 LD 构建 SNP 互作图。*
- [ ] **群体隔离**：按 `round` 标签切分样本子集。
- [ ] **LD 滑动窗口**：在窗口内计算 $R^2$ 边，构建拓扑结构。

### 阶段 3: GNN 模型开发 (GAT Training)
*目标：利用注意力机制捕捉高阶互作。*
- [ ] **模型设计**：实现多层 GAT 模型，输入为 SNP 编码（训练时转为 Float），输出为预测残差。
- [ ] **训练循环**：支持 Early Stopping 和学习率调度。

### 阶段 4: 权重解释与定位 (Interpretation)
- [ ] **Attention 提取**：解析多头注意力权重。
- [ ] **显著性筛选**：定位具有高注意力的跨染色体/长程 SNP 对。

---

## 三、 硬件优化目标
- [ ] **MPS 加速**：在苹果 M 系列芯片上实现张量化矩阵运算。
- [ ] **零拷贝 I/O**：通过 Parquet 映射直接加载到内存。
