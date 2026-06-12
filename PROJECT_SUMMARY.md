# Project Summary: GAT-GWAS Phenotype Prediction

本项目旨在利用图神经网络（Graph Neural Networks, GNN）结合图注意力机制（Graph Attention Networks, GAT），挖掘基因组中的复杂遗传相互作用（如上位效应），并对表型残差进行高精度预测。

## 1. 模型架构 (Model Architecture)

模型基于 `GAT_GWAS` 类实现，采用分阶段的特征提取与聚合策略。

### 1.1 核心组件
- **图注意力卷积层 (GAT Layers)**:
    - **Layer 1**: 采用多头注意力机制 (`heads=2`)，将 SNP 的剂量特征（0/1/2）映射到隐藏空间。负责学习局部连锁不平衡（LD）区域内的位点交互。
    - **Layer 2**: 单头注意力层，用于捕获跨区域的高阶遗传互作信息。
- **归一化与激活**: 每层卷积后紧跟 `BatchNorm1d` 和 `ELU` 激活函数，通过 `Dropout`（20%）增强模型的泛化能力。
- **全局池化 (Readout)**: 使用 `global_mean_pool` 将所有 SNP 节点的嵌入向量聚合为一个个体级别的特征向量。
- **回归预测头 (MLP)**: 两层全连接层。第一层带激活函数和 Dropout，第二层输出最终预测值。

### 1.2 关键架构参数 (Hyperparameters)
| 参数名 | 取值 | 说明 |
| :--- | :--- | :--- |
| `num_node_features` | 1 | 输入节点特征维度（SNP 剂量） |
| `hidden_dim` | 16 | 隐藏层神经元数量 |
| `heads` | 2 | 第一层 GAT 的注意力头数 |
| `dropout` | 0.2 | 随机失活概率 |

## 2. 训练与优化配置 (Optimization & Training)

### 2.1 优化器架构
- **优化器**: `Adam` (Adaptive Moment Estimation)
- **学习率 (Learning Rate)**: `0.001`
- **权重衰减 (Weight Decay)**: `1e-4` (L2 正则化，防止过拟合)

### 2.2 损失函数与评估
- **损失函数**: `MSELoss` (均方误差)
- **评估指标**: 验证集平均损失 (`avg_val_loss`)。
- **最佳模型选择**: 基于验证集表现自动保存最优模型权重 (`src/best_gat_model.pth`)。

## 3. 完整流程步骤 (Workflow Steps)

1.  **残差计算 (REML)**: 
    - 使用混合线性模型 (LMM) 估计遗传方差和残差方差。
    - 扣除固定效应（如 `round`, `cageid`）后，提取表型残差作为 GNN 的训练目标（Label）。
2.  **图结构构建 (Graph Construction)**:
    - 基于连锁不平衡 (LD) 计算位点间的 $R^2$。
    - 在 100kb 窗口内，将 $R^2 > 0.2$ 的位点通过边连接，构建遗传互作图。
3.  **数据装载 (DataLoader)**:
    - 将每个个体的 SNP 数据封装为 `torch_geometric.data.Data` 对象。
    - 按 8:2 划分训练集和验证集。
4.  **模型迭代**:
    - 前向传播计算预测值 -> 计算 MSE Loss -> 反向传播更新权重。

## 4. 依赖包与技术栈 (Tech Stack)

| 类别 | 库名称 | 作用 |
| :--- | :--- | :--- |
| 深度学习框架 | `torch` (>=2.11.0) | 核心计算与神经网络 |
| 图学习扩展 | `torch-geometric` (>=2.7.0) | 提供 GAT 卷积与图数据处理 |
| 数据处理 | `pandas`, `pyarrow`, `polars` | Parquet 文件读取与矩阵操作 |
| 进度管理 | `tqdm` | 训练进度可视化 |
| 生物信息工具 | `src.REML`, `src.LD` | 遗传方差分析与 LD 计算 |

## 5. 性能参考 (Current Performance)

- **目标变量方差**: ~4.695 (`bw0_residuals`)
- **最佳 MSE 损失**: ~3.8660
- **解释力度 ($R^2$)**: 约 17.7% (相对于残差部分)
- **综合表型解释率**: ~57% (固定效应 + 遗传 GAT 效应)
