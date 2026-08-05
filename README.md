# SNP-VGAE：基于 SNP-VAE + VGAE 的基因组预测框架

SNP-VGAE 是一个两阶段深度学习框架，用于基因组表型预测：先通过 **SNP-VAE** 对 SNP 基因型做无监督预训练、学习位点级低维嵌入，再基于遗传关系矩阵（GRM）构建样本 KNN 图，利用 **变分图自编码器（VGAE）** 联合优化表型预测与图结构重建，并通过 SNP 级归因解释功能位点。

## ⚠️ 开发状态说明

> **当前主要开发成果位于 [`feature/vgae`](https://github.com/Georicl/SNP-VGAE/tree/feature/vgae) 分支。**
>
> 完整的模型实现（SNP-VAE 预训练、样本图构建、VGAE 联合训练、SNP 归因）、统计验证（5 折交叉验证 + 置换检验 + GWAS 交叉验证）以及详细文档（含框架流程图与关键参数说明）均在 `feature/vgae` 分支上维护。`main` 分支仅作为稳定入口，暂未包含最新开发成果。

## 快速导航

| 内容 | 位置 |
| --- | --- |
| 完整实现与文档 | [`feature/vgae` 分支](https://github.com/Georicl/SNP-VGAE/tree/feature/vgae) |
| 分支最新 README | [`feature/vgae` README](https://github.com/Georicl/SNP-VGAE/blob/feature/vgae/README.md) |
| 实验结果与图表 | `feature/vgae` 分支 `test/result/` 目录 |

## 许可证

仅用于学术研究用途。
