# 评测完整性修复（2026-09-30）

基于 feature/vgae 的 `f1d56f02ff30fce9ca0e2e2ea067f836193bdfb9`。本轮只修复实现和增加小型回归测试，没有重新运行真实队列评测或更新已提交的指标/图表。

## 行为变化

- BLUP：λ只在train拟合、val评分；选择后才合并train+val重拟合，test必须独立。两个评测脚本共用同一调参实现
- 置换验证脚本：观测与置换统一使用200 epochs、patience=30、warmup=30。运行时长可能比原快速null配置更长；20次置换的最低p值仍为1/21，未增加计算预算或声称统计证据变强
- C++ PLINK：cM改为double，bp为int64_t；按独立字段转换，畸形BIM/FAM行及截断BED明确报错。有效BED的0/1/2/-9编码不变
- VGAE checkpoint：训练仍使用warmup β；验证固定为目标β，并固定负边抽样，避免各epoch评分目标漂移和验证消耗训练随机状态
- SNP-VAE checkpoint与学习率调度：按同一目标β计算训练重建/KL评分。该评分仍来自训练batch，不是独立验证集，不作为泛化估计
- VGAE eval：使用后验均值mu，训练模式保留随机重参数化。旧checkpoint仍可加载，但新推断指标不应与旧单次潜变量采样输出混作同一协议
- 预训练增加seed参数（默认42）；run_kfold_cv为每折设置seed+fold_index，覆盖初始化、dropout和训练采样。只保证同一设备/依赖条件下可复现，不保证跨CPU/CUDA/MPS逐位一致
- GWAS以SNP ID对齐BIM，存在TEST列时只选ADD；缺失、额外、重复ID、无效p值、坐标冲突均报错，不静默填充/删行。染色体标记必须精确一致，位置要求正数，需调用者显式统一命名和基因组版本
- top5% + 跨折CV<0.5明确为探索性稳定高归因候选。新输出为snp_candidates.tsv、stable_snp_candidates.tsv、snp_candidates_summary.json及snp_candidate_*图，不覆盖旧significance文件；nominal GWAS overlap不称为独立验证或非线性效应

## 快速测试（无需test_data）

在仓库根目录、Python >=3.13环境中：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=pytorch/src uv run pytest \
  test/test_audit_regressions.py test/test_snp_alignment.py test/test_pretrain.py \
  test/test_vgae_model.py::test_gcn_conv \
  test/test_vgae_model.py::test_vgae_model \
  test/test_vgae_model.py::test_vgae_loss -q
```

C++需要编译器和Eigen3头文件，详见test/test_plink_reader.cpp顶部命令；EIGEN_INCLUDE_DIR指向包含Eigen/的目录。测试用临时文件构造BIM/FAM/BED，避免真实数据下载。

## 已执行验证与边界

- Python 3.13.15、NumPy 2.5.3、SciPy 1.18.1、PyTorch 2.14.0+cpu、pytest 9.1.1、Matplotlib 3.11.2：快速测试通过；具体计数和完整输出随修复交付记录保存
- 原始版本上新增Python行为回归6项均失败；修改后通过。C++38项原版失败29项，修复后全部通过
- C++ AddressSanitizer + UndefinedBehaviorSanitizer通过；执行环境不支持LeakSanitizer，因此关闭了泄漏检测，不能声称内存泄漏扫描通过
- 没有执行真实数据集训练、CUDA/MPS测试、整仓库所有pytest入口或远端CI。原脚本式集成测试仍需要真实数据/参数，不应把本快速集合通过表述为整仓库测试通过
- 现有历史结果仍需按新协议另行重算。转导CV、按亲缘分组/外部验证、SNP发现的null校准、归因abs_sum/完整邻居贡献语义等研究问题未在此补丁重新设计

## 发布前简化

移除测试中仅用于兼容旧实现的hasattr/getattr路径，删除重复的配置别名与配置单测（实际调用链回归保留），去掉_validate未使用的β参数和统计脚本未使用的lambda累积。输入格式、SNP身份对应以及训练/验证分离校验保留。
