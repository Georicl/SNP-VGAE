"""
M0: 表型文件读取模块
====================

读取 PLINK 格式的 .pheno 文件（FID IID 表型值），
支持缺失值识别（-9 = missing）和多样表型文件解析。

可扩展性:
  - 当前仅支持单列表型，后续可扩展为多列表型（如 --pheno-col）
  - 可集成样本对齐逻辑，自动校验与 GRM/FAM 的样本一致性
  - 支持 -9 / NaN / 空值等多种缺失值标记
"""

import numpy as np


def pheno_reader(
    pheno_path: str,
    missing_val: float = -9.0,
) -> tuple[list[str], np.ndarray]:
    """
    读取 PLINK .pheno 文件。

    文件格式: 每行 FID IID phenotype_value，空格/制表符分隔。
    -9.0 视为缺失值。

    参数:
        pheno_path:  .pheno 文件路径
        missing_val: 缺失值标记，默认 -9.0

    返回:
        iids:       样本 IID 列表 (N,)
        phenotypes: 表型值 np.ndarray (N,), 缺失值为 np.nan

    可扩展性:
      - 后续可返回 FID+IID 元组以支持家系数据
      - 可支持表头行自动检测（以 # 开头的行）
    """
    iids: list[str] = []
    values: list[float] = []

    with open(pheno_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue

            fid, iid = parts[0], parts[1]
            try:
                pheno_val = float(parts[2])
            except ValueError:
                continue

            iids.append(iid)
            if pheno_val == missing_val:
                values.append(np.nan)
            else:
                values.append(pheno_val)

    phenotypes = np.array(values, dtype=np.float32)
    return iids, phenotypes
