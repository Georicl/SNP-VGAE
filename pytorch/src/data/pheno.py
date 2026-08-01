"""
表型文件读取模块
================

读取 PLINK 格式的表型文件（.pheno），解析每个样本的表型观测值，
支持缺失值识别（-9 = missing）和注释行跳过（# 开头）。

.pheno 文件格式:
  每行: FID  IID  表型值，空格/制表符分隔
  -9.0 视为缺失值，转换为 np.nan

作者: Xiang Yang
邮箱: Georicl@outlook.com
创建时间: 2026-07-23
"""

import numpy as np


def pheno_reader(
    pheno_path: str,
    missing_val: float = -9.0,
) -> tuple[list[str], np.ndarray]:
    """
    读取 PLINK .pheno 文件，返回样本 IID 列表和表型值数组。

    参数:
        pheno_path:  .pheno 文件路径
        missing_val: 缺失值标记，默认 -9.0

    返回:
        iids:       样本 IID 列表 (N,)
        phenotypes: 表型值 np.ndarray (N,)，缺失值为 np.nan
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
