"""
PLINK .fam 文件解析模块
========================

解析 PLINK 二进制基因型数据集的 .fam 伴生文件，
提取样本的家系 ID、个体 ID 及可选的家系特征信息
（父本 ID、母本 ID、性别）。

.fam 文件格式（每行 6 列，空格分隔）:
  FID  IID  父本ID  母本ID  性别(1=男/2=女)  表型值

作者: Xiang Yang
邮箱: Georicl@outlook.com
创建时间: 2026-07-06
"""

from typing import Literal, overload


@overload
def fam_reader(
    fam_path: str, return_feature: Literal[True]) -> tuple[list, list]: ...


@overload
def fam_reader(fam_path: str, return_feature: Literal[False]) -> list: ...


def fam_reader(fam_path: str, return_feature: bool) -> list | tuple[list, list]:
    """
    解析 PLINK .fam 文件，提取样本 ID 及可选的家系特征。

    参数:
        fam_path:       .fam 文件路径
        return_feature: 是否同时返回家系特征（父本、母本、性别）

    返回:
        - return_feature=False: [(FID, IID), ...] 样本 ID 列表
        - return_feature=True:  ([(FID, IID), ...], [(父本, 母本, 性别), ...])
    """
    fam_list_id: list = []
    fam_list_feature: list = []
    with open(fam_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts: list = line.split()
            fid, iid, father, mother, sex, *_ = parts
            fam_list_id.append((fid, iid))
            if return_feature:
                fam_list_feature.append((father, mother, sex))
    if return_feature:
        return fam_list_id, fam_list_feature
    return fam_list_id
