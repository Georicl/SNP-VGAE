"""
协变量文件读取模块
==================

解析 PLINK 格式的协变量文件，将每个样本的协变量信息
组织为以 IID 为键的字典结构，供下游模型训练时作为
固定效应或分组变量使用。

文件格式: 每行包含 FID、IID 和若干协变量列，以空格分隔。
当前动态解析所有协变量列（parts[2:]），无需硬编码列数。

作者: Xiang Yang
邮箱: Georicl@outlook.com
创建时间: 2026-07-06
"""

from typing import Dict


def covar_reader(covar_file: str) -> Dict:
    """
    读取协变量文件，解析为以 IID 为键的嵌套字典。

    参数:
        covar_file: 协变量文件路径

    返回:
        以 IID 为键的字典，每个值包含:
        - fid:    家系 ID
        - iid:    个体 ID
        - covar_N: 第 N 个协变量值（动态解析）
    """
    covar_dict: Dict[str, dict] = {}
    with open(covar_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            # 以 IID（parts[1]）作为字典的键
            # 动态解析所有协变量列（parts[2:]），避免硬编码列数
            covar_dict[parts[1]] = {
                "fid": parts[0],
                "iid": parts[1],
                **{f"covar_{i+1}": parts[2 + i]
                   for i in range(len(parts) - 2)},
            }

    return covar_dict
