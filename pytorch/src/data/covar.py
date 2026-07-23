"""
协变量文件读取模块（临时实现）。

当前为测试阶段的简易实现，仅支持固定格式的协变量文件解析。
后续可替换为更成熟的协变量导入方案，例如：
    - 支持表头自动解析，动态识别协变量列名与数量
    - 集成 pandas 读取，支持缺失值处理与类型推断
    - 支持多种协变量格式（如 PLINK .covar / .qcovar）
    - 与样本对齐模块联动，自动校验协变量与 GRM/表型的样本一致性
"""

from typing import Dict


def covar_reader(covar_file: str) -> Dict:
    """
    读取协变量文件，解析为以 IID 为键的字典。

    注意：此为临时测试实现，文件格式和列数均为硬编码，
    后续将被更通用的协变量导入方案替代。

    文件格式：每行包含 FID、IID 和若干协变量列，以空格分隔。
    当前默认解析前两列协变量（covar1, covar2）。

    参数:
        covar_file: 协变量文件路径

    返回:
        以 IID 为键的字典，每个值为包含 fid、iid 及协变量的字典。

    可扩展性说明:
        当前为临时方案，后续可替换为支持动态列解析、
        多格式兼容、缺失值处理的成熟协变量导入模块。
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
