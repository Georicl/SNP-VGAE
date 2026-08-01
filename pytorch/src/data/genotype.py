"""
PLINK .bed 二进制基因型读取模块
====================================

从 PLINK .bed 二进制文件中解码基因型数据，将 2-bit 压缩编码
还原为标准的 0/1/2 加性编码矩阵。

.bed 文件结构:
  - 前 3 字节: 魔数 (0x6C, 0x1B, 0x01)，标识 SNP-major 存储顺序
  - 后续数据: 每个 SNP 占用 ceil(N/4) 字节，每 2-bit 编码一个样本

2-bit 编码映射:
  00 → 0 (AA, 第一等位基因纯合)
  01 → -9 (缺失)
  10 → 1 (AB, 杂合)
  11 → 2 (BB, 第二等位基因纯合)

作者: Xiang Yang
邮箱: Georicl@outlook.com
创建时间: 2026-07-06
"""

import numpy as np


def genotype_read(genotype_file: str, num_samples: int, num_snps: int) -> np.ndarray:
    """
    从 PLINK .bed 文件读取基因型矩阵，解码 2-bit 压缩为 0/1/2/-9 编码。

    参数:
        genotype_file: PLINK .bed 文件路径
        num_samples:   样本数量（用于确定每个 SNP 的字节块大小）
        num_snps:      SNP 总数（用于重塑矩阵维度）

    返回:
        np.ndarray: 基因型矩阵 (num_snps, num_samples)，
                    值为 0/1/2 加性编码，-9 表示缺失

    异常:
        ValueError: 当 .bed 文件魔数不匹配时抛出
    """
    with open(genotype_file, "rb") as f:
        magic = f.read(3)  # 依据bed文件格式, 前三个字节为魔法块, 判断顺序模式
        data = f.read()  # 载入后续所有SNP

    if magic[0] == 0x6C and magic[1] == 0x1B and magic[2] == 0x01:
        # 载入data到数组里, 此时模式为bed文件的标准模式(SNP为顺序)
        buffer = np.frombuffer(data, dtype=np.uint8)
        block: int = (num_samples + 3) // 4  # 记录每个SNP会占据多少个block(向上取整)

        # 转换为一个行为snps, 列为block计数的矩阵, 实际为一个二维数组<snps<block>>
        raw = buffer.reshape(num_snps, block)
        # 对每个字节取前两位进行位运算, 以计算基因型
        g0: np.ndarray = raw & 0x03
        g1: np.ndarray = (raw >> 2) & 0x03
        g2: np.ndarray = (raw >> 4) & 0x03
        g3: np.ndarray = (raw >> 6) & 0x03

        # 堆叠到一张矩阵中, 维数为[num_snps x num_samples] 行为SNP, 列为基因型
        genotype: np.ndarray = np.stack(
            (g0, g1, g2, g3), axis=-1).reshape(num_snps, -1)[:, :num_samples]
        # 查表将基因型的二进制转换为实际基因型矩阵
        # 编码映射: 0->0(第一等位基因纯合), 1->-9(缺失),
        # 2->1(杂合), 3->2(第二等位基因纯合)
        lookup = np.array([0, -9, 1, 2], dtype=np.int32)
        geno_matrix = lookup[genotype]

        return geno_matrix
    else:
        raise ValueError(
            f"无效的 PLINK .bed 文件: 魔数不匹配 "
            f"(期望 0x6C 0x1B 0x01, 实际 "
            f"{magic[0]:#04x} {magic[1]:#04x} {magic[2]:#04x})"
        )
