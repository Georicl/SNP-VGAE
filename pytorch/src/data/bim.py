"""
PLINK .bim 文件解析模块
========================

解析 PLINK 二进制基因型数据集的 .bim 伴生文件，
提取每个 SNP 位点的染色体编号、名称、遗传距离、
物理位置及等位基因信息。

.bim 文件格式（每行 6 列，空格分隔）:
  染色体编号  SNP名称  遗传距离(cM)  物理位置(bp)  等位基因1  等位基因2

作者: Xiang Yang
邮箱: Georicl@outlook.com
创建时间: 2026-07-06
"""

from typing import Dict, List


def bim_reader(bim_file: str) -> List[Dict]:
    """
    解析 PLINK .bim 文件，返回所有 SNP 位点的元信息列表。

    参数:
        bim_file: .bim 文件的绝对或相对路径

    返回:
        字典列表，每个字典包含一个 SNP 的完整信息:
        - chrom:    染色体编号 (str)
        - snp_id:   SNP 标识符 (str)
        - cM:       遗传距离，单位为厘摩 (str)
        - position: 物理位置，单位为碱基对 (str)
        - allele1:  第一等位基因 (str)
        - allele2:  第二等位基因 (str)
    """
    bim_list = []
    with open(bim_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts: list = line.split()
            bim_list.append({
                "chrom": parts[0],
                "snp_id": parts[1],
                "cM": parts[2],
                "position": parts[3],
                "allele1": parts[4],
                "allele2": parts[5]
            })
    return bim_list
