import numpy as np


def genotype_read(genotype_file: str, num_samples: int, num_snps: int) -> np.ndarray:
    """
    基因型载入函数, 提供基因型载入窗口, 接受Plink的输出文件, 并将基因型的二进制编码还原为0/1/2
    genotype_file: plink输出文件地址, .bed文件为基因型文件, 其头三字节为魔数, 后所有为SNP或样本顺序排列的SNP基因型.
    num_samples: 样本数量, 当bed文件为SNP顺序排列时为样本数量, 当bed文件为样本顺序排列时为SNP数量.
    num_snps: snp总数, 用于展开bed文件

    return: 一个np数组,为基因型矩阵(0/1/2编码)
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
        zero_matrix = np.zeros((1, 1), dtype=np.int16)
        return zero_matrix
