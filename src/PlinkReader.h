#ifndef PLINK_READER_H
#define PLINK_READER_H

#include <Eigen/Dense>
#include <cstddef>
#include <string>
#include <vector>

// SNP信息结构体
struct SNPinfo {
  std::string snp_name;  // SNP名称
  std::string snp_chr;   // 染色体编号
  std::string allele1;   // 等位基因1
  std::string allele2;   // 等位基因2
  double snp_position;   // SNP位置
  long cM;               // 遗传距离(cM)
};

// PLINK数据结构体,整合三个文件的信息
struct PlinkData {
  size_t num_snps;                       // SNP数量
  std::vector<SNPinfo> snp_info;         // SNP信息列表
  std::vector<std::string> fid;          // 家系ID列表
  std::vector<std::string> iid;          // 个体ID列表
  std::vector<std::string> samples_ids;  // 样本ID列表 (FID_IID, 兼容旧代码)
  size_t num_samples;                    // 样本数量
  Eigen::MatrixXi genotypes;  // 基因型矩阵 (SNP x Sample), NA=-9
};

/*
 * 读取PLINK格式文件(.bed/.bim/.fam),构建完整的数据结构
 * 参数:
 *   bedFile - 二进制基因型文件路径
 *   bimFile - SNP信息文件路径
 *   famFile - 家系信息文件路径
 * 返回:
 *   PlinkData结构体,包含所有SNP信息和基因型矩阵
 */
PlinkData PlinkReader(const std::string& bedFile, const std::string& bimFile,
                      const std::string& famFile);

#endif  // PLINK_READER_H
