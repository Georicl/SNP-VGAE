#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>

#include "GenotypeData.h"

// 函数声明
MAF CalculateMAF(const std::string& snp_name, const std::vector<int>& genotype);
std::vector<MAF> CalculateAllMAF(const GenotypeData& data);
std::vector<MAF> Geno2MAF(const std::string GenotypeFile);

MAF CalculateMAF(const std::string& snp_name,
                 const std::vector<int>& genotype) {
  MAF result;
  result.snp_name = snp_name;
  result.a2 = 0;
  result.na = 0;
  result.total_count = 0;
  result.maf = 0.00;
  // 遍历genotype中的snp的基因型, 并统计总值
  for (int g : genotype) {
    if (g < 0) {
      result.na++;
    } else {
      result.total_count++;
      result.a2 += g;
    }
  }
  /*
  a2等位基因频率 = a2出现的次数(g可能的值只有0/1/2) /
  (非NA的样本出现数量 * 2(二倍体))
  */
  double p = static_cast<double>(result.a2) / (result.total_count * 2);
  // maf = min(p, 1 - p)
  result.maf = std::min(p, 1 - p);
  return result;
}

// 从 GenotypeData 结构体直接计算所有 SNP 的 MAF
std::vector<MAF> CalculateAllMAF(const GenotypeData& data) {
  std::vector<MAF> maf_list;

  // 遍历每个 SNP (列)
  for (size_t snp_idx = 0; snp_idx < data.snp_names.size(); ++snp_idx) {
    // 提取该 SNP 在所有样本中的基因型
    std::vector<int> genotype;
    genotype.reserve(data.sample_ids.size());

    for (size_t sample_idx = 0; sample_idx < data.sample_ids.size();
         ++sample_idx) {
      genotype.push_back(data.genotypes[sample_idx][snp_idx]);
    }

    // 计算该 SNP 的 MAF
    maf_list.push_back(CalculateMAF(data.snp_names[snp_idx], genotype));
  }

  return maf_list;
}

std::vector<MAF> Geno2MAF(const std::string GenotypeFile) {
  // 输入的基因型文件(行为样本, 列为SNP, 第一列为SampleID)
  std::ifstream inFile(GenotypeFile);
  if (!inFile.is_open()) {
    std::cerr << "打开基因型文件失败, 路径为:" << GenotypeFile
              << "请检查是否存在该文件" << std::endl;
    return {};
  }

  // 首行
  std::string header_line;
  std::getline(inFile, header_line);
  std::istringstream iss(header_line);
  // 记录所有snp
  std::string tmp_token;
  std::vector<std::string> snp_name;
  iss >> tmp_token;  // 跳过SampleID
  while (iss >> tmp_token) {
    // 得到含有所有snp名字的vector
    snp_name.push_back(tmp_token);
  }
  // 后续为SNP的基因型行
  std::string line;
  // 基因型二维vector
  int snps_number = snp_name.size();
  // 初始, snps长度为snp数量
  std::vector<std::vector<int>> snps(snps_number);

  while (std::getline(inFile, line)) {
    /*
    计算方法: snps是一个二维数组, snps[i][j] 结构为第i个SNP, 第J个样本的基因型,
    while: 逐行读取除了首行的其他行(编码基因型)
    对于一个基因矩阵
            snp1   snp2
    sample1  0  1
    sample2  1  0
    有以下放置方法
    snp[0]: [0] <- 首个 样本
    snp[1]: [1] <- 第二个SNP的第一个样本的基因型
    For: 循环为每次循环使得i个snp位点塞入一个样本的所有编码基因型
    */
    std::stringstream ss(line);
    ss >> tmp_token;  // 跳过第一列
    for (int i = 0; i < snps_number; ++i) {
      ss >> tmp_token;
      if (tmp_token == "NA") {
        snps[i].push_back(-1);
      } else {
        snps[i].push_back(std::stoi(tmp_token));
      }
    }
  }
  std::vector<MAF> maf_vector;  // 返回一个含有每个SNP的MAF结构体
  for (int i = 0; i < snps_number; ++i) {
    maf_vector.push_back(CalculateMAF(snp_name[i], snps[i]));
  }
  return maf_vector;
}