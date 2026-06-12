#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "GenotypeData.h"

// 读取 .alleles文件获取SNP信息, 返回一个包含了等位基因信息的vector
std::vector<SNP> readAlleles(const std::string& path) {
  std::vector<SNP> snps;
  std::ifstream file(path);
  std::string line, type, name, allele;
  SNP current;

  while (std::getline(file, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    // 记录marker
    ss >> type;
    if (type == "marker") {
      if (!current.name.empty()) snps.push_back(current);
      ss >> current.name;
      // 初始化snp位点
      current.a1 = "";
      current.a2 = "";
    } else if (type == "allele") {
      ss >> allele;
      if (allele != "NA") {
        // 如果a1不为空则给a2
        current.a1.empty() ? current.a1 = allele : current.a2 = allele;
      }
    }
  }
  snps.push_back(current);
  return snps;
}

GenotypeData ProcessGenotype(const std::string& datapath,
                             const std::vector<SNP>& snps) {
  GenotypeData result;

  // 预分配SNP名称
  for (const auto& s : snps) {
    result.snp_names.push_back(s.name);
  }

  std::ifstream file(datapath);
  if (!file.is_open()) {
    std::cerr << "无法打开data文件: " << datapath << std::endl;
    return result;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string id1, id2, others;

    ss >> id1 >> id2;
    // 跳过接下来4个字段
    for (int i = 0; i < 4; ++i) ss >> others;

    // 保存样本ID
    std::string sample_id = id1 + "_" + id2;
    result.sample_ids.push_back(sample_id);

    // 为该样本预分配基因型向量
    std::vector<int> sample_genotypes;
    sample_genotypes.reserve(snps.size());

    for (const auto& s : snps) {
      std::string g1, g2;
      if (g1 == "NA" || g2 == "NA") {
        // NA值标记为-1
        sample_genotypes.push_back(-1);
      } else {
        int count = 0;
        // 计算a2等位基因的数量：0=纯合a1, 1=杂合, 2=纯合a2
        if (g1 == s.a2) count++;
        if (g2 == s.a2) count++;
        sample_genotypes.push_back(count);
      }
    }

    result.genotypes.push_back(std::move(sample_genotypes));
  }

  return result;
}