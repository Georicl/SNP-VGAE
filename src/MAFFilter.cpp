#include <algorithm>
#include <iostream>
#include <iterator>
#include <unordered_set>
#include <vector>

#include "GenotypeData.h"

std::vector<MAF> FilterMAF(const std::vector<MAF>& maf_list,
                           double threshold = 0.01) {
  // 筛选MAF大于指定阈值的SNPs用于后续的分析, 阈值默认为0.01
  std::vector<MAF> maf_filter;
  std::copy_if(maf_list.begin(), maf_list.end(), std::back_inserter(maf_filter),
               [threshold](const MAF& maf) { return maf.maf >= threshold; });
  return maf_filter;
}

GenotypeData SNPResult(const GenotypeData& rawsnps,
                       const std::vector<MAF>& maf_filter) {
  // 根据MAF筛选后的SNP名单,从原始基因型数据中提取对应的SNP
  GenotypeData result;

  // 构建筛选后的SNP名称集合,用于快速查找
  std::unordered_set<std::string> filtered_snp_names;
  for (const auto& maf : maf_filter) {
    filtered_snp_names.insert(maf.snp_name);
  }

  // 复制样本ID
  result.sample_ids = rawsnps.sample_ids;

  // 遍历原始SNP,只保留在筛选列表中的SNP
  for (size_t i = 0; i < rawsnps.snp_names.size(); ++i) {
    if (filtered_snp_names.count(rawsnps.snp_names[i]) > 0) {
      // 该SNP在筛选列表中,添加到结果中
      result.snp_names.push_back(rawsnps.snp_names[i]);
      result.genotypes.push_back(rawsnps.genotypes[i]);
    }
  }

  return result;
}