#pragma once
#include "GenotypeData.h"

// 按 MAF 阈值筛选 SNP, 默认阈值 0.01
std::vector<MAF> FilterMAF(const std::vector<MAF>& maf_list,
                           double threshold = 0.01);

// 根据 MAF 筛选结果提取对应 SNP 的基因型数据
GenotypeData SNPResult(const GenotypeData& rawsnps,
                       const std::vector<MAF>& maf_filter);
