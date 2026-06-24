#pragma once
#include "GenotypeData.h"

// 计算单个 SNP 的 MAF 统计量
MAF CalculateMAF(const std::string& snp_name, const std::vector<int>& genotype);

// 从 GenotypeData 计算所有 SNP 的 MAF
std::vector<MAF> CalculateAllMAF(const GenotypeData& data);

// 从原始基因型文本文件计算所有 SNP 的 MAF
std::vector<MAF> Geno2MAF(const std::string GenotypeFile);
