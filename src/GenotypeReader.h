#pragma once
#include "GenotypeData.h"

// 解析 .alleles 文件, 提取 SNP 元信息
std::vector<SNP> readAlleles(const std::string& path);

// 解析 .data 文件, 将双碱基转换为 0/1/2 基因型编码
GenotypeData ProcessGenotype(const std::string& datapath,
                             const std::vector<SNP>& snps);
