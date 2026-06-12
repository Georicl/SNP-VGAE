#pragma once
#include <string>
#include <vector>

struct SNP {
  std::string name;
  std::string a1;
  std::string a2;
};

struct GenotypeData {
  std::vector<std::string> sample_ids;
  std::vector<std::string> snp_names;
  std::vector<std::vector<int>> genotypes;  // -1表示NA
};

struct MAF {
  std::string snp_name;
  double maf;
  int a2;
  int na;
  int total_count;
};