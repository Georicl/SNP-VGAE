#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "GenotypeData.h"
#include "GRMBuilder.h"

// 打印矩阵, 用于检查结果
void PrintMatrix(const std::vector<std::vector<double>>& mat,
                 const std::vector<std::string>& ids) {
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "           ";
  for (const auto& id : ids) {
    std::cout << std::setw(8) << id;
  }
  std::cout << std::endl;

  for (size_t i = 0; i < ids.size(); ++i) {
    std::cout << std::setw(10) << ids[i] << " ";
    for (size_t j = 0; j < ids.size(); ++j) {
      std::cout << std::setw(8) << mat[i][j];
    }
    std::cout << std::endl;
  }
}

int main() {
  std::cout << "===== GRMBuilder 测试 =====" << std::endl << std::endl;

  // 构建测试数据: 3 个样本 × 5 个 SNP
  // SNP3 全 0 (单态), SNP4 全 2 (单态) — 应被跳过
  GenotypeData data;
  data.sample_ids = {"S1", "S2", "S3"};
  data.snp_names = {"rs001", "rs002", "rs003", "rs_mono0", "rs_mono2"};

  // genotypes[sample_idx][snp_idx]
  data.genotypes = {
      {0, 2, 1, 0, 2},   // S1: rs001=0, rs002=2, rs003=1, ...
      {1, 2, 0, 0, 2},   // S2
      {2, 0, 1, 0, 2}    // S3
  };

  // 计算 GRM
  GRMResult result = GRMBuilder::ComputeYangGRM(data);

  std::cout << std::endl;
  PrintMatrix(result.grm, result.sample_ids);

  // 验证结果
  std::cout << std::endl << "===== 期望值验证 =====" << std::endl;

  // 手工计算 (3个有效SNP)
  // rs001: p=0.5,    2p(1-p)=0.5
  //  Z = [-1.414, 0.000, 1.414]
  // rs002: p=0.667,  2p(1-p)=0.444
  //  Z = [1.000, 1.000, -2.000]
  // rs003: p=0.333,  2p(1-p)=0.444
  //  Z = [0.500, -1.000, 0.500]
  // rs_mono0: p=0,    跳过
  // rs_mono2: p=1,    跳过

  // GRM[0][0] = (2.0 + 1.0 + 0.25) / 3 = 3.25/3 = 1.0833
  double expected_00 = 3.25 / 3.0;
  double expected_01 = 0.5 / 3.0;
  double expected_02 = -3.75 / 3.0;
  double expected_11 = 2.0 / 3.0;
  double expected_12 = -2.5 / 3.0;
  double expected_22 = 6.25 / 3.0;

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "GRM[0][0]: 计算=" << result.grm[0][0]
            << ", 期望=" << expected_00
            << (std::fabs(result.grm[0][0] - expected_00) < 1e-3 ? " ✓"
                                                                : " ✗")
            << std::endl;
  std::cout << "GRM[0][1]: 计算=" << result.grm[0][1]
            << ", 期望=" << expected_01
            << (std::fabs(result.grm[0][1] - expected_01) < 1e-3 ? " ✓"
                                                                : " ✗")
            << std::endl;
  std::cout << "GRM[0][2]: 计算=" << result.grm[0][2]
            << ", 期望=" << expected_02
            << (std::fabs(result.grm[0][2] - expected_02) < 1e-3 ? " ✓"
                                                                : " ✗")
            << std::endl;
  std::cout << "GRM[1][1]: 计算=" << result.grm[1][1]
            << ", 期望=" << expected_11
            << (std::fabs(result.grm[1][1] - expected_11) < 1e-3 ? " ✓"
                                                                : " ✗")
            << std::endl;
  std::cout << "GRM[1][2]: 计算=" << result.grm[1][2]
            << ", 期望=" << expected_12
            << (std::fabs(result.grm[1][2] - expected_12) < 1e-3 ? " ✓"
                                                                : " ✗")
            << std::endl;
  std::cout << "GRM[2][2]: 计算=" << result.grm[2][2]
            << ", 期望=" << expected_22
            << (std::fabs(result.grm[2][2] - expected_22) < 1e-3 ? " ✓"
                                                                : " ✗")
            << std::endl;

  // 验证对称性
  std::cout << std::endl << "对称性验证: ";
  bool symmetric = true;
  for (size_t i = 0; i < result.grm.size() && symmetric; ++i) {
    for (size_t j = 0; j < i && symmetric; ++j) {
      if (std::fabs(result.grm[i][j] - result.grm[j][i]) > 1e-10) {
        symmetric = false;
      }
    }
  }
  std::cout << (symmetric ? "✓ 矩阵对称" : "✗ 矩阵不对称!") << std::endl;

  return 0;
}
