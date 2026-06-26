#include "PlinkReader.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

PlinkData PlinkReader(const std::string& bedFile, const std::string& bimFile,
                      const std::string& famFile) {
  // 步骤1: 读取.fam文件,获取样本数量和ID
  std::ifstream fam(famFile);
  if (!fam.is_open()) {
    throw std::runtime_error("Cannot open fam file: " + famFile);
  }

  std::string line;
  std::vector<std::string> sample_ids;
  int num_samples = 0;
  while (std::getline(fam, line)) {
    std::string fid;
    std::string iid;
    std::string sample_id;
    std::istringstream ss(line);
    ss >> fid >> iid;
    sample_id = fid + "_" + iid;
    sample_ids.push_back(sample_id);

    ++num_samples;
  }

  // 步骤2: 读取.bim文件,获取SNP信息
  std::ifstream bim(bimFile);
  if (!bim.is_open()) {
    throw std::runtime_error("Cannot open bim file: " + bimFile);
  }
  std::vector<SNPinfo> snp_list;
  int num_snps = 0;
  while (std::getline(bim, line)) {
    std::string snp_name;
    std::string snp_chr;
    std::string allele1;
    std::string allele2;
    double snp_position;
    long cM;

    std::istringstream ss(line);
    ss >> snp_chr >> snp_name >> cM >> snp_position >> allele1 >> allele2;
    SNPinfo SNP{.snp_name = snp_name,
                .snp_chr = snp_chr,
                .allele1 = allele1,
                .allele2 = allele2,
                .snp_position = snp_position,
                .cM = cM};
    snp_list.push_back(SNP);
    ++num_snps;
  }

  // 步骤3: 读取.bed二进制文件,构建基因型矩阵
  std::ifstream bed(bedFile, std::ios::binary);
  if (!bed.is_open()) {
    throw std::runtime_error("Cannot open bed file: " + bedFile);
  }

  // 读取并验证魔数(magic number): 前3字节固定为0x6C, 0x1B, 0x01
  char magic[3];
  bed.read(magic, 3);
  if (static_cast<unsigned char>(magic[0]) != 0x6C ||
      static_cast<unsigned char>(magic[1]) != 0x1B ||
      static_cast<unsigned char>(magic[2]) != 0x01) {
    throw std::runtime_error("Invalid BED file: bad magic number");
  }

  // 计算每个SNP占用的字节数: block = ceil(num_samples / 4)
  int block = (num_samples + 3) / 4;
  size_t total_bytes = static_cast<size_t>(block) * num_snps;

  // 一次性读取所有基因型数据到缓冲区
  std::vector<unsigned char> buffer(total_bytes);
  bed.read(reinterpret_cast<char*>(buffer.data()), total_bytes);
  if (!bed.good()) {
    throw std::runtime_error("Failed to read genotype data from BED file");
  }

  // 定义基因型矩阵并解析二进制数据 (使用整数矩阵,NA=-9)
  Eigen::MatrixXi genotype(num_snps, num_samples);

  for (size_t snp_idx = 0; snp_idx < num_snps; ++snp_idx) {
    // 定义字节块的大小, 当前SNP的第一个样本的字节开始位置
    size_t current_snp_buffer = snp_idx * block;

    for (size_t byte_idx = 0; byte_idx < block; ++byte_idx) {
      // 取当前的字节块的内容, 每个字节块内有最多4个样本
      unsigned char byte_data = buffer[current_snp_buffer + byte_idx];

      for (size_t sample_byte = 0; sample_byte < 4; ++sample_byte) {
        // 计算当前样本是第几个, 每两个比特是一个样本, 每一个字节包含4个样本
        // 计算公式为:
        // 当前样本 = 已经过的字节数量(byte_idx) + 当前字节中的样本序
        int global_sample_idx = byte_idx * 4 + sample_byte;
        // 因为是SNP为序, 所以这里是一共最多是有num_samples个有效的SNP序
        if (global_sample_idx >= num_samples) break;
        // 位运算提取 2-bit 编码
        unsigned char bits = (byte_data >> (2 * sample_byte)) & 0b11;

        int value = -9;  // 默认值为 -9 (缺失数据)
        switch (bits) {
          case 0:
            value = 0;   // AA: 纯合参考
            break;
          case 1:
            value = -9;  // Missing: 缺失数据
            break;
          case 2:
            value = 1;   // AB: 杂合子
            break;
          case 3:
            value = 2;   // BB: 纯合替代
            break;
        }

        genotype(snp_idx, global_sample_idx) = value;
      }
    }
  }

  // 返回完整的数据结构
  PlinkData result;
  result.num_snps = num_snps;
  result.snp_info = snp_list;
  result.samples_ids = sample_ids;
  result.num_samples = num_samples;
  result.genotypes = genotype;

  return result;
}
