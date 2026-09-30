/**
 * @file PlinkReader.cpp
 * @brief PLINK 二进制基因型文件读取器实现
 *
 * 实现 PLINK .bed/.bim/.fam 文件的解析逻辑，
 * 将二进制基因型数据解码为 0/1/2/-9 编码的 Eigen 矩阵。
 *
 * @author Xiang Yang
 * @email Georicl@outlook.com
 * @date 2026-06-27
 */

#include "PlinkReader.h"

#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

std::runtime_error InvalidRow(const std::string& path, size_t line_number,
                              const std::string& reason) {
  return std::runtime_error(path + ": line " + std::to_string(line_number) +
                            ": " + reason);
}

// Parse tokens first: extracting a decimal into an integer can otherwise shift
// the remaining columns without making the complete row extraction fail.
std::array<std::string, 6> ReadFields(const std::string& line,
                                    const std::string& path,
                                    size_t line_number) {
  std::istringstream stream(line);
  std::array<std::string, 6> fields;
  for (auto& field : fields) {
    if (!(stream >> field)) {
      throw InvalidRow(path, line_number, "expected exactly 6 fields");
    }
  }
  std::string extra;
  if (stream >> extra) {
    throw InvalidRow(path, line_number, "expected exactly 6 fields");
  }
  return fields;
}

template <typename T>
T ReadNumber(const std::string& token, const std::string& path,
             size_t line_number, const std::string& field_name) {
  std::istringstream stream(token);
  stream.imbue(std::locale::classic());
  T value{};
  if (!(stream >> value) || !stream.eof()) {
    throw InvalidRow(path, line_number, "invalid " + field_name + ": " + token);
  }
  return value;
}

}  // namespace

PlinkData PlinkReader(const std::string& bedFile, const std::string& bimFile,
                      const std::string& famFile) {
  // 步骤1: 读取.fam文件,获取样本数量和ID
  std::ifstream fam(famFile);
  if (!fam.is_open()) {
    throw std::runtime_error("Cannot open fam file: " + famFile);
  }

  std::string line;
  std::vector<std::string> fid_list;
  std::vector<std::string> iid_list;
  std::vector<std::string> sample_ids;
  size_t line_number = 0;
  while (std::getline(fam, line)) {
    const auto fields = ReadFields(line, famFile, ++line_number);
    fid_list.push_back(fields[0]);
    iid_list.push_back(fields[1]);
    sample_ids.push_back(fields[0] + "_" + fields[1]);
  }
  if (!fam.eof()) {
    throw std::runtime_error("Failed to read fam file: " + famFile);
  }
  const size_t num_samples = sample_ids.size();
  if (num_samples == 0) {
    throw std::runtime_error("FAM file contains no samples: " + famFile);
  }

  // 步骤2: 读取.bim文件,获取SNP信息
  std::ifstream bim(bimFile);
  if (!bim.is_open()) {
    throw std::runtime_error("Cannot open bim file: " + bimFile);
  }
  std::vector<SNPinfo> snp_list;
  line_number = 0;
  while (std::getline(bim, line)) {
    const auto fields = ReadFields(line, bimFile, ++line_number);
    const double cM =
        ReadNumber<double>(fields[2], bimFile, line_number, "cM");
    if (!std::isfinite(cM)) {
      throw InvalidRow(bimFile, line_number, "cM must be finite");
    }
    const auto position = ReadNumber<std::int64_t>(
        fields[3], bimFile, line_number, "integer bp position");
    snp_list.push_back(
        SNPinfo{fields[1], fields[0], fields[4], fields[5], position, cM});
  }
  if (!bim.eof()) {
    throw std::runtime_error("Failed to read bim file: " + bimFile);
  }
  const size_t num_snps = snp_list.size();
  if (num_snps == 0) {
    throw std::runtime_error("BIM file contains no variants: " + bimFile);
  }

  // 步骤3: 读取.bed二进制文件,构建基因型矩阵
  std::ifstream bed(bedFile, std::ios::binary);
  if (!bed.is_open()) {
    throw std::runtime_error("Cannot open bed file: " + bedFile);
  }

  // 读取并验证魔数(magic number): 前3字节固定为0x6C, 0x1B, 0x01
  char magic[3]{};
  if (!bed.read(magic, sizeof(magic))) {
    throw std::runtime_error("Invalid BED file: truncated header");
  }
  if (static_cast<unsigned char>(magic[0]) != 0x6C ||
      static_cast<unsigned char>(magic[1]) != 0x1B ||
      static_cast<unsigned char>(magic[2]) != 0x01) {
    throw std::runtime_error("Invalid BED file: bad magic number");
  }

  // 计算每个SNP占用的字节数: block = ceil(num_samples / 4)
  const size_t block = num_samples / 4 + (num_samples % 4 != 0);
  const auto max_index =
      static_cast<size_t>(std::numeric_limits<Eigen::Index>::max());
  if (num_samples > max_index || num_snps > max_index / num_samples ||
      num_snps > std::numeric_limits<size_t>::max() / block) {
    throw std::runtime_error("PLINK dimensions are too large");
  }
  const size_t total_bytes = block * num_snps;
  if (total_bytes >
      static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error("BED payload is too large");
  }

  // 一次性读取所有基因型数据到缓冲区
  std::vector<unsigned char> buffer(total_bytes);
  if (!bed.read(reinterpret_cast<char*>(buffer.data()),
                static_cast<std::streamsize>(total_bytes))) {
    throw std::runtime_error("Failed to read genotype data from BED file");
  }
  if (bed.peek() != std::ifstream::traits_type::eof()) {
    throw std::runtime_error("Invalid BED file: unexpected trailing data");
  }
  if (bed.bad()) {
    throw std::runtime_error("Failed to read genotype data from BED file");
  }

  // 定义基因型矩阵并解析二进制数据 (使用整数矩阵,NA=-9)
  Eigen::MatrixXi genotype(static_cast<Eigen::Index>(num_snps),
                           static_cast<Eigen::Index>(num_samples));

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
        size_t global_sample_idx = byte_idx * 4 + sample_byte;
        // 因为是SNP为序, 所以这里是一共最多是有num_samples个有效的SNP序
        if (global_sample_idx >= num_samples) break;
        // 位运算提取 2-bit 编码
        unsigned char bits = (byte_data >> (2 * sample_byte)) & 0b11;

        int value = -9;  // 默认值为 -9 (缺失数据)
        switch (bits) {
          case 0:
            value = 0;   // Homozygous allele1 (not necessarily reference)
            break;
          case 1:
            value = -9;  // Missing: 缺失数据
            break;
          case 2:
            value = 1;   // AB: 杂合子
            break;
          case 3:
            value = 2;   // Homozygous allele2
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
  result.fid = fid_list;
  result.iid = iid_list;
  result.samples_ids = sample_ids;
  result.num_samples = num_samples;
  result.genotypes = genotype;

  return result;
}
