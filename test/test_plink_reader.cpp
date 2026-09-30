/*
Standalone C++ regression test; no real dataset or Python packages required.
From the repository root (Eigen 3 headers must be installed):
  g++ -std=c++17 -Wall -Wextra -Werror -pedantic -Isrc \
    -isystem "${EIGEN_INCLUDE_DIR:-/usr/include/eigen3}" \
    src/PlinkReader.cpp test/test_plink_reader.cpp -o /tmp/test_plink_reader
  /tmp/test_plink_reader
*/

#include "PlinkReader.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

class Fixture {
 public:
  Fixture() {
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    for (int attempt = 0; attempt < 100; ++attempt) {
      const auto candidate = std::filesystem::temp_directory_path() /
                             ("snp_vgae_plink_" + std::to_string(stamp) + "_" +
                              std::to_string(attempt));
      if (std::filesystem::create_directory(candidate)) {
        directory_ = candidate;
        return;
      }
    }
    throw std::runtime_error("Could not create an isolated test directory");
  }

  ~Fixture() {
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
  }

  std::string Path(const std::string& extension) const {
    return (directory_ / ("tiny." + extension)).string();
  }

  void Write(const std::string& extension, const std::string& text) const {
    std::ofstream file(Path(extension), std::ios::binary);
    Check(static_cast<bool>(file << text), "Could not write test fixture");
  }

  void WriteBed(const std::vector<unsigned char>& bytes) const {
    std::ofstream file(Path("bed"), std::ios::binary);
    for (unsigned char byte : bytes) file.put(static_cast<char>(byte));
    Check(static_cast<bool>(file), "Could not write BED fixture");
  }

  void Reset() const {
    Write("fam", "F1 I1 0 0 1 -9\n"
                 "F1 I2 0 0 2 1\n"
                 "F2 I3 I1 I2 0 2\n"
                 "F2 I4 0 0 1 3.5\n"
                 "F_3 I_5 0 0 0 NA\n");
    Write("bim", "1 rs_decimal 0.125 12345 A G\n"
                 "X rs_scientific 2.5e-2 67890 AT A\n");
    // Five samples require two bytes per variant. Low-order bits come first.
    // SNP 1: 00, 01, 10, 11, 10; SNP 2: 11, 10, 01, 00, 11.
    WriteBed({0x6c, 0x1b, 0x01, 0xe4, 0x02, 0x1b, 0x03});
  }

  PlinkData Read() const {
    return PlinkReader(Path("bed"), Path("bim"), Path("fam"));
  }

  void ExpectError(const std::string& message) const {
    try {
      Read();
    } catch (const std::runtime_error& error) {
      Check(std::string(error.what()).find(message) != std::string::npos,
            "Wrong error: " + std::string(error.what()));
      return;
    }
    throw std::runtime_error("Expected failure containing: " + message);
  }

 private:
  std::filesystem::path directory_;
};

}  // namespace

int main() {
  Fixture fixture;
  int passed = 0;
  int failed = 0;
  const auto run = [&](const std::string& name,
                       const std::function<void()>& test) {
    try {
      fixture.Reset();
      test();
      ++passed;
      std::cout << "PASS " << name << '\n';
    } catch (const std::exception& error) {
      ++failed;
      std::cerr << "FAIL " << name << ": " << error.what() << '\n';
    }
  };

  run("decimal cM preserves all BIM columns", [&] {
    const auto data = fixture.Read();
    Check(data.num_snps == 2 && data.snp_info.size() == 2, "SNP count");
    const auto& snp = data.snp_info[0];
    Check(snp.snp_name == "rs_decimal" && snp.snp_chr == "1", "SNP identity");
    Check(std::abs(snp.cM - 0.125) < 1e-12, "Decimal cM was lost");
    Check(snp.snp_position == 12345, "bp column shifted");
    Check(snp.allele1 == "A" && snp.allele2 == "G", "Allele columns shifted");
    const auto& second = data.snp_info[1];
    Check(std::abs(second.cM - 0.025) < 1e-12, "Scientific cM was lost");
    Check(second.snp_chr == "X" && second.snp_position == 67890 &&
              second.allele1 == "AT" && second.allele2 == "A",
          "Chromosome or multibase allele changed");
    Check(std::is_floating_point<decltype(snp.cM)>::value,
          "cM must be floating");
    Check(std::is_integral<decltype(snp.snp_position)>::value,
          "bp must be an integer");
  });

  run("BED all genotype codes, variant order and partial-byte sample", [&] {
    const auto data = fixture.Read();
    Check(data.genotypes.rows() == 2 && data.genotypes.cols() == 5,
          "Matrix must be SNP x sample");
    const int expected[2][5] = {{0, -9, 1, 2, 1}, {2, 1, -9, 0, 2}};
    for (int snp = 0; snp < 2; ++snp) {
      for (int sample = 0; sample < 5; ++sample) {
        Check(data.genotypes(snp, sample) == expected[snp][sample],
              "Incorrect genotype at " + std::to_string(snp) + "," +
                  std::to_string(sample));
      }
    }
    Check(data.num_samples == 5 && data.fid.size() == 5 && data.iid.size() == 5,
          "Sample count");
    Check(data.fid[4] == "F_3" && data.iid[4] == "I_5" &&
              data.samples_ids[4] == "F_3_I_5",
          "Sample identifiers or order changed");
  });

  run("BED exact four-sample block", [&] {
    fixture.Write("fam", "F I1 0 0 1 -9\nF I2 0 0 2 -9\n"
                         "F I3 0 0 0 -9\nF I4 0 0 0 -9\n");
    fixture.WriteBed({0x6c, 0x1b, 0x01, 0xe4, 0x1b});
    const auto data = fixture.Read();
    Check(data.genotypes.cols() == 4 && data.genotypes(0, 3) == 2 &&
              data.genotypes(1, 0) == 2,
          "Byte-aligned block decoding");
  });

  run("whitespace, CRLF and missing final newline", [&] {
    fixture.Write("fam", "  F\tI\t0 0 0 -9  \r\n");
    fixture.Write("bim", "  1\trs_zero\t0 2147483646 A G  \r\n"
                         "2 rs_signed -0.5 -1 0 C");
    fixture.WriteBed({0x6c, 0x1b, 0x01, 0x02, 0x03});
    const auto data = fixture.Read();
    Check(data.snp_info[0].cM == 0 &&
              data.snp_info[0].snp_position == 2147483646,
          "Integer cM or bp changed");
    Check(data.snp_info[1].cM == -0.5 && data.snp_info[1].snp_position == -1,
          "Signed numeric metadata changed");
    Check(data.genotypes(0, 0) == 1 && data.genotypes(1, 0) == 2,
          "Single-sample decoding");
  });

  for (const std::string extension : {"fam", "bim"}) {
    const std::string valid = extension == "fam" ? "F I 0 0 1 -9\n"
                                                : "1 rs_ok 0 100 A G\n";
    for (const std::string bad :
         {"", " ", "one", "1 2 3 4 5", "1 2 3 4 5 6 7"}) {
      run(extension + " rejects malformed row: [" + bad + "]", [&] {
        fixture.Write(extension, valid + bad + "\n");
        fixture.ExpectError(fixture.Path(extension) + ": line 2:");
      });
    }
    run(extension + " rejects empty file", [&] {
      fixture.Write(extension, "");
      fixture.ExpectError(extension == "fam" ? "no samples" : "no variants");
    });
  }

  for (const std::string token : {"bad", "0.1junk", "nan", "inf", "1e999"}) {
    run("BIM rejects invalid cM: " + token, [&] {
      fixture.Write("bim", "1 rs_bad " + token + " 100 A G\n");
      fixture.ExpectError(fixture.Path("bim") + ": line 1:");
    });
  }

  for (const std::string token : {"bad", "123.5", "1e3", "12junk", "nan",
                                  "9223372036854775808"}) {
    run("BIM rejects invalid integer bp: " + token, [&] {
      fixture.Write("bim", "1 rs_bad 0.5 " + token + " A G\n");
      fixture.ExpectError(fixture.Path("bim") + ": line 1: invalid integer bp");
    });
  }

  const std::vector<unsigned char> header{0x6c, 0x1b, 0x01};
  for (size_t length = 0; length < header.size(); ++length) {
    run("BED rejects header length " + std::to_string(length), [&] {
      fixture.WriteBed(std::vector<unsigned char>(header.begin(),
                                                   header.begin() + length));
      fixture.ExpectError("truncated header");
    });
  }
  run("BED rejects bad magic", [&] {
    fixture.WriteBed({0x00, 0x1b, 0x01, 0xe4, 0x02, 0x1b, 0x03});
    fixture.ExpectError("bad magic number");
  });
  run("BED rejects sample-major mode", [&] {
    fixture.WriteBed({0x6c, 0x1b, 0x00, 0xe4, 0x02, 0x1b, 0x03});
    fixture.ExpectError("bad magic number");
  });
  run("BED rejects missing payload", [&] {
    fixture.WriteBed(header);
    fixture.ExpectError("Failed to read genotype data");
  });
  run("BED rejects truncated payload", [&] {
    fixture.WriteBed({0x6c, 0x1b, 0x01, 0xe4, 0x02, 0x1b});
    fixture.ExpectError("Failed to read genotype data");
  });
  run("BED rejects extra payload", [&] {
    fixture.WriteBed({0x6c, 0x1b, 0x01, 0xe4, 0x02, 0x1b, 0x03, 0x00});
    fixture.ExpectError("unexpected trailing data");
  });

  for (const std::string extension : {"fam", "bim", "bed"}) {
    run("missing " + extension + " reports path", [&] {
      std::filesystem::remove(fixture.Path(extension));
      fixture.ExpectError("Cannot open " + extension + " file: " +
                          fixture.Path(extension));
    });
  }

  std::cout << passed << " passed, " << failed << " failed\n";
  return failed == 0 ? 0 : 1;
}
