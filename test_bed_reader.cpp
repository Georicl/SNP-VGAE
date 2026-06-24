#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
#include <iomanip>

// PLINK .bed 文件格式:
// - 前3字节: magic number (0x6c, 0x1b, 0x01)
// - 之后每4位(bits)代表一个个体的基因型:
//   00 = homozygous major (AA)
//   01 = missing (NA)
//   10 = heterozygous (AB)
//   11 = homozygous minor (BB)
// - 每个SNP占用 ceil(N_individuals / 4) 字节

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <bed_file>" << std::endl;
        return 1;
    }

    std::string bed_file = argv[1];
    std::ifstream file(bed_file, std::ios::binary);
    
    if (!file) {
        std::cerr << "Error: Cannot open file " << bed_file << std::endl;
        return 1;
    }

    // 1. 读取并验证 magic number (前3字节)
    uint8_t magic[3];
    file.read(reinterpret_cast<char*>(magic), 3);
    
    if (!file.good() || magic[0] != 0x6c || magic[1] != 0x1b || magic[2] != 0x01) {
        std::cerr << "Error: Invalid .bed file format (bad magic number)" << std::endl;
        std::cerr << "Expected: 0x6c 0x1b 0x01" << std::endl;
        std::cerr << "Got: ";
        for (int i = 0; i < 3; i++) {
            std::cerr << "0x" << std::hex << std::setw(2) << std::setfill('0') 
                      << static_cast<int>(magic[i]) << " ";
        }
        std::cerr << std::dec << std::endl;
        return 1;
    }
    
    std::cout << "✓ Valid PLINK .bed file detected" << std::endl;
    std::cout << "Magic number: 0x6c 0x1b 0x01" << std::endl;
    std::cout << std::endl;

    // 2. 读取前100个字节查看数据结构
    const int SAMPLE_BYTES = 100;  // 读取前100字节作为样本
    std::vector<uint8_t> buffer(SAMPLE_BYTES);
    file.read(reinterpret_cast<char*>(buffer.data()), SAMPLE_BYTES);
    
    std::streamsize bytes_read = file.gcount();
    std::cout << "First " << bytes_read << " bytes after magic number:" << std::endl;
    std::cout << std::endl;
    
    // 以十六进制显示
    std::cout << "Hex dump:" << std::endl;
    for (int i = 0; i < bytes_read; i++) {
        std::cout << "0x" << std::hex << std::setw(2) << std::setfill('0') 
                  << static_cast<int>(buffer[i]);
        if ((i + 1) % 16 == 0) {
            std::cout << std::endl;
        } else {
            std::cout << " ";
        }
    }
    std::cout << std::dec << std::endl;
    std::cout << std::endl;

    // 3. 尝试解码第一个SNP的基因型（假设我们知道样本数）
    // 注意：实际使用时需要从 .fam 文件获取样本数
    std::cout << "Note: To decode genotypes, you need to know the number of samples" << std::endl;
    std::cout << "      from the corresponding .fam file." << std::endl;
    std::cout << std::endl;
    
    // 示例：假设有N个样本，展示如何解码
    std::cout << "Example genotype decoding (assuming N samples):" << std::endl;
    std::cout << "Each byte contains 4 genotypes (2 bits each):" << std::endl;
    std::cout << "  Bit pattern -> Genotype" << std::endl;
    std::cout << "  00 (0x00)   -> Homozygous major (AA)" << std::endl;
    std::cout << "  01 (0x01)   -> Missing (NA)" << std::endl;
    std::cout << "  10 (0x02)   -> Heterozygous (AB)" << std::endl;
    std::cout << "  11 (0x03)   -> Homozygous minor (BB)" << std::endl;
    std::cout << std::endl;
    
    // 展示第一个字节的解码
    if (bytes_read > 0) {
        uint8_t first_byte = buffer[0];
        std::cout << "First byte: 0x" << std::hex << std::setw(2) << std::setfill('0') 
                  << static_cast<int>(first_byte) << std::dec << std::endl;
        std::cout << "  Individual 1 (bits 0-1): ";
        switch (first_byte & 0x03) {
            case 0x00: std::cout << "AA (homozygous major)"; break;
            case 0x01: std::cout << "NA (missing)"; break;
            case 0x02: std::cout << "AB (heterozygous)"; break;
            case 0x03: std::cout << "BB (homozygous minor)"; break;
        }
        std::cout << std::endl;
        
        std::cout << "  Individual 2 (bits 2-3): ";
        switch ((first_byte >> 2) & 0x03) {
            case 0x00: std::cout << "AA (homozygous major)"; break;
            case 0x01: std::cout << "NA (missing)"; break;
            case 0x02: std::cout << "AB (heterozygous)"; break;
            case 0x03: std::cout << "BB (homozygous minor)"; break;
        }
        std::cout << std::endl;
        
        std::cout << "  Individual 3 (bits 4-5): ";
        switch ((first_byte >> 4) & 0x03) {
            case 0x00: std::cout << "AA (homozygous major)"; break;
            case 0x01: std::cout << "NA (missing)"; break;
            case 0x02: std::cout << "AB (heterozygous)"; break;
            case 0x03: std::cout << "BB (homozygous minor)"; break;
        }
        std::cout << std::endl;
        
        std::cout << "  Individual 4 (bits 6-7): ";
        switch ((first_byte >> 6) & 0x03) {
            case 0x00: std::cout << "AA (homozygous major)"; break;
            case 0x01: std::cout << "NA (missing)"; break;
            case 0x02: std::cout << "AB (heterozygous)"; break;
            case 0x03: std::cout << "BB (homozygous minor)"; break;
        }
        std::cout << std::endl;
    }

    file.close();
    return 0;
}
