#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../pairhmm_internal.h"
#include "../../genotype/genotype_macors.h"
#include "../../genotype/genotype_struct.h"
#include "../../common/utils/rovaca_memory_pool.h"
#include "../../common/enum.h"

using namespace rovaca;

// 将字符转换为 uint8_t (A=0x1, C=0x2, T=0x4, G=0x8, N=0xF)
uint8_t char_to_base(char c)
{
    switch (c) {
        case 'A':
        case 'a':
            return 0x1;
        case 'C':
        case 'c':
            return 0x2;
        case 'T':
        case 't':
            return 0x4;
        case 'G':
        case 'g':
            return 0x8;
        case 'N':
        case 'n':
            return 0xF;
        default:
            return 0xF;  // 未知字符当作 N
    }
}

// 读取 fastq 文件，提取序列
std::vector<std::string> read_fastq_sequences(const std::string& filename, size_t max_reads = 1000)
{
    std::vector<std::string> sequences;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return sequences;
    }

    std::string line;
    size_t line_count = 0;
    while (std::getline(file, line) && sequences.size() < max_reads) {
        line_count++;
        // FastQ 格式：第1行是 @开头的标识符，第2行是序列，第3行是+，第4行是质量值
        if (line_count % 4 == 2) {
            // 移除换行符和空白字符
            line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
            if (!line.empty()) {
                sequences.push_back(line);
            }
        }
    }
    return sequences;
}

// 创建 pBases 从字符串
pBases create_bases_from_string(const std::string& seq, RovacaMemoryPool* pool)
{
    const size_t len = seq.size();
    pBases bases = new ALLOC_FLEXIBLE_IN_POOL(pool, Bases, len, uint8_t) Bases{static_cast<uint32_t>(len)};
    for (size_t i = 0; i < len; ++i) {
        bases->data[i] = char_to_base(seq[i]);
    }
    return bases;
}

// 创建 gap_qual，初始值全部为 '-'
pBases create_gap_qual(uint32_t len, RovacaMemoryPool* pool)
{
    pBases gap_qual = new ALLOC_FLEXIBLE_IN_POOL(pool, Bases, len, uint8_t) Bases{len};
    // '-' 的 ASCII 值是 45，但这里应该使用质量值
    // 根据代码逻辑，gap_qual 应该是质量值，初始值设为最大值 40
    for (uint32_t i = 0; i < len; ++i) {
        gap_qual->data[i] = 40;  // 初始质量值设为 40
    }
    return gap_qual;
}

// 复制 gap_qual
pBases copy_gap_qual(pBases src, RovacaMemoryPool* pool)
{
    pBases dst = new ALLOC_FLEXIBLE_IN_POOL(pool, Bases, src->num, uint8_t) Bases{src->num};
    std::memcpy(dst->data, src->data, src->num * sizeof(uint8_t));
    return dst;
}

// 比较两个 gap_qual 是否一致
bool compare_gap_qual(pBases gap1, pBases gap2)
{
    if (gap1->num != gap2->num) {
        return false;
    }
    for (uint32_t i = 0; i < gap1->num; ++i) {
        if (gap1->data[i] != gap2->data[i]) {
            return false;
        }
    }
    return true;
}

// 打印 gap_qual 用于调试
void print_gap_qual(pBases gap_qual, const std::string& label)
{
    std::cout << label << ": ";
    for (uint32_t i = 0; i < gap_qual->num; ++i) {
        std::cout << static_cast<int>(gap_qual->data[i]) << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <fastq_file> [max_reads]" << std::endl;
        return 1;
    }

    std::string fastq_file = argv[1];
    size_t max_reads = 1000;
    if (argc >= 3) {
        max_reads = std::stoul(argv[2]);
    }

    // 读取 fastq 文件
    std::cout << "Reading FASTQ file: " << fastq_file << std::endl;
    std::vector<std::string> sequences = read_fastq_sequences(fastq_file, max_reads);
    if (sequences.empty()) {
        std::cerr << "Error: No sequences found in file" << std::endl;
        return 1;
    }
    std::cout << "Loaded " << sequences.size() << " sequences" << std::endl;

    // 创建内存池
    const size_t pool_size = 10 * 1024 * 1024;  // 10MB
    std::vector<uint8_t> pool_buffer(pool_size);
    RovacaMemoryPool pool(pool_buffer.data(), pool_size);

    // 统计信息
    size_t total_tests = 0;
    size_t passed_tests = 0;
    size_t failed_tests = 0;
    double total_time_original = 0.0;
    double total_time_rolling = 0.0;

    // 测试所有 PCR 模式
    std::vector<PcrIndelModel> pcr_modes = {
        PcrIndelModel::HOSTILE,
        PcrIndelModel::AGGRESSIVE,
        PcrIndelModel::CONSERVATIVE
    };

    for (const auto& seq : sequences) {
        if (seq.empty() || seq.size() > 200) {
            continue;  // 跳过空序列或过长的序列
        }

        // 创建 bases
        pBases bases = create_bases_from_string(seq, &pool);

        for (auto pcr_mode : pcr_modes) {
            // 创建两个 gap_qual 副本
            pBases gap_qual_original = create_gap_qual(bases->num, &pool);
            pBases gap_qual_rolling = create_gap_qual(bases->num, &pool);

            // 测试原始版本
            auto start_original = std::chrono::high_resolution_clock::now();
            apply_pcr_error_model(bases, gap_qual_original, pcr_mode);
            auto end_original = std::chrono::high_resolution_clock::now();
            auto duration_original = std::chrono::duration_cast<std::chrono::microseconds>(end_original - start_original);
            total_time_original += duration_original.count() / 1000.0;  // 转换为毫秒

            // 重新创建 gap_qual_rolling（因为原始函数会修改它）
            gap_qual_rolling = create_gap_qual(bases->num, &pool);

            // 测试 rolling hash 版本
            auto start_rolling = std::chrono::high_resolution_clock::now();
            apply_pcr_error_model_rolling(bases, gap_qual_rolling, pcr_mode);
            auto end_rolling = std::chrono::high_resolution_clock::now();
            auto duration_rolling = std::chrono::duration_cast<std::chrono::microseconds>(end_rolling - start_rolling);
            total_time_rolling += duration_rolling.count() / 1000.0;  // 转换为毫秒

            // 比较结果
            total_tests++;
            bool match = compare_gap_qual(gap_qual_original, gap_qual_rolling);
            if (match) {
                passed_tests++;
            } else {
                failed_tests++;
                std::cout << "\nMismatch found!" << std::endl;
                std::cout << "Sequence: " << seq.substr(0, 50) << (seq.size() > 50 ? "..." : "") << std::endl;
                std::cout << "PCR Mode: " << static_cast<int>(pcr_mode) << std::endl;
                print_gap_qual(gap_qual_original, "Original");
                print_gap_qual(gap_qual_rolling, "Rolling");
            }
        }

        // 每处理 100 个序列打印一次进度
        if (total_tests % 300 == 0) {
            std::cout << "Processed " << total_tests << " tests, " << passed_tests << " passed, " << failed_tests << " failed" << std::endl;
        }
    }

    // 打印统计信息
    std::cout << "\n========================================\n";
    std::cout << "Test Summary:\n";
    std::cout << "  Total tests: " << total_tests << std::endl;
    std::cout << "  Passed: " << passed_tests << std::endl;
    std::cout << "  Failed: " << failed_tests << std::endl;
    std::cout << "\nPerformance:\n";
    std::cout << "  Original method total time: " << total_time_original << " ms" << std::endl;
    std::cout << "  Rolling hash method total time: " << total_time_rolling << " ms" << std::endl;
    if (total_tests > 0) {
        std::cout << "  Original method avg time: " << (total_time_original / total_tests) << " ms" << std::endl;
        std::cout << "  Rolling hash method avg time: " << (total_time_rolling / total_tests) << " ms" << std::endl;
        if (total_time_original > 0) {
            double speedup = total_time_original / total_time_rolling;
            std::cout << "  Speedup: " << speedup << "x" << std::endl;
        }
    }
    std::cout << "========================================\n";

    return (failed_tests == 0) ? 0 : 1;
}

