#ifndef PAIRHMM_COMMON_H_
#define PAIRHMM_COMMON_H_

#include <cstdint>
#include <x86intrin.h>

namespace pairhmm {
namespace common {

// 内存对齐宏
#define ALIGNED32   __attribute__((aligned(32)))
#define ALIGNED64   __attribute__((aligned(64)))

// 分支预测优化
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

// 常量定义
constexpr uint32_t k_ambig_char = 4;
constexpr uint32_t k_bits_per_byte = 8;
constexpr uint32_t k_num_distinct_chars = 5;  // A, C, G, T, N

/**
 * @brief 测试用例结构
 * 包含 read 序列、haplotype 序列和质量分数
 */
struct TestCase {
    uint32_t rslen;   // read 序列长度
    uint32_t haplen;  // haplotype 序列长度
    const ALIGNED32 uint8_t* ALIGNED32 hap;   // haplotype 序列
    const ALIGNED32 uint8_t* ALIGNED32 rs;    // read 序列
    const ALIGNED32 uint8_t* ALIGNED32 q;     // base quality
    const ALIGNED32 uint8_t* ALIGNED32 i;     // insertion quality
    const ALIGNED32 uint8_t* ALIGNED32 d;     // deletion quality
    const ALIGNED32 uint8_t* ALIGNED32 c;     // gap continuation quality
};

/**
 * @brief 字符转换工具
 * 将 DNA 碱基字符转换为整数索引
 */
struct ConvertChar {
    static inline uint8_t k_conversion_table[20];  // 'T' - 'A' + 1 = 20
    
    static inline void init() {
        k_conversion_table['A' - 'A'] = 0;
        k_conversion_table['C' - 'A'] = 1;
        k_conversion_table['T' - 'A'] = 2;
        k_conversion_table['G' - 'A'] = 3;
        k_conversion_table['N' - 'A'] = 4;
    }
    
    static inline uint8_t get(uint8_t input) {
        return k_conversion_table[input - 'A'];
    }
};

inline void init_native()
{
    // enable FTZ
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
}

}  // namespace common
}  // namespace pairhmm

#endif  // PAIRHMM_COMMON_H_

