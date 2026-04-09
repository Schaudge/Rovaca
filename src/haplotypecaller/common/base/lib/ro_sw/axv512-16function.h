#ifndef RO_SW_AXV512_16_FUNCTION_H
#define RO_SW_AXV512_16_FUNCTION_H

#include <immintrin.h>

// AVX512 INT16: 使用 32 个 int16_t lanes (一个 __m512i)
// 一轮执行两次 MAIN_CODE，变成 64 个元素/迭代
#define RO_SW_LANES 32
#define RO_SW_VEC_TYPE __m512i
#define RO_SW_MASK_TYPE __mmask32

#define RO_SW_LOADU(ptr) _mm512_loadu_si512((__m512i*)(ptr))
#define RO_SW_STORE(ptr, v) _mm512_store_si512((__m512i*)(ptr), (v))
#define RO_SW_STOREU(ptr, v) _mm512_storeu_si512((__m512i*)(ptr), (v))
#define RO_SW_STREAM(ptr, v) _mm512_stream_si512((__m512i*)(ptr), (v))

#define RO_SW_SET_ZERO() _mm512_setzero_si512()
#define RO_SW_SET1(val) _mm512_set1_epi16((int16_t)(val))

#define RO_SW_ADD(v1, v2) _mm512_add_epi16((v1), (v2))
#define RO_SW_MAX(v1, v2) _mm512_max_epi16((v1), (v2))

#define RO_SW_CMPGT(v1, v2) _mm512_movm_epi16(_mm512_cmpgt_epi16_mask((v1), (v2)))
#define RO_SW_CMPGT_MASK(v1, v2) _mm512_cmpgt_epi16_mask((v1), (v2))
#define RO_SW_CMPEQ_MASK(v1, v2) _mm512_cmpeq_epi16_mask((v1), (v2))

#define RO_SW_AND(v1, v2) _mm512_and_si512((v1), (v2))
#define RO_SW_OR(v1, v2) _mm512_or_si512((v1), (v2))
#define RO_SW_ANDNOT(v1, v2) _mm512_andnot_si512((v1), (v2))

#define RO_SW_BLEND(v1, v2, mask) _mm512_mask_blend_epi16((mask), (v1), (v2))

// AVX512 INT16: 两次 MAIN_CODE 后，每个 __m512i 包含 32 个 int16_t
// 总共 64 个 int16_t，需要 permute 重新排列
// 注意：AVX512 没有直接的 permute2var_epi16，使用 permutex2var_epi32
#define RO_SW_BACKTRACK_PERMUTE_EVEN(v1, v2) (v1)
#define RO_SW_BACKTRACK_PERMUTE_ODD(v1, v2) (v2)
// INT16 版本：pack 操作返回第一个向量（用于统一接口）
#define RO_SW_BACKTRACK_PACK(v1, v2) (v1)
// INT16 版本需要存储两个向量（64 个 int16_t）
#define RO_SW_BACKTRACK_STORE_TWO(ptr, v1, v2) \
    do { \
        RO_SW_STREAM((ptr), (v1)); \
        RO_SW_STREAM((ptr) + 32, (v2)); \
    } while(0)

#define RO_SW_INIT_CONSTANTS 

#endif  // RO_SW_AXV512_16_FUNCTION_H

