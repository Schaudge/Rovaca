#ifndef RO_SW_AXV2_16_FUNCTION_H
#define RO_SW_AXV2_16_FUNCTION_H

#include <immintrin.h>

// AVX2 INT16: 使用 16 个 int16_t lanes (两个 __m256i)
#define RO_SW_LANES 16
#define RO_SW_VEC_TYPE __m256i
#define RO_SW_MASK_TYPE __m256i

#define RO_SW_LOADU(ptr) _mm256_loadu_si256((__m256i*)(ptr))
#define RO_SW_STORE(ptr, v) _mm256_store_si256((__m256i*)(ptr), (v))
#define RO_SW_STOREU(ptr, v) _mm256_storeu_si256((__m256i*)(ptr), (v))
#define RO_SW_STREAM(ptr, v) _mm256_stream_si256((__m256i*)(ptr), (v))

#define RO_SW_SET_ZERO() _mm256_setzero_si256()
#define RO_SW_SET1(val) _mm256_set1_epi16((int16_t)(val))

#define RO_SW_ADD(v1, v2) _mm256_add_epi16((v1), (v2))
#define RO_SW_MAX(v1, v2) _mm256_max_epi16((v1), (v2))

#define RO_SW_CMPGT(v1, v2) _mm256_cmpgt_epi16((v1), (v2))
#define RO_SW_CMPGT_MASK(v1, v2) _mm256_cmpgt_epi16((v1), (v2))
#define RO_SW_CMPEQ_MASK(v1, v2) _mm256_cmpeq_epi16((v1), (v2))

#define RO_SW_AND(v1, v2) _mm256_and_si256((v1), (v2))
#define RO_SW_OR(v1, v2) _mm256_or_si256((v1), (v2))
#define RO_SW_ANDNOT(v1, v2) _mm256_andnot_si256((v1), (v2))

#define RO_SW_BLEND(v1, v2, mask) \
    (__m256i)_mm256_blendv_epi8((v1), (v2), (mask))

// INT16 版本：两次 MAIN_CODE 后，每个 __m256i 包含 16 个 int16_t
// 总共 32 个 int16_t，直接存储，不需要 pack
#define RO_SW_BACKTRACK_PERMUTE_EVEN(v1, v2) (v1)
//_mm256_permute2f128_si256((v1), (v2), 0x20)
#define RO_SW_BACKTRACK_PERMUTE_ODD(v1, v2) (v2)
//_mm256_permute2f128_si256((v1), (v2), 0x31)
// INT16 版本：pack 操作返回第一个向量（用于统一接口）
#define RO_SW_BACKTRACK_PACK(v1, v2) (v1)
// INT16 版本需要存储两个向量（32 个 int16_t）
#define RO_SW_BACKTRACK_STORE_TWO(ptr, v1, v2) \
    do { \
        RO_SW_STREAM((ptr), (v1)); \
        RO_SW_STREAM((ptr) + 16, (v2)); \
    } while(0)

#define RO_SW_INIT_CONSTANTS

#endif  // RO_SW_AXV2_16_FUNCTION_H

