#ifndef RO_SW_AXV2_32_FUNCTION_H
#define RO_SW_AXV2_32_FUNCTION_H

#include <immintrin.h>

#define RO_SW_LANES 8
#define RO_SW_VEC_TYPE __m256i
#define RO_SW_MASK_TYPE __m256i

#define RO_SW_LOADU(ptr) _mm256_loadu_si256((__m256i*)(ptr))
#define RO_SW_STORE(ptr, v) _mm256_store_si256((__m256i*)(ptr), (v))
#define RO_SW_STOREU(ptr, v) _mm256_storeu_si256((__m256i*)(ptr), (v))
#define RO_SW_STREAM(ptr, v) _mm256_stream_si256((__m256i*)(ptr), (v))

#define RO_SW_SET_ZERO() _mm256_setzero_si256()
#define RO_SW_SET1(val) _mm256_set1_epi32((val))

#define RO_SW_ADD(v1, v2) _mm256_add_epi32((v1), (v2))
#define RO_SW_MAX(v1, v2) _mm256_max_epi32((v1), (v2))

#define RO_SW_CMPGT(v1, v2) _mm256_cmpgt_epi32((v1), (v2))
#define RO_SW_CMPGT_MASK(v1, v2) _mm256_cmpgt_epi32((v1), (v2))
#define RO_SW_CMPEQ_MASK(v1, v2) _mm256_cmpeq_epi32((v1), (v2))

#define RO_SW_AND(v1, v2) _mm256_and_si256((v1), (v2))
#define RO_SW_OR(v1, v2) _mm256_or_si256((v1), (v2))
#define RO_SW_ANDNOT(v1, v2) _mm256_andnot_si256((v1), (v2))

#define RO_SW_BLEND(v1, v2, mask) \
    (__m256i)_mm256_blendv_ps((__m256)(v1), (__m256)(v2), (__m256)(mask))

#define RO_SW_BACKTRACK_PERMUTE_EVEN(v1, v2) _mm256_permute2f128_si256((v1), (v2), 0x20)
#define RO_SW_BACKTRACK_PERMUTE_ODD(v1, v2) _mm256_permute2f128_si256((v1), (v2), 0x31)
#define RO_SW_BACKTRACK_PACK(v1, v2) _mm256_packs_epi32((v1), (v2))
// INT32 版本：pack 后存储一个向量（16 个 int16_t）
#define RO_SW_BACKTRACK_STORE_TWO(ptr, v1, v2) \
    do { \
        RO_SW_VEC_TYPE packed = RO_SW_BACKTRACK_PACK((v1), (v2)); \
        RO_SW_STREAM((ptr), packed); \
    } while(0)

#define RO_SW_INIT_CONSTANTS

#endif  // RO_SW_AXV2_32_FUNCTION_H

