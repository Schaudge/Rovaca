#ifndef RO_SW_AXV512_32_FUNCTION_H
#define RO_SW_AXV512_32_FUNCTION_H

#include <immintrin.h>

#define RO_SW_LANES 16
#define RO_SW_VEC_TYPE __m512i
#define RO_SW_MASK_TYPE __mmask16

#define RO_SW_LOADU(ptr) _mm512_loadu_si512((__m512i*)(ptr))
#define RO_SW_STORE(ptr, v) _mm512_store_si512((__m512i*)(ptr), (v))
#define RO_SW_STOREU(ptr, v) _mm512_storeu_si512((__m512i*)(ptr), (v))
#define RO_SW_STREAM(ptr, v) _mm512_stream_si512((__m512i*)(ptr), (v))

#define RO_SW_SET_ZERO() _mm512_setzero_si512()
#define RO_SW_SET1(val) _mm512_set1_epi32((val))

#define RO_SW_ADD(v1, v2) _mm512_add_epi32((v1), (v2))
#define RO_SW_MAX(v1, v2) _mm512_max_epi32((v1), (v2))

#define RO_SW_CMPGT(v1, v2) _mm512_movm_epi32(_mm512_cmpgt_epi32_mask((v1), (v2)))
#define RO_SW_CMPGT_MASK(v1, v2) _mm512_cmpgt_epi32_mask((v1), (v2))
#define RO_SW_CMPEQ_MASK(v1, v2) _mm512_cmpeq_epi32_mask((v1), (v2))

#define RO_SW_AND(v1, v2) _mm512_and_si512((v1), (v2))
#define RO_SW_OR(v1, v2) _mm512_or_si512((v1), (v2))
#define RO_SW_ANDNOT(v1, v2) _mm512_andnot_si512((v1), (v2))

#define RO_SW_BLEND(v1, v2, mask) _mm512_mask_blend_epi32((mask), (v1), (v2))

// AVX512 INT32: 需要 permute 和 pack
#define RO_SW_BACKTRACK_PERMUTE_EVEN(v1, v2) \
    _mm512_permutex2var_epi32((v1), idx_even, (v2))
#define RO_SW_BACKTRACK_PERMUTE_ODD(v1, v2) \
    _mm512_permutex2var_epi32((v1) , idx_odd, (v2))
#define RO_SW_BACKTRACK_PACK(v1, v2) _mm512_packs_epi32((v1), (v2))
// INT32 版本：pack 后存储一个向量（32 个 int16_t）
#define RO_SW_BACKTRACK_STORE_TWO(ptr, v1, v2) \
    do { \
        RO_SW_VEC_TYPE packed = RO_SW_BACKTRACK_PACK((v1), (v2)); \
        RO_SW_STREAM((ptr), packed); \
    } while(0)

#define RO_SW_INIT_CONSTANTS \
    __m512i idx_even = _mm512_set_epi32(27, 26, 25, 24, 19, 18, 17, 16, 11, 10, 9, 8, 3, 2, 1, 0); \
    __m512i idx_odd = _mm512_set_epi32(31, 30, 29, 28, 23, 22, 21, 20, 15, 14, 13, 12, 7, 6, 5, 4);

#endif  // RO_SW_AXV512_32_FUNCTION_H

