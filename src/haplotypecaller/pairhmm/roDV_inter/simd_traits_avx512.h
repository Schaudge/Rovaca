#ifndef PAIRHMM_INTER_SIMD_TRAITS_AVX512_H_
#define PAIRHMM_INTER_SIMD_TRAITS_AVX512_H_

#include "../roDV_common/common.h"
#include "../roDV_common/context.h"
#include <boost/integer_fwd.hpp>
#include <immintrin.h>
#include <x86intrin.h>

namespace pairhmm {
using namespace common;
namespace inter {
/**
 * @brief AVX512 SIMD Traits for Inter-PairHMM
 *
 * 支持多个 reads 与多个单倍型的并发计算
 * 使用 AVX512 指令集（512-bit）
 */

// ============================================================================
// AVX512 Float Traits (512-bit, 16 个 float)
// ============================================================================
struct AVX512FloatTraits {
  using MainType = float;
  using SimdType = __m512;
  using SeqType = int32_t;
  using SimdIntType = __m512i;
  using MaskType = __mmask16;

  static constexpr uint32_t simd_width = 16;
  static constexpr uint32_t simd_bits = 512;
  static constexpr uint32_t alignment = 64;
  // 基础 SIMD 操作
  static inline MaskType set1_all_mask() { return 0xffff; }
  static inline SimdType set1(MainType v) { return _mm512_set1_ps(v); }
  static inline SimdType setzero() { return _mm512_setzero_ps(); }
  static inline SimdType add(SimdType a, SimdType b) {
    return _mm512_add_ps(a, b);
  }
  static inline SimdType sub(SimdType a, SimdType b) {
    return _mm512_sub_ps(a, b);
  }
  static inline SimdType mul(SimdType a, SimdType b) {
    return _mm512_mul_ps(a, b);
  }
  static inline SimdType div(SimdType a, SimdType b) {
    return _mm512_div_ps(a, b);
  }
  static inline SimdType load(const MainType *ptr) {
    return _mm512_load_ps(ptr);
  }

  static inline SimdIntType load_seqs(const SeqType *ptr) {
    return _mm512_load_epi32(ptr);
  }

  static inline MaskType test_cmpeq(SimdIntType a, SimdIntType b) {
    return _mm512_cmpeq_epi32_mask(a, b);
  }

  static inline SimdType mask_blend(MaskType mask, SimdType a, SimdType b) {
    return _mm512_mask_blend_ps(mask, a, b);
  }
  static inline void store(MainType *ptr, SimdType v) {
    _mm512_store_ps(ptr, v);
  }
  // 特殊操作：用于生成长度掩码
  static inline MaskType generate_length_mask(uint32_t read_idx,
                                              const uint32_t *lens) {
    __m512i idx = _mm512_set1_epi32(read_idx);
    __m512i len = _mm512_set_epi32(lens[15], lens[14], lens[13], lens[12], lens[11], lens[10],
                         lens[9], lens[8], lens[7], lens[6], lens[5], lens[4],
                         lens[3], lens[2], lens[1], lens[0]);
    return _mm512_cmpgt_epi32_mask(len, idx);
  }

  static inline SimdType set_init_d(const uint32_t *hap_lens) {
    MainType init_const = Context<MainType>::INITIAL_CONSTANT;
    return _mm512_set_ps(init_const / static_cast<MainType>(hap_lens[15]),
                         init_const / static_cast<MainType>(hap_lens[14]),
                         init_const / static_cast<MainType>(hap_lens[13]),
                         init_const / static_cast<MainType>(hap_lens[12]),
                         init_const / static_cast<MainType>(hap_lens[11]),
                         init_const / static_cast<MainType>(hap_lens[10]),
                         init_const / static_cast<MainType>(hap_lens[9]),
                         init_const / static_cast<MainType>(hap_lens[8]),
                         init_const / static_cast<MainType>(hap_lens[7]),
                         init_const / static_cast<MainType>(hap_lens[6]),
                         init_const / static_cast<MainType>(hap_lens[5]),
                         init_const / static_cast<MainType>(hap_lens[4]),
                         init_const / static_cast<MainType>(hap_lens[3]),
                         init_const / static_cast<MainType>(hap_lens[2]),
                         init_const / static_cast<MainType>(hap_lens[1]),
                         init_const / static_cast<MainType>(hap_lens[0]));
  }
  static inline MaskType mask_and(MaskType a, MaskType b) {
    return _kand_mask16(a, b);
  }
};

// ============================================================================
// AVX512 Double Traits (512-bit, 8 个 double)
// ============================================================================
struct AVX512DoubleTraits {
  using MainType = double;
  using SimdType = __m512d;
  using SimdIntType = __m512i;
  using SeqType = int64_t;
  using MaskType = __mmask8;

  static constexpr uint32_t simd_width = 8;
  static constexpr uint32_t simd_bits = 512;
  static constexpr uint32_t alignment = 64;
  // 基础 SIMD 操作
  static inline MaskType set1_all_mask() { return 0xff; }
  static inline SimdType set1(MainType v) { return _mm512_set1_pd(v); }
  static inline SimdType setzero() { return _mm512_setzero_pd(); }
  static inline SimdType add(SimdType a, SimdType b) {
    return _mm512_add_pd(a, b);
  }
  static inline SimdType sub(SimdType a, SimdType b) {
    return _mm512_sub_pd(a, b);
  }
  static inline SimdType mul(SimdType a, SimdType b) {
    return _mm512_mul_pd(a, b);
  }
  static inline SimdType div(SimdType a, SimdType b) {
    return _mm512_div_pd(a, b);
  }
  static inline SimdType load(const MainType *ptr) {
    return _mm512_load_pd(ptr);
  }

  static inline void store(MainType *ptr, SimdType v) {
    _mm512_store_pd(ptr, v);
  }
  static inline SimdIntType load_seqs(const SeqType *ptr) {
    return _mm512_load_epi64(ptr);
  }
  static inline MaskType test_cmpeq(SimdIntType a, SimdIntType b) {
    return _mm512_cmpeq_epi64_mask(a, b);
  }
  static inline SimdType mask_blend(MaskType mask, SimdType a, SimdType b) {
    return _mm512_mask_blend_pd(mask, a, b);
  }

  static inline SimdType set_init_d(const uint32_t *hap_lens) {
    MainType init_const = Context<MainType>::INITIAL_CONSTANT;
    return _mm512_set_pd(init_const / static_cast<MainType>(hap_lens[7]),
                         init_const / static_cast<MainType>(hap_lens[6]),
                         init_const / static_cast<MainType>(hap_lens[5]),
                         init_const / static_cast<MainType>(hap_lens[4]),
                         init_const / static_cast<MainType>(hap_lens[3]),
                         init_const / static_cast<MainType>(hap_lens[2]),
                         init_const / static_cast<MainType>(hap_lens[1]),
                         init_const / static_cast<MainType>(hap_lens[0]));
  }
  // 特殊操作：用于生成长度掩码
  static inline MaskType generate_length_mask(uint32_t read_idx,
                                              const uint32_t *lens) {
    __m512i idx = _mm512_set1_epi64(read_idx);
    __m512i len = _mm512_set_epi64(lens[7], lens[6], lens[5], lens[4], lens[3],
                                   lens[2], lens[1], lens[0]);
    return _mm512_cmpgt_epi64_mask(len, idx);
  }

  static inline MaskType mask_and(MaskType a, MaskType b) {
    return _kand_mask8(a, b);
  }
};

} // namespace inter
} // namespace pairhmm

#endif // PAIRHMM_INTER_SIMD_TRAITS_AVX512_H_
