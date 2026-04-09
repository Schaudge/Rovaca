#ifndef SIMD_TRAITS_AVX2_H_
#define SIMD_TRAITS_AVX2_H_

#include "../roDV_common/common.h"
#include <immintrin.h>
#include <x86intrin.h>

namespace pairhmm {
namespace intra {

/**
 * @brief AVX2 SIMD Traits
 * 
 * 使用编译时多态（模板）实现零开销抽象
 * 仅在编译 AVX2 版本时包含此文件
 */

// ============================================================================
// AVX2 Float (256-bit, 8 个 float)
// ============================================================================
struct AVX2FloatTraits {
    using MainType = float;
    using SimdType = __m256;
    using SimdIntType = __m256i;
    using VecIntType = __m256i;
    using MaskType = uint32_t;
    
    static constexpr uint32_t simd_width = 8;
    static constexpr uint32_t simd_bits = 256;
    static constexpr uint32_t alignment = 32;
    static constexpr MaskType mask_all_ones = 0xFFFFFFFF;
    
    // SIMD 操作 - 全部内联，编译后零开销
    static inline SimdType set1(MainType v) { return _mm256_set1_ps(v); }
    static inline SimdIntType setzero_int() { return _mm256_set1_epi32(0); }
    static inline SimdType setzero() { return _mm256_setzero_ps(); }
    static inline SimdType add(SimdType a, SimdType b) { return _mm256_add_ps(a, b); }
    static inline SimdType sub(SimdType a, SimdType b) { return _mm256_sub_ps(a, b); }
    static inline SimdType mul(SimdType a, SimdType b) { return _mm256_mul_ps(a, b); }
    static inline SimdType div(SimdType a, SimdType b) { return _mm256_div_ps(a, b); }
    static inline SimdType load(const MainType* ptr) { return _mm256_load_ps(ptr); }
    static inline void store(MainType* ptr, SimdType v) { _mm256_store_ps(ptr, v); }
    
    // 向量移位操作
    static inline void vector_shift(SimdType& x, MainType shift_in, MainType& shift_out) {
        SimdType reversed_x = _mm256_permutevar8x32_ps(x, _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7));
        shift_out = _mm256_cvtss_f32(reversed_x);
        x = _mm256_blend_ps(reversed_x, _mm256_set1_ps(shift_in), 0b00000001);
    }
    
    static inline void vector_shift_last(SimdType& x, MainType shift_in) {
        SimdType reversed_x = _mm256_permutevar8x32_ps(x, _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7));
        x = _mm256_blend_ps(reversed_x, _mm256_set1_ps(shift_in), 0b00000001);
    }
    
    // 新增方法：用于模板实现
    static inline SimdType set_lse(MainType v) {
        return _mm256_set_ps(0, 0, 0, 0, 0, 0, 0, v);
    }
       
    static inline SimdIntType set_epi_from_array(const MaskType* arr, const uint8_t* rs_arr) {
        return _mm256_set_epi32(
            static_cast<MaskType>(arr[rs_arr[7]]), static_cast<MaskType>(arr[rs_arr[6]]), 
            static_cast<MaskType>(arr[rs_arr[5]]), static_cast<MaskType>(arr[rs_arr[4]]),
            static_cast<MaskType>(arr[rs_arr[3]]), static_cast<MaskType>(arr[rs_arr[2]]),
            static_cast<MaskType>(arr[rs_arr[1]]), static_cast<MaskType>(arr[rs_arr[0]])
        );
    }
    
    static inline SimdIntType get_forward_shift_vector() {
        return _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    }
    
    static inline SimdIntType get_reserved_mask() {
        return _mm256_set_epi32(0x7F, 0x3F, 0x1F, 0x0F, 0x07, 0x03, 0x01, 0x00);
    }
    
    static inline SimdIntType get_backward_shift_vector() {
        return _mm256_set_epi32(25, 26, 27, 28, 29, 30, 31, 32);
    }
    
    static inline SimdIntType forward_shift(SimdIntType a, SimdIntType b) {
        return _mm256_srlv_epi32(a, b);
    }
    
    static inline SimdIntType backward_shift(SimdIntType a, SimdIntType b) {
        return _mm256_sllv_epi32(a, b);
    }
     
    static inline SimdIntType or_si(SimdIntType a, SimdIntType b) {
        return _mm256_or_si256(a, b);
    }
    
    static inline SimdIntType and_si(SimdIntType a, SimdIntType b) {
        return _mm256_and_si256(a, b);
    }
    
    // compute_dist_vec 实现
    static inline void compute_dist_vec(VecIntType& bit_mask_vec, SimdType& distm_chosen, 
                                       const SimdType& distm, const SimdType& _1_distm) {
        distm_chosen = _mm256_blendv_ps(distm, _1_distm, _mm256_castsi256_ps(bit_mask_vec));
        bit_mask_vec = _mm256_slli_epi32(bit_mask_vec, 1);
    }
};

// ============================================================================
// AVX2 Double (256-bit, 4 个 double)
// ============================================================================
struct AVX2DoubleTraits {
    using MainType = double;
    using SimdType = __m256d;
    using SimdIntType = __m256i;
    using VecIntType = __m256i;
    using MaskType = uint64_t;
    
    static constexpr uint32_t simd_width = 4;
    static constexpr uint32_t simd_bits = 256;
    static constexpr uint32_t alignment = 32;
    static constexpr MaskType mask_all_ones = 0xFFFFFFFFFFFFFFFF;
    
    static inline SimdType set1(MainType v) { return _mm256_set1_pd(v); }
    static inline SimdIntType setzero_int() { return _mm256_set1_epi64x(0); }
    static inline SimdType setzero() { return _mm256_setzero_pd(); }
    static inline SimdType add(SimdType a, SimdType b) { return _mm256_add_pd(a, b); }
    static inline SimdType sub(SimdType a, SimdType b) { return _mm256_sub_pd(a, b); }
    static inline SimdType mul(SimdType a, SimdType b) { return _mm256_mul_pd(a, b); }
    static inline SimdType div(SimdType a, SimdType b) { return _mm256_div_pd(a, b); }
    static inline SimdType load(const MainType* ptr) { return _mm256_load_pd(ptr); }
    static inline void store(MainType* ptr, SimdType v) { _mm256_store_pd(ptr, v); }
    
    // 整数向量操作
    static inline SimdIntType set_epi_from_array(const MaskType* arr, const uint8_t* rs_arr) {
        return _mm256_set_epi64x(
            static_cast<MaskType>(arr[rs_arr[3]]), static_cast<MaskType>(arr[rs_arr[2]]),
            static_cast<MaskType>(arr[rs_arr[1]]), static_cast<MaskType>(arr[rs_arr[0]])
        );
    }
    
    static inline SimdIntType forward_shift(SimdIntType a, SimdIntType b) {
        return _mm256_srlv_epi64(a, b);
    }
    
    static inline SimdIntType backward_shift(SimdIntType a, SimdIntType b) {
        return _mm256_sllv_epi64(a, b);
    }
    
    static inline SimdIntType or_si(SimdIntType a, SimdIntType b) {
        return _mm256_or_si256(a, b);
    }
    
    static inline SimdIntType and_si(SimdIntType a, SimdIntType b) {
        return _mm256_and_si256(a, b);
    }
    
    static inline void vector_shift(SimdType& x, MainType shift_in, MainType& shift_out) {
        SimdType reversed_x = _mm256_permute4x64_pd(x, 0b10010011);
        shift_out = _mm256_cvtsd_f64(reversed_x);
        x = _mm256_blend_pd(reversed_x, _mm256_set1_pd(shift_in), 0b0001);
    }
    
    static inline void vector_shift_last(SimdType& x, MainType shift_in) {
        SimdType reversed_x = _mm256_permute4x64_pd(x, 0b10010011);
        x = _mm256_blend_pd(reversed_x, _mm256_set1_pd(shift_in), 0b0001);
    }
    
    // 新增方法：用于模板实现
    static inline SimdType set_lse(MainType v) {
        return _mm256_set_pd(0, 0, 0, v);
    }
    
    static inline SimdIntType get_forward_shift_vector() {
        return _mm256_set_epi64x(3, 2, 1, 0);
    }
    
    static inline SimdIntType get_reserved_mask() {
        return _mm256_set_epi64x(0b111, 0b11, 0b1, 0b0);
    }
    
    static inline SimdIntType get_backward_shift_vector() {
        return _mm256_set_epi64x(61, 62, 63, 64);
    }
    
    // compute_dist_vec 实现
    static inline void compute_dist_vec(VecIntType& bit_mask_vec, SimdType& distm_chosen, 
                                       const SimdType& distm, const SimdType& _1_distm) {
        distm_chosen = _mm256_blendv_pd(distm, _1_distm, _mm256_castsi256_pd(bit_mask_vec));
        bit_mask_vec = _mm256_slli_epi64(bit_mask_vec, 1);
    }
};

}  // namespace intra
}  // namespace pairhmm

#endif  // SIMD_TRAITS_AVX2_H_

