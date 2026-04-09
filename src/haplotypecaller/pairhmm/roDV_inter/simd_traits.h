#ifndef PAIRHMM_INTER_SIMD_TRAITS_H_
#define PAIRHMM_INTER_SIMD_TRAITS_H_

/**
 * @brief Inter-PairHMM SIMD Traits 统一接口
 * 
 * 支持多个 reads 与多个单倍型的并发计算
 * 根据编译时宏定义自动包含对应的指令集实现
 */

// 根据编译标志选择性包含对应的 traits
#if defined(__AVX512F__)
    #include "simd_traits_avx512.h"
#else
    // 默认使用 AVX2（大多数现代 CPU 都支持）
    #include "simd_traits_avx2.h"
#endif

#endif  // PAIRHMM_INTER_SIMD_TRAITS_H_
