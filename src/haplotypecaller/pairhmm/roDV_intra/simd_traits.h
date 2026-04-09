#ifndef SIMD_TRAITS_H_
#define SIMD_TRAITS_H_

/**
 * @brief SIMD Traits 统一接口
 * 
 * 根据编译时宏定义自动包含对应的指令集实现
 * 这样可以避免编译时的 ABI 警告
 * 
 * 编译时需要定义以下宏之一：
 * - __AVX2__     : 使用 AVX2 指令集
 * - __AVX512F__  : 使用 AVX512 指令集
 */

// 根据编译标志选择性包含对应的 traits
#if defined(__AVX512F__)
    #include "simd_traits_avx512.h"
#elif defined(__AVX2__)
    #include "simd_traits_avx2.h"
#else
    #error "No SIMD instruction set defined. Please compile with -mavx2 or -mavx512f"
#endif

#endif  // SIMD_TRAITS_H_
