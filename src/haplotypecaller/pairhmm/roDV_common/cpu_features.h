#ifndef PAIRHMM_CPU_FEATURES_H_
#define PAIRHMM_CPU_FEATURES_H_

#include <cstdint>

#ifndef __APPLE__
#include <cpuid.h>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace pairhmm {
namespace common {

/**
 * @brief 检查 XCR0 寄存器是否支持 YMM 寄存器（AVX/AVX2）
 * @return 如果支持返回非零值，否则返回 0
 */
inline int check_xcr0_ymm() {
    uint32_t xcr0;
#if defined(_MSC_VER)
    xcr0 = (uint32_t)_xgetbv(0);
#else
    __asm__("xgetbv" : "=a"(xcr0) : "c"(0) : "%edx");
#endif
    return ((xcr0 & 6) == 6);
}

/**
 * @brief 检查 XCR0 寄存器是否支持 ZMM 寄存器（AVX512）
 * @return 如果支持返回非零值，否则返回 0
 */
inline int check_xcr0_zmm() {
    uint32_t xcr0;
    uint32_t zmm_ymm_xmm = (7 << 5) | (1 << 2) | (1 << 1);
#if defined(_MSC_VER)
    /* min VS2010 SP1 compiler is required */
    xcr0 = (uint32_t)_xgetbv(0);
#else
    __asm__("xgetbv" : "=a"(xcr0) : "c"(0) : "%edx");
#endif
    /* check if xmm, ymm and zmm state are enabled in XCR0 */
    return ((xcr0 & zmm_ymm_xmm) == zmm_ymm_xmm);
}

/**
 * @brief CPU 指令集特性检测工具
 * 
 * 提供静态方法来检测 CPU 是否支持特定的 SIMD 指令集
 */
class CpuFeatures {
public:
    /**
     * @brief 检测当前 CPU 是否支持 AVX2 指令集
     * @return true 如果支持 AVX2，false 否则
     */
    static bool hasAVX2Support() {
        static bool checked = false;
        static bool result = false;
        
        if (!checked) {
            result = checkAVX2Support();
            checked = true;
        }
        
        return result;
    }
    
    /**
     * @brief 检测当前 CPU 是否支持 AVX512 指令集
     * @return true 如果支持 AVX512，false 否则
     */
    static bool hasAVX512Support() {
        static bool checked = false;
        static bool result = false;
        
        if (!checked) {
            result = checkAVX512Support();
            checked = true;
        }
        
        return result;
    }

private:
    /**
     * @brief 实际检测 AVX2 支持的实现
     */
    static bool checkAVX2Support() {
#ifndef __APPLE__
        uint32_t a, b, c, d;
        uint32_t osxsave_mask = (1 << 27);  // OSXSAVE
        uint32_t avx2_mask = (1 << 5);      // AVX2
        
        // step 1 - must ensure OS supports extended processor state management
        __cpuid_count(1, 0, a, b, c, d);
        if ((c & osxsave_mask) != osxsave_mask) {
            return false;
        }
        
        // step 2 - must ensure OS supports YMM registers (and XMM)
        if (!check_xcr0_ymm()) {
            return false;
        }
        
        // step 3 - must ensure AVX2 is supported
        __cpuid_count(7, 0, a, b, c, d);
        if ((b & avx2_mask) != avx2_mask) {
            return false;
        }
        
        return true;
#else
        return false;
#endif
    }
    
    /**
     * @brief 实际检测 AVX512 支持的实现
     */
    static bool checkAVX512Support() {
#ifndef __APPLE__
        uint32_t a, b, c, d;
        uint32_t osxsave_mask = (1 << 27);     // OSXSAVE
        uint32_t avx512_skx_mask = (1 << 16) | // AVX-512F
                                   (1 << 17) | // AVX-512DQ
                                   (1 << 30) | // AVX-512BW
                                   (1 << 31);  // AVX-512VL
        
        // step 1 - must ensure OS supports extended processor state management
        __cpuid_count(1, 0, a, b, c, d);
        if ((c & osxsave_mask) != osxsave_mask) {
            return false;
        }
        
        // step 2 - must ensure OS supports ZMM registers (and YMM, and XMM)
        if (!check_xcr0_zmm()) {
            return false;
        }
        
        // step 3 - must ensure AVX512 is supported
        __cpuid_count(7, 0, a, b, c, d);
        if ((b & avx512_skx_mask) != avx512_skx_mask) {
            return false;
        }
        
        return true;
#else
        return false;
#endif
    }
};

}  // namespace common
}  // namespace pairhmm

#endif  // PAIRHMM_CPU_FEATURES_H_

