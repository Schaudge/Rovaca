#include <immintrin.h>
#include <stdint.h>

#include "ro_sw.h"
#include "sw_internal.h"

// AVX512 INT32 实现
#define RO_SW_VARIANT_NAME ro_sw_avx512_run_int32
#define RO_SW_BACKTRACK_NAME ro_sw_avx512_backtrack_int32
#define RO_SW_SCORE_TYPE int32_t
#define RO_SW_LOW_INIT_VALUE RO_SW_LOW_INIT_VALUE_32
#define RO_SW_MIN_CUTOFF RO_SW_MIN_CUTOFF_32

#include "axv512-32function.h"
#include "sw_core_template.h"

// AVX512 INT16 实现
#define RO_SW_VARIANT_NAME ro_sw_avx512_run_int16
#define RO_SW_BACKTRACK_NAME ro_sw_avx512_backtrack_int16
#define RO_SW_SCORE_TYPE int16_t
#define RO_SW_LOW_INIT_VALUE RO_SW_LOW_INIT_VALUE_16
#define RO_SW_MIN_CUTOFF RO_SW_MIN_CUTOFF_16

#include "axv512-16function.h"
#include "sw_core_template.h"

