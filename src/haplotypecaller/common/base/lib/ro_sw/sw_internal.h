#ifndef RO_SW_SW_INTERNAL_H
#define RO_SW_SW_INTERNAL_H

#include <immintrin.h>
#include <stdint.h>

#include "hc_marco.h"
#include "smithwaterman_common.h"

#ifdef __cplusplus
extern "C" {
#endif

void ro_sw_get_cigar(p_lib_sw_avx p);

#ifdef __cplusplus
}
#endif

#define RO_SW_LOW_INIT_VALUE_32  (INT32_MIN / 2)
#define RO_SW_LOW_INIT_VALUE_16  (INT16_MIN / 2)
#define RO_SW_MIN_CUTOFF_32      (-100000000)
#define RO_SW_MIN_CUTOFF_16      (INT16_MIN / 4)

#endif  // RO_SW_SW_INTERNAL_H

