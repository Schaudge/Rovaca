#ifndef RO_SW_RO_SW_H
#define RO_SW_RO_SW_H

#include "smithwaterman_common.h"

#ifdef __cplusplus
extern "C" {
#endif

void ro_sw_avx2_run(p_lib_sw_avx p);
void ro_sw_avx512_run(p_lib_sw_avx p);

int ro_sw_should_use_int32_avx2(const p_lib_sw_avx p);
int ro_sw_should_use_int32_avx512(const p_lib_sw_avx p);

void ro_sw_avx2_run_int32(p_lib_sw_avx p);
void ro_sw_avx2_run_int16(p_lib_sw_avx p);
void ro_sw_avx512_run_int32(p_lib_sw_avx p);
void ro_sw_avx512_run_int16(p_lib_sw_avx p);

#ifdef __cplusplus
}
#endif

#endif  // RO_SW_RO_SW_H

