#include <limits.h>

#include "ro_sw.h"

static inline int ro_sw_min_int(int a, int b)
{
    return (a < b) ? a : b;
}

inline int ro_sw_should_use_int32_avx2(const p_lib_sw_avx p)
{
    const int read_len = p->len2;
    if (read_len < 8) {
        return 1;
    }

    if (p->para.match <= 0) {
        return 1;
    }

    const int min_len = ro_sw_min_int(p->len1, p->len2);
    const long long max_score = (long long)min_len * (long long)p->para.match;
    return max_score > INT16_MAX;
}

 inline int ro_sw_should_use_int32_avx512(const p_lib_sw_avx p)
{
    const int read_len = p->len2;
    if (read_len < 16) {
        return 1;
    }

    if (p->para.match <= 0) {
        return 1;
    }

    const int min_len = ro_sw_min_int(p->len1, p->len2);
    const long long max_score = (long long)min_len * (long long)p->para.match;
    return max_score > INT16_MAX;
}

void ro_sw_avx2_run(p_lib_sw_avx p)
{
    if (ro_sw_should_use_int32_avx2(p)) {
        ro_sw_avx2_run_int32(p);
    }
    else {
        ro_sw_avx2_run_int16(p);
    }
}

void ro_sw_avx512_run(p_lib_sw_avx p)
{
    if (ro_sw_should_use_int32_avx512(p)) {
        ro_sw_avx512_run_int32(p);
    }
    else {
        ro_sw_avx512_run_int16(p);
    }
}

