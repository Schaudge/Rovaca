#include <immintrin.h>

#include "hc_marco.h"
#include "sw_internal.h"

static inline uint32_t ro_sw_get_new_cigar(uint8_t cigar_op, uint32_t cigar_len) { return (cigar_len << BAM_CIGAR_SHIFT) | cigar_op; }

static inline uint32_t ro_sw_make_element(uint32_t state, int length)
{
    uint8_t op = WORKER_SIGAR_MAX;
    switch (state) {
        case LIB_SW_AVX_OVERHANG_STRATEGY_MATCH: op = WORKER_SIGAR_STATUS_MATCH; break;
        case LIB_SW_AVX_OVERHANG_STRATEGY_INSERT: op = WORKER_SIGAR_STATUS_INSERTION; break;
        case LIB_SW_AVX_OVERHANG_STRATEGY_DELETE: op = WORKER_SIGAR_STATUS_DELETION; break;
        case LIB_SW_AVX_OVERHANG_STRATEGY_SOFT_CLIP: op = WORKER_SIGAR_SOFT_CLIP; break;
        default: return 0;
    }
    return ro_sw_get_new_cigar(op, length);
}
#define ENABLE_PREFETCH

void ro_sw_get_cigar(p_lib_sw_avx p)
{
    int16_t* btrack = p->btrack;
    int16_t max_i = p->max_i;
    int16_t max_j = p->max_j;
    int16_t nrow = p->len1;
    int16_t ncol = p->len2;
    int32_t max_seq_len = p->max_seq_len;
    int32_t cigar_buf_length = p->cigar_max_len;
    int16_t i, j;
    int16_t* cigar_array = p->mm_aloc.cigar_buf;

    int32_t cigar_id = 0;

    i = max_i;
    j = max_j;
    if (j < ncol) {
        cigar_array[cigar_id * 2] = LIB_SW_AVX_OVERHANG_STRATEGY_SOFT_CLIP;
        cigar_array[cigar_id * 2 + 1] = (int16_t)(ncol - j);
        cigar_id++;
    }
    int state = 0;
    // 预取第一个 btrInd 对应的值
#ifdef ENABLE_PREFETCH
    int32_t first_anti_diag = i + j;
    int32_t first_btrInd;
    if (first_anti_diag <= nrow) {
        first_btrInd = first_anti_diag * max_seq_len + j - 1;
    }
    else {
        int32_t jLo = first_anti_diag - nrow - 1;
        first_btrInd = first_anti_diag * max_seq_len + j - jLo - 1;
    }
    __builtin_prefetch(&btrack[first_btrInd], 0, 3);  // 预取到 L1 缓存，用于读取
#endif
    while ((i > 0) && (j > 0)) {
        int32_t anti_diag = i + j;
        int32_t btrInd;
        if (anti_diag <= nrow) {
            btrInd = anti_diag * max_seq_len + j - 1;
        }
        else {
            int32_t jLo = anti_diag - nrow - 1;
            btrInd = anti_diag * max_seq_len + j - jLo - 1;
        }

        int32_t btr = btrack[btrInd];
#ifdef ENABLE_PREFETCH
        // 根据当前 btr 和 state 预取下一次迭代可能访问的位置
        // 这样可以提前加载数据到缓存，减少内存访问延迟
        int32_t next_i = i, next_j = j;
        if (state == LIB_SW_AVX_OVERHANG_STRATEGY_INSERT_EXT) {
            next_j = j - 1;
        }
        else if (state == LIB_SW_AVX_OVERHANG_STRATEGY_DELETE_EXT) {
            next_i = i - 1;
        }
        else {
            switch (btr & 3) {
                case LIB_SW_AVX_OVERHANG_STRATEGY_MATCH:
                    next_i = i - 1;
                    next_j = j - 1;
                    break;
                case LIB_SW_AVX_OVERHANG_STRATEGY_INSERT: next_j = j - 1; break;
                case LIB_SW_AVX_OVERHANG_STRATEGY_DELETE: next_i = i - 1; break;
            }
        }

        // 预取下一次迭代的位置
        if (next_i > 0 && next_j > 0) {
            int32_t next_anti_diag = next_i + next_j;
            int32_t next_btrInd;
            if (next_anti_diag <= nrow) {
                next_btrInd = next_anti_diag * max_seq_len + next_j - 1;
            }
            else {
                int32_t next_jLo = next_anti_diag - nrow - 1;
                next_btrInd = next_anti_diag * max_seq_len + next_j - next_jLo - 1;
            }
            __builtin_prefetch(&btrack[next_btrInd], 0, 2);  // 预取到 L2 缓存
        }
#endif
        if (state == LIB_SW_AVX_OVERHANG_STRATEGY_INSERT_EXT) {
            j--;
            cigar_array[cigar_id * 2 - 1]++;
            state = btr & LIB_SW_AVX_OVERHANG_STRATEGY_INSERT_EXT;
        }
        else if (state == LIB_SW_AVX_OVERHANG_STRATEGY_DELETE_EXT) {
            i--;
            cigar_array[cigar_id * 2 - 1]++;
            state = btr & LIB_SW_AVX_OVERHANG_STRATEGY_DELETE_EXT;
        }
        else {
            switch (btr & 3) {
                case LIB_SW_AVX_OVERHANG_STRATEGY_MATCH:
                    i--;
                    j--;
                    cigar_array[cigar_id * 2] = LIB_SW_AVX_OVERHANG_STRATEGY_MATCH;
                    cigar_array[cigar_id * 2 + 1] = 1;
                    state = 0;
                    cigar_id++;
                    break;
                case LIB_SW_AVX_OVERHANG_STRATEGY_INSERT:
                    j--;
                    cigar_array[cigar_id * 2] = LIB_SW_AVX_OVERHANG_STRATEGY_INSERT;
                    cigar_array[cigar_id * 2 + 1] = 1;
                    state = btr & LIB_SW_AVX_OVERHANG_STRATEGY_INSERT_EXT;
                    cigar_id++;
                    break;
                case LIB_SW_AVX_OVERHANG_STRATEGY_DELETE:
                    i--;
                    cigar_array[cigar_id * 2] = LIB_SW_AVX_OVERHANG_STRATEGY_DELETE;
                    cigar_array[cigar_id * 2 + 1] = 1;
                    state = btr & LIB_SW_AVX_OVERHANG_STRATEGY_DELETE_EXT;
                    cigar_id++;
                    break;
            }
        }
    }
    if (j > 0) {
        cigar_array[cigar_id * 2] = LIB_SW_AVX_OVERHANG_STRATEGY_SOFT_CLIP;
        cigar_array[cigar_id * 2 + 1] = j;
        cigar_id++;
    }
    p->alignment_offset = i;
    int16_t new_id = 0;
    int16_t prev = cigar_array[new_id * 2];
    for (i = 1; i < cigar_id; i++) {
        int16_t cur = cigar_array[i * 2];
        if (cur == prev) {
            cigar_array[new_id * 2 + 1] = (int16_t)(cigar_array[new_id * 2 + 1] + cigar_array[i * 2 + 1]);
        }
        else {
            new_id++;
            cigar_array[new_id * 2] = cur;
            cigar_array[new_id * 2 + 1] = cigar_array[i * 2 + 1];
            prev = cur;
        }
    }

    int cur_size = 0;
    for (i = new_id; i >= 0; i--) {
        if (cur_size < 0 || cigar_array[2 * i + 1] <= 0 || cur_size + 1 >= cigar_buf_length) {
            continue;
        }
        p->cigar[cur_size] = ro_sw_make_element(cigar_array[2 * i], cigar_array[2 * i + 1]);
        cur_size += 1;
    }
    p->cigar_count = cur_size;
}
