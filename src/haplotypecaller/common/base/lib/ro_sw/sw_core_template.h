#ifndef RO_SW_VARIANT_NAME
#error "RO_SW_VARIANT_NAME must be defined before including sw_core_template.h"
#endif

#ifndef RO_SW_BACKTRACK_NAME
#error "RO_SW_BACKTRACK_NAME must be defined before including sw_core_template.h"
#endif

#ifndef RO_SW_SCORE_TYPE
#error "RO_SW_SCORE_TYPE must be defined before including sw_core_template.h"
#endif

#ifndef RO_SW_LOW_INIT_VALUE
#error "RO_SW_LOW_INIT_VALUE must be defined before including sw_core_template.h"
#endif

#ifndef RO_SW_MIN_CUTOFF
#error "RO_SW_MIN_CUTOFF must be defined before including sw_core_template.h"
#endif

#ifndef RO_SW_VEC_TYPE
#error "RO_SW_VEC_TYPE must be defined before including sw_core_template.h"
#endif

#ifndef RO_SW_MASK_TYPE
#error "RO_SW_MASK_TYPE must be defined before including sw_core_template.h"
#endif

#ifndef RO_SW_LANES
#error "RO_SW_LANES must be defined before including sw_core_template.h"
#endif

#ifndef RO_SW_INIT_CONSTANTS
#define RO_SW_INIT_CONSTANTS
#endif

#ifndef RO_SW_LOADU
#error "RO_SW_LOADU must be defined before including sw_core_template.h"
#endif

#ifndef RO_SW_STOREU
#error "RO_SW_STOREU must be defined before including sw_core_template.h"
#endif

#ifndef RO_SW_STREAM
#define RO_SW_STREAM(ptr, v) RO_SW_STOREU((ptr), (v))
#endif

#ifndef RO_SW_SET_ZERO
#error "RO_SW_SET_ZERO must be defined before including sw_core_template.h"
#endif

#ifndef RO_SW_SET1
#error "RO_SW_SET1 must be defined before including sw_core_template.h"
#endif

#ifndef RO_SW_ADD
#error "RO_SW_ADD must be defined before including sw_core_template.h"
#endif

#ifndef RO_SW_MAX
#error "RO_SW_MAX must be defined before including sw_core_template.h"
#endif

#ifndef RO_SW_CMPGT
#error "RO_SW_CMPGT must be defined before including sw_core_template.h"
#endif

#ifndef RO_SW_CMPGT_MASK
#error "RO_SW_CMPGT_MASK must be defined before including sw_core_template.h"
#endif

#ifndef RO_SW_CMPEQ_MASK
#error "RO_SW_CMPEQ_MASK must be defined before including sw_core_template.h"
#endif

#ifndef RO_SW_AND
#error "RO_SW_AND must be defined before including sw_core_template.h"
#endif

#ifndef RO_SW_OR
#error "RO_SW_OR must be defined before including sw_core_template.h"
#endif

#ifndef RO_SW_ANDNOT
#error "RO_SW_ANDNOT must be defined before including sw_core_template.h"
#endif

#ifndef RO_SW_BLEND
#error "RO_SW_BLEND must be defined before including sw_core_template.h"
#endif

#ifndef RO_SW_BACKTRACK_PERMUTE_EVEN
#error "RO_SW_BACKTRACK_PERMUTE_EVEN must be defined before including sw_core_template.h"
#endif

#ifndef RO_SW_BACKTRACK_PERMUTE_ODD
#error "RO_SW_BACKTRACK_PERMUTE_ODD must be defined before including sw_core_template.h"
#endif

#ifndef RO_SW_BACKTRACK_PACK
#error "RO_SW_BACKTRACK_PACK must be defined before including sw_core_template.h"
#endif

#ifndef RO_SW_BACKTRACK_STORE_TWO
#error "RO_SW_BACKTRACK_STORE_TWO must be defined before including sw_core_template.h"
#endif

#include "sw_internal.h"

static void RO_SW_BACKTRACK_NAME(p_lib_sw_avx p)
{
    int32_t max_seq_len = p->max_seq_len;
    RO_SW_SCORE_TYPE* seq1_rev = (RO_SW_SCORE_TYPE*)(p->cacl_cache.seq1_rev);
    RO_SW_SCORE_TYPE* seq2 = (RO_SW_SCORE_TYPE*)(p->cacl_cache.seq2);
    int32_t nrow = p->len1;
    int32_t ncol = p->len2;
    int32_t match = p->para.match;
    int32_t mismatch = p->para.mismatch;
    int32_t open = p->para.gap_open;
    int32_t extend = p->para.gap_extend;

    RO_SW_VEC_TYPE w_match_vec = RO_SW_SET1(match);
    RO_SW_VEC_TYPE w_mismatch_vec = RO_SW_SET1(mismatch);
    RO_SW_VEC_TYPE w_open_vec = RO_SW_SET1(open);
    RO_SW_VEC_TYPE w_extend_vec = RO_SW_SET1(extend);

    int32_t i, j;
    RO_SW_SCORE_TYPE low_init_value = (RO_SW_SCORE_TYPE)RO_SW_LOW_INIT_VALUE;
    RO_SW_VEC_TYPE low_init_value_vec = RO_SW_SET1(low_init_value);
    RO_SW_VEC_TYPE min_cutoff_vec = RO_SW_SET1(RO_SW_MIN_CUTOFF);
    RO_SW_VEC_TYPE ins_vec = RO_SW_SET1(LIB_SW_AVX_OVERHANG_STRATEGY_INSERT);
    RO_SW_VEC_TYPE del_vec = RO_SW_SET1(LIB_SW_AVX_OVERHANG_STRATEGY_DELETE);
    RO_SW_VEC_TYPE ins_ext_vec = RO_SW_SET1(LIB_SW_AVX_OVERHANG_STRATEGY_INSERT_EXT);
    RO_SW_VEC_TYPE del_ext_vec = RO_SW_SET1(LIB_SW_AVX_OVERHANG_STRATEGY_DELETE_EXT);
    RO_SW_INIT_CONSTANTS;
    int32_t hwidth = max_seq_len + RO_SW_LANES;
    int32_t ewidth = max_seq_len + RO_SW_LANES;

    RO_SW_SCORE_TYPE* matrix = (RO_SW_SCORE_TYPE*)(p->mm_aloc.e);
    RO_SW_SCORE_TYPE* E = matrix;
    RO_SW_SCORE_TYPE* F = E + ewidth;
    RO_SW_SCORE_TYPE* H = E + 2 * ewidth;
    for (j = 0; j <= ncol; j += RO_SW_LANES) {
        RO_SW_STORE(F + j, low_init_value_vec);
    }
    for (i = 0; i <= nrow; i += RO_SW_LANES) {
        RO_SW_STORE(E + i, low_init_value_vec);
    }

    H[max_seq_len >> 1] = 0;

    for (i = 0; i < nrow; i++) {
        seq1_rev[max_seq_len - 1 - i] = p->seq1[i];
    }
    for (i = 0; i < ncol; i++) {
        seq2[i] = p->seq2[i];
    }

    int16_t* back_track = p->btrack;

    RO_SW_SCORE_TYPE max_score;
    if (sizeof(RO_SW_SCORE_TYPE) == sizeof(int16_t)) {
        max_score = (RO_SW_SCORE_TYPE)INT16_MIN;
    } else {
        max_score = (RO_SW_SCORE_TYPE)INT32_MIN;
    }
    int32_t max_i = 0;
    int32_t max_j = 0;

    int32_t anti_diag;
    int32_t prev, cur;
    for (anti_diag = 1; anti_diag <= (nrow + ncol); anti_diag++) {
        int32_t ilo = assemble_min(anti_diag, nrow + 1);
        int32_t jhi = assemble_min(anti_diag, ncol + 1);
        int32_t ihi = anti_diag - jhi;
        int32_t jlo = anti_diag - ilo;

        prev = ((anti_diag - 1) & 1) * hwidth;
        cur = (anti_diag & 1) * hwidth;

        for (j = (jlo + 1); j < (jhi - RO_SW_LANES);) {
            int32_t back_track_ind = j - jlo - 1;
            RO_SW_VEC_TYPE bt_vec_0, bt_vec_1;
            {
                i = anti_diag - j;
                int32_t diag = j - i;
                int32_t diagInd = max_seq_len + diag;
                int32_t inde = max_seq_len - i;
                int32_t indf = j;
                int32_t hLeftInd = prev + ((diagInd - 1) >> 1);
                int32_t hTopInd = hLeftInd + 1;
                int32_t hCurInd = cur + (diagInd >> 1);
                int32_t seq2Ind = j - 1;

                RO_SW_VEC_TYPE e10 = RO_SW_LOADU(&E[inde]);
                RO_SW_VEC_TYPE ext_score_h = RO_SW_ADD(e10, w_extend_vec);
                RO_SW_VEC_TYPE h10 = RO_SW_LOADU(&H[hLeftInd]);
                RO_SW_VEC_TYPE open_score_h = RO_SW_ADD(h10, w_open_vec);
                RO_SW_VEC_TYPE e11 = RO_SW_MAX(open_score_h, ext_score_h);
                RO_SW_VEC_TYPE open_gt_ext_h = RO_SW_CMPGT(open_score_h, ext_score_h);
                RO_SW_VEC_TYPE ext_vec = RO_SW_ANDNOT(open_gt_ext_h, ins_ext_vec);
                RO_SW_STOREU(&E[inde], e11);
                RO_SW_VEC_TYPE f01 = RO_SW_LOADU(&F[indf]);
                RO_SW_VEC_TYPE ext_score_v = RO_SW_ADD(f01, w_extend_vec);
                RO_SW_VEC_TYPE h01 = RO_SW_LOADU(&H[hTopInd]);
                RO_SW_VEC_TYPE open_score_v = RO_SW_ADD(h01, w_open_vec);
                RO_SW_VEC_TYPE f11 = RO_SW_MAX(ext_score_v, open_score_v);
                RO_SW_VEC_TYPE open_gt_ext_v = RO_SW_CMPGT(open_score_v, ext_score_v);
                ext_vec = RO_SW_OR(ext_vec, RO_SW_ANDNOT(open_gt_ext_v, del_ext_vec));
                RO_SW_STOREU(&F[indf], f11);
                RO_SW_VEC_TYPE h00 = RO_SW_LOADU(&H[hCurInd]);
                RO_SW_VEC_TYPE s1 = RO_SW_LOADU((RO_SW_VEC_TYPE*)(seq1_rev + inde));
                RO_SW_VEC_TYPE s2 = RO_SW_LOADU((RO_SW_VEC_TYPE*)(seq2 + seq2Ind));
                RO_SW_MASK_TYPE cmp11 = RO_SW_CMPEQ_MASK(s1, s2);
                RO_SW_VEC_TYPE sbt11 = RO_SW_BLEND(w_mismatch_vec, w_match_vec, cmp11);
                RO_SW_VEC_TYPE m11 = RO_SW_ADD(h00, sbt11);
                RO_SW_VEC_TYPE h11 = RO_SW_MAX(min_cutoff_vec, m11);
                RO_SW_VEC_TYPE e11_gt_h11 = RO_SW_CMPGT(e11, h11);
                h11 = RO_SW_MAX(h11, e11);
                bt_vec_0 = RO_SW_AND(ins_vec, e11_gt_h11);
                cmp11 = RO_SW_CMPGT_MASK(f11, h11);
                h11 = RO_SW_MAX(h11, f11);
                bt_vec_0 = RO_SW_BLEND(bt_vec_0, del_vec, cmp11);
                bt_vec_0 = RO_SW_OR(bt_vec_0, ext_vec);
                RO_SW_STOREU(&H[hCurInd], h11);
                j = j + RO_SW_LANES;
            }
            {
                i = anti_diag - j;
                int32_t diag = j - i;
                int32_t diagInd = max_seq_len + diag;
                int32_t inde = max_seq_len - i;
                int32_t indf = j;
                int32_t hLeftInd = prev + ((diagInd - 1) >> 1);
                int32_t hTopInd = hLeftInd + 1;
                int32_t hCurInd = cur + (diagInd >> 1);
                int32_t seq2Ind = j - 1;

                RO_SW_VEC_TYPE e10 = RO_SW_LOADU(&E[inde]);
                RO_SW_VEC_TYPE ext_score_h = RO_SW_ADD(e10, w_extend_vec);
                RO_SW_VEC_TYPE h10 = RO_SW_LOADU(&H[hLeftInd]);
                RO_SW_VEC_TYPE open_score_h = RO_SW_ADD(h10, w_open_vec);
                RO_SW_VEC_TYPE e11 = RO_SW_MAX(open_score_h, ext_score_h);
                RO_SW_VEC_TYPE open_gt_ext_h = RO_SW_CMPGT(open_score_h, ext_score_h);
                RO_SW_VEC_TYPE ext_vec = RO_SW_ANDNOT(open_gt_ext_h, ins_ext_vec);
                RO_SW_STOREU(&E[inde], e11);
                RO_SW_VEC_TYPE f01 = RO_SW_LOADU(&F[indf]);
                RO_SW_VEC_TYPE ext_score_v = RO_SW_ADD(f01, w_extend_vec);
                RO_SW_VEC_TYPE h01 = RO_SW_LOADU(&H[hTopInd]);
                RO_SW_VEC_TYPE open_score_v = RO_SW_ADD(h01, w_open_vec);
                RO_SW_VEC_TYPE f11 = RO_SW_MAX(ext_score_v, open_score_v);
                RO_SW_VEC_TYPE open_gt_ext_v = RO_SW_CMPGT(open_score_v, ext_score_v);
                ext_vec = RO_SW_OR(ext_vec, RO_SW_ANDNOT(open_gt_ext_v, del_ext_vec));
                RO_SW_STOREU(&F[indf], f11);
                RO_SW_VEC_TYPE h00 = RO_SW_LOADU(&H[hCurInd]);
                RO_SW_VEC_TYPE s1 = RO_SW_LOADU((RO_SW_VEC_TYPE*)(seq1_rev + inde));
                RO_SW_VEC_TYPE s2 = RO_SW_LOADU((RO_SW_VEC_TYPE*)(seq2 + seq2Ind));
                RO_SW_MASK_TYPE cmp11 = RO_SW_CMPEQ_MASK(s1, s2);
                RO_SW_VEC_TYPE sbt11 = RO_SW_BLEND(w_mismatch_vec, w_match_vec, cmp11);
                RO_SW_VEC_TYPE m11 = RO_SW_ADD(h00, sbt11);
                RO_SW_VEC_TYPE h11 = RO_SW_MAX(min_cutoff_vec, m11);
                RO_SW_VEC_TYPE e11_gt_h11 = RO_SW_CMPGT(e11, h11);
                h11 = RO_SW_MAX(h11, e11);
                bt_vec_1 = RO_SW_AND(ins_vec, e11_gt_h11);
                cmp11 = RO_SW_CMPGT_MASK(f11, h11);
                h11 = RO_SW_MAX(h11, f11);
                bt_vec_1 = RO_SW_BLEND(bt_vec_1, del_vec, cmp11);
                bt_vec_1 = RO_SW_OR(bt_vec_1, ext_vec);
                RO_SW_STOREU(&H[hCurInd], h11);
                j = j + RO_SW_LANES;
            }
            RO_SW_VEC_TYPE bt_vec_even = RO_SW_BACKTRACK_PERMUTE_EVEN(bt_vec_0, bt_vec_1);
            RO_SW_VEC_TYPE bt_vec_odd = RO_SW_BACKTRACK_PERMUTE_ODD(bt_vec_0, bt_vec_1);
            RO_SW_BACKTRACK_STORE_TWO(back_track + anti_diag * max_seq_len + back_track_ind, bt_vec_even, bt_vec_odd);
        }
        if (j < jhi) {
            i = anti_diag - j;
            int32_t diag = j - i;
            int32_t diagInd = max_seq_len + diag;
            int32_t inde = max_seq_len - i;
            int32_t indf = j;
            int32_t hLeftInd = prev + ((diagInd - 1) >> 1);
            int32_t hTopInd = hLeftInd + 1;
            int32_t hCurInd = cur + (diagInd >> 1);
            int32_t seq2Ind = j - 1;
            RO_SW_VEC_TYPE bt_vec_0;
            RO_SW_VEC_TYPE bt_vec_1 = RO_SW_SET_ZERO();

            RO_SW_VEC_TYPE e10 = RO_SW_LOADU(&E[inde]);
            RO_SW_VEC_TYPE ext_score_h = RO_SW_ADD(e10, w_extend_vec);
            RO_SW_VEC_TYPE h10 = RO_SW_LOADU(&H[hLeftInd]);
            RO_SW_VEC_TYPE open_score_h = RO_SW_ADD(h10, w_open_vec);
            RO_SW_VEC_TYPE e11 = RO_SW_MAX(open_score_h, ext_score_h);
            RO_SW_VEC_TYPE open_gt_ext_h = RO_SW_CMPGT(open_score_h, ext_score_h);
            RO_SW_VEC_TYPE ext_vec = RO_SW_ANDNOT(open_gt_ext_h, ins_ext_vec);
            RO_SW_STOREU(&E[inde], e11);
            RO_SW_VEC_TYPE f01 = RO_SW_LOADU(&F[indf]);
            RO_SW_VEC_TYPE ext_score_v = RO_SW_ADD(f01, w_extend_vec);
            RO_SW_VEC_TYPE h01 = RO_SW_LOADU(&H[hTopInd]);
            RO_SW_VEC_TYPE open_score_v = RO_SW_ADD(h01, w_open_vec);
            RO_SW_VEC_TYPE f11 = RO_SW_MAX(ext_score_v, open_score_v);
            RO_SW_VEC_TYPE open_gt_ext_v = RO_SW_CMPGT(open_score_v, ext_score_v);
            ext_vec = RO_SW_OR(ext_vec, RO_SW_ANDNOT(open_gt_ext_v, del_ext_vec));
            RO_SW_STOREU(&F[indf], f11);
            RO_SW_VEC_TYPE h00 = RO_SW_LOADU(&H[hCurInd]);
            RO_SW_VEC_TYPE s1 = RO_SW_LOADU((RO_SW_VEC_TYPE*)(seq1_rev + inde));
            RO_SW_VEC_TYPE s2 = RO_SW_LOADU((RO_SW_VEC_TYPE*)(seq2 + seq2Ind));
            RO_SW_MASK_TYPE cmp11 = RO_SW_CMPEQ_MASK(s1, s2);
            RO_SW_VEC_TYPE sbt11 = RO_SW_BLEND(w_mismatch_vec, w_match_vec, cmp11);
            RO_SW_VEC_TYPE m11 = RO_SW_ADD(h00, sbt11);
            RO_SW_VEC_TYPE h11 = RO_SW_MAX(min_cutoff_vec, m11);
            RO_SW_VEC_TYPE e11_gt_h11 = RO_SW_CMPGT(e11, h11);
            h11 = RO_SW_MAX(h11, e11);
            bt_vec_0 = RO_SW_AND(ins_vec, e11_gt_h11);
            cmp11 = RO_SW_CMPGT_MASK(f11, h11);
            h11 = RO_SW_MAX(h11, f11);
            bt_vec_0 = RO_SW_BLEND(bt_vec_0, del_vec, cmp11);
            bt_vec_0 = RO_SW_OR(bt_vec_0, ext_vec);
            RO_SW_STOREU(&H[hCurInd], h11);

            RO_SW_VEC_TYPE bt_vec_even = RO_SW_BACKTRACK_PERMUTE_EVEN(bt_vec_0, bt_vec_1);
            RO_SW_VEC_TYPE bt_vec_odd = RO_SW_BACKTRACK_PERMUTE_ODD(bt_vec_0, bt_vec_1);
            int32_t back_track_ind = j - jlo - 1;
            RO_SW_BACKTRACK_STORE_TWO(back_track + anti_diag * max_seq_len + back_track_ind, bt_vec_even, bt_vec_odd);
        }
        H[cur + ((max_seq_len + 2 * jhi - anti_diag) >> 1)] = 0;
        H[cur + ((max_seq_len + 2 * jlo - anti_diag) >> 1)] = 0;
        F[jhi] = low_init_value;
        E[max_seq_len - ilo] = low_init_value;

        if (ilo == (nrow + 1)) {
            RO_SW_SCORE_TYPE score = H[cur + ((max_seq_len + jlo + 1 - (ilo - 1)) >> 1)];
            if ((max_score < score) || ((max_score == score) && (abs(ilo - jlo - 2) < abs(max_i - max_j)))) {
                max_score = score;
                max_i = ilo - 1;
                max_j = jlo + 1;
            }
        }
        if (jhi == (ncol + 1)) {
            RO_SW_SCORE_TYPE score = H[cur + ((max_seq_len + jhi - 1 - (ihi + 1)) >> 1)];
            if ((max_score < score) || ((max_score == score) && ((max_j == ncol) || (abs(ihi - jhi + 2) <= abs(max_i - max_j))))) {
                max_score = score;
                max_i = ihi + 1;
                max_j = jhi - 1;
            }
        }
    }
    p->score = (int32_t)max_score;
    p->max_i = (int16_t)max_i;
    p->max_j = (int16_t)max_j;
}

void RO_SW_VARIANT_NAME(p_lib_sw_avx p)
{
    int len = assemble_max(p->len1, p->len2);

    while (len >= p->max_seq_len) {
        p->alignment_offset = 1;
        return;
    }

    RO_SW_BACKTRACK_NAME(p);
    ro_sw_get_cigar(p);
}

#undef RO_SW_VARIANT_NAME
#undef RO_SW_BACKTRACK_NAME
#undef RO_SW_SCORE_TYPE
#undef RO_SW_LOW_INIT_VALUE
#undef RO_SW_MIN_CUTOFF
#undef RO_SW_VEC_TYPE
#undef RO_SW_MASK_TYPE
#undef RO_SW_LANES
#undef RO_SW_INIT_CONSTANTS
#undef RO_SW_LOADU
#undef RO_SW_STOREU
#undef RO_SW_STREAM
#undef RO_SW_SET_ZERO
#undef RO_SW_SET1
#undef RO_SW_ADD
#undef RO_SW_MAX
#undef RO_SW_CMPGT
#undef RO_SW_CMPGT_MASK
#undef RO_SW_CMPEQ_MASK
#undef RO_SW_AND
#undef RO_SW_OR
#undef RO_SW_ANDNOT
#undef RO_SW_BLEND
#undef RO_SW_BACKTRACK_PERMUTE_EVEN
#undef RO_SW_BACKTRACK_PERMUTE_ODD
#undef RO_SW_BACKTRACK_PACK
#undef RO_SW_BACKTRACK_STORE_TWO

