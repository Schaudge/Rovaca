#ifndef __PAIRHMM_INTERNAL_H__
#define __PAIRHMM_INTERNAL_H__
#include <cstdint>

#include "../common/enum.h"
#include "../genotype/forward.h"
#include "adapter.h"

namespace rovaca
{

struct TestCase;



void init_native();

bool string_equal(const uint8_t* str1, const uint8_t* str2, int32_t len);

void normalize_likelihoods(DoubleVector2D& log_likelihoods);

int32_t find_tandem_repeat_units(pBases bases, uint32_t offset);
int32_t find_tandem_repeat_units_rolling(pBases bases, uint32_t offset);

void apply_pcr_error_model(pBases bases, pBases gap_qual, PcrIndelModel pcr_option);
void apply_pcr_error_model_rolling(pBases bases, pBases gap_qual, PcrIndelModel pcr_option);

void filter_poorly_modelled_evidence(ReadVector& reads, DoubleVector2D& likelihoods, pMemoryPool pool);

void transpose_likelihood_matrix(const DoubleVector2D& source, DoubleVector2D& target);

int32_t find_number_of_repetitions(pBases repeat_unit_full, int32_t offset_in_repeat_unit_full, int32_t repeat_unit_length,
                                   pBases test_string_full, int32_t offset_in_test_string_full, int32_t test_string_length,
                                   bool leading_repeats);



// 使用 roDV 优化版本的 PairHMM 计算
DoubleVector2D call_roDV_pairhmm_scheduled(const HaplotypeVector& hs, ReadVector& rs, int32_t min_quality_threshold, PcrIndelModel pcr_option,
                                           pMemoryPool pool);

}  // namespace rovaca

#endif  // __PAIRHMM_INTERNAL_H__