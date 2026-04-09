#include "pairhmm_inter_api.h"
#include "../roDV_common/cpu_features.h"
#include "../roDV_adapter/rovaca_pool_allocator.h"
#include "pairhmm_inter.h"
#include <algorithm>
#include <cmath>
#include "simd_traits.h"
#include <string.h>

namespace pairhmm {
namespace inter {

#if defined(__AVX512F__)
// 无参数版本（使用 DefaultAllocator）
bool compute_inter_pairhmm_AVX512_float(TestCase *tc, uint32_t num,
                                        double *results, bool islog10) {
  rovaca::roDV_adapter::DefaultAllocator allocator;
  return compute_inter_pairhmm_AVX512_float(tc, num, results, allocator, islog10);
}

// 带 allocator 参数的模板版本
template <typename ALLOCATOR>
bool compute_inter_pairhmm_AVX512_float(TestCase *tc, uint32_t num,
                                        double *results, ALLOCATOR &allocator,
                                        bool islog10) {
  if (num < AVX512FloatTraits::simd_width)
    return false;
  MultiTestCase<AVX512FloatTraits> mtc;
  memcpy(mtc.test_cases, tc, num * sizeof(TestCase));
  InterPairHMMComputer<AVX512FloatTraits, ALLOCATOR>::compute(mtc, allocator);
  for (uint32_t i = 0; i < AVX512FloatTraits::simd_width; i++) {
    results[i] = islog10 ? log10(mtc.results[i]) - LOG10_INITIAL_CONSTANT_F
                         : mtc.results[i];
  }
  return true;
}

// 无参数版本（使用 DefaultAllocator）
bool compute_inter_pairhmm_AVX512_double(TestCase *tc, uint32_t num,
                                         double *results, bool islog10) {
  rovaca::roDV_adapter::DefaultAllocator allocator;
  return compute_inter_pairhmm_AVX512_double(tc, num, results, allocator, islog10);
}

// 带 allocator 参数的模板版本
template <typename ALLOCATOR>
bool compute_inter_pairhmm_AVX512_double(TestCase *tc, uint32_t num,
                                         double *results, ALLOCATOR &allocator,
                                         bool islog10) {
  if (num < AVX512DoubleTraits::simd_width)
    return false;
  MultiTestCase<AVX512DoubleTraits> mtc;
  memcpy(mtc.test_cases, tc, num * sizeof(TestCase));
  InterPairHMMComputer<AVX512DoubleTraits, ALLOCATOR>::compute(mtc, allocator);
  for (uint32_t i = 0; i < AVX512DoubleTraits::simd_width; i++) {
    results[i] = islog10 ? log10(mtc.results[i]) - LOG10_INITIAL_CONSTANT_D
                         : mtc.results[i];
  }
  return true;
}
#elif defined(__AVX2__)
// 无参数版本（使用 DefaultAllocator）
bool compute_inter_pairhmm_AVX2_float(TestCase *tc, uint32_t num,
                                      double *results, bool islog10) {
  rovaca::roDV_adapter::DefaultAllocator allocator;
  return compute_inter_pairhmm_AVX2_float(tc, num, results, allocator, islog10);
}

// 带 allocator 参数的模板版本
template <typename ALLOCATOR>
bool compute_inter_pairhmm_AVX2_float(TestCase *tc, uint32_t num,
                                      double *results, ALLOCATOR &allocator,
                                      bool islog10) {
  if (num < AVX2FloatTraits::simd_width)
    return false;
  MultiTestCase<AVX2FloatTraits> mtc;
  memcpy(mtc.test_cases, tc, num * sizeof(TestCase));
  InterPairHMMComputer<AVX2FloatTraits, ALLOCATOR>::compute(mtc, allocator);
  for (uint32_t i = 0; i < AVX2FloatTraits::simd_width; i++) {
    results[i] = islog10 ? log10(mtc.results[i]) - LOG10_INITIAL_CONSTANT_F
                         : mtc.results[i];
  }
  return true;
}

// 无参数版本（使用 DefaultAllocator）
bool compute_inter_pairhmm_AVX2_double(TestCase *tc, uint32_t num,
                                       double *results, bool islog10) {
  rovaca::roDV_adapter::DefaultAllocator allocator;
  return compute_inter_pairhmm_AVX2_double(tc, num, results, allocator, islog10);
}

// 带 allocator 参数的模板版本
template <typename ALLOCATOR>
bool compute_inter_pairhmm_AVX2_double(TestCase *tc, uint32_t num,
                                       double *results, ALLOCATOR &allocator,
                                       bool islog10) {
  if (num < AVX2DoubleTraits::simd_width)
    return false;
  MultiTestCase<AVX2DoubleTraits> mtc;
  memcpy(mtc.test_cases, tc, num * sizeof(TestCase));
  InterPairHMMComputer<AVX2DoubleTraits, ALLOCATOR>::compute(mtc, allocator);
  for (uint32_t i = 0; i < AVX2DoubleTraits::simd_width; i++) {
    results[i] = islog10 ? log10(mtc.results[i]) - LOG10_INITIAL_CONSTANT_D
                         : mtc.results[i];
  }
  return true;
}
#endif

// 显式实例化模板函数（用于链接）
#if defined(__AVX512F__)
template bool compute_inter_pairhmm_AVX512_float<rovaca::roDV_adapter::DefaultAllocator>(
    TestCase *tc, uint32_t num, double *results, rovaca::roDV_adapter::DefaultAllocator &allocator, bool islog10);
template bool compute_inter_pairhmm_AVX512_double<rovaca::roDV_adapter::DefaultAllocator>(
    TestCase *tc, uint32_t num, double *results, rovaca::roDV_adapter::DefaultAllocator &allocator, bool islog10);
template bool compute_inter_pairhmm_AVX512_float<rovaca::roDV_adapter::RovacaMemoryPoolAllocator>(
    TestCase *tc, uint32_t num, double *results, rovaca::roDV_adapter::RovacaMemoryPoolAllocator &allocator, bool islog10);
template bool compute_inter_pairhmm_AVX512_double<rovaca::roDV_adapter::RovacaMemoryPoolAllocator>(
    TestCase *tc, uint32_t num, double *results, rovaca::roDV_adapter::RovacaMemoryPoolAllocator &allocator, bool islog10);
template bool compute_inter_pairhmm_AVX512_float<rovaca::roDV_adapter::PairHMMAllocator>(
    TestCase *tc, uint32_t num, double *results, rovaca::roDV_adapter::PairHMMAllocator &allocator, bool islog10);
template bool compute_inter_pairhmm_AVX512_double<rovaca::roDV_adapter::PairHMMAllocator>(
    TestCase *tc, uint32_t num, double *results, rovaca::roDV_adapter::PairHMMAllocator &allocator, bool islog10);
#elif defined(__AVX2__)
template bool compute_inter_pairhmm_AVX2_float<rovaca::roDV_adapter::DefaultAllocator>(
    TestCase *tc, uint32_t num, double *results, rovaca::roDV_adapter::DefaultAllocator &allocator, bool islog10);
template bool compute_inter_pairhmm_AVX2_double<rovaca::roDV_adapter::DefaultAllocator>(
    TestCase *tc, uint32_t num, double *results, rovaca::roDV_adapter::DefaultAllocator &allocator, bool islog10);
template bool compute_inter_pairhmm_AVX2_float<rovaca::roDV_adapter::RovacaMemoryPoolAllocator>(
    TestCase *tc, uint32_t num, double *results, rovaca::roDV_adapter::RovacaMemoryPoolAllocator &allocator, bool islog10);
template bool compute_inter_pairhmm_AVX2_double<rovaca::roDV_adapter::RovacaMemoryPoolAllocator>(
    TestCase *tc, uint32_t num, double *results, rovaca::roDV_adapter::RovacaMemoryPoolAllocator &allocator, bool islog10);
template bool compute_inter_pairhmm_AVX2_float<rovaca::roDV_adapter::PairHMMAllocator>(
    TestCase *tc, uint32_t num, double *results, rovaca::roDV_adapter::PairHMMAllocator &allocator, bool islog10);
template bool compute_inter_pairhmm_AVX2_double<rovaca::roDV_adapter::PairHMMAllocator>(
    TestCase *tc, uint32_t num, double *results, rovaca::roDV_adapter::PairHMMAllocator &allocator, bool islog10);
#endif

} // namespace inter
} // namespace pairhmm
