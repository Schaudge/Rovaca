#ifndef PAIRHMM_INTER_API_H_
#define PAIRHMM_INTER_API_H_

#include "pairhmm_inter.h"

namespace pairhmm {
namespace inter {
constexpr float LOG10_INITIAL_CONSTANT_F = 36.1236000061;
constexpr double LOG10_INITIAL_CONSTANT_D = 307.050595577260822;
inline double loglikelihoodfloat(double x) {
  return log10(x) - LOG10_INITIAL_CONSTANT_F;
}
inline double loglikelihooddouble(double x) {
  return log10(x) - LOG10_INITIAL_CONSTANT_D;
}

// 无参数版本（使用 DefaultAllocator）
bool compute_inter_pairhmm_AVX512_float(TestCase *tc, uint32_t num,
                                        double *results, bool islog10 = true);
bool compute_inter_pairhmm_AVX512_double(TestCase *tc, uint32_t num,
                                         double *results, bool islog10 = true);
bool compute_inter_pairhmm_AVX2_float(TestCase *tc, uint32_t num,
                                      double *results, bool islog10 = true);
bool compute_inter_pairhmm_AVX2_double(TestCase *tc, uint32_t num,
                                       double *results, bool islog10 = true);

// 带 allocator 参数的模板版本
template <typename ALLOCATOR>
bool compute_inter_pairhmm_AVX512_float(TestCase *tc, uint32_t num,
                                        double *results, ALLOCATOR &allocator,
                                        bool islog10 = true);
template <typename ALLOCATOR>
bool compute_inter_pairhmm_AVX512_double(TestCase *tc, uint32_t num,
                                         double *results, ALLOCATOR &allocator,
                                         bool islog10 = true);
template <typename ALLOCATOR>
bool compute_inter_pairhmm_AVX2_float(TestCase *tc, uint32_t num,
                                      double *results, ALLOCATOR &allocator,
                                      bool islog10 = true);
template <typename ALLOCATOR>
bool compute_inter_pairhmm_AVX2_double(TestCase *tc, uint32_t num,
                                       double *results, ALLOCATOR &allocator,
                                       bool islog10 = true);

} // namespace inter
} // namespace pairhmm

#endif // PAIRHMM_INTER_API_H_
