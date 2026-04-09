#include "pairhmm_api.h"
#include "pairhmm_impl.h"

#define MIN_ACCEPTED 1e-28f

namespace pairhmm {
namespace intra {

constexpr float LOG10_INITIAL_CONSTANT_F = 36.1236000061;
constexpr double LOG10_INITIAL_CONSTANT_D = 307.050595577260822;

// 注意：AVX512 编译时 GCC 也会定义 __AVX2__，所以需要先检查 AVX512
#if defined(__AVX512F__)
double computeLikelihoodsAVX512(const TestCase &tc, bool use_double) {
  float result_float =
      use_double ? 0 : PairHMMComputer<AVX512FloatTraits>::compute(tc);
  if (result_float < MIN_ACCEPTED) {
    double result_double = PairHMMComputer<AVX512DoubleTraits>::compute(tc);
    return log10(result_double) - LOG10_INITIAL_CONSTANT_D;
  } else {
    return log10f(result_float) - LOG10_INITIAL_CONSTANT_F;
  }
}
#elif defined(__AVX2__)
double computeLikelihoodsAVX2(const TestCase &tc, bool use_double) {

  float result_float =
      use_double ? 0 : PairHMMComputer<AVX2FloatTraits>::compute(tc);
  if (result_float < MIN_ACCEPTED) {
    double result_double = PairHMMComputer<AVX2DoubleTraits>::compute(tc);
    return log10(result_double) - LOG10_INITIAL_CONSTANT_D;
  } else {
    return log10f(result_float) - LOG10_INITIAL_CONSTANT_F;
  }
}
#endif

} // namespace intra
} // namespace pairhmm