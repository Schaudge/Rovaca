#ifndef PAIRHMM_INTER_H_
#define PAIRHMM_INTER_H_

#include "../roDV_common/common.h"
#include "simd_traits.h"
#include <cstdint>

namespace pairhmm {
namespace inter {

using namespace pairhmm::common;

template <typename Traits> struct MultiTestCase {
  uint32_t min_haplen;
  uint32_t max_haplen;
  uint32_t min_rslen;
  uint32_t max_rslen;
  // alloc must alignment to Traits::alignment
  typename Traits::SeqType *rs_seqs;
  typename Traits::SeqType *hap_seqs;

  typename Traits::MainType *mm;
  typename Traits::MainType *mi;
  typename Traits::MainType *ii;
  typename Traits::MainType *md;
  typename Traits::MainType *dd;
  typename Traits::MainType *gapm;
  typename Traits::MainType *distm;
  typename Traits::MainType *_1_distm;

  typename Traits::MainType results[Traits::simd_width];
  TestCase test_cases[Traits::simd_width];
};

/**
 * @brief Inter-PairHMM 计算器
 *
 * 使用模板实现多个 reads 与多个单倍型的并发计算
 * 支持 AVX2 和 AVX512 指令集
 *
 * @tparam Traits SIMD 特征类
 */
template <typename Traits, typename ALLOCATOR> class InterPairHMMComputer {
public:
  using MainType = typename Traits::MainType;
  using SimdType = typename Traits::SimdType;
  using SimdIntType = typename Traits::SimdIntType;
  using MaskType = typename Traits::MaskType;
  using SeqType = typename Traits::SeqType;


  static constexpr uint32_t simd_width = Traits::simd_width;

  static void compute(MultiTestCase<Traits> &tc, ALLOCATOR &allocator);
private:

  static inline void precompute(MultiTestCase<Traits> &tc, ALLOCATOR &allocator);

  static inline void finalize(MultiTestCase<Traits> &tc, ALLOCATOR &allocator);


  static inline void initialize_matrices(const MultiTestCase<Traits> &tc, SimdType *mm,
                                  SimdType *ii, SimdType *dd,
                                  uint32_t *hap_lens);

  __attribute__((always_inline, hot)) static inline void process_matrix_cell(
      const SimdIntType &rbase, const SimdIntType &h, const SimdType &distm,
      const SimdType &_1_distm, const SimdType &p_mm, const SimdType &p_gapm,
      const SimdType &p_mx, const SimdType &p_xx, const SimdType &p_my,
      const SimdType &p_yy, SimdType &M, SimdType &I, SimdType &D,
      SimdType &M_i1, SimdType &I_i1, SimdType &D_i1, SimdType &M_j1,
      SimdType &I_j1, SimdType &D_j1, SimdType &M_i1j1, SimdType &I_i1j1,
      SimdType &D_i1j1, SimdType *__restrict__ mm, SimdType *__restrict__ ii,
      SimdType *__restrict__ dd, int j, MaskType len_mask);

  static inline void load_parameters_for_read(const MultiTestCase<Traits> &tc, int i,
                                       SimdType &distm, SimdType &_1_distm,
                                       SimdType &p_gapm, SimdType &p_mm,
                                       SimdType &p_mx, SimdType &p_xx,
                                       SimdType &p_my, SimdType &p_yy);

  static inline void init_row_states(
      uint32_t i, const uint32_t *hap_lens, SimdType *mm, SimdType *ii,
      SimdType *dd, SimdType &M_j1, SimdType &I_j1, SimdType &D_j1,
      SimdType &M_i1j1, SimdType &I_i1j1, SimdType &D_i1j1, SimdType &M_i1,
      SimdType &I_i1, SimdType &D_i1);
};

} // namespace inter
} // namespace pairhmm

#endif // PAIRHMM_INTER_H_
