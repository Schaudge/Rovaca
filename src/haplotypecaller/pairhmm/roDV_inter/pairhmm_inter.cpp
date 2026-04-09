#include "pairhmm_inter.h"
#include "../roDV_common/context.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include "../roDV_adapter/rovaca_pool_allocator.h"
using namespace pairhmm::common; // 使用 common 命名空间的 Context


namespace pairhmm {
namespace inter {


template <typename Traits, typename ALLOCATOR>
void InterPairHMMComputer<Traits, ALLOCATOR>::precompute(MultiTestCase<Traits> &tc,
                                              ALLOCATOR &allocator) {

  uint32_t min_rslen = UINT32_MAX;
  uint32_t max_rslen = 0;
  uint32_t min_haplen = UINT32_MAX;
  uint32_t max_haplen = 0;
  for (uint32_t i = 0; i < Traits::simd_width; i++) {
    min_rslen = std::min(min_rslen, tc.test_cases[i].rslen);
    max_rslen = std::max(max_rslen, tc.test_cases[i].rslen);
    min_haplen = std::min(min_haplen, tc.test_cases[i].haplen);
    max_haplen = std::max(max_haplen, tc.test_cases[i].haplen);
  }
  tc.min_rslen = min_rslen;
  tc.max_rslen = max_rslen;
  tc.min_haplen = min_haplen;
  tc.max_haplen = max_haplen;

  int alloc_bytes = tc.max_rslen * Traits::simd_width * sizeof(MainType);
  tc.distm = static_cast<MainType *>(
      allocator.allocate(alloc_bytes, Traits::alignment));
  tc._1_distm = static_cast<MainType *>(
      allocator.allocate(alloc_bytes, Traits::alignment));
  tc.gapm = static_cast<MainType *>(
      allocator.allocate(alloc_bytes, Traits::alignment));
  tc.mm = static_cast<MainType *>(
      allocator.allocate(alloc_bytes, Traits::alignment));
  tc.mi = static_cast<MainType *>(
      allocator.allocate(alloc_bytes, Traits::alignment));
  tc.ii = static_cast<MainType *>(
      allocator.allocate(alloc_bytes, Traits::alignment));
  tc.md = static_cast<MainType *>(
      allocator.allocate(alloc_bytes, Traits::alignment));
  tc.dd = static_cast<MainType *>(
      allocator.allocate(alloc_bytes, Traits::alignment));

  alloc_bytes = tc.max_rslen * Traits::simd_width * sizeof(SeqType);
  tc.rs_seqs = static_cast<SeqType *>(
      allocator.allocate(alloc_bytes, Traits::alignment));
  alloc_bytes = tc.max_haplen * Traits::simd_width * sizeof(SeqType);
  tc.hap_seqs = static_cast<SeqType *>(
      allocator.allocate(alloc_bytes, Traits::alignment));
  Context<MainType> ctx;
  constexpr MainType one_third = MainType(1.0) / MainType(3.0);
  constexpr MainType one = MainType(1.0);

  for(uint32_t i = 0; i < tc.min_rslen; i++) {
    for(uint32_t j = 0; j < Traits::simd_width; j++) {
      tc.rs_seqs[i*Traits::simd_width + j] = tc.test_cases[j].rs[i];
      int _i = tc.test_cases[j].i[i] & 127;
      int _d = tc.test_cases[j].d[i] & 127;
      int _c = tc.test_cases[j].c[i] & 127;
      int _q = tc.test_cases[j].q[i] & 127;
      MainType dist = Context<MainType>::ph2pr[_q];
      tc.distm[i*Traits::simd_width + j] = dist * one_third;
      tc._1_distm[i*Traits::simd_width + j] = one - dist;
      tc.gapm[i*Traits::simd_width + j] = one - Context<MainType>::ph2pr[_c];
      tc.mm[i*Traits::simd_width + j] = ctx.set_mm_prob(_i, _d);
      tc.mi[i*Traits::simd_width + j] = Context<MainType>::ph2pr[_i];
      tc.ii[i*Traits::simd_width + j] = Context<MainType>::ph2pr[_c];
      tc.md[i*Traits::simd_width + j] = Context<MainType>::ph2pr[_d];
      tc.dd[i*Traits::simd_width + j] = Context<MainType>::ph2pr[_c];
    }
  }
  for (uint32_t i = 0; i < Traits::simd_width; i++) {
    for (uint32_t j = tc.min_rslen; j < tc.max_rslen; j++) {
      tc.rs_seqs[i + j * Traits::simd_width] = tc.test_cases[i].rs[j];
      int _i = tc.test_cases[i].i[j] & 127;
      int _d = tc.test_cases[i].d[j] & 127;
      int _c = tc.test_cases[i].c[j] & 127;
      int _q = tc.test_cases[i].q[j] & 127;
      MainType dist = Context<MainType>::ph2pr[_q];
      tc.distm[i + j * Traits::simd_width] = dist * one_third;
      tc._1_distm[i + j * Traits::simd_width] = one - dist;
      tc.gapm[i + j * Traits::simd_width] = one - Context<MainType>::ph2pr[_c];
      tc.mm[i + j * Traits::simd_width] = ctx.set_mm_prob(_i, _d);
      tc.mi[i + j * Traits::simd_width] = Context<MainType>::ph2pr[_i];
      tc.ii[i + j * Traits::simd_width] = Context<MainType>::ph2pr[_c];
      tc.md[i + j * Traits::simd_width] = Context<MainType>::ph2pr[_d];
      tc.dd[i + j * Traits::simd_width] = Context<MainType>::ph2pr[_c];
    }
    for (uint32_t j = 0; j < tc.test_cases[i].haplen; j++) {
      tc.hap_seqs[i + j * Traits::simd_width] = tc.test_cases[i].hap[j];
    }
  }
}

// 带allocator参数的finalize实现
template <typename Traits, typename ALLOCATOR>
void InterPairHMMComputer<Traits, ALLOCATOR>::finalize(MultiTestCase<Traits> &tc,
                                            ALLOCATOR &allocator) {

  int alloc_bytes = tc.max_rslen * Traits::simd_width * sizeof(MainType);
  allocator.deallocate(tc.distm, alloc_bytes, Traits::alignment);
  allocator.deallocate(tc._1_distm, alloc_bytes, Traits::alignment);
  allocator.deallocate(tc.gapm, alloc_bytes, Traits::alignment);
  allocator.deallocate(tc.mm, alloc_bytes, Traits::alignment);
  allocator.deallocate(tc.mi, alloc_bytes, Traits::alignment);
  allocator.deallocate(tc.ii, alloc_bytes, Traits::alignment);
  allocator.deallocate(tc.md, alloc_bytes, Traits::alignment);
  allocator.deallocate(tc.dd, alloc_bytes, Traits::alignment);

  alloc_bytes = tc.max_rslen * Traits::simd_width * sizeof(SeqType);
  allocator.deallocate(tc.rs_seqs, alloc_bytes, Traits::alignment);
  alloc_bytes = tc.max_haplen * Traits::simd_width * sizeof(SeqType);
  allocator.deallocate(tc.hap_seqs, alloc_bytes, Traits::alignment);
}

template <typename Traits, typename ALLOCATOR>
void InterPairHMMComputer<Traits, ALLOCATOR>::load_parameters_for_read(
    const MultiTestCase<Traits> &tc, int i, SimdType &distm, SimdType &_1_distm,
    SimdType &p_gapm, SimdType &p_mm, SimdType &p_mx, SimdType &p_xx,
    SimdType &p_my, SimdType &p_yy) {
  distm = Traits::load(tc.distm + i * Traits::simd_width);
  _1_distm = Traits::load(tc._1_distm + i * Traits::simd_width);
  p_gapm = Traits::load(tc.gapm + i * Traits::simd_width);
  p_mm = Traits::load(tc.mm + i * Traits::simd_width);
  p_mx = Traits::load(tc.mi + i * Traits::simd_width);
  p_xx = Traits::load(tc.ii + i * Traits::simd_width);
  p_my = Traits::load(tc.md + i * Traits::simd_width);
  p_yy = Traits::load(tc.dd + i * Traits::simd_width);
}

template <typename Traits, typename ALLOCATOR>
void InterPairHMMComputer<Traits, ALLOCATOR>::compute(MultiTestCase<Traits> &tc, ALLOCATOR &allocator) {

  uint32_t hap_lens[Traits::simd_width];
  uint32_t rs_lens[Traits::simd_width];

  precompute(tc, allocator);
  for (uint32_t i = 0; i < Traits::simd_width; i++) {
    hap_lens[i] = tc.test_cases[i].haplen;
    rs_lens[i] = tc.test_cases[i].rslen;
  }
  SimdType mm[tc.max_haplen + 1], ii[tc.max_haplen + 1], dd[tc.max_haplen + 1];
  SimdType distm, _1_distm;
  SimdType p_gapm, p_mm, p_mx, p_xx, p_my, p_yy;
  SimdType M, M_i1, M_j1, M_i1j1, I, I_i1, I_j1, I_i1j1, D, D_i1, D_j1, D_i1j1;

  initialize_matrices(tc, mm, ii, dd, hap_lens);

  for (uint32_t i = 0; i < tc.min_rslen; i++) {
    SimdIntType rbase = Traits::load_seqs(tc.rs_seqs + i * Traits::simd_width);
    init_row_states(i, hap_lens, mm, ii, dd, M_j1, I_j1, D_j1, M_i1j1, I_i1j1,
                    D_i1j1, M_i1, I_i1, D_i1);
    load_parameters_for_read(tc, i, distm, _1_distm, p_gapm, p_mm, p_mx, p_xx,
                             p_my, p_yy);

    for (uint32_t j = 0; j < tc.min_haplen; j++) {
      SimdIntType h = Traits::load_seqs(tc.hap_seqs + j * Traits::simd_width);
      MaskType mask = Traits::test_cmpeq(rbase, h);
      SimdType distm_chosen = Traits::mask_blend(mask, distm, _1_distm);
      // 优化：预先计算中间值，减少临时变量
      SimdType temp1 = Traits::mul(M_i1j1, p_mm);
      SimdType temp2 = Traits::mul(I_i1j1, p_gapm);
      SimdType temp3 = Traits::mul(D_i1j1, p_gapm);
      SimdType sum = Traits::add(Traits::add(temp1, temp2), temp3);
      M = Traits::mul(sum, distm_chosen);
      // I 和 D 的计算可以并行
      I = Traits::add(Traits::mul(M_i1, p_mx), Traits::mul(I_i1, p_xx));
      D = Traits::add(Traits::mul(M_j1, p_my), Traits::mul(D_j1, p_yy));

      M_i1j1 = M_i1;
      I_i1j1 = I_i1;
      D_i1j1 = D_i1;

      M_j1 = M;
      I_j1 = I;
      D_j1 = D;

      mm[j] = M;
      ii[j] = I;
      dd[j] = D;

      // 准备下一次迭代
      M_i1 = mm[j + 1];
      I_i1 = ii[j + 1];
      D_i1 = dd[j + 1];
    }
    // MASK Haplotypes
    for (uint32_t j = tc.min_haplen; j < tc.max_haplen; j++) {
      SimdIntType h = Traits::load_seqs(tc.hap_seqs + j * Traits::simd_width);
      process_matrix_cell(rbase, h, distm, _1_distm, p_mm, p_gapm, p_mx, p_xx,
                          p_my, p_yy, M, I, D, M_i1, I_i1, D_i1, M_j1, I_j1,
                          D_j1, M_i1j1, I_i1j1, D_i1j1, mm, ii, dd, j,
                          Traits::generate_length_mask(j, hap_lens));
    }
  }
  for (uint32_t i = tc.min_rslen; i < tc.max_rslen; i++) {
    SimdIntType rbase = Traits::load_seqs(tc.rs_seqs + i * Traits::simd_width);
    // MASK Reads
    load_parameters_for_read(tc, i, distm, _1_distm, p_gapm, p_mm, p_mx, p_xx,
                             p_my, p_yy);
    // MASK Reads and Haplotypes
    MaskType reads_mask = Traits::generate_length_mask(i, rs_lens);
    init_row_states(i, hap_lens, mm, ii, dd, M_j1, I_j1, D_j1, M_i1j1, I_i1j1,
                    D_i1j1, M_i1, I_i1, D_i1);
    for (uint32_t j = 0; j < tc.min_haplen; j++) {
      SimdIntType h = Traits::load_seqs(tc.hap_seqs + j * Traits::simd_width);
      process_matrix_cell(rbase, h, distm, _1_distm, p_mm, p_gapm, p_mx, p_xx,
                          p_my, p_yy, M, I, D, M_i1, I_i1, D_i1, M_j1, I_j1,
                          D_j1, M_i1j1, I_i1j1, D_i1j1, mm, ii, dd, j,
                          reads_mask);
    }
    for (uint32_t j = tc.min_haplen; j < tc.max_haplen; j++) {
      SimdIntType h = Traits::load_seqs(tc.hap_seqs + j * Traits::simd_width);
      MaskType hap_mask = Traits::generate_length_mask(j, hap_lens);
      process_matrix_cell(rbase, h, distm, _1_distm, p_mm, p_gapm, p_mx, p_xx,
                          p_my, p_yy, M, I, D, M_i1, I_i1, D_i1, M_j1, I_j1,
                          D_j1, M_i1j1, I_i1j1, D_i1j1, mm, ii, dd, j,
                          Traits::mask_and(reads_mask, hap_mask));
    }
  }
  SimdType sum_m = Traits::setzero();
  SimdType sum_i = Traits::setzero();
  for (uint32_t i = 0; i < tc.max_haplen; i++) {
    sum_m = Traits::add(sum_m, mm[i]);
    sum_i = Traits::add(sum_i, ii[i]);
  }
  MainType m_result_temp[Traits::simd_width]
      __attribute__((aligned(Traits::alignment)));
  MainType i_result_temp[Traits::simd_width]
      __attribute__((aligned(Traits::alignment)));
  Traits::store(m_result_temp, sum_m);
  Traits::store(i_result_temp, sum_i);
  for (uint32_t i = 0; i < Traits::simd_width; i++) {
    tc.results[i] = m_result_temp[i] + i_result_temp[i];
  }
  finalize(tc, allocator);
}

template <typename Traits, typename ALLOCATOR >
void InterPairHMMComputer<Traits, ALLOCATOR>::init_row_states(
    uint32_t i, const uint32_t *hap_lens, SimdType *mm, SimdType *ii,
    SimdType *dd, SimdType &M_j1, SimdType &I_j1, SimdType &D_j1,
    SimdType &M_i1j1, SimdType &I_i1j1, SimdType &D_i1j1, SimdType &M_i1,
    SimdType &I_i1, SimdType &D_i1) {
  M_j1 = I_j1 = D_j1 = M_i1j1 = I_i1j1 = Traits::setzero();
  if (i == 0) {
    D_i1j1 = Traits::set_init_d(hap_lens);
  } else {
    D_i1j1 = Traits::setzero();
  }
  M_i1 = mm[0];
  I_i1 = ii[0];
  D_i1 = dd[0];
}
template <typename Traits, typename ALLOCATOR>
void InterPairHMMComputer<Traits, ALLOCATOR>::initialize_matrices(
    const MultiTestCase<Traits> &tc, SimdType *mm, SimdType *ii, SimdType *dd,
    uint32_t *hap_lens) {
  for (uint32_t i = 0; i <= tc.max_haplen; ++i) {
    mm[i] = Traits::setzero();
    ii[i] = Traits::setzero();
    dd[i] = Traits::set_init_d(hap_lens);
  }
}

template <typename Traits, typename ALLOCATOR>
__attribute__((always_inline, hot)) inline void
InterPairHMMComputer<Traits, ALLOCATOR>::process_matrix_cell(
    const SimdIntType &rbase, const SimdIntType &h, const SimdType &distm,
    const SimdType &_1_distm, const SimdType &p_mm, const SimdType &p_gapm,
    const SimdType &p_mx, const SimdType &p_xx, const SimdType &p_my,
    const SimdType &p_yy, SimdType &M, SimdType &I, SimdType &D, SimdType &M_i1,
    SimdType &I_i1, SimdType &D_i1, SimdType &M_j1, SimdType &I_j1,
    SimdType &D_j1, SimdType &M_i1j1, SimdType &I_i1j1, SimdType &D_i1j1,
    SimdType *__restrict__ mm, SimdType *__restrict__ ii,
    SimdType *__restrict__ dd, int j, MaskType len_mask) {

  MaskType mask = Traits::test_cmpeq(rbase, h);
  SimdType distm_chosen = Traits::mask_blend(mask, distm, _1_distm);

  // 计算新的矩阵值
  SimdType temp1 = Traits::mul(M_i1j1, p_mm);
  SimdType temp2 = Traits::mul(I_i1j1, p_gapm);
  SimdType temp3 = Traits::mul(D_i1j1, p_gapm);
  SimdType sum = Traits::add(Traits::add(temp1, temp2), temp3);
  M = Traits::mul(sum, distm_chosen);
  I = Traits::add(Traits::mul(M_i1, p_mx), Traits::mul(I_i1, p_xx));
  D = Traits::add(Traits::mul(M_j1, p_my), Traits::mul(D_j1, p_yy));

  M = Traits::mask_blend(len_mask, M_i1, M);
  I = Traits::mask_blend(len_mask, I_i1, I);
  D = Traits::mask_blend(len_mask, D_i1, D);
  // 更新状态变量
  M_i1j1 = M_i1;
  I_i1j1 = I_i1;
  D_i1j1 = D_i1;

  M_j1 = M;
  I_j1 = I;
  D_j1 = D;

  mm[j] = M;
  ii[j] = I;
  dd[j] = D;

  // 准备下一次迭代
  M_i1 = mm[j + 1];
  I_i1 = ii[j + 1];
  D_i1 = dd[j + 1];
}
// 显式实例化
#if defined(__AVX512F__)
template class InterPairHMMComputer<AVX512FloatTraits, rovaca::roDV_adapter::DefaultAllocator>;
template class InterPairHMMComputer<AVX512DoubleTraits, rovaca::roDV_adapter::DefaultAllocator>;
template class InterPairHMMComputer<AVX512FloatTraits, rovaca::roDV_adapter::RovacaMemoryPoolAllocator>;
template class InterPairHMMComputer<AVX512DoubleTraits, rovaca::roDV_adapter::RovacaMemoryPoolAllocator>;
template class InterPairHMMComputer<AVX512FloatTraits, rovaca::roDV_adapter::PairHMMAllocator>;
template class InterPairHMMComputer<AVX512DoubleTraits, rovaca::roDV_adapter::PairHMMAllocator>;
#elif defined(__AVX2__)
template class InterPairHMMComputer<AVX2FloatTraits, rovaca::roDV_adapter::DefaultAllocator>;
template class InterPairHMMComputer<AVX2DoubleTraits, rovaca::roDV_adapter::DefaultAllocator>;
template class InterPairHMMComputer<AVX2FloatTraits, rovaca::roDV_adapter::RovacaMemoryPoolAllocator>;
template class InterPairHMMComputer<AVX2DoubleTraits, rovaca::roDV_adapter::RovacaMemoryPoolAllocator>;
template class InterPairHMMComputer<AVX2FloatTraits, rovaca::roDV_adapter::PairHMMAllocator>;
template class InterPairHMMComputer<AVX2DoubleTraits, rovaca::roDV_adapter::PairHMMAllocator>;
#endif

} // namespace inter
} // namespace pairhmm
