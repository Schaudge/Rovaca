#include "pairhmm_impl.h"
#include <cstring>
#include <algorithm>

using namespace pairhmm::common;
namespace pairhmm {
namespace intra {

// ============================================================================
// 模板实现：precompute_masks
// ============================================================================
template <typename Traits>
inline void PairHMMComputer<Traits>::precompute_masks(
    const TestCase& tc,
    uint32_t cols,
    uint32_t num_mask_vecs,
    MaskType (*mask_arr)[k_num_distinct_chars])
{
    constexpr uint32_t mask_bit_cnt = main_type_size;
    
    // 初始化 mask 数组
    for (uint32_t vi = 0; vi < num_mask_vecs; ++vi) {
        for (uint32_t rs = 0; rs < k_num_distinct_chars; ++rs) {
            mask_arr[vi][rs] = 0;
        }
        mask_arr[vi][k_ambig_char] = Traits::mask_all_ones;
    }
    
    // 预计算每列的 mask
    for (uint32_t col = 1; col < cols; ++col) {
        uint32_t index = (col - 1) / mask_bit_cnt;
        uint32_t offset = (col - 1) % mask_bit_cnt;
        uint8_t hap_char = ConvertChar::get(tc.hap[col - 1]);
        MaskType bit_mask = MaskType(1) << (mask_bit_cnt - 1 - offset);
        mask_arr[index][hap_char] |= bit_mask;
        
        // 如果是模糊碱基 N，所有碱基都匹配
        if (unlikely(hap_char == k_ambig_char)) {
            for (uint32_t ci = 0; ci < k_num_distinct_chars; ++ci) {
                mask_arr[index][ci] |= bit_mask;
            }
        }
    }
}

// ============================================================================
// 模板实现：initialize_vectors
// ============================================================================
template <typename Traits>
inline void PairHMMComputer<Traits>::initialize_vectors(
    uint32_t rows,
    uint32_t cols,
    MainType* shift_out_m,
    MainType* shift_out_x,
    MainType* shift_out_y,
    Context<MainType>& ctx,
    const TestCase& tc,
    const ProbabilityArrays& pa,
    SimdType* distm1d)
{
    const MainType zero = Context<MainType>::_(0.0);
    const MainType init_y = Context<MainType>::INITIAL_CONSTANT / static_cast<MainType>(tc.haplen);
    
    // 初始化 shift 缓冲区
    for (uint32_t s = 0; s < rows + cols + simd_width; s++) {
        shift_out_m[s] = zero;
        shift_out_x[s] = zero;
        shift_out_y[s] = init_y;
    }
    
    // 初始化概率数组（强制转换为 MainType*）
    auto* ptr_p_mm = reinterpret_cast<MainType*>(pa.p_mm_arr);
    auto* ptr_p_gapm = reinterpret_cast<MainType*>(pa.p_gapm_arr);
    auto* ptr_p_mx = reinterpret_cast<MainType*>(pa.p_mx_arr);
    auto* ptr_p_xx = reinterpret_cast<MainType*>(pa.p_xx_arr);
    auto* ptr_p_my = reinterpret_cast<MainType*>(pa.p_my_arr);
    auto* ptr_p_yy = reinterpret_cast<MainType*>(pa.p_yy_arr);
    auto* ptr_distm1d = reinterpret_cast<MainType*>(distm1d);
    
    ptr_p_mm[0] = zero;
    ptr_p_gapm[0] = zero;
    ptr_p_mx[0] = zero;
    ptr_p_xx[0] = zero;
    ptr_p_my[0] = zero;
    ptr_p_yy[0] = zero;
    
    // 填充质量分数转概率
    for (uint32_t r = 1; r < rows; r++) {
        const uint8_t _i = tc.i[r - 1] & 127;
        const uint8_t _d = tc.d[r - 1] & 127;
        const uint8_t _c = tc.c[r - 1] & 127;
        const uint8_t _q = tc.q[r - 1] & 127;
        
        if constexpr (Traits::simd_bits == 256) {
            ptr_p_mm[r - 1] = ctx.set_mm_prob(_i, _d);
            ptr_p_gapm[r - 1] = Context<MainType>::_(1.0) - Context<MainType>::ph2pr[_c];
            ptr_p_mx[r - 1] = Context<MainType>::ph2pr[_i];
            ptr_p_xx[r - 1] = Context<MainType>::ph2pr[_c];
            ptr_p_my[r - 1] = Context<MainType>::ph2pr[_d];
            ptr_p_yy[r - 1] = Context<MainType>::ph2pr[_c];
            ptr_distm1d[r - 1] = Context<MainType>::ph2pr[_q];
        }else{
            int index =((r - 1)/ Traits::simd_width) * Traits::simd_width;
            int offset = Traits::simd_width - 1 - (r - 1) % Traits::simd_width;
            ptr_p_mm[index+offset] = ctx.set_mm_prob(_i, _d);
            ptr_p_gapm[index+offset] = Context<MainType>::_(1.0) - Context<MainType>::ph2pr[_c];
            ptr_p_mx[index+offset] = Context<MainType>::ph2pr[_i];
            ptr_p_xx[index+offset] = Context<MainType>::ph2pr[_c];
            ptr_p_my[index+offset] = Context<MainType>::ph2pr[_d];
            ptr_p_yy[index+offset] = Context<MainType>::ph2pr[_c];
            ptr_distm1d[index+offset] = Context<MainType>::ph2pr[_q];
        }
       

    }
}

// ============================================================================
// 模板实现：stripe_initialization
// ============================================================================
template <typename Traits>
void PairHMMComputer<Traits>::stripe_initialization(
    uint32_t stripe_idx,
    SimdType& p_gapm, SimdType& p_mm, SimdType& p_mx, SimdType& p_xx,
    SimdType& p_my, SimdType& p_yy,
    SimdType& distm, SimdType& _1_distm,
    const SimdType* distm1d,
    const SimdType* p_mm_arr,
    const SimdType* p_gapm_arr,
    const SimdType* p_mx_arr,
    const SimdType* p_xx_arr,
    const SimdType* p_my_arr,
    const SimdType* p_yy_arr,
    SimdType& m_t_1, SimdType& m_t_2,
    SimdType& x_t_1, SimdType& x_t_2,
    SimdType& y_t_1, SimdType& y_t_2,
    SimdType& m_t_1_y,
    const MainType* shift_out_x,
    const MainType* shift_out_m,
    const TestCase& tc)
{
    p_mm = p_mm_arr[stripe_idx];
    p_gapm = p_gapm_arr[stripe_idx];
    p_mx = p_mx_arr[stripe_idx];
    p_xx = p_xx_arr[stripe_idx];
    p_my = p_my_arr[stripe_idx];
    p_yy = p_yy_arr[stripe_idx];
    
    const MainType init_y = Context<MainType>::INITIAL_CONSTANT / static_cast<MainType>(tc.haplen);
    
    const SimdType packed1 = Traits::set1(Context<MainType>::_(1.0));
    const SimdType packed3 = Traits::set1(Context<MainType>::_(3.0));
    
    distm = distm1d[stripe_idx];
    _1_distm = Traits::sub(packed1, distm);
    distm = Traits::div(distm, packed3);
    
    m_t_2 = Traits::setzero();
    x_t_2 = Traits::setzero();
    
    if (stripe_idx == 0) {
        m_t_1 = Traits::setzero();
        x_t_1 = Traits::setzero();
        y_t_2 = Traits::set_lse(init_y);
        y_t_1 = Traits::setzero();
    } else {
        x_t_1 = Traits::set_lse(shift_out_x[simd_width]);
        m_t_1 = Traits::set_lse(shift_out_m[simd_width]);
        y_t_2 = Traits::setzero();
        y_t_1 = Traits::setzero();
    }
    
    m_t_1_y = m_t_1;
}

// ============================================================================
// 模板实现：init_masks_for_row
// ============================================================================
template <typename Traits>
inline void PairHMMComputer<Traits>::init_masks_for_row(
    const TestCase& tc,
    uint8_t* rs_arr,
    typename Traits::VecIntType& last_mask_shift_out,
    uint32_t begin_row_index,
    uint32_t num_rows_to_process)
{
    const uint8_t* dest = tc.rs + begin_row_index - 1;
    last_mask_shift_out = Traits::setzero_int();
    
    if (likely(num_rows_to_process == simd_width)) {
        // 展开循环以优化性能
        for (uint32_t ri = 0; ri < simd_width; ++ri) {
            rs_arr[ri] = ConvertChar::k_conversion_table[dest[ri] - 'A'];
        }
    } else {
        for (uint32_t ri = 0; ri < num_rows_to_process; ++ri) {
            rs_arr[ri] = ConvertChar::get(dest[ri]);
        }
        for (uint32_t ri = num_rows_to_process; ri < simd_width; ++ri) {
            rs_arr[ri] = 0;
        }
    }
}
// FixMe: 这个函数需要修改
// ============================================================================
// 模板实现：update_masks_for_cols
// ============================================================================
template <typename Traits>
inline void PairHMMComputer<Traits>::update_masks_for_cols(
    uint32_t mask_index,
    typename Traits::VecIntType& bit_mask_vec,
    MaskType (*mask_arr)[k_num_distinct_chars],
    const uint8_t* rs_arr,
    typename Traits::VecIntType& last_mask_shift_out)
{
    MaskType* arr = mask_arr[mask_index];
    
    // 构建 SIMD 向量
    typename Traits::VecIntType src = Traits::set_epi_from_array(arr, rs_arr);
    typename Traits::VecIntType forward_shift_vec = Traits::get_forward_shift_vector();
    typename Traits::VecIntType mask_vec = Traits::forward_shift(src, forward_shift_vec);
    bit_mask_vec = Traits::or_si(mask_vec, last_mask_shift_out);
    
    // 计算保留的 mask
    typename Traits::VecIntType reserved_mask = Traits::get_reserved_mask();
    typename Traits::VecIntType reserved_src = Traits::and_si(src, reserved_mask);
    typename Traits::VecIntType backward_shift_vec = Traits::get_backward_shift_vector();
    last_mask_shift_out = Traits::backward_shift(reserved_src, backward_shift_vec);
}

// ============================================================================
// 模板实现：compute_mxy
// ============================================================================
template <typename Traits>
inline void PairHMMComputer<Traits>::compute_mxy(
    SimdType& m_t,
    SimdType& m_t_y,
    SimdType& x_t,
    SimdType& y_t,
    const SimdType& m_t_2,
    const SimdType& x_t_2,
    const SimdType& y_t_2,
    const SimdType& m_t_1,
    const SimdType& x_t_1,
    const SimdType& m_t_1_y,
    const SimdType& y_t_1,
    const SimdType& p_mm,
    const SimdType& p_gapm,
    const SimdType& p_mx,
    const SimdType& p_xx,
    const SimdType& p_my,
    const SimdType& p_yy,
    const SimdType& distm_sel)
{
    // Match 状态: M[i][j] = distm * (p_mm*M[i-1][j-1] + p_gapm*I[i-1][j-1] + p_gapm*D[i-1][j-1])
    m_t = Traits::mul(
        Traits::add(
            Traits::add(
                Traits::mul(m_t_2, p_mm),
                Traits::mul(x_t_2, p_gapm)
            ),
            Traits::mul(y_t_2, p_gapm)
        ),
        distm_sel
    );
    
    m_t_y = m_t;
    
    // Insertion 状态: I[i][j] = p_mx*M[i-1][j] + p_xx*I[i-1][j]
    x_t = Traits::add(
        Traits::mul(m_t_1, p_mx),
        Traits::mul(x_t_1, p_xx)
    );
    
    // Deletion 状态: D[i][j] = p_my*M[i][j-1] + p_yy*D[i][j-1]
    y_t = Traits::add(
        Traits::mul(m_t_1_y, p_my),
        Traits::mul(y_t_1, p_yy)
    );
}

// ============================================================================
// 模板实现：compute (主函数)
// ============================================================================
template <typename Traits>
typename Traits::MainType PairHMMComputer<Traits>::compute(const TestCase& tc)
{
    // 初始化字符转换表    
    uint32_t rows = tc.rslen + 1;
    uint32_t cols = tc.haplen + 1;
    uint32_t mavx_count = (rows + simd_width - 1) / simd_width;
    
    // 分配概率数组（栈上）
    SimdType p_mm_arr[mavx_count], p_gapm_arr[mavx_count], p_mx_arr[mavx_count];
    SimdType p_xx_arr[mavx_count], p_my_arr[mavx_count], p_yy_arr[mavx_count];
    ProbabilityArrays pa{p_mm_arr, p_gapm_arr, p_mx_arr, p_xx_arr, p_my_arr, p_yy_arr};
    
    // 预计算 distm
    SimdType distm1d_arr[mavx_count];
    
    // Shift 缓冲区
    MainType shift_out_m[rows + cols + simd_width];
    MainType shift_out_x[rows + cols + simd_width];
    MainType shift_out_y[rows + cols + simd_width];
    
    // Context
    Context<MainType> ctx;
    
    // 初始化向量
    initialize_vectors(rows, cols, shift_out_m, shift_out_x, shift_out_y,
                      ctx, tc, pa, distm1d_arr);
    
    // 预计算 masks
    constexpr uint32_t mask_bit_cnt = main_type_size;
    const uint32_t num_mask_vecs = (cols + rows + mask_bit_cnt - 1) / mask_bit_cnt;
    MaskType mask_arr[num_mask_vecs][k_num_distinct_chars];
    precompute_masks(tc, cols, num_mask_vecs, mask_arr);
    
    // HMM 状态向量
    SimdType m_t, m_t_1, m_t_2, x_t, x_t_1, x_t_2, y_t, y_t_1, y_t_2, m_t_y, m_t_1_y;
    SimdType p_gapm, p_mm, p_mx, p_xx, p_my, p_yy;
    SimdType distm, _1_distm, distm_chosen;
    
    uint32_t remaining_rows = (rows - 1) % simd_width;
    uint32_t stripe_cnt = ((rows - 1) / simd_width) + (remaining_rows != 0);
    
    uint8_t rs_arr[simd_width];
    typename Traits::VecIntType bit_mask_vec;
    typename Traits::VecIntType last_mask_shift_out;
    uint32_t shift_idx;
    uint32_t num_mask_bits_to_process;
    
    // 处理完整的 stripes
    for (uint32_t i = 0; i < stripe_cnt - 1; ++i) {
        
        stripe_initialization(i, p_gapm, p_mm, p_mx, p_xx, p_my, p_yy, distm, _1_distm,
                            distm1d_arr, p_mm_arr, p_gapm_arr, p_mx_arr, p_xx_arr, p_my_arr, p_yy_arr,
                            m_t_1, m_t_2, x_t_1, x_t_2, y_t_1, y_t_2, m_t_1_y,
                            shift_out_x, shift_out_m, tc);
        
        init_masks_for_row(tc, rs_arr, last_mask_shift_out, i * simd_width + 1, simd_width);
        
        for (uint32_t begin_d = 1; begin_d < cols + simd_width; begin_d += main_type_size) {
            num_mask_bits_to_process = std::min(main_type_size, cols + simd_width - begin_d);
            
            update_masks_for_cols((begin_d - 1) / main_type_size, bit_mask_vec, mask_arr, rs_arr, last_mask_shift_out);
            
            for (uint32_t mbi = 0; mbi < num_mask_bits_to_process; ++mbi) {
                shift_idx = begin_d + mbi + simd_width;
                
                Traits::compute_dist_vec(bit_mask_vec, distm_chosen, distm, _1_distm);
                
                compute_mxy(m_t, m_t_y, x_t, y_t, m_t_2, x_t_2, y_t_2, m_t_1, x_t_1, m_t_1_y, y_t_1,
                           p_mm, p_gapm, p_mx, p_xx, p_my, p_yy, distm_chosen);
                
                Traits::vector_shift(m_t, shift_out_m[shift_idx], shift_out_m[begin_d + mbi]);
                Traits::vector_shift(x_t, shift_out_x[shift_idx], shift_out_x[begin_d + mbi]);
                Traits::vector_shift(y_t_1, shift_out_y[shift_idx], shift_out_y[begin_d + mbi]);
                
                m_t_2 = m_t_1;
                m_t_1 = m_t;
                x_t_2 = x_t_1;
                x_t_1 = x_t;
                y_t_2 = y_t_1;
                y_t_1 = y_t;
                m_t_1_y = m_t_y;
            }
        }
    }
    
    // 处理最后一个 stripe
    {
        uint32_t i = stripe_cnt - 1;
        
        stripe_initialization(i, p_gapm, p_mm, p_mx, p_xx, p_my, p_yy, distm, _1_distm,
                            distm1d_arr, p_mm_arr, p_gapm_arr, p_mx_arr, p_xx_arr, p_my_arr, p_yy_arr,
                            m_t_1, m_t_2, x_t_1, x_t_2, y_t_1, y_t_2, m_t_1_y,
                            shift_out_x, shift_out_m, tc);
        
        if (remaining_rows == 0) {
            remaining_rows = simd_width;
        }
        init_masks_for_row(tc, rs_arr, last_mask_shift_out, i * simd_width + 1, remaining_rows);
        
        SimdType sum_m = Traits::setzero();
        SimdType sum_x = Traits::setzero();
        
        for (uint32_t begin_d = 1; begin_d < cols + remaining_rows - 1; begin_d += main_type_size) {
            num_mask_bits_to_process = std::min(main_type_size, cols + remaining_rows - 1 - begin_d);
            
            update_masks_for_cols((begin_d - 1) / main_type_size, bit_mask_vec, mask_arr, rs_arr, last_mask_shift_out);
            
            for (uint32_t mbi = 0; mbi < num_mask_bits_to_process; ++mbi) {
                shift_idx = begin_d + mbi + simd_width;
                
                Traits::compute_dist_vec(bit_mask_vec, distm_chosen, distm, _1_distm);
                
                compute_mxy(m_t, m_t_y, x_t, y_t, m_t_2, x_t_2, y_t_2, m_t_1, x_t_1, m_t_1_y, y_t_1,
                           p_mm, p_gapm, p_mx, p_xx, p_my, p_yy, distm_chosen);
                
                sum_m = Traits::add(sum_m, m_t);
                sum_x = Traits::add(sum_x, x_t);
                Traits::vector_shift_last(m_t, shift_out_m[shift_idx]);
                Traits::vector_shift_last(x_t, shift_out_x[shift_idx]);
                Traits::vector_shift_last(y_t_1, shift_out_y[shift_idx]);
                
                m_t_2 = m_t_1;
                m_t_1 = m_t;
                x_t_2 = x_t_1;
                x_t_1 = x_t;
                y_t_2 = y_t_1;
                y_t_1 = y_t;
                m_t_1_y = m_t_y;
            }
        }
        
        // 提取结果
        alignas(64) MainType m_result[simd_width];
        alignas(64) MainType x_result[simd_width];
        Traits::store(m_result, sum_m);
        Traits::store(x_result, sum_x);
        if  constexpr (Traits::simd_bits == 256) {
            return m_result[remaining_rows - 1] + x_result[remaining_rows - 1];
        } else {
            return m_result[simd_width - remaining_rows] + x_result[simd_width - remaining_rows];
        }
    }
}

// ============================================================================
// 显式实例化（编译器会生成具体代码）
// 根据编译标志选择性实例化
// 注意：AVX512 编译时 GCC 也会定义 __AVX2__，所以需要先检查 AVX512
// ============================================================================
#if defined(__AVX512F__)
// 实例化 AVX512 版本
template class PairHMMComputer<AVX512FloatTraits>;
template class PairHMMComputer<AVX512DoubleTraits>;
#elif defined(__AVX2__)
// 实例化 AVX2 版本
template class PairHMMComputer<AVX2FloatTraits>;
template class PairHMMComputer<AVX2DoubleTraits>;
#endif

}  // namespace intra
}  // namespace pairhmm