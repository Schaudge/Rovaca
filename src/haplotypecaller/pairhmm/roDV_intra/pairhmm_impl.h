#ifndef PAIRHMM_IMPL_H_
#define PAIRHMM_IMPL_H_

#include "../roDV_common/common.h"
#include "../roDV_common/context.h"
#include "simd_traits.h"
#include <algorithm>
#include <cstdint>

using namespace pairhmm::common;

namespace pairhmm {
namespace intra {

/**
 * @brief PairHMM 算法的统一模板实现
 * 
 * 使用 SIMD Traits 模板参数实现不同指令集和精度的零开销抽象
 * 
 * @tparam Traits SIMD 特征类（AVX2FloatTraits, AVX2DoubleTraits, 等）
 */
template <typename Traits>
class PairHMMComputer {
public:
    using MainType = typename Traits::MainType;
    using SimdType = typename Traits::SimdType;
    using SimdIntType = typename Traits::SimdIntType;
    using MaskType = typename Traits::MaskType;
    
    static constexpr uint32_t simd_width = Traits::simd_width;
    static constexpr uint32_t bits_per_byte = k_bits_per_byte;
    static constexpr uint32_t main_type_size = sizeof(MainType) * bits_per_byte;
    
    /**
     * @brief 计算 PairHMM 前向算法概率
     * 
     * @param tc 测试用例（包含序列和质量分数）
     * @return MainType 计算得到的对数概率
     */
    static MainType compute(const TestCase& tc);
    
private:
    // 概率数组结构
    struct ProbabilityArrays {
        SimdType* p_mm_arr;   // match to match
        SimdType* p_gapm_arr; // gap penalty
        SimdType* p_mx_arr;   // match to insertion
        SimdType* p_xx_arr;   // insertion to insertion
        SimdType* p_my_arr;   // match to deletion
        SimdType* p_yy_arr;   // deletion to deletion
    };
    
    // 预计算 mask（用于快速碱基匹配）
    static inline void precompute_masks(
        const TestCase& tc,
        uint32_t cols,
        uint32_t num_mask_vecs,
        MaskType (*mask_arr)[k_num_distinct_chars]);
    
    // 初始化概率向量
    static inline void initialize_vectors(
        uint32_t rows,
        uint32_t cols,
        MainType* shift_out_m,
        MainType* shift_out_x,
        MainType* shift_out_y,
        Context<MainType>& ctx,
        const TestCase& tc,
        const ProbabilityArrays& pa,
        SimdType* distm1d);
    
    // 初始化每个 stripe
    static inline void stripe_initialization(
        uint32_t stripe_idx,
        SimdType& p_gapm,
        SimdType& p_mm,
        SimdType& p_mx,
        SimdType& p_xx,
        SimdType& p_my,
        SimdType& p_yy,
        SimdType& distm,
        SimdType& _1_distm,
        const SimdType* distm1d,
        const SimdType* p_mm_arr,
        const SimdType* p_gapm_arr,
        const SimdType* p_mx_arr,
        const SimdType* p_xx_arr,
        const SimdType* p_my_arr,
        const SimdType* p_yy_arr,
        SimdType& m_t_1,
        SimdType& m_t_2,
        SimdType& x_t_1,
        SimdType& x_t_2,
        SimdType& y_t_1,
        SimdType& y_t_2,
        SimdType& m_t_1_y,
        const MainType* shift_out_x,
        const MainType* shift_out_m,
        const TestCase& tc);
    
    // 计算 HMM 状态转移
    static inline void compute_mxy(
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
        const SimdType& distm_sel);
    
    // 初始化行 mask
    static inline void init_masks_for_row(
        const TestCase& tc,
        uint8_t* rs_arr,
        typename Traits::VecIntType& last_mask_shift_out,
        uint32_t start_row,
        uint32_t num_rows);
    
    // 更新列 mask
    static inline void update_masks_for_cols(
        uint32_t col_idx,
        typename Traits::VecIntType& bit_mask_vec,
        MaskType (*mask_arr)[k_num_distinct_chars],
        const uint8_t* rs_arr,
        typename Traits::VecIntType& last_mask_shift_out);
    
    
};

}  // namespace intra
}  // namespace pairhmm

#endif  // PAIRHMM_IMPL_H_

