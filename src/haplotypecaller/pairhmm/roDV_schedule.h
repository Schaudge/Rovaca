

#ifndef PAIRHMM_SCHEDULE_H_
#define PAIRHMM_SCHEDULE_H_
#include "roDV_common/common.h"
#include <cstdint>
#include <vector>

// 包含必要的类型定义
#include "../genotype/forward.h"
#include "../common/enum.h"
#include "roDV_adapter/rovaca_pool_allocator.h"
using namespace pairhmm::common;

namespace pairhmm {
namespace schedule {

bool schedule_pairhmm(
    const std::vector<std::vector<uint8_t>> &haplotypes,
    const std::vector<std::vector<uint8_t>> &reads,
    std::vector<std::vector<double>> &result,
    const std::vector<std::vector<uint8_t>> &quality,
    const std::vector<std::vector<uint8_t>> &insertion_qualities,
    const std::vector<std::vector<uint8_t>> &deletion_qualities,
    const std::vector<std::vector<uint8_t>> &gap_contiguous_qualities,
    bool use_double = false,
    double max_idle_ratio_float = 0.1,
    double max_idle_ratio_double = 0.1,
    bool verbose = false);

// 带 allocator 的模板版本
template <typename Allocator>
bool schedule_pairhmm_with_allocator(
    const std::vector<std::vector<uint8_t>> &haplotypes,
    const std::vector<std::vector<uint8_t>> &reads,
    std::vector<std::vector<double>> &result,
    const std::vector<std::vector<uint8_t>> &quality,
    const std::vector<std::vector<uint8_t>> &insertion_qualities,
    const std::vector<std::vector<uint8_t>> &deletion_qualities,
    const std::vector<std::vector<uint8_t>> &gap_contiguous_qualities,
    Allocator& allocator,
    bool use_double = false,
    double max_idle_ratio_float = 0.02,
    double max_idle_ratio_double = 0.02,
    bool verbose = false);

// 直接接受 HaplotypeVector 和 ReadVector 的版本（避免数据转换）
bool schedule_pairhmm_with_allocator(
    const rovaca::HaplotypeVector& hs,
    rovaca::ReadVector& rs,
    rovaca::DoubleVector2D* result,
    rovaca::roDV_adapter::RovacaMemoryPoolAllocator& allocator,
    int32_t min_quality_threshold = 18,
    PcrIndelModel pcr_option = PcrIndelModel::NONE,
    bool use_double = false,
    double max_idle_ratio_float = 0.02,
    double max_idle_ratio_double = 0.02,
    bool verbose = false);
}
} // namespace pairhmm

#endif