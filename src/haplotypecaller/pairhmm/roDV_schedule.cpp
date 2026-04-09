#include <net/if.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#include "../genotype/genotype_macors.h"
#include "../genotype/haplotype.h"
#include "../genotype/read_record.h"
#include "adapter.h"
#include "rovaca_logger.h"
#include "pairhmm_internal.h"
#include "roDV_adapter/rovaca_pool_allocator.h"
#include "roDV_common/cpu_features.h"
#include "roDV_inter/pairhmm_inter.h"
#include "roDV_inter/pairhmm_inter_api.h"
#include "roDV_intra/pairhmm_api.h"
#include "roDV_schedule.h"

using namespace pairhmm::common;
using namespace pairhmm::inter;
using namespace pairhmm::intra;

namespace pairhmm
{
namespace schedule
{

#define MIN_ACCEPTED 1e-28f
// 辅助结构：表示一个单倍型-read对
struct HapReadPair
{
    static constexpr uint8_t kContainsNMask = 0x1;
    static constexpr uint8_t kUsedMask = 0x2;

    uint32_t read_idx;
    uint16_t hap_idx;
    uint16_t hap_len;
    uint16_t read_len;
    uint8_t flags;

    HapReadPair(uint32_t h_idx, uint32_t r_idx, uint16_t h_len, uint16_t r_len, bool contains_n)
        : read_idx(r_idx)
        , hap_idx(static_cast<uint16_t>(h_idx))
        , hap_len(h_len)
        , read_len(r_len)
        , flags(static_cast<uint8_t>(contains_n ? kContainsNMask : 0))
    {}

    bool contains_n() const { return (flags & kContainsNMask) != 0; }
    bool is_used() const { return (flags & kUsedMask) != 0; }
    void set_used(bool value)
    {
        if (value) {
            flags |= kUsedMask;
        }
        else {
            flags &= static_cast<uint8_t>(~kUsedMask);
        }
    }
};
static_assert(sizeof(HapReadPair) <= 12, "HapReadPair should remain compact");

constexpr uint32_t kMaxAllowedHaplotypeIndex = 4095;    // 原始 2048，留有余量
constexpr uint32_t kMaxAllowedReadIndex = 140'000'000;  // 原始 128M，适度放宽
constexpr uint16_t kMaxAllowedHaplotypeLength = 1536;   // 原始 1024，适度放宽
constexpr uint16_t kMaxAllowedReadLength = 768;         // 原始 512，适度放宽

// 辅助结构：表示一组（模板化以支持不同的容器类型）
template <typename IndexContainer = std::vector<size_t>>
struct Group
{
    IndexContainer pair_indices;  // 指向pairs数组的索引
    uint32_t max_hap_len = 0;
    uint32_t max_read_len = 0;
    double idle_ratio = 0.0;
};

// 贪心分组算法（模板化以支持不同的容器类型）
template <typename PairContainer, typename GroupContainer>
GroupContainer greedy_grouping(PairContainer &pairs, double max_idle_ratio, uint32_t SimdWidth)
{
    GroupContainer groups;
    if (SimdWidth == 0) {
        return groups;
    }

    const size_t total = pairs.size();
    for (size_t i = 0; i < total; ++i) {
        if (pairs[i].is_used() || pairs[i].contains_n()) continue;

        // 使用 GroupContainer 的元素类型来创建 Group
        using GroupType = typename GroupContainer::value_type;
        GroupType group;
        group.pair_indices.reserve(SimdWidth);

        size_t idx = i;
        while (idx < total && group.pair_indices.size() < SimdWidth) {
            if (!pairs[idx].is_used() && !pairs[idx].contains_n()) {
                group.pair_indices.push_back(idx);
            }
            ++idx;
        }

        if (group.pair_indices.size() < SimdWidth) {
            break;
        }

        uint32_t max_hap_len = 0;
        uint32_t max_read_len = 0;
        uint64_t sum_elements = 0;
        for (size_t pair_idx : group.pair_indices) {
            const auto &pair = pairs[pair_idx];
            max_hap_len = std::max(max_hap_len, static_cast<uint32_t>(pair.hap_len));
            max_read_len = std::max(max_read_len, static_cast<uint32_t>(pair.read_len));
            sum_elements += static_cast<uint64_t>(pair.hap_len) * pair.read_len;
        }

        uint64_t total_capacity = static_cast<uint64_t>(max_hap_len) * max_read_len * SimdWidth;
        uint64_t idle = total_capacity > sum_elements ? total_capacity - sum_elements : 0;
        double idle_ratio = total_capacity == 0 ? 0.0 : static_cast<double>(idle) / static_cast<double>(total_capacity);

        group.max_hap_len = max_hap_len;
        group.max_read_len = max_read_len;
        group.idle_ratio = idle_ratio;

        if (idle_ratio <= max_idle_ratio) {
            for (size_t pair_idx : group.pair_indices) {
                pairs[pair_idx].set_used(true);
            }
            groups.push_back(std::move(group));
        }
    }

    return groups;
}

bool containsN(const std::vector<uint8_t> &sequence)
{
    for (uint8_t c : sequence) {
        if (c == 'N' || c == 'n') {
            return true;
        }
    }
    return false;
}

bool schedule_pairhmm(const std::vector<std::vector<uint8_t>> &haplotypes, const std::vector<std::vector<uint8_t>> &reads,
                      std::vector<std::vector<double>> &result, const std::vector<std::vector<uint8_t>> &quality,
                      const std::vector<std::vector<uint8_t>> &insertion_qualities,
                      const std::vector<std::vector<uint8_t>> &deletion_qualities,
                      const std::vector<std::vector<uint8_t>> &gap_contiguous_qualities, bool use_double, double max_idle_ratio_float,
                      double max_idle_ratio_double, bool verbose)
{
    rovaca::roDV_adapter::DefaultAllocator allocator;
    return schedule_pairhmm_with_allocator(haplotypes, reads, result, quality, insertion_qualities, deletion_qualities,
                                           gap_contiguous_qualities, allocator, use_double, max_idle_ratio_float, max_idle_ratio_double,
                                           verbose);
}

// 带 allocator 的模板版本实现
template <typename Allocator>
bool schedule_pairhmm_with_allocator(const std::vector<std::vector<uint8_t>> &haplotypes, const std::vector<std::vector<uint8_t>> &reads,
                                     std::vector<std::vector<double>> &result, const std::vector<std::vector<uint8_t>> &quality,
                                     const std::vector<std::vector<uint8_t>> &insertion_qualities,
                                     const std::vector<std::vector<uint8_t>> &deletion_qualities,
                                     const std::vector<std::vector<uint8_t>> &gap_contiguous_qualities, Allocator &allocator,
                                     bool use_double, double max_idle_ratio_float, double max_idle_ratio_double, bool verbose)
{
    const size_t M = haplotypes.size();
    const size_t N = reads.size();

    if (M == 0 || N == 0) {
        return false;
    }

    // 初始化结果矩阵 [N][M] (read x haplotype)
    result.resize(N);
    for (size_t r = 0; r < N; ++r) {
        result[r].resize(M, 0.0);
    }

    // 生成所有单倍型-read对
    std::vector<HapReadPair> pairs;
    pairs.reserve(M * N);
    std::vector<bool> contains_n_hap_vec(M, false);
    std::vector<bool> contains_n_read_vec(N, false);
    if (verbose) {
        std::cerr << "Pairs: " << pairs.size() << std::endl;
        for (const auto &pair : pairs) {
            std::cerr << "Pair: " << pair.hap_idx << "," << pair.read_idx << " hap_len: " << pair.hap_len << " read_len: " << pair.read_len
                      << std::endl;
        }
    }
    for (size_t h = 0; h < M; ++h) {
        contains_n_hap_vec[h] = containsN(haplotypes[h]);
    }
    for (size_t r = 0; r < N; ++r) {
        contains_n_read_vec[r] = containsN(reads[r]);
    }
    for (size_t h = 0; h < M; ++h) {
        for (size_t r = 0; r < N; ++r) {
            pairs.emplace_back(h, r, static_cast<uint32_t>(haplotypes[h].size()), static_cast<uint32_t>(reads[r].size()),
                               contains_n_hap_vec[h] || contains_n_read_vec[r]);
        }
    }

    std::sort(pairs.begin(), pairs.end(), [](const HapReadPair &lhs, const HapReadPair &rhs) {
        if (lhs.hap_len == rhs.hap_len) {
            return lhs.read_len > rhs.read_len;
        }
        return lhs.hap_len > rhs.hap_len;
    });

    uint32_t float_simd_width = 0;
    uint32_t double_simd_width = 0;

    if (CpuFeatures::hasAVX512Support()) {
        float_simd_width = 16;
        double_simd_width = 8;
    }
    else if (CpuFeatures::hasAVX2Support()) {
        float_simd_width = 8;
        double_simd_width = 4;
    }

    // 标记哪些对已经被处理
    std::vector<bool> processed_float(M * N, false);
    std::vector<bool> processed_double(M * N, false);

    int total_num_pairs_float = 0;
    // 第一步：尝试float类型分组
    if (!use_double && float_simd_width > 0) {
        auto float_groups = greedy_grouping<std::vector<HapReadPair>, std::vector<Group<std::vector<size_t>>>>(pairs, max_idle_ratio_float,
                                                                                                               float_simd_width);

        // 使用模板直接调用 InterPairHMMComputer，传入 allocator
        for (const auto &group : float_groups) {
            pairhmm::common::TestCase *tc = new pairhmm::common::TestCase[float_simd_width];
            double *results = new double[float_simd_width];

            for (uint32_t i = 0; i < float_simd_width; i++) {
                const auto &pair = pairs[group.pair_indices[i]];
                tc[i].hap = haplotypes[pair.hap_idx].data();
                tc[i].rs = reads[pair.read_idx].data();
                tc[i].q = quality[pair.read_idx].data();
                tc[i].i = insertion_qualities[pair.read_idx].data();
                tc[i].d = deletion_qualities[pair.read_idx].data();
                tc[i].c = gap_contiguous_qualities[pair.read_idx].data();
                tc[i].haplen = pair.hap_len;
                tc[i].rslen = pair.read_len;
                total_num_pairs_float++;
            }

            // 根据 CPU 特性选择对应的 Traits 并调用模板方法（传入 allocator）
            if (CpuFeatures::hasAVX512Support()) {
                compute_inter_pairhmm_AVX512_float(tc, float_simd_width, results, allocator, false);
            }
            else if (CpuFeatures::hasAVX2Support()) {
                compute_inter_pairhmm_AVX2_float(tc, float_simd_width, results, allocator, false);
            }

            for (uint32_t i = 0; i < float_simd_width; i++) {
                if (results[i] < MIN_ACCEPTED) {
                    pairs[group.pair_indices[i]].set_used(false);
                }
                else {
                    pairs[group.pair_indices[i]].set_used(true);
                    const auto &pair = pairs[group.pair_indices[i]];
                    result[pair.read_idx][pair.hap_idx] = inter::loglikelihoodfloat(results[i]);
                    processed_double[pairs[group.pair_indices[i]].hap_idx * N + pairs[group.pair_indices[i]].read_idx] = true;
                }
                processed_float[pairs[group.pair_indices[i]].hap_idx * N + pairs[group.pair_indices[i]].read_idx] = true;
            }

            delete[] tc;
            delete[] results;
        }
    }

    // 第二步：尝试double类型分组
    if (double_simd_width > 0) {
        pairhmm::common::TestCase *tc = new pairhmm::common::TestCase[double_simd_width];
        double *results = new double[double_simd_width];

        for (auto &pair : pairs) {
            if (!processed_float[pair.hap_idx * N + pair.read_idx]) pair.set_used(true);
        }
        auto double_groups = greedy_grouping<std::vector<HapReadPair>, std::vector<Group<std::vector<size_t>>>>(
            pairs, max_idle_ratio_double, double_simd_width);

        for (const auto &group : double_groups) {
            for (uint32_t i = 0; i < double_simd_width; i++) {
                const auto &pair = pairs[group.pair_indices[i]];
                tc[i].hap = haplotypes[pair.hap_idx].data();
                tc[i].rs = reads[pair.read_idx].data();
                tc[i].q = quality[pair.read_idx].data();
                tc[i].i = insertion_qualities[pair.read_idx].data();
                tc[i].d = deletion_qualities[pair.read_idx].data();
                tc[i].c = gap_contiguous_qualities[pair.read_idx].data();
                tc[i].haplen = pair.hap_len;
                tc[i].rslen = pair.read_len;
            }

            // 根据 CPU 特性选择对应的 Traits 并调用模板方法（传入 allocator）
            if (CpuFeatures::hasAVX512Support()) {
                compute_inter_pairhmm_AVX512_double(tc, double_simd_width, results, allocator, true);
            }
            else if (CpuFeatures::hasAVX2Support()) {
                compute_inter_pairhmm_AVX2_double(tc, double_simd_width, results, allocator, true);
            }

            for (uint32_t i = 0; i < double_simd_width; i++) {
                const auto &pair = pairs[group.pair_indices[i]];
                result[pair.read_idx][pair.hap_idx] = results[i];
                processed_double[pairs[group.pair_indices[i]].hap_idx * N + pairs[group.pair_indices[i]].read_idx] = true;
            }
        }
        delete[] tc;
        delete[] results;
    }

    // 第三步：处理剩余未分组的对，使用intra策略
    int total_num_pairs_intra = 0;
    for (size_t h = 0; h < M; ++h) {
        for (size_t r = 0; r < N; ++r) {
            if (!processed_double[h * N + r]) {
                pairhmm::common::TestCase tc;
                bool use_double = processed_float[h * N + r];
                double results = 0.0;
                tc.hap = haplotypes[h].data();
                tc.rs = reads[r].data();
                tc.q = quality[r].data();
                tc.i = insertion_qualities[r].data();
                tc.d = deletion_qualities[r].data();
                tc.c = gap_contiguous_qualities[r].data();
                tc.haplen = haplotypes[h].size();
                tc.rslen = reads[r].size();
                if (CpuFeatures::hasAVX512Support()) {
                    results = intra::computeLikelihoodsAVX512(tc, use_double);
                }
                else if (CpuFeatures::hasAVX2Support()) {
                    results = intra::computeLikelihoodsAVX2(tc, use_double);
                }
                result[r][h] = results;
                total_num_pairs_intra++;
            }
        }
    }
    if (verbose) {
        std::cerr << "Total num pairs float: " << total_num_pairs_float << std::endl;
        std::cerr << "Total num pairs intra: " << total_num_pairs_intra << std::endl;
    }
    return true;
}

// 直接接受 HaplotypeVector 和 ReadVector 的版本实现
bool schedule_pairhmm_with_allocator(const rovaca::HaplotypeVector &hs, rovaca::ReadVector &rs, rovaca::DoubleVector2D *result,
                                     rovaca::roDV_adapter::RovacaMemoryPoolAllocator &allocator, int32_t min_quality_threshold,
                                     PcrIndelModel pcr_option, bool use_double, double max_idle_ratio_float, double max_idle_ratio_double,
                                     bool verbose)
{
    using namespace rovaca;

    const size_t M = hs.size();
    const size_t N = rs.size();

    if (M > static_cast<size_t>(kMaxAllowedHaplotypeIndex) + 1) {
        RovacaLogger::error("Haplotype index exceeds the maximum allowed value");
        return false;
    }
    if (N > static_cast<size_t>(kMaxAllowedReadIndex) + 1) {
        RovacaLogger::error("Read index exceeds the maximum allowed value");
        return false;
    }

    if (M == 0 || N == 0 || !result) {
        return false;
    }

    // 获取内存池（从 allocator 获取，如果是 RovacaMemoryPoolAllocator）
    pMemoryPool pool = allocator.get_pool();

    // 初始化结果矩阵 [N][M] (read x haplotype)
    result->resize(N);
    for (size_t i = 0; i < N; ++i) {
        (*result)[i].resize(M, 0.0);
    }

    uint32_t max_rslen = 0;
    uint32_t max_haplen = 0;
    for (size_t r = 0; r < N; ++r) {
        if (rs[r] && !rs[r]->is_empty()) {
            max_rslen = std::max(max_rslen, static_cast<uint32_t>(rs[r]->seq_length()));
        }
    }
    for (size_t h = 0; h < M; ++h) {
        if (hs[h] && hs[h]->get_bases()) {
            max_haplen = std::max(max_haplen, static_cast<uint32_t>(hs[h]->get_bases()->num));
        }
    }
    size_t scratch_bytes = rovaca::roDV_adapter::PairHMMAllocator::scratch_bytes(max_rslen, max_haplen, CpuFeatures::hasAVX512Support());
    rovaca::roDV_adapter::PairHMMAllocator pairhmm_allocator = rovaca::roDV_adapter::PairHMMAllocator(scratch_bytes, pool);
    // 生成所有单倍型-read对
    std::pmr::vector<HapReadPair> pairs(pool);
    pairs.reserve(M * N);
    std::pmr::vector<bool> contains_n_hap_vec(M, false, pool);
    std::pmr::vector<bool> contains_n_read_vec(N, false, pool);

    // 检查包含 N 的情况
    for (size_t h = 0; h < M; ++h) {
        if (hs[h]) {
            pBases hap_bases = hs[h]->get_bases();
            if (hap_bases && hap_bases->num > 0) {
                for (uint32_t i = 0; i < hap_bases->num; ++i) {
                    if (hap_bases->data[i] == 'N') {  // 'N'
                        contains_n_hap_vec[h] = true;
                        break;
                    }
                }
            }
        }
    }

    for (size_t r = 0; r < N; ++r) {
        if (rs[r] && !rs[r]->is_empty()) {
            uint32_t len = rs[r]->seq_length();
            for (uint32_t i = 0; i < len; ++i) {
                if (rs[r]->seq_i(i) == 'N') {  // 'N'
                    contains_n_read_vec[r] = true;
                    break;
                }
            }
        }
    }

    // 创建 pairs
    for (size_t h = 0; h < M; ++h) {
        if (h > kMaxAllowedHaplotypeIndex) {
            return false;
        }
        uint32_t hap_len = hs[h] && hs[h]->get_bases() ? hs[h]->get_bases()->num : 0;
        if (hap_len > kMaxAllowedHaplotypeLength) {
            return false;
        }
        for (size_t r = 0; r < N; ++r) {
            if (r > kMaxAllowedReadIndex) {
                return false;
            }
            uint32_t read_len = rs[r] && !rs[r]->is_empty() ? rs[r]->seq_length() : 0;
            if (read_len > kMaxAllowedReadLength) {
                return false;
            }
            if (hap_len > 0 && read_len > 0) {
                pairs.emplace_back(h, r, hap_len, read_len, contains_n_hap_vec[h] || contains_n_read_vec[r]);
            }
        }
    }

    if (pairs.empty()) {
        return true;
    }

    std::sort(pairs.begin(), pairs.end(), [](const HapReadPair &lhs, const HapReadPair &rhs) {
        if (lhs.hap_len == rhs.hap_len) {
            return lhs.read_len > rhs.read_len;
        }
        return lhs.hap_len > rhs.hap_len;
    });

    uint32_t float_simd_width = 0;
    uint32_t double_simd_width = 0;

    if (CpuFeatures::hasAVX512Support()) {
        float_simd_width = 16;
        double_simd_width = 8;
    }
    else if (CpuFeatures::hasAVX2Support()) {
        float_simd_width = 8;
        double_simd_width = 4;
    }

    // 标记哪些对已经被处理
    std::pmr::vector<bool> processed_float(M * N, false, pool);
    std::pmr::vector<bool> processed_double(M * N, false, pool);

    // 为每个 read 缓存解码后的数据（避免重复解码）
    struct ReadData
    {
        pBases read_base{nullptr};
        const uint8_t *read_qual{nullptr};
        pBases read_qual_storage{nullptr};
        const uint8_t *ins_gops{nullptr};
        pBases ins_gops_storage{nullptr};
        const uint8_t *del_gops{nullptr};
        pBases del_gops_storage{nullptr};
        const uint8_t *gap_conts{nullptr};
        pBases gap_conts_storage{nullptr};
        bool valid{false};
    };
    std::pmr::vector<ReadData> read_cache(N, pool);

    constexpr uint32_t kCachedReadBufferLen = 1000;
    thread_local std::array<uint8_t, kCachedReadBufferLen> kDefaultGapConts = []() {
        std::array<uint8_t, kCachedReadBufferLen> arr{};
        arr.fill(static_cast<uint8_t>('+' - 33));
        return arr;
    }();
    thread_local std::array<uint8_t, kCachedReadBufferLen> kDefaultGapPenalty = []() {
        std::array<uint8_t, kCachedReadBufferLen> arr{};
        arr.fill(static_cast<uint8_t>('-'));
        return arr;
    }();

    // 预解码所有 read 数据
    for (size_t r = 0; r < N; ++r) {
        read_cache[r].valid = false;
        if (!rs[r] || rs[r]->is_empty()) {
            continue;
        }

        // 解码 read 序列
        pBases read_base = rs[r]->decode_to_str(pool);
        if (!read_base || read_base->num == 0) {
            continue;
        }

        const uint32_t read_len = read_base->num;

        // 质量值：仅在需要修改时才拷贝
        pBases read_qual_storage = nullptr;
        uint8_t *read_qual_ptr = nullptr;
        {
            uint8_t *raw_qual = rs[r]->qual();
            bool need_copy = (raw_qual == nullptr);
            uint8_t mq = rs[r]->mapping_quality();
            if (!need_copy) {
                for (uint32_t x = 0; x < read_len; ++x) {
                    uint8_t original = raw_qual[x];
                    if (original > mq || original < min_quality_threshold) {
                        need_copy = true;
                        break;
                    }
                }
            }
            if (need_copy) {
                read_qual_storage = new ALLOC_FLEXIBLE_IN_POOL(pool, Bases, read_len, uint8_t) Bases{read_len};
                rs[r]->decode_qual(read_qual_storage);
                for (uint32_t x = 0; x < read_len; ++x) {
                    uint8_t value = std::min<uint8_t>(read_qual_storage->data[x], mq);
                    value = value < min_quality_threshold ? static_cast<uint8_t>(MIN_QUALITY) : value;
                    read_qual_storage->data[x] = value;
                }
                read_qual_ptr = read_qual_storage->data;
            }
            else {
                read_qual_ptr = raw_qual;
            }
        }

        const bool can_use_cached_buffers = read_len <= kCachedReadBufferLen;
        const bool share_gap_penalty = (pcr_option == PcrIndelModel::NONE);

        // 插入/删除质量
        pBases ins_gops_storage = nullptr;
        pBases del_gops_storage = nullptr;
        const uint8_t *ins_gops_ptr = nullptr;
        const uint8_t *del_gops_ptr = nullptr;
        if (share_gap_penalty && can_use_cached_buffers) {
            ins_gops_ptr = kDefaultGapPenalty.data();
            del_gops_ptr = kDefaultGapPenalty.data();
        }
        else if (share_gap_penalty) {
            ins_gops_storage = new ALLOC_FLEXIBLE_IN_POOL(pool, Bases, read_len, uint8_t) Bases{read_len};
            rs[r]->ins_gops(ins_gops_storage);
            ins_gops_ptr = ins_gops_storage->data;
            del_gops_ptr = ins_gops_storage->data;
            del_gops_storage = ins_gops_storage;
        }
        else {
            ins_gops_storage = new ALLOC_FLEXIBLE_IN_POOL(pool, Bases, read_len, uint8_t) Bases{read_len};
            rs[r]->ins_gops(ins_gops_storage);
            del_gops_storage = ins_gops_storage;
            ins_gops_ptr = ins_gops_storage->data;
            del_gops_ptr = del_gops_storage->data;
        }

        // gap 连续质量
        pBases gap_conts_storage = nullptr;
        const uint8_t *gap_conts_ptr = nullptr;
        if (can_use_cached_buffers) {
            gap_conts_ptr = kDefaultGapConts.data();
        }
        else {
            gap_conts_storage = new ALLOC_FLEXIBLE_IN_POOL(pool, Bases, read_len, uint8_t) Bases{read_len};
            rs[r]->gap_conts(gap_conts_storage);
            gap_conts_ptr = gap_conts_storage->data;
        }

        // 应用 PCR 错误模型（只在需要修改时）
        if (pcr_option != PcrIndelModel::NONE && ins_gops_storage) {
            apply_pcr_error_model_rolling(read_base, ins_gops_storage, pcr_option);
        }
        if (verbose) {
            std::cerr << "Read " << r << " base: ";
            for (uint32_t i = 0; i < read_base->num; i++) {
                std::cerr << (char)read_base->data[i];
            }
            std::cerr << std::endl;
            std::cerr << " qual: ";
            for (uint32_t i = 0; i < read_len; i++) {
                std::cerr << static_cast<int>(read_qual_ptr ? read_qual_ptr[i] : 0) << " ";
            }
            std::cerr << std::endl;
            std::cerr << " ins_gops: ";
            for (uint32_t i = 0; i < read_len; i++) {
                std::cerr << static_cast<int>(ins_gops_ptr ? ins_gops_ptr[i] : 0) << " ";
            }
            std::cerr << std::endl;
            std::cerr << " del_gops: ";
            for (uint32_t i = 0; i < read_len; i++) {
                std::cerr << static_cast<int>(del_gops_ptr ? del_gops_ptr[i] : 0) << " ";
            }
            std::cerr << std::endl;
            std::cerr << " gap_conts: ";
            for (uint32_t i = 0; i < read_len; i++) {
                std::cerr << static_cast<int>(gap_conts_ptr ? gap_conts_ptr[i] : 0);
            }
            std::cerr << " contains_n_read_vec: " << contains_n_read_vec[r] << std::endl;
            std::cerr << std::endl;
        }

        read_cache[r].read_base = read_base;
        read_cache[r].read_qual = read_qual_ptr;
        read_cache[r].read_qual_storage = read_qual_storage;
        read_cache[r].ins_gops = ins_gops_ptr;
        read_cache[r].ins_gops_storage = ins_gops_storage;
        read_cache[r].del_gops = del_gops_ptr;
        read_cache[r].del_gops_storage = del_gops_storage;
        read_cache[r].gap_conts = gap_conts_ptr;
        read_cache[r].gap_conts_storage = gap_conts_storage;
        read_cache[r].valid = true;
    }

    // 第一步：尝试float类型分组
    if (!use_double && float_simd_width > 0) {
        using GroupType = Group<std::pmr::vector<size_t>>;
        std::pmr::vector<GroupType> float_groups(pool);
        float_groups = greedy_grouping<decltype(pairs), decltype(float_groups)>(pairs, max_idle_ratio_float, float_simd_width);
        pairhmm::common::TestCase *tc = (pairhmm::common::TestCase *)pool->allocate(float_simd_width * sizeof(pairhmm::common::TestCase));
        double *results = (double *)pool->allocate(float_simd_width * sizeof(double));
        for (const auto &group : float_groups) {
            for (uint32_t i = 0; i < float_simd_width; i++) {
                const auto &pair = pairs[group.pair_indices[i]];

                // 获取 haplotype 数据
                if (hs[pair.hap_idx] && hs[pair.hap_idx]->get_bases()) {
                    pBases hap_bases = hs[pair.hap_idx]->get_bases();
                    tc[i].hap = hap_bases->data;
                    tc[i].haplen = hap_bases->num;
                }
                else {
                    tc[i].hap = nullptr;
                    tc[i].haplen = 0;
                }

                // 获取 read 数据（使用缓存的解码数据）
                if (read_cache[pair.read_idx].valid) {
                    const auto &rd = read_cache[pair.read_idx];
                    tc[i].rs = rd.read_base->data;
                    tc[i].rslen = rd.read_base->num;
                    tc[i].q = rd.read_qual;
                    tc[i].i = rd.ins_gops;
                    tc[i].d = rd.del_gops;
                    tc[i].c = rd.gap_conts;
                }
                else {
                    tc[i].rs = nullptr;
                    tc[i].rslen = 0;
                    tc[i].q = nullptr;
                    tc[i].i = nullptr;
                    tc[i].d = nullptr;
                    tc[i].c = nullptr;
                }
            }

            // 根据 CPU 特性选择对应的 Traits 并调用模板方法
            if (CpuFeatures::hasAVX512Support()) {
                compute_inter_pairhmm_AVX512_float(tc, float_simd_width, results, pairhmm_allocator, false);
            }
            else if (CpuFeatures::hasAVX2Support()) {
                compute_inter_pairhmm_AVX2_float(tc, float_simd_width, results, pairhmm_allocator, false);
            }
            pairhmm_allocator.reset();

            for (uint32_t i = 0; i < float_simd_width; i++) {
                const auto &pair = pairs[group.pair_indices[i]];
                if (results[i] < MIN_ACCEPTED) {
                    pairs[group.pair_indices[i]].set_used(false);
                }
                else {
                    pairs[group.pair_indices[i]].set_used(true);
                    (*result)[pair.read_idx][pair.hap_idx] = inter::loglikelihoodfloat(results[i]);
                    processed_double[pair.hap_idx * N + pair.read_idx] = true;
                }
                processed_float[pair.hap_idx * N + pair.read_idx] = true;
            }
        }
    }

    // 第二步：尝试double类型分组
    if (double_simd_width > 0) {
        pairhmm::common::TestCase *tc = (pairhmm::common::TestCase *)pool->allocate(double_simd_width * sizeof(pairhmm::common::TestCase));
        double *results = (double *)pool->allocate(double_simd_width * sizeof(double));

        for (auto &pair : pairs) {
            if (!processed_float[pair.hap_idx * N + pair.read_idx]) {
                pair.set_used(true);
            }
        }
        using GroupType = Group<std::pmr::vector<size_t>>;
        std::pmr::vector<GroupType> double_groups(pool);
        double_groups = greedy_grouping<decltype(pairs), decltype(double_groups)>(pairs, max_idle_ratio_double, double_simd_width);

        for (const auto &group : double_groups) {
            for (uint32_t i = 0; i < double_simd_width; i++) {
                const auto &pair = pairs[group.pair_indices[i]];

                // 获取 haplotype 数据
                if (hs[pair.hap_idx] && hs[pair.hap_idx]->get_bases()) {
                    pBases hap_bases = hs[pair.hap_idx]->get_bases();
                    tc[i].hap = hap_bases->data;
                    tc[i].haplen = hap_bases->num;
                }
                else {
                    tc[i].hap = nullptr;
                    tc[i].haplen = 0;
                }

                // 获取 read 数据（使用缓存的解码数据）
                if (read_cache[pair.read_idx].valid) {
                    const auto &rd = read_cache[pair.read_idx];
                    tc[i].rs = rd.read_base->data;
                    tc[i].rslen = rd.read_base->num;
                    tc[i].q = rd.read_qual;
                    tc[i].i = rd.ins_gops;
                    tc[i].d = rd.del_gops;
                    tc[i].c = rd.gap_conts;
                }
                else {
                    tc[i].rs = nullptr;
                    tc[i].rslen = 0;
                    tc[i].q = nullptr;
                    tc[i].i = nullptr;
                    tc[i].d = nullptr;
                    tc[i].c = nullptr;
                }
            }

            // 根据 CPU 特性选择对应的 Traits 并调用模板方法
            if (CpuFeatures::hasAVX512Support()) {
                compute_inter_pairhmm_AVX512_double(tc, double_simd_width, results, pairhmm_allocator, true);
            }
            else if (CpuFeatures::hasAVX2Support()) {
                compute_inter_pairhmm_AVX2_double(tc, double_simd_width, results, pairhmm_allocator, true);
            }
            pairhmm_allocator.reset();
            for (uint32_t i = 0; i < double_simd_width; i++) {
                const auto &pair = pairs[group.pair_indices[i]];
                (*result)[pair.read_idx][pair.hap_idx] = results[i];
                processed_double[pair.hap_idx * N + pair.read_idx] = true;
            }
        }
    }

    // 第三步：处理剩余未分组的对，使用intra策略
    for (size_t h = 0; h < M; ++h) {
        for (size_t r = 0; r < N; ++r) {
            if (!processed_double[h * N + r]) {
                pairhmm::common::TestCase tc;
                bool use_double = processed_float[h * N + r];
                double results = 0.0;

                // 获取 haplotype 数据
                if (hs[h] && hs[h]->get_bases()) {
                    pBases hap_bases = hs[h]->get_bases();
                    tc.hap = hap_bases->data;
                    tc.haplen = hap_bases->num;
                }
                else {
                    tc.hap = nullptr;
                    tc.haplen = 0;
                }

                // 获取 read 数据（使用缓存的解码数据）
                if (read_cache[r].valid) {
                    const auto &rd = read_cache[r];
                    tc.rs = rd.read_base->data;
                    tc.rslen = rd.read_base->num;
                    tc.q = rd.read_qual;
                    tc.i = rd.ins_gops;
                    tc.d = rd.del_gops;
                    tc.c = rd.gap_conts;
                }
                else {
                    tc.rs = nullptr;
                    tc.rslen = 0;
                    tc.q = nullptr;
                    tc.i = nullptr;
                    tc.d = nullptr;
                    tc.c = nullptr;
                }

                if (tc.hap && tc.rs && tc.haplen > 0 && tc.rslen > 0) {
                    if (CpuFeatures::hasAVX512Support()) {
                        results = intra::computeLikelihoodsAVX512(tc, use_double);
                    }
                    else if (CpuFeatures::hasAVX2Support()) {
                        results = intra::computeLikelihoodsAVX2(tc, use_double);
                    }
                    (*result)[r][h] = results;
                }
            }
        }
    }

    return true;
}

// 显式实例化模板函数
template bool schedule_pairhmm_with_allocator<rovaca::roDV_adapter::RovacaMemoryPoolAllocator>(
    const std::vector<std::vector<uint8_t>> &haplotypes, const std::vector<std::vector<uint8_t>> &reads,
    std::vector<std::vector<double>> &result, const std::vector<std::vector<uint8_t>> &quality,
    const std::vector<std::vector<uint8_t>> &insertion_qualities, const std::vector<std::vector<uint8_t>> &deletion_qualities,
    const std::vector<std::vector<uint8_t>> &gap_contiguous_qualities, rovaca::roDV_adapter::RovacaMemoryPoolAllocator &allocator,
    bool use_double, double max_idle_ratio_float, double max_idle_ratio_double, bool verbose);

}  // namespace schedule
}  // namespace pairhmm
