#include <cmath>
#include <cstring>  // for std::memcmp
#include <limits>
#include <memory_resource>
#include <vector>

#include "../common/utils/rovaca_memory_pool.h"
#include "../genotype/haplotype.h"
#include "../genotype/read_record.h"
#include "adapter.h"
#include "rovaca_logger.h"
#include "pairhmm_internal.h"
#include "roDV_adapter/rovaca_pool_allocator.h"
#include "roDV_common/common.h"
#include "roDV_common/context.h"
#include "roDV_schedule.h"
namespace rovaca
{

namespace
{
template <typename Container>
void collect_poorly_modelled_indices(const ReadVector& reads, const DoubleVector2D& likelihoods, Container& out)
{
    const size_t count = std::min(reads.size(), likelihoods.size());
    for (size_t i = 0; i < count; ++i) {
        pReadRecord read_i = reads[i];
        if (!read_i) {
            continue;
        }
        const DoubleVector& likelihoods_i = likelihoods[i];
        double best_likelihood = -std::numeric_limits<double>::infinity();
        for (double val : likelihoods_i) {
            if (!std::isnan(val)) {
                best_likelihood = std::max(best_likelihood, val);
            }
        }
        double log10_min_likelihood =
            std::min(2.0, std::ceil(read_i->seq_length() * EXPECTED_ERROR_RATE_PER_BASE)) * LOG10_QUALITY_PER_BASE;
        if (best_likelihood < log10_min_likelihood) {
            out.push_back(i);
        }
    }
}
}  // namespace

namespace
{
// 前向声明
class RollingHashView;

// 线程局部存储的缓冲区池
class RollingHashViewPool
{
private:
    struct ViewStorage
    {
        std::vector<uint64_t> prefix_;
        std::vector<uint64_t> power_;
        size_t capacity_ = 0;

        void resize(size_t new_size)
        {
            if (new_size > capacity_) {
                capacity_ = new_size;
                prefix_.resize(new_size);
                power_.resize(new_size);
            }
        }

        void clear()
        {
            capacity_ = 0;
            prefix_.clear();
            power_.clear();
            prefix_.shrink_to_fit();
            power_.shrink_to_fit();
        }
    };

    // 使用 thread_local 确保线程安全，每个线程有独立的缓冲区
    thread_local static ViewStorage storage_;

public:
    // 创建 RollingHashView，使用线程局部缓冲区
    static RollingHashView create(pBases bases);

    // 清理线程局部缓冲区（可选，用于释放内存）
    static void clear() { storage_.clear(); }
};

// 定义 thread_local 静态成员
thread_local RollingHashViewPool::ViewStorage RollingHashViewPool::storage_;

// 修改后的 RollingHashView 类
class RollingHashView
{
public:
    // 构造函数：使用外部缓冲区（不拥有内存）
    RollingHashView(size_t len, const uint64_t* prefix_buf, const uint64_t* power_buf)
        : length_(len)
        , prefix_ptr_(prefix_buf)
        , power_ptr_(power_buf)
    {}

    [[nodiscard]] size_t size() const { return length_; }

    [[nodiscard]] uint64_t substring(size_t begin, size_t end) const
    {
        return prefix_ptr_[end] - prefix_ptr_[begin] * power_ptr_[end - begin];
    }

private:
    size_t length_;

    // 使用指针访问外部缓冲区
    const uint64_t* prefix_ptr_;
    const uint64_t* power_ptr_;
};

// 实现 RollingHashViewPool::create
RollingHashView RollingHashViewPool::create(pBases bases)
{
    if (!bases || bases->num == 0) {
        return RollingHashView(0, nullptr, nullptr);
    }

    size_t len = bases->num;
    size_t size = len + 1;

    // 确保缓冲区足够大
    storage_.resize(size);

    // 计算 prefix 和 power
    storage_.power_[0] = 1;
    storage_.prefix_[0] = 0;
    for (size_t i = 0; i < len; ++i) {
        storage_.power_[i + 1] = storage_.power_[i] * 1315423911ULL;
        storage_.prefix_[i + 1] = storage_.prefix_[i] * 1315423911ULL + static_cast<uint64_t>(seq_nt16_int[bases->data[i]]) + 1ULL;
    }

    // 返回使用外部缓冲区的 RollingHashView
    return RollingHashView(len, storage_.prefix_.data(), storage_.power_.data());
}

int32_t find_number_of_repetitions_rolling(const RollingHashView& hash_view, int32_t repeat_unit_offset, int32_t repeat_unit_length,
                                           int32_t test_offset, int32_t test_length, bool leading, const uint8_t* seq_data)
{
    if (test_length <= 0 || repeat_unit_length <= 0 || repeat_unit_length > test_length) {
        return 0;
    }
    if (repeat_unit_offset < 0 || test_offset < 0) {
        return 0;
    }

    const size_t hash_size = hash_view.size();
    const size_t repeat_end = static_cast<size_t>(repeat_unit_offset) + static_cast<size_t>(repeat_unit_length);
    const size_t test_end = static_cast<size_t>(test_offset) + static_cast<size_t>(test_length);
    if (repeat_end > hash_size || test_end > hash_size) {
        return 0;
    }

    const uint64_t repeat_hash = hash_view.substring(static_cast<size_t>(repeat_unit_offset), repeat_end);
    int32_t num_repeats = 0;
    
    // 辅助函数：验证两个子串是否真的相等（碰撞检测）
    // 对于短序列（< 1000），碰撞概率极低，可以跳过验证以提高性能
    // 对于长序列，必须验证以防止哈希碰撞导致的误判
    auto verify_match = [&](int32_t pos1, int32_t pos2, int32_t len) -> bool {
        if (seq_data == nullptr) {
            // 如果没有提供序列数据，只能信任哈希值（向后兼容）
            return true;
        }
        // 使用 memcmp 进行快速比较（编译器通常会优化为 SIMD）
        return std::memcmp(seq_data + pos1, seq_data + pos2, len * sizeof(uint8_t)) == 0;
    };
    
    if (leading) {
        for (int32_t start = 0; start + repeat_unit_length <= test_length; start += repeat_unit_length) {
            const size_t l = static_cast<size_t>(test_offset + start);
            const size_t r = l + static_cast<size_t>(repeat_unit_length);
            const uint64_t test_hash = hash_view.substring(l, r);
            
            // 先比较哈希值（快速路径）
            if (test_hash == repeat_hash) {
                // 碰撞检测：验证实际字符串是否匹配（慢速路径，但保证正确性）
                if (verify_match(repeat_unit_offset, test_offset + start, repeat_unit_length)) {
                    ++num_repeats;
                } else {
                    // 发生碰撞！停止搜索
                    break;
                }
            }
            else {
                break;
            }
        }
    }
    else {
        for (int32_t start = test_length - repeat_unit_length; start >= 0; start -= repeat_unit_length) {
            const size_t l = static_cast<size_t>(test_offset + start);
            const size_t r = l + static_cast<size_t>(repeat_unit_length);
            const uint64_t test_hash = hash_view.substring(l, r);
            
            // 先比较哈希值（快速路径）
            if (test_hash == repeat_hash) {
                // 碰撞检测：验证实际字符串是否匹配（慢速路径，但保证正确性）
                if (verify_match(repeat_unit_offset, test_offset + start, repeat_unit_length)) {
                    ++num_repeats;
                } else {
                    // 发生碰撞！停止搜索
                    break;
                }
            }
            else {
                break;
            }
        }
    }
    return num_repeats;
}
}  // namespace

inline void init_native()
{
    // enable FTZ
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
}

/*
 * 'A' : 0x1
 * 'C' : 0x2
 * 'T' : 0x4
 * 'G' : 0x8
 * 'N' : 0xF
 */

bool string_equal(const uint8_t* str1, const uint8_t* str2, int32_t len)
{
    for (int32_t i{0}; i < len; ++i) {
        if (str1[i] != str2[i]) {
            return false;
        }
    }
    return true;
}

void normalize_likelihoods(DoubleVector2D& log_likelihoods)
{
    double best, cmp;
    for (DoubleVector& likelihoods : log_likelihoods) {
        best = std::max_element(likelihoods.begin(), likelihoods.end()).operator*();
        cmp = best + MAXIMUM_BEST_ALT_LIKELIHOOD_DIFFERENCE;
        for (double& val : likelihoods) {
            val = val < cmp ? cmp : val;
        }
    }
}

int32_t find_tandem_repeat_units(pBases bases, uint32_t offset1)
{
    int32_t offset = static_cast<int32_t>(offset1);
    int32_t base_num = static_cast<int32_t>(bases->num);

    int32_t max_bw = 0;
    std::pair<int32_t, int32_t> best_bw_repeat_unit(offset, 1);
    for (int32_t str{1}; str <= MAX_STR_UNIT_LENGTH; str++) {
        if (offset + 1 - str < 0) {
            break;
        }
        max_bw = find_number_of_repetitions(bases, offset - str + 1, str, bases, 0, offset + 1, false);
        if (max_bw > 1) {
            best_bw_repeat_unit.first = offset - str - 1;
            best_bw_repeat_unit.second = str;
            break;
        }
    }

    int32_t max_rl = max_bw;
    std::pair<int32_t, int32_t> best_repeat_unit = best_bw_repeat_unit;

    if (offset < base_num - 1) {
        std::pair<int32_t, int32_t> best_FW_repeat_unit(offset + 1, 1);
        int32_t max_fw = 0;

        for (int32_t str{1}; str <= MAX_STR_UNIT_LENGTH; str++) {
            if (offset + str + 1 > base_num) {
                break;
            }

            max_fw = find_number_of_repetitions(bases, offset + 1, str, bases, offset + 1, bases->num - offset - 1, true);
            if (max_fw > 1) {
                best_FW_repeat_unit.first = offset + 1;
                best_FW_repeat_unit.second = str;
                break;
            }
        }
        if (best_FW_repeat_unit == best_bw_repeat_unit) {
            max_rl = max_bw + max_fw;
            best_repeat_unit = best_FW_repeat_unit;
        }
        else {
            max_bw = find_number_of_repetitions(bases, best_FW_repeat_unit.first, best_FW_repeat_unit.second, bases, 0, offset + 1, false);
            max_rl = max_bw + max_fw;
            best_repeat_unit = best_FW_repeat_unit;
        }
    }

    if (max_rl > MAX_REPEAT_LENGTH) {
        max_rl = MAX_REPEAT_LENGTH;
    }

    return max_rl;
}

// 内部辅助函数，接受 RollingHashView 引用
int32_t find_tandem_repeat_units_rolling_impl(const RollingHashView& hash_view, pBases bases, uint32_t offset1)
{
    if (!bases || bases->num == 0) {
        return 0;
    }

    auto repeat_counter = [&](int32_t repeat_offset, int32_t repeat_len, int32_t test_offset, int32_t test_length, bool leading) {
        return find_number_of_repetitions_rolling(hash_view, repeat_offset, repeat_len, test_offset, test_length, leading, bases->data);
    };

    int32_t offset = static_cast<int32_t>(offset1);
    int32_t base_num = static_cast<int32_t>(bases->num);

    int32_t max_bw = 0;
    std::pair<int32_t, int32_t> best_bw_repeat_unit(offset, 1);
    for (int32_t str{1}; str <= MAX_STR_UNIT_LENGTH; ++str) {
        if (offset + 1 - str < 0) {
            break;
        }
        max_bw = repeat_counter(offset - str + 1, str, 0, offset + 1, false);
        if (max_bw > 1) {
            best_bw_repeat_unit.first = offset - str - 1;
            best_bw_repeat_unit.second = str;
            break;
        }
    }

    int32_t max_rl = max_bw;
    std::pair<int32_t, int32_t> best_repeat_unit = best_bw_repeat_unit;

    if (offset < base_num - 1) {
        std::pair<int32_t, int32_t> best_FW_repeat_unit(offset + 1, 1);
        int32_t max_fw = 0;

        for (int32_t str{1}; str <= MAX_STR_UNIT_LENGTH; ++str) {
            if (offset + str + 1 > base_num) {
                break;
            }

            max_fw = repeat_counter(offset + 1, str, offset + 1, base_num - offset - 1, true);
            if (max_fw > 1) {
                best_FW_repeat_unit.first = offset + 1;
                best_FW_repeat_unit.second = str;
                break;
            }
        }
        if (best_FW_repeat_unit == best_bw_repeat_unit) {
            max_rl = max_bw + max_fw;
            best_repeat_unit = best_FW_repeat_unit;
        }
        else {
            max_bw = repeat_counter(best_FW_repeat_unit.first, best_FW_repeat_unit.second, 0, offset + 1, false);
            max_rl = max_bw + max_fw;
            best_repeat_unit = best_FW_repeat_unit;
        }
    }

    if (max_rl > MAX_REPEAT_LENGTH) {
        max_rl = MAX_REPEAT_LENGTH;
    }

    return max_rl;
}

int32_t find_tandem_repeat_units_rolling(pBases bases, uint32_t offset1)
{
    if (!bases || bases->num == 0) {
        return 0;
    }

    RollingHashView hash_view = RollingHashViewPool::create(bases);
    return find_tandem_repeat_units_rolling_impl(hash_view, bases, offset1);
}

void apply_pcr_error_model(pBases bases, pBases gap_qual, PcrIndelModel pcr_option)
{
    int32_t repeat_length = 0;
    switch (pcr_option) {
        case PcrIndelModel::HOSTILE: {
            for (uint32_t i = 1; i < bases->num; i++) {
                repeat_length = find_tandem_repeat_units(bases, i - 1);
                gap_qual->data[i - 1] = gap_qual->data[i - 1] < PCR_INDEL_MODEL_CACHE_HOSTILE[repeat_length]
                                            ? gap_qual->data[i - 1]
                                            : PCR_INDEL_MODEL_CACHE_HOSTILE[repeat_length];
            }
            break;
        }

        case PcrIndelModel::AGGRESSIVE: {
            for (uint32_t i = 1; i < bases->num; i++) {
                repeat_length = find_tandem_repeat_units(bases, i - 1);
                gap_qual->data[i - 1] = gap_qual->data[i - 1] < PCR_INDEL_MODEL_CACHE_AGGRESSIVE[repeat_length]
                                            ? gap_qual->data[i - 1]
                                            : PCR_INDEL_MODEL_CACHE_AGGRESSIVE[repeat_length];
            }
            break;
        }

        case PcrIndelModel::CONSERVATIVE: {
            for (uint32_t i = 1; i < bases->num; i++) {
                repeat_length = find_tandem_repeat_units(bases, i - 1);
                gap_qual->data[i - 1] = gap_qual->data[i - 1] < PCR_INDEL_MODEL_CACHE_CONSERVATIVE[repeat_length]
                                            ? gap_qual->data[i - 1]
                                            : PCR_INDEL_MODEL_CACHE_CONSERVATIVE[repeat_length];
            }
            break;
        }

        default: {
            break;
        }
    }
}

void apply_pcr_error_model_rolling(pBases bases, pBases gap_qual, PcrIndelModel pcr_option)
{
    if (!bases || bases->num == 0) {
        return;
    }

    // 使用对象池创建 RollingHashView，避免内存分配
    RollingHashView hash_view = RollingHashViewPool::create(bases);

    int32_t repeat_length = 0;
    switch (pcr_option) {
        case PcrIndelModel::HOSTILE: {
            for (uint32_t i = 1; i < bases->num; i++) {
                repeat_length = find_tandem_repeat_units_rolling_impl(hash_view, bases, i - 1);
                gap_qual->data[i - 1] = gap_qual->data[i - 1] < PCR_INDEL_MODEL_CACHE_HOSTILE[repeat_length]
                                            ? gap_qual->data[i - 1]
                                            : PCR_INDEL_MODEL_CACHE_HOSTILE[repeat_length];
            }
            break;
        }

        case PcrIndelModel::AGGRESSIVE: {
            for (uint32_t i = 1; i < bases->num; i++) {
                repeat_length = find_tandem_repeat_units_rolling_impl(hash_view, bases, i - 1);
                gap_qual->data[i - 1] = gap_qual->data[i - 1] < PCR_INDEL_MODEL_CACHE_AGGRESSIVE[repeat_length]
                                            ? gap_qual->data[i - 1]
                                            : PCR_INDEL_MODEL_CACHE_AGGRESSIVE[repeat_length];
            }
            break;
        }

        case PcrIndelModel::CONSERVATIVE: {
            for (uint32_t i = 1; i < bases->num; i++) {
                repeat_length = find_tandem_repeat_units_rolling_impl(hash_view, bases, i - 1);
                gap_qual->data[i - 1] = gap_qual->data[i - 1] < PCR_INDEL_MODEL_CACHE_CONSERVATIVE[repeat_length]
                                            ? gap_qual->data[i - 1]
                                            : PCR_INDEL_MODEL_CACHE_CONSERVATIVE[repeat_length];
            }
            break;
        }

        default: {
            break;
        }
    }
}

void filter_poorly_modelled_evidence(ReadVector& reads, DoubleVector2D& likelihoods, pMemoryPool pool)
{
    std::pmr::vector<size_t> remove_idx{pool};
    collect_poorly_modelled_indices(reads, likelihoods, remove_idx);

    for (auto it = remove_idx.rbegin(); it != remove_idx.rend(); ++it) {
        reads.erase(reads.begin() + *it);
        likelihoods.erase(likelihoods.begin() + *it);
    }
}

std::vector<size_t> collect_poorly_modelled_indices(ReadVector& reads, DoubleVector2D& likelihoods)
{
    std::vector<size_t> remove_idx;
    collect_poorly_modelled_indices(reads, likelihoods, remove_idx);
    for (auto it = remove_idx.rbegin(); it != remove_idx.rend(); ++it) {
        likelihoods.erase(likelihoods.begin() + *it);
    }
    return remove_idx;
}

void transpose_likelihood_matrix(const DoubleVector2D& source, DoubleVector2D& target)
{
    size_t raw = source.size(), col = target.size();
    for (size_t i{0}; i < raw; ++i) {
        const DoubleVector& source_i = source[i];
        for (size_t j{0}; j < col; ++j) {
            target[j][i] = source_i[j];
        }
    }
}

int32_t find_number_of_repetitions(pBases repeat_unit_full, int32_t offset_in_repeat_unit_full, int32_t repeat_unit_length,
                                   pBases test_string_full, int32_t offset_in_test_string_full, int32_t test_string_length,
                                   bool leading_repeats)
{
    if (test_string_length == 0) {
        return 0;
    }

    int32_t num_repeats = 0;
    int32_t length_diff = test_string_length - repeat_unit_length;

    if (leading_repeats) {
        for (int32_t start{0}; start <= length_diff; start += repeat_unit_length) {
            if (string_equal(test_string_full->data + start + offset_in_test_string_full,
                             repeat_unit_full->data + offset_in_repeat_unit_full, repeat_unit_length)) {
                ++num_repeats;
            }
            else {
                return num_repeats;
            }
        }
    }
    else {
        for (int32_t start = length_diff; start >= 0; start -= repeat_unit_length) {
            if (string_equal(test_string_full->data + start + offset_in_test_string_full,
                             repeat_unit_full->data + offset_in_repeat_unit_full, repeat_unit_length)) {
                ++num_repeats;
            }
            else {
                return num_repeats;
            }
        }
    }
    return num_repeats;
}

void debug_print(uint32_t* arr, uint32_t len = k_avx512_float_read_concurrency)
{
    for (uint32_t i{0}; i < len; ++i) {
        printf("%d\t", arr[i]);
    }

    printf("\n");
    fflush(stdout);
}

void debug_tools_print_arr(const uint8_t* arr, int32_t len, uint8_t sub_num = 0)
{
    for (int32_t i{0}; i < len; ++i) {
        printf("%c", arr[i] + sub_num);
    }
    printf("\n");
}

void debug_tools_print_temp(const ::TestCase& temp)
{
    printf("%d \t %d\n", temp.rslen, temp.haplen);
    debug_tools_print_arr(temp.hap, temp.haplen);
    debug_tools_print_arr(temp.rs, temp.rslen);
    debug_tools_print_arr(temp.q, temp.rslen, 33);
    debug_tools_print_arr(temp.i, temp.rslen, 33);
    debug_tools_print_arr(temp.d, temp.rslen, 33);
    debug_tools_print_arr(temp.c, temp.rslen, 33);
    fflush(stdout);
}

DoubleVector2D call_roDV_pairhmm_scheduled(const HaplotypeVector& hs, ReadVector& rs, int32_t min_quality_threshold,
                                           PcrIndelModel pcr_option, pMemoryPool pool)
{
    // 使用 MemoryPoolGuard 管理内存
    // 注意：需要将 pool 转换为 RovacaMemoryPool* 类型
    DoubleVector2D result(hs.size(), DoubleVector(rs.size(), NAN, pool), pool);
    std::vector<size_t> remove_indices;
    RovacaMemoryPool* rovaca_pool = dynamic_cast<RovacaMemoryPool*>(pool);

    if (!rovaca_pool) {
        RovacaLogger::error("pool is not RovacaMemoryPool");
        return result;
    }
    {
        MemoryPoolGuard guard(rovaca_pool);
        // 直接使用新接口，避免数据转换
        roDV_adapter::RovacaMemoryPoolAllocator allocator(pool);
        DoubleVector2D read_major(rs.size(), DoubleVector(hs.size(), NAN, pool), pool);

        // 初始化结果矩阵：[read][haplotype] 用于调度输出，再转置成 [haplotype][read]

        // 调用新版本的 schedule_pairhmm_with_allocator（直接接受 HaplotypeVector 和 ReadVector）
        bool success = pairhmm::schedule::schedule_pairhmm_with_allocator(hs, rs, &read_major, allocator, min_quality_threshold, pcr_option,
                                                                          false,  // use_double
                                                                          0.02,   // max_idle_ratio_float
                                                                          0.02,   // max_idle_ratio_double
                                                                          false   // verbose
        );

        if (!success) {
            // 如果失败，返回空结果
            RovacaLogger::error("schedule_pairhmm_with_allocator failed");
            DoubleVector2D empty_result(hs.size(), DoubleVector(rs.size(), NAN, pool), pool);
            return empty_result;
        }
        normalize_likelihoods(read_major);
        remove_indices = collect_poorly_modelled_indices(rs, read_major);
        transpose_likelihood_matrix(read_major, result);
    }
    // guard 离开作用域后再同步删除对应的 reads 及结果列
    for (auto it = remove_indices.rbegin(); it != remove_indices.rend(); ++it) {
        size_t idx = *it;
        rs.erase(rs.begin() + idx);
    }

    // 结果已经是 [haplotype][read] 格式，直接返回
    return result;
}

}  // namespace rovaca