#ifndef PAIRHMM_TEST_CASE_COMMON_H_
#define PAIRHMM_TEST_CASE_COMMON_H_

#include "../roDV_common/common.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <string>
#include <vector>

using pairhmm::common::TestCase;

/**
 * @brief 测试用例数据结构
 */
struct TestCaseData {
  std::string hap_bases;
  std::string read_bases;
  std::vector<uint8_t> read_qual;
  std::vector<uint8_t> read_ins_qual;
  std::vector<uint8_t> read_del_qual;
  std::vector<uint8_t> gcp;
  double expected_result;
  uint32_t line_number;
};

/**
 * @brief 对齐内存分配器
 */
template <size_t Alignment> class AlignedAllocator {
public:
  using value_type = uint8_t;
  static constexpr size_t align_value = Alignment;

  uint8_t *allocate(size_t n) {
    size_t size = n + Alignment - 1;
    void *ptr = nullptr;

// 使用 posix_memalign 进行对齐分配
#ifdef _GNU_SOURCE
    if (posix_memalign(&ptr, Alignment, size) != 0) {
      ptr = nullptr;
    }
#else
    // 回退到普通的 malloc
    ptr = std::malloc(size);
#endif

    if (!ptr) {
      throw std::bad_alloc();
    }
    return static_cast<uint8_t *>(ptr);
  }

  void deallocate(uint8_t *ptr, size_t) {
    if (ptr) {
      std::free(ptr);
    }
  }
};

template <size_t Alignment> struct AlignmentChecker {
  // 检查是否是2的幂次
  static constexpr bool is_power_of_two =
      (Alignment > 0) && ((Alignment & (Alignment - 1)) == 0);

  // 检查是否能被32整除
  static constexpr bool is_divisible_by_32 = (Alignment % 32 == 0);

  // 检查是否至少32字节
  static constexpr bool is_at_least_32 = (Alignment >= 32);

  // 综合检查
  static constexpr bool is_valid =
      is_power_of_two && is_divisible_by_32 && is_at_least_32;

  static_assert(is_power_of_two, "Alignment must be a power of 2");
  static_assert(is_divisible_by_32, "Alignment must be divisible by 32");
  static_assert(is_at_least_32, "Alignment must be at least 32 bytes");
};

/**
 * @brief 测试用例包装器，确保内存安全
 */
// 检查alignment是否为32或64
template <size_t alignment> class TestCaseWrapper {
  static_assert(AlignmentChecker<alignment>::is_valid,
                "Alignment must be a power of 2 and divisible by 32 and at "
                "least 32 bytes");

public:
  TestCaseWrapper(const TestCaseData &data)
      : hap_data_(nullptr), rs_data_(nullptr), q_data_(nullptr),
        i_data_(nullptr), d_data_(nullptr), c_data_(nullptr) {
    // 使用对齐分配器分配内存
    size_t hap_size = data.hap_bases.size() + alignment;
    size_t rs_size = data.read_bases.size() + alignment;
    size_t q_size = data.read_qual.size() + alignment;
    size_t i_size = data.read_ins_qual.size() + alignment;
    size_t d_size = data.read_del_qual.size() + alignment;
    size_t c_size = data.gcp.size() + alignment;

    hap_data_ = allocator_.allocate(hap_size);
    rs_data_ = allocator_.allocate(rs_size);
    q_data_ = allocator_.allocate(q_size);
    i_data_ = allocator_.allocate(i_size);
    d_data_ = allocator_.allocate(d_size);
    c_data_ = allocator_.allocate(c_size);

    // 填充数据
    for (size_t i = 0; i < data.hap_bases.size(); i++) {
      hap_data_[i] = data.hap_bases[i];
    }

    for (size_t i = 0; i < data.read_bases.size(); i++) {
      rs_data_[i] = data.read_bases[i];
    }

    std::copy(data.read_qual.begin(), data.read_qual.end(), q_data_);
    std::copy(data.read_ins_qual.begin(), data.read_ins_qual.end(), i_data_);
    std::copy(data.read_del_qual.begin(), data.read_del_qual.end(), d_data_);
    std::copy(data.gcp.begin(), data.gcp.end(), c_data_);

    // 设置TestCase结构
    tc_.haplen = static_cast<uint32_t>(data.hap_bases.size());
    tc_.rslen = static_cast<uint32_t>(data.read_bases.size());
    tc_.hap = hap_data_;
    tc_.rs = rs_data_;
    tc_.q = q_data_;
    tc_.i = i_data_;
    tc_.d = d_data_;
    tc_.c = c_data_;
  }

  ~TestCaseWrapper() {
    allocator_.deallocate(hap_data_, 0);
    allocator_.deallocate(rs_data_, 0);
    allocator_.deallocate(q_data_, 0);
    allocator_.deallocate(i_data_, 0);
    allocator_.deallocate(d_data_, 0);
    allocator_.deallocate(c_data_, 0);
  }

  // 禁用拷贝构造和赋值操作
  TestCaseWrapper(const TestCaseWrapper &) = delete;
  TestCaseWrapper &operator=(const TestCaseWrapper &) = delete;

  const TestCase &getTestCase() const { return tc_; }
  TestCase &getTestCase() { return tc_; }

private:
  AlignedAllocator<alignment> allocator_;
  uint8_t *hap_data_;
  uint8_t *rs_data_;
  uint8_t *q_data_;
  uint8_t *i_data_;
  uint8_t *d_data_;
  uint8_t *c_data_;
  TestCase tc_;
};

#endif  // PAIRHMM_TEST_CASE_COMMON_H_

