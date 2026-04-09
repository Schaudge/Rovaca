#include "../roDV_common/cpu_features.h"
#include "../roDV_inter/pairhmm_inter_api.h"
#include "../roDV_intra/pairhmm_api.h"
#include "../roDV_adapter/rovaca_pool_allocator.h"
#include "test_case_common.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <vector>

using namespace pairhmm::intra;
using pairhmm::common::CpuFeatures;
using pairhmm::common::ConvertChar;
/**
 * @brief PairHMM Unit Tests using GoogleTest
 * 参考 Intel GKL PairHMM 测试用例
 *
 * 测试数据格式:
 * hap-bases read-bases read-qual read-ins-qual read-del-qual gcp
 * expected-result
 */

/**
 * @brief 测试数据解析器
 */
class TestDataParser {
public:
  /**
   * @brief 从文件加载测试数据
   */
  static std::vector<TestCaseData> loadTestData(const std::string &filename) {
    std::vector<TestCaseData> test_cases;
    std::ifstream file(filename);

    if (!file.is_open()) {
      throw std::runtime_error("Cannot open test data file: " + filename);
    }

    std::string line;
    uint32_t line_num = 0;

    // 跳过第一行（标题行）
    if (std::getline(file, line)) {
      line_num++;
    }

    while (std::getline(file, line)) {
      line_num++;

      std::istringstream iss(line);
      TestCaseData data;
      std::string hap_bases, read_bases, read_qual_str, read_ins_qual_str;
      std::string read_del_qual_str, gcp_str, expected_str;

      if (iss >> hap_bases >> read_bases >> read_qual_str >>
          read_ins_qual_str >> read_del_qual_str >> gcp_str >> expected_str) {

        data.hap_bases = hap_bases;
        data.read_bases = read_bases;
        data.read_qual = parseQualityString(read_qual_str);
        data.read_ins_qual = parseQualityString(read_ins_qual_str);
        data.read_del_qual = parseQualityString(read_del_qual_str);
        data.gcp = parseQualityString(gcp_str);
        data.expected_result = std::stod(expected_str);
        data.line_number = line_num;

        test_cases.push_back(data);
      }
    }

    return test_cases;
  }

private:
  /**
   * @brief 解析质量字符串为uint8_t数组
   */
  static std::vector<uint8_t> parseQualityString(const std::string &qual_str) {
    std::vector<uint8_t> qual;
    for (char c : qual_str) {
      qual.push_back(static_cast<uint8_t>(c) - 33);
    }
    return qual;
  }
};

/**
 * @brief 简单测试用例 - 用于快速验证
 * 参考 Intel GKL PairHmmUnitTest.java 中的 simpleTest
 */
class PairHMMSimpleTest : public ::testing::Test {
  void SetUp() override {
    ConvertChar::init();
    pairhmm::common::init_native();
  }
protected:
  /**
   * @brief 创建简单的测试用例
   */
  TestCaseData createSimpleTestCase() {
    TestCaseData data;
    // 简单的序列数据
    data.hap_bases = "ACGT";
    data.read_bases = "ACGT";

    data.read_qual = {43, 43, 43, 43};
    data.read_ins_qual = {43, 43, 43, 43};
    data.read_del_qual = {43, 43, 43, 43};
    data.gcp = {43, 43, 43, 43};

    // 完全匹配应该有非常高的似然度（接近0的对数似然度）
    data.expected_result = -6.022797e-01; // 大约的期望值
    data.line_number = 0;

    return data;
  }
  TestCaseData createSimpleTest1Case() {
    TestCaseData data;
    // 简单的序列数据
    data.hap_bases = "ACGTA";
    data.read_bases = "ACGTA";

    data.read_qual = {43, 43, 43, 43, 43};
    data.read_ins_qual = {43, 43, 43, 43, 43};
    data.read_del_qual = {43, 43, 43, 43, 43};
    data.gcp = {43, 43, 43, 43, 43};

    // 完全匹配应该有非常高的似然度（接近0的对数似然度）
    data.expected_result = -0.69925308227539062; // 大约的期望值
    data.line_number = 0;

    return data;
  }
};
/**
 * @brief 简单测试 - AVX2版本
 */
TEST_F(PairHMMSimpleTest, SimpleMatchAVX2) {
  auto data = createSimpleTestCase();
  TestCaseWrapper<32> wrapper(data);
  double result = computeLikelihoodsAVX2(
      wrapper.getTestCase(), false); // 完美匹配应该产生接近0的对数似然度
  EXPECT_NEAR(result, data.expected_result, 1e-5);
  auto data1 = createSimpleTest1Case();
  TestCaseWrapper<32> wrapper1(data1);
  double result1 = computeLikelihoodsAVX2(wrapper1.getTestCase(), false);
  EXPECT_NEAR(result1, data1.expected_result, 1e-5);
}

/**
 * @brief 简单测试 - AVX512版本
 */
TEST_F(PairHMMSimpleTest, SimpleMatchAVX512) {
  if (!CpuFeatures::hasAVX512Support()) {
    GTEST_SKIP() << "AVX512 not supported on this system";
  }

  auto data = createSimpleTestCase();
  TestCaseWrapper<64> wrapper(data);
  double result = computeLikelihoodsAVX512(wrapper.getTestCase(), false);

  EXPECT_NEAR(result, data.expected_result, 1e-5);
}

TEST_F(PairHMMSimpleTest, SimpleInterMatchAVX512) {
  if (!CpuFeatures::hasAVX512Support()) {
    GTEST_SKIP() << "AVX512 not supported on this system";
  }

  // 创建16个测试用例（AVX512的simd_width）
  double *results = new double[16];
  std::vector<TestCaseData> test_data(16);
  for (int i = 0; i < 16; ++i) {
    test_data[i] = createSimpleTestCase();
  }

  // 创建16个TestCaseWrapper，生成TestCase数组
  std::vector<std::unique_ptr<TestCaseWrapper<64>>> wrappers;
  std::vector<TestCase> test_cases(16);

  for (int i = 0; i < 16; ++i) {
    wrappers.emplace_back(std::make_unique<TestCaseWrapper<64>>(test_data[i]));
    test_cases[i] = wrappers[i]->getTestCase();
  }

  // 调用compute_inter_pairhmm_AVX512_float接口（使用 DefaultAllocator）
  rovaca::roDV_adapter::DefaultAllocator allocator;
  bool success = pairhmm::inter::compute_inter_pairhmm_AVX512_float(
      test_cases.data(), 16, results, allocator);
  EXPECT_TRUE(success) << "compute_inter_pairhmm_AVX512_float failed";

  if (success) {

    for (int i = 0; i < 16; ++i) {
      SCOPED_TRACE("Test case " + std::to_string(i));

      EXPECT_NEAR(results[i], test_data[i].expected_result, 1e-5);
    }
  }
  // 调用compute_inter_pairhmm_AVX512_double接口
  success = pairhmm::inter::compute_inter_pairhmm_AVX512_double(
      test_cases.data(), 8, results, allocator);
  EXPECT_TRUE(success) << "compute_inter_pairhmm_AVX512_double failed";

  if (success) {
    for (int i = 0; i < 8; ++i) {
      EXPECT_NEAR(results[i], test_data[i].expected_result, 1e-5);
    }
  }
  delete[] results;
}
TEST_F(PairHMMSimpleTest, SimpleInterMatchAVX2) {

  if (!CpuFeatures::hasAVX2Support()) {
    GTEST_SKIP() << "AVX2 not supported on this system";
  }

  // 创建8个测试用例（AVX2的simd_width）
  double *results = new double[8];
  std::vector<TestCaseData> test_data(8);
  for (int i = 0; i < 8; ++i) {
    test_data[i] = createSimpleTest1Case();
  }

  // 创建8个TestCaseWrapper，生成TestCase数组
  std::vector<std::unique_ptr<TestCaseWrapper<32>>> wrappers;
  std::vector<TestCase> test_cases(8);

  for (int i = 0; i < 8; ++i) {
    wrappers.emplace_back(std::make_unique<TestCaseWrapper<32>>(test_data[i]));
    test_cases[i] = wrappers[i]->getTestCase();
  }

  // 调用compute_inter_pairhmm_AVX2_float接口
  rovaca::roDV_adapter::DefaultAllocator allocator;
  bool success = pairhmm::inter::compute_inter_pairhmm_AVX2_float(
      test_cases.data(), 8, results, allocator);
  EXPECT_TRUE(success) << "compute_inter_pairhmm_AVX2_float failed";

  if (success) {
    for (int i = 0; i < 8; ++i) {
      EXPECT_NEAR(results[i], test_data[i].expected_result, 1e-5);
    }
  }
  // 调用compute_inter_pairhmm_AVX2_double接口
  success = pairhmm::inter::compute_inter_pairhmm_AVX2_double(test_cases.data(),
                                                              4, results, allocator);
  EXPECT_TRUE(success) << "compute_inter_pairhmm_AVX2_double failed";

  if (success) {
    for (int i = 0; i < 4; ++i) {
      EXPECT_NEAR(results[i], test_data[i].expected_result, 1e-5);
    }
  }
  delete[] results;
}
TEST_F(PairHMMSimpleTest, SimpleInterDiffLenghMatchAVX2) {
  if (!CpuFeatures::hasAVX2Support()) {
    GTEST_SKIP() << "AVX2 not supported on this system";
  }

  // 创建8个测试用例（AVX2的simd_width）
  double *results = new double[8];
  std::vector<TestCaseData> test_data(8);
  for (int i = 0; i < 8; ++i) {
    if(i % 2 == 0) {
      test_data[i] = createSimpleTestCase();
    } else {
      test_data[i] = createSimpleTest1Case();
    }
  }

  // 创建8个TestCaseWrapper，生成TestCase数组
  std::vector<std::unique_ptr<TestCaseWrapper<32>>> wrappers;
  std::vector<TestCase> test_cases(8);

  for (int i = 0; i < 8; ++i) {
    wrappers.emplace_back(std::make_unique<TestCaseWrapper<32>>(test_data[i]));
    test_cases[i] = wrappers[i]->getTestCase();
  }

  // 调用compute_inter_pairhmm_AVX2_float接口
  rovaca::roDV_adapter::DefaultAllocator allocator;
  bool success = pairhmm::inter::compute_inter_pairhmm_AVX2_float(
      test_cases.data(), 8, results, allocator);
  EXPECT_TRUE(success) << "compute_inter_pairhmm_AVX2_float failed";

  if (success) {
    for (int i = 0; i < 8; ++i) {
      EXPECT_NEAR(results[i], test_data[i].expected_result, 1e-5) << "Test case " + std::to_string(i) + " failed";
    }
  }
  // 调用compute_inter_pairhmm_AVX2_double接口
  success = pairhmm::inter::compute_inter_pairhmm_AVX2_double(test_cases.data(),
                                                              4, results, allocator);
  EXPECT_TRUE(success) << "compute_inter_pairhmm_AVX2_double failed";

  if (success) {
    for (int i = 0; i < 4; ++i) {
      EXPECT_NEAR(results[i], test_data[i].expected_result, 1e-5) << "Test case " + std::to_string(i) + " failed";
    }
  }
  delete[] results;
}

TEST_F(PairHMMSimpleTest, SimpleInterDiffLenghMatchAVX512) {
  if (!CpuFeatures::hasAVX512Support()) {
    GTEST_SKIP() << "AVX512 not supported on this system";
  }

  // 创建16个测试用例（AVX512的simd_width）
  double *results = new double[16];
  std::vector<TestCaseData> test_data(16);
  for( int i = 0; i < 16; i++) {
    if(i % 2 == 0) {
      test_data[i] = createSimpleTestCase();
    } else {
      test_data[i] = createSimpleTest1Case();
    }
  }
  std::vector<std::unique_ptr<TestCaseWrapper<64>>> wrappers;
  std::vector<TestCase> test_cases(16);
  for( int i = 0; i < 16; i++) {
    wrappers.emplace_back(std::make_unique<TestCaseWrapper<64>>(test_data[i]));
    test_cases[i] = wrappers[i]->getTestCase();
  }
  rovaca::roDV_adapter::DefaultAllocator allocator;
  bool success = pairhmm::inter::compute_inter_pairhmm_AVX512_float(test_cases.data(), 16, results, allocator);
  EXPECT_TRUE(success) << "compute_inter_pairhmm_AVX512_float failed";
  if(success) {
    for( int i = 0; i < 16; i++) {
      ASSERT_NEAR(results[i], test_data[i].expected_result, 1e-5) << "Test case " + std::to_string(i) + " failed";
    }
  }
  success = pairhmm::inter::compute_inter_pairhmm_AVX512_double(test_cases.data(), 8, results, allocator);
  EXPECT_TRUE(success) << "compute_inter_pairhmm_AVX512_double failed";
  if(success) {
    for( int i = 0; i < 8; i++) {
      ASSERT_NEAR(results[i], test_data[i].expected_result, 1e-5) << "Test case " + std::to_string(i) + " failed";
    }
  }
  delete[] results;
  wrappers.clear();
}
/**
 * @brief 测试基类，包含通用测试逻辑
 */
class PairHMMTestBase : public ::testing::Test {
protected:
  void SetUp() override {
    // 尝试多个可能的路径
    std::vector<std::string> possible_paths = {
        "../../src/haplotypecaller/pairhmm/resource/pairhmm-testdata.txt",
        "../src/haplotypecaller/pairhmm/resource/pairhmm-testdata.txt",
        "src/haplotypecaller/pairhmm/resource/pairhmm-testdata.txt",
        "../../resource/pairhmm-testdata.txt"
    };

    bool data_loaded = false;
    for (const auto &path : possible_paths) {
      try {
        test_data_ = TestDataParser::loadTestData(path);
        if (!test_data_.empty()) {
          data_loaded = true;
          break;
        }
      } catch (const std::exception &) {
        continue;
      }
      ConvertChar::init();
      pairhmm::common::init_native();
    }

    ASSERT_TRUE(data_loaded) << "Cannot load test data from any expected path";
  }

  /**
   * @brief 检查结果精度是否在合理范围内
   * 新API统一返回double，但内部可能使用float或double计算
   */
  template <typename T>
  void checkResultAccuracy(T actual, double expected,
                           const std::string &test_name) {
    double actual_d = static_cast<double>(actual);
    double rel_error = std::abs(actual_d - expected) / std::abs(expected);

    // 新API内部会优先尝试float，如果精度不够才用double
    // 使用较宽松的精度要求：1e-5
    double tolerance = 1e-5;

    ASSERT_LT(rel_error, tolerance)
        << "Test: " << test_name << ", Expected: " << expected
        << ", Actual: " << actual_d << ", Relative error: " << rel_error;
  }

  std::vector<TestCaseData> test_data_;
};

/**
 * @brief AVX2 测试 - 统一接口，内部自动选择精度
 */
class PairHMMAVX2Test : public PairHMMTestBase {
  void SetUp() override {
    ConvertChar::init();
    pairhmm::common::init_native();
  }
};

TEST_F(PairHMMAVX2Test, AllTestCases) {
  for (const auto &data : test_data_) {
    TestCaseWrapper<32> wrapper(data);
    double result = computeLikelihoodsAVX2(wrapper.getTestCase(), false);

    SCOPED_TRACE("Line " + std::to_string(data.line_number) +
                 " Haplength: " + std::to_string(data.hap_bases.size()) +
                 " RSlength: " + std::to_string(data.read_bases.size()));
    checkResultAccuracy(result, data.expected_result, "AVX2 Float");
    result = computeLikelihoodsAVX2(wrapper.getTestCase(), true);
    checkResultAccuracy(result, data.expected_result, "AVX2 double");
  }
}

TEST_F(PairHMMAVX2Test,ALLInterMatchAVX2) {

  if (!CpuFeatures::hasAVX2Support()) {
    GTEST_SKIP() << "AVX2 not supported on this system";
  }
  rovaca::roDV_adapter::DefaultAllocator allocator;
  double *results = new double[8];
  std::vector<std::unique_ptr<TestCaseWrapper<32>>> wrappers;
  std::vector<TestCase> test_cases(8);
  for( size_t i = 0; i < test_data_.size(); i += 8) {
    for( int j = 0; j < 8; j++) {
      wrappers.emplace_back(std::make_unique<TestCaseWrapper<32>>(test_data_[i + j]));
      test_cases[j] = wrappers[j]->getTestCase();
    }
    bool success = pairhmm::inter::compute_inter_pairhmm_AVX2_float(test_cases.data(), 8, results, allocator);
    EXPECT_TRUE(success) << "compute_inter_pairhmm_AVX2_float failed";
    if(success) {
      for( int j = 0; j < 8; j++) {
        ASSERT_NEAR(results[j], test_data_[i + j].expected_result, 1e-5) << "Test case " + std::to_string(i + j) + " failed";
      }
    }
    for( int  j = 0 ; j < 2 ;j ++) {
      success = pairhmm::inter::compute_inter_pairhmm_AVX2_double(test_cases.data() + j * 4, 4, results + j * 4, allocator);
      EXPECT_TRUE(success) << "compute_inter_pairhmm_AVX2_double failed";
      if(success) {
        for( int k= 0; k < 4; k++) {
          ASSERT_NEAR(results[j * 4 + k], test_data_[i + j * 4 + k].expected_result, 1e-5) << "Test case " + std::to_string(i + j * 4 + k) + " failed";
        }
      }
    }
    wrappers.clear();
  }
  delete[] results;
}


/**
 * @brief AVX512 测试 - 统一接口，内部自动选择精度
 */
class PairHMMAVX512Test : public PairHMMTestBase {
protected:
  void SetUp() override {
    PairHMMTestBase::SetUp();
    if (!CpuFeatures::hasAVX512Support()) {
      GTEST_SKIP() << "AVX512 not supported on this system";
    }
    ConvertChar::init();
    pairhmm::common::init_native();
  }
};
TEST_F(PairHMMAVX512Test,ALLInterMatchAVX512) {
  if (!CpuFeatures::hasAVX512Support()) {
    GTEST_SKIP() << "AVX512 not supported on this system";
  }
  rovaca::roDV_adapter::DefaultAllocator allocator;
  double *results = new double[16];
  std::vector<std::unique_ptr<TestCaseWrapper<64>>> wrappers;
  std::vector<TestCase> test_cases(16);
  for( size_t i = 0; i < test_data_.size(); i += 16) {
    if(i + 16 > test_data_.size()) {
      break;
    }
    for( int j = 0; j < 16; j++) {
      wrappers.emplace_back(std::make_unique<TestCaseWrapper<64>>(test_data_[i + j]));
      test_cases[j] = wrappers[j]->getTestCase();
    }
    bool success = pairhmm::inter::compute_inter_pairhmm_AVX512_float(test_cases.data(), 16, results, allocator);
    EXPECT_TRUE(success) << "compute_inter_pairhmm_AVX512_float failed";
    if(success) {
      for( int j = 0; j < 16; j++) {
        ASSERT_NEAR(results[j], test_data_[i + j].expected_result, 1e-5) << "Test case " + std::to_string(i + j) + " failed";
      }
    }
    for( int  j = 0 ; j < 2 ;j ++) {
      success = pairhmm::inter::compute_inter_pairhmm_AVX512_double(test_cases.data() + j * 8, 8, results + j * 8, allocator);
      EXPECT_TRUE(success) << "compute_inter_pairhmm_AVX512_double failed";
      if(success) {
        for( int k= 0; k < 8; k++) {
          ASSERT_NEAR(results[j * 8 + k], test_data_[i + j * 8 + k].expected_result, 1e-5) << "Test case " + std::to_string(i + j * 8 + k) + " failed";
        }
      }
    }
    wrappers.clear();
  }
  delete[] results;
}

TEST_F(PairHMMAVX512Test, AllTestCases) {
  if (!CpuFeatures::hasAVX512Support()) {
    GTEST_SKIP() << "AVX512 not supported on this system";
  }

  for (const auto &data : test_data_) {
    TestCaseWrapper<64> wrapper(data);
    double result = computeLikelihoodsAVX512(wrapper.getTestCase(), false);

    SCOPED_TRACE("Line " + std::to_string(data.line_number));
    checkResultAccuracy(result, data.expected_result, "AVX512 Float");
    result = computeLikelihoodsAVX512(wrapper.getTestCase(), true);
    checkResultAccuracy(result, data.expected_result, "AVX512 double");
  }
}

// 精度一致性测试已删除 - 新API内部自动选择合适的精度

/**
 * @brief 指令集一致性测试
 */
class PairHMMInstructionSetConsistencyTest : public PairHMMTestBase {};

TEST_F(PairHMMInstructionSetConsistencyTest, AVX2VsAVX512) {
  if (!CpuFeatures::hasAVX512Support()) {
    GTEST_SKIP() << "AVX512 not supported on this system";
  }

  for (size_t i = 0; i < std::min(test_data_.size(), size_t(10)); ++i) {
    const auto &data = test_data_[i];
    TestCaseWrapper<32> wrapper(data);

    double result_avx2 = computeLikelihoodsAVX2(wrapper.getTestCase());
    double result_avx512 = computeLikelihoodsAVX512(wrapper.getTestCase());

    double rel_error =
        std::abs(result_avx2 - result_avx512) / std::abs(result_avx512);

    SCOPED_TRACE("Line " + std::to_string(data.line_number) +
                 " - AVX2 vs AVX512");
    ASSERT_LT(rel_error, 1e-5)
        << "AVX2: " << result_avx2 << ", AVX512: " << result_avx512
        << ", Relative error: " << rel_error;
  }
}

/**
 * @brief 测试程序入口点
 */
#ifndef PAIRHMM_UNITTEST_MAIN_DISABLED
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // 打印系统信息
  if (CpuFeatures::hasAVX512Support()) {
    std::cout << "AVX512 support detected - running all tests\n";
  } else {
    std::cout << "AVX512 not supported - skipping AVX512 tests\n";
  }

  return RUN_ALL_TESTS();
}
#endif // PAIRHMM_UNITTEST_MAIN_DISABLED

