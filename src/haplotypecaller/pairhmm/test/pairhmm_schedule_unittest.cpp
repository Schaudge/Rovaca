#include "../roDV_schedule.h"
#include "../roDV_common/cpu_features.h"
#include "../roDV_intra/pairhmm_api.h"
#include "test_case_common.h"
#include <gtest/gtest.h>
#include <random>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <unordered_map>

using namespace pairhmm::schedule;
using namespace pairhmm::intra;
using namespace pairhmm::common;
using pairhmm::common::CpuFeatures;
using pairhmm::common::ConvertChar;

/**
 * @brief 突变类型
 */
enum class MutationType {
  MISMATCH = 0,  // 错配
  INSERTION = 1, // 插入
  DELETION = 2   // 缺失
};

/**
 * @brief 单个突变
 */
struct Mutation {
  size_t position;      // 位置（0-based）
  MutationType type;    // 类型
  char base;            // 对于mismatch和insertion，新的碱基
};

/**
 * @brief 测试用例生成器
 * 按照用户指定策略生成测试数据
 */
class TestCaseGenerator {
public:
  TestCaseGenerator(uint32_t seed = 42) : rng_(seed) {
    bases_ = {'A', 'C', 'G', 'T'};
  }

  /**
   * @brief 生成固定长度的随机碱基序列（无N）
   */
  std::string generateBaseSequence(size_t length) {
    std::uniform_int_distribution<size_t> base_dist(0, bases_.size() - 1);
    std::string seq;
    seq.reserve(length);
    for (size_t i = 0; i < length; ++i) {
      seq += bases_[base_dist(rng_)];
    }
    return seq;
  }

  /**
   * @brief 生成多个随机突变位置
   * @param sequence_len 序列长度
   * @param num_mutations 突变数量
   * @param mutations 输出的突变列表
   */
  void generateRandomMutations(size_t sequence_len, size_t num_mutations,
                                std::vector<Mutation> &mutations) {
    mutations.clear();
    mutations.reserve(num_mutations);

    std::uniform_int_distribution<size_t> pos_dist(0, sequence_len - 1);
    std::uniform_int_distribution<int> type_dist(0, 2);
    std::uniform_int_distribution<size_t> base_dist(0, bases_.size() - 1);

    std::unordered_set<size_t> used_positions;

    for (size_t i = 0; i < num_mutations; ++i) {
      size_t pos;
      // 确保位置不重复
      size_t attempts = 0;
      do {
        pos = pos_dist(rng_);
        ++attempts;
        if (attempts > 1000) {
          // 如果尝试太多次，扩大范围
          pos_dist = std::uniform_int_distribution<size_t>(0, sequence_len - 1);
          used_positions.clear();
          break;
        }
      } while (used_positions.find(pos) != used_positions.end());

      used_positions.insert(pos);
      
      Mutation mut;
      mut.position = pos;
      mut.type = static_cast<MutationType>(type_dist(rng_));
      
      // 生成新的碱基（对于mismatch和insertion）
      if (mut.type != MutationType::DELETION) {
        mut.base = bases_[base_dist(rng_)];
      }
      
      mutations.push_back(mut);
    }

    // 按位置排序
    std::sort(mutations.begin(), mutations.end(),
              [](const Mutation &a, const Mutation &b) {
                return a.position < b.position;
              });
  }

  /**
   * @brief 应用突变到序列
   * @param base_seq 基础序列
   * @param mutations 突变列表
   * @param use_mask 使用哪些突变（bitset，每个bit对应一个突变）
   * @return 突变后的序列
   */
  std::string applyMutations(const std::string &base_seq,
                            const std::vector<Mutation> &mutations,
                            uint32_t use_mask) {
    std::string result = base_seq;
    
    // 从后往前应用，避免位置偏移
    for (int i = static_cast<int>(mutations.size()) - 1; i >= 0; --i) {
      if (!(use_mask & (1u << i))) continue; // 跳过未使用的突变
      
      const Mutation &mut = mutations[i];
      size_t pos = mut.position;

      switch (mut.type) {
      case MutationType::MISMATCH:
        // 替换碱基
        if (pos < result.length()) {
          result[pos] = mut.base;
        }
        break;
      case MutationType::INSERTION:
        // 在位置后插入
        if (pos < result.length()) {
          result.insert(result.begin() + pos + 1, mut.base);
        }
        break;
      case MutationType::DELETION:
        // 删除位置上的碱基
        if (pos < result.length()) {
          result.erase(result.begin() + pos);
        }
        break;
      }
    }
    
    return result;
  }

  /**
   * @brief 生成所有可能的突变组合（2^num_mutations 条序列）
   */
  std::vector<std::string> generateAllMutationCombinations(
      const std::string &base_seq, const std::vector<Mutation> &mutations) {
    size_t num_combinations = 1u << mutations.size();
    std::vector<std::string> sequences;
    sequences.reserve(num_combinations);

    for (uint32_t mask = 0; mask < num_combinations; ++mask) {
      sequences.push_back(applyMutations(base_seq, mutations, mask));
    }

    return sequences;
  }

  /**
   * @brief 从序列中随机选择子序列
   */
  std::string randomSubsequence(const std::string &seq, size_t target_len) {
    if (seq.length() <= target_len) {
      return seq;
    }

    std::uniform_int_distribution<size_t> start_dist(0, seq.length() - target_len);
    size_t start = start_dist(rng_);
    return seq.substr(start, target_len);
  }

  /**
   * @brief 在序列上添加1-2个突变
   */
  std::string addRandomMutationsToRead(const std::string &seq, 
                                        size_t max_mutations = 2) {
    if (seq.empty()) return seq;
    
    std::uniform_int_distribution<size_t> num_mut_dist(1, max_mutations);
    size_t actual_muts = num_mut_dist(rng_);
    
    std::vector<Mutation> muts;
    generateRandomMutations(seq.length(), actual_muts, muts);
    
    uint32_t full_mask = (1u << actual_muts) - 1; // 使用所有突变
    return applyMutations(seq, muts, full_mask);
  }

  /**
   * @brief 生成质量值（6-43之间）
   */
  std::vector<uint8_t> generateQuality(size_t length) {
    std::uniform_int_distribution<uint8_t> qual_dist(6, 43);
    std::vector<uint8_t> qual;
    qual.reserve(length);
    for (size_t i = 0; i < length; ++i) {
      qual.push_back(qual_dist(rng_));
    }
    return qual;
  }

  std::mt19937 &getRNG() { return rng_; }

private:
  std::mt19937 rng_;
  std::vector<char> bases_;
};

/**
 * @brief 生成的测试数据
 */
struct GeneratedTestData {
  std::vector<std::string> haplotypes;
  std::vector<std::string> reads;
  std::vector<std::vector<uint8_t>> quality;
  std::vector<std::vector<uint8_t>> insertion_qualities;
  std::vector<std::vector<uint8_t>> deletion_qualities;
  std::vector<std::vector<uint8_t>> gap_contiguous_qualities;
  std::vector<std::vector<double>> expected_results;
};

/**
 * @brief 生成测试用例的函数
 * 
 * @param base_seq_len 基础序列长度（默认500）
 * @param num_mutations 突变数量（默认10）
 * @param M 单倍型数量
 * @param N reads数量
 * @param seed 随机数种子
 * @return GeneratedTestData 生成的测试数据
 */
GeneratedTestData generateTestCases(
    size_t base_seq_len = 500,
    size_t num_mutations = 10,
    size_t M = 16,
    size_t N = 16,
    uint32_t seed = 42) {
  
  GeneratedTestData test_data;
  TestCaseGenerator generator(seed);
  
  // 1. 生成500长度的基础序列
  std::string base_sequence = generator.generateBaseSequence(base_seq_len);
  
  // 2. 生成10个随机突变，然后生成2^10=1024条变体序列
  std::vector<Mutation> mutations;
  generator.generateRandomMutations(base_seq_len, num_mutations, mutations);
  
  std::vector<std::string> variant_sequences = 
      generator.generateAllMutationCombinations(base_sequence, mutations);
  
  // 3. 随机生成M个单倍型（长度30-base_seq_len）
  std::mt19937 &rng = generator.getRNG();
  std::uniform_int_distribution<size_t> hap_len_dist(30, base_seq_len);
  std::uniform_int_distribution<size_t> seq_sel(0, variant_sequences.size() - 1);
  
  test_data.haplotypes.reserve(M);
  for (size_t i = 0; i < M; ++i) {
    size_t target_len = hap_len_dist(rng);
    size_t seq_idx = seq_sel(rng);
    test_data.haplotypes.push_back(
        generator.randomSubsequence(variant_sequences[seq_idx], target_len));
  }
  
  // 4. 随机生成N个reads（长度30-150），加上1-2个突变，生成质量值
  std::uniform_int_distribution<size_t> read_len_dist(145, 150);
  
  test_data.reads.reserve(N);
  test_data.quality.reserve(N);
  test_data.insertion_qualities.reserve(N);
  test_data.deletion_qualities.reserve(N);
  test_data.gap_contiguous_qualities.reserve(N);
  
  for (size_t i = 0; i < N; ++i) {
    size_t target_len = read_len_dist(rng);
    size_t seq_idx = seq_sel(rng);
    
    // 选择子序列并添加1-2个突变
    std::string read_seq = generator.randomSubsequence(
        variant_sequences[seq_idx], target_len);
    read_seq = generator.addRandomMutationsToRead(read_seq, 2);
    
    test_data.reads.push_back(read_seq);
    
    // 生成质量值
    size_t actual_len = read_seq.length();
    test_data.quality.push_back(generator.generateQuality(actual_len));
    test_data.insertion_qualities.push_back(generator.generateQuality(actual_len));
    test_data.deletion_qualities.push_back(generator.generateQuality(actual_len));
    test_data.gap_contiguous_qualities.push_back(generator.generateQuality(actual_len));
  }
  
  // 5. 调用M*N次intra API生成expected results
  test_data.expected_results.resize(M);
  for (size_t h = 0; h < M; ++h) {
    test_data.expected_results[h].resize(N);
    for (size_t r = 0; r < N; ++r) {
      TestCaseData data;
      data.hap_bases = test_data.haplotypes[h];
      data.read_bases = test_data.reads[r];
      data.read_qual = test_data.quality[r];
      data.read_ins_qual = test_data.insertion_qualities[r];
      data.read_del_qual = test_data.deletion_qualities[r];
      data.gcp = test_data.gap_contiguous_qualities[r];

      TestCaseWrapper<64> wrapper(data);
      const TestCase &tc = wrapper.getTestCase();

      // 使用运行时CPU特性检测来选择API
      if (CpuFeatures::hasAVX512Support()) {
        test_data.expected_results[h][r] = computeLikelihoodsAVX512(tc, false);
      } else if (CpuFeatures::hasAVX2Support()) {
        test_data.expected_results[h][r] = computeLikelihoodsAVX2(tc, false);
      } else {
        // 如果不支持AVX2，则无法计算（需要至少AVX2支持）
        test_data.expected_results[h][r] = 0.0;
      }
    }
  }
  
  return test_data;
}

/**
 * @brief Schedule 测试类
 */
class SchedulePairHMMTest : public ::testing::Test {
protected:
  void SetUp() override {
    ConvertChar::init();
    pairhmm::common::init_native();
  }
};

/**
 * @brief 测试1：按照用户思路生成的测试用例
 */
TEST_F(SchedulePairHMMTest, GeneratedTestCase) {
  // 生成测试数据
  GeneratedTestData test_data = generateTestCases(500, 10, 16, 16, 42);
  
  const size_t M = test_data.haplotypes.size();
  const size_t N = test_data.reads.size();
  
  // 转换格式并调用schedule_pairhmm
  std::vector<std::vector<uint8_t>> hap_vecs, read_vecs;
  std::vector<std::vector<uint8_t>> qual_vecs, ins_vecs, del_vecs, gcp_vecs;
  
  for (const auto &hap : test_data.haplotypes) {
    hap_vecs.emplace_back(hap.begin(), hap.end());
  }
  for (const auto &read : test_data.reads) {
    read_vecs.emplace_back(read.begin(), read.end());
  }
  for (const auto &qual : test_data.quality) {
    qual_vecs.push_back(qual);
  }
  for (const auto &ins : test_data.insertion_qualities) {
    ins_vecs.push_back(ins);
  }
  for (const auto &del : test_data.deletion_qualities) {
    del_vecs.push_back(del);
  }
  for (const auto &g : test_data.gap_contiguous_qualities) {
    gcp_vecs.push_back(g);
  }
  
  std::vector<std::vector<double>> schedule_results;
  bool success = schedule_pairhmm(hap_vecs, read_vecs, schedule_results,
                                   qual_vecs, ins_vecs, del_vecs, gcp_vecs, false);
  
  ASSERT_TRUE(success) << "schedule_pairhmm failed";
  ASSERT_EQ(schedule_results.size(), N) << "Result size mismatch (N reads)";
  
  for (size_t r = 0; r < N; ++r) {
    ASSERT_EQ(schedule_results[r].size(), M) << "Result size mismatch (M haplotypes) for read " << r;
    for (size_t h = 0; h < M; ++h) {
      ASSERT_FALSE(std::isnan(schedule_results[r][h]))
          << "Result[" << r << "][" << h << "] is NaN";
      ASSERT_FALSE(std::isinf(schedule_results[r][h]))
          << "Result[" << r << "][" << h << "] is Inf";

      ASSERT_NEAR(schedule_results[r][h], test_data.expected_results[h][r], 1e-5)
          << "Read" << r << "_Hap" << h;
    }
  }
}

/**
 * @brief 测试2：不同规模的数据集
 */
TEST_F(SchedulePairHMMTest, DifferentSizes) {
  // 测试小规模（M=4, N=4）
  GeneratedTestData test_data_small = generateTestCases(200, 5, 10, 20, 100);
  
  std::vector<std::vector<uint8_t>> hap_vecs, read_vecs;
  std::vector<std::vector<uint8_t>> qual_vecs, ins_vecs, del_vecs, gcp_vecs;
  
  for (const auto &hap : test_data_small.haplotypes) {
    hap_vecs.emplace_back(hap.begin(), hap.end());
  }
  for (const auto &read : test_data_small.reads) {
    read_vecs.emplace_back(read.begin(), read.end());
  }
  for (const auto &qual : test_data_small.quality) {
    qual_vecs.push_back(qual);
  }
  for (const auto &ins : test_data_small.insertion_qualities) {
    ins_vecs.push_back(ins);
  }
  for (const auto &del : test_data_small.deletion_qualities) {
    del_vecs.push_back(del);
  }
  for (const auto &g : test_data_small.gap_contiguous_qualities) {
    gcp_vecs.push_back(g);
  }
  
  std::vector<std::vector<double>> schedule_results;
  bool success = schedule_pairhmm(hap_vecs, read_vecs, schedule_results,
                                qual_vecs, ins_vecs, del_vecs, gcp_vecs, false);
  
  ASSERT_TRUE(success);
  ASSERT_EQ(schedule_results.size(), test_data_small.reads.size());
  
  for (size_t r = 0; r < schedule_results.size(); ++r) {
    ASSERT_EQ(schedule_results[r].size(), test_data_small.haplotypes.size());
    for (size_t h = 0; h < schedule_results[r].size(); ++h) {
      ASSERT_NEAR(schedule_results[r][h], test_data_small.expected_results[h][r], 1e-5)
          << "Small_Read" << r << "_Hap" << h;
    }
  }
}

/**
 * @brief 测试3：验证所有对都被处理
 */
TEST_F(SchedulePairHMMTest, AllPairsProcessed) {
  GeneratedTestData test_data = generateTestCases(300, 8, 8, 8, 200);
  
  std::vector<std::vector<uint8_t>> hap_vecs, read_vecs;
  std::vector<std::vector<uint8_t>> qual_vecs, ins_vecs, del_vecs, gcp_vecs;
  
  for (const auto &hap : test_data.haplotypes) {
    hap_vecs.emplace_back(hap.begin(), hap.end());
  }
  for (const auto &read : test_data.reads) {
    read_vecs.emplace_back(read.begin(), read.end());
  }
  for (const auto &qual : test_data.quality) {
    qual_vecs.push_back(qual);
  }
  for (const auto &ins : test_data.insertion_qualities) {
    ins_vecs.push_back(ins);
  }
  for (const auto &del : test_data.deletion_qualities) {
    del_vecs.push_back(del);
  }
  for (const auto &g : test_data.gap_contiguous_qualities) {
    gcp_vecs.push_back(g);
  }
  
  std::vector<std::vector<double>> schedule_results;
  schedule_pairhmm(hap_vecs, read_vecs, schedule_results,
                  qual_vecs, ins_vecs, del_vecs, gcp_vecs, false);
  
  // 验证所有结果都被计算（不为0或NaN）
  for (size_t r = 0; r < schedule_results.size(); ++r) {
    for (size_t h = 0; h < schedule_results[r].size(); ++h) {
      EXPECT_FALSE(std::isnan(schedule_results[r][h]))
          << "Result[" << r << "][" << h << "] is NaN";
      EXPECT_FALSE(std::isinf(schedule_results[r][h]))
          << "Result[" << r << "][" << h << "] is Inf";
      ASSERT_NEAR(schedule_results[r][h], test_data.expected_results[h][r], 1e-5)
          << "Result[" << r << "][" << h << "] is not equal to expected result";
    }
  }
}

