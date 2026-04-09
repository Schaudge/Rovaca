#include "assemble_log_parser.h"
#include "test_case_common.h"

#include "../roDV_schedule.h"
#include "../roDV_common/cpu_features.h"
#include "../roDV_intra/pairhmm_api.h"
#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

using namespace pairhmm::test;
using namespace pairhmm::schedule;
using namespace pairhmm::intra;
using namespace pairhmm::common;
using pairhmm::common::CpuFeatures;

TEST(AssembleLogParserTest, ParseAndTestSchedule) {
  using namespace pairhmm::test;
  ConvertChar::init();
  pairhmm::common::init_native();
  std::vector<std::string> possible_paths = {
      "../../src/haplotypecaller/pairhmm/resource/pairhmm_debug.txt",
      "../src/haplotypecaller/pairhmm/resource/pairhmm_debug.txt",
      "src/haplotypecaller/pairhmm/resource/pairhmm_debug.txt",
      "../../resource/pairhmm_debug.txt"
  };

  std::string log_file;
  std::ifstream test_file;
  for (const auto &path : possible_paths) {
    test_file.open(path);
    if (test_file.is_open()) {
      log_file = path;
      test_file.close();
      break;
    }
  }

  if (log_file.empty()) {
    // 打印当前目录
    std::cout << "Current directory: " << getcwd(nullptr, 0) << std::endl;
    GTEST_SKIP() << "Cannot find log file in any of the expected locations";
    return;
  }

  std::vector<ParsedRegion> regions;

  try {
    regions = AssembleLogParser::parseLogFile(log_file);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Cannot parse log file: " << e.what();
    return;
  }

  ASSERT_GT(regions.size(), 0) << "No regions found in log file";

  for (size_t region_idx = 0; region_idx < regions.size(); ++region_idx) {
    const auto &region = regions[region_idx];

    SCOPED_TRACE("Region: " + region.region_str + " (index: " +
                 std::to_string(region_idx) + ")");

    const size_t M = region.haplotypes.size();
    const size_t N = region.reads.size();

    ASSERT_GT(M, 0) << "No haplotypes in region";
    ASSERT_GT(N, 0) << "No reads in region";

    std::vector<std::vector<double>> expected_results(M);

    for (size_t h = 0; h < M; ++h) {
      expected_results[h].resize(N);
      for (size_t r = 0; r < N; ++r) {
        TestCaseData data;
        data.hap_bases = region.haplotypes[h];
        data.read_bases = region.reads[r].sequence;
        data.read_qual = region.reads[r].read_qual;
        data.read_ins_qual = region.reads[r].read_ins_qual;
        data.read_del_qual = region.reads[r].read_del_qual;
        data.gcp = region.reads[r].gcp;

        size_t read_len = data.read_bases.length();
        if (data.read_qual.size() != read_len ||
            data.read_ins_qual.size() != read_len ||
            data.read_del_qual.size() != read_len ||
            data.gcp.size() != read_len) {
          expected_results[h][r] = 0.0;
          continue;
        }

        TestCaseWrapper<64> wrapper(data);
        const TestCase &tc = wrapper.getTestCase();

        if (CpuFeatures::hasAVX512Support()) {
          expected_results[h][r] = computeLikelihoodsAVX512(tc, false);
        } else if (CpuFeatures::hasAVX2Support()) {
          expected_results[h][r] = computeLikelihoodsAVX2(tc, false);
        } else {
          GTEST_SKIP() << "CPU does not support AVX2 or AVX512";
          return;
        }

        ASSERT_FALSE(std::isnan(expected_results[h][r]))
            << "Expected result is NaN for H" << h << "_R" << r;
        ASSERT_FALSE(std::isinf(expected_results[h][r]))
            << "Expected result is Inf for H" << h << "_R" << r;
      }
    }

    std::vector<std::vector<uint8_t>> hap_vecs, read_vecs;
    std::vector<std::vector<uint8_t>> qual_vecs, ins_vecs, del_vecs, gcp_vecs;

    hap_vecs.reserve(M);
    for (const auto &hap : region.haplotypes) {
      hap_vecs.emplace_back(hap.begin(), hap.end());
    }

    read_vecs.reserve(N);
    qual_vecs.reserve(N);
    ins_vecs.reserve(N);
    del_vecs.reserve(N);
    gcp_vecs.reserve(N);

    for (const auto &read : region.reads) {
      read_vecs.emplace_back(read.sequence.begin(), read.sequence.end());
      qual_vecs.push_back(read.read_qual);
      ins_vecs.push_back(read.read_ins_qual);
      del_vecs.push_back(read.read_del_qual);
      gcp_vecs.push_back(read.gcp);
    }

    std::vector<std::vector<double>> schedule_results;
    bool success = schedule_pairhmm(
        hap_vecs, read_vecs, schedule_results, qual_vecs, ins_vecs, del_vecs,
        gcp_vecs, false, 0.5, 0.7, false);

    ASSERT_TRUE(success) << "schedule_pairhmm failed for region " << region_idx;
    ASSERT_EQ(schedule_results.size(), N)
        << "Result size mismatch (reads) for region " << region_idx;

    for (size_t r = 0; r < N; ++r) {
      ASSERT_EQ(schedule_results[r].size(), M)
          << "Result size mismatch (haplotypes) for read " << r
          << " in region " << region_idx;

      for (size_t h = 0; h < M; ++h) {
        ASSERT_FALSE(std::isnan(schedule_results[r][h]))
            << "Schedule result is NaN for R" << r << "_H" << h
            << " in region " << region_idx;
        ASSERT_FALSE(std::isinf(schedule_results[r][h]))
            << "Schedule result is Inf for R" << r << "_H" << h
            << " in region " << region_idx;

        if (expected_results[h][r] == 0.0) {
          continue;
        }

        ASSERT_NEAR(schedule_results[r][h], expected_results[h][r], 1e-5)
            << "Mismatch for R" << r << "_H" << h << " in region "
            << region_idx << " (" << region.region_str << ")"
            << "\nExpected: " << expected_results[h][r]
            << "\nGot: " << schedule_results[r][h]
            << "\nDifference: "
            << std::abs(schedule_results[r][h] - expected_results[h][r]);
      }
    }
  }
}
