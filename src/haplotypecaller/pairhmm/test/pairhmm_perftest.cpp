#include "assemble_log_parser.h"
#include "test_case_common.h"

#include "../roDV_common/cpu_features.h"
#include "../roDV_intra/pairhmm_api.h"
#include "../roDV_schedule.h"
#include "../pairhmm_internal.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../../genotype/forward.h"
#include "../../genotype/haplotype.h"
#include "../../genotype/read_record.h"
#include "../../common/utils/rovaca_memory_pool.h"
#include "../roDV_adapter/rovaca_pool_allocator.h"
#include "htslib/sam.h"

using namespace pairhmm::test;
using namespace pairhmm::schedule;
using namespace pairhmm::intra;
using namespace pairhmm::common;
using pairhmm::common::CpuFeatures;

namespace {
std::unique_ptr<bam_hdr_t, decltype(&bam_hdr_destroy)> makeFakeHeader() {
  bam_hdr_t *hdr = bam_hdr_init();
  hdr->n_targets = 1;
  hdr->target_len = static_cast<uint32_t *>(calloc(1, sizeof(uint32_t)));
  hdr->target_name = static_cast<char **>(calloc(1, sizeof(char *)));
  hdr->target_len[0] = 1'000'000;
  hdr->target_name[0] = strdup("chr1");
  return {hdr, bam_hdr_destroy};
}

bam1_t *makeBamRecord(const ParsedRead &read, const std::string &qname) {
  bam1_t *b = bam_init1();
  const size_t l_qseq = read.sequence.size();
  const size_t l_qname = qname.size() + 1;

  std::string seq = read.sequence;
  for (auto &ch : seq) ch = std::toupper(static_cast<unsigned char>(ch));
  std::string qual(l_qseq, static_cast<char>(30));
  if (!read.read_qual.empty()) {
    for (size_t i = 0; i < l_qseq; ++i) {
      uint8_t q = i < read.read_qual.size() ? read.read_qual[i] : 30;
      qual[i] = static_cast<char>(q);
    }
  }

  std::vector<uint32_t> cigar(1);
  cigar[0] = bam_cigar_gen(static_cast<uint32_t>(l_qseq), BAM_CMATCH);

  if (bam_set1(b, l_qname, qname.c_str(), 0, 0, 0, 60,
               cigar.size(), cigar.data(), -1, 0, 0,
               static_cast<int32_t>(l_qseq), seq.c_str(), qual.c_str(), 0) < 0) {
    bam_destroy1(b);
    throw std::runtime_error("bam_set1 failed");
  }
  return b;
}

rovaca::HaplotypeVector buildHaplotypes(const std::vector<std::string> &seqs,
                                      rovaca::RovacaMemoryPool &pool) {
  rovaca::HaplotypeVector haplotypes{&pool};
  for (const auto &seq : seqs) {
    rovaca::pHaplotype hap = rovaca::Haplotype::create(&pool);
    hap->init_haplotype(seq.c_str(), static_cast<uint32_t>(seq.size()), 0, &pool);
    haplotypes.push_back(hap);
  }
  return haplotypes;
}

struct ReadHolder {
  std::unique_ptr<bam1_t, decltype(&bam_destroy1)> bam;
  rovaca::pReadRecord record;
};

std::pair<rovaca::ReadVector, std::vector<ReadHolder>>
buildReads(const ParsedRegion &region, rovaca::RovacaMemoryPool &pool,
           bam_hdr_t *hdr) {
  rovaca::ReadVector reads{&pool};
  std::vector<ReadHolder> holders;
  holders.reserve(region.reads.size());

  for (size_t i = 0; i < region.reads.size(); ++i) {
    const auto &r = region.reads[i];
    std::string qname = "READ" + std::to_string(i);
    bam1_t *bam = makeBamRecord(r, qname);
    ReadHolder holder{
        std::unique_ptr<bam1_t, decltype(&bam_destroy1)>(bam, bam_destroy1),
        nullptr};
    holder.record = rovaca::ReadRecord::create(&pool, hdr, bam);
    reads.push_back(holder.record);
    holders.push_back(std::move(holder));
  }
  return {std::move(reads), std::move(holders)};
}
} // namespace

class HighPrecisionTimer {
public:
  void start() { start_time_ = std::chrono::high_resolution_clock::now(); }
  void stop() { end_time_ = std::chrono::high_resolution_clock::now(); }
  double elapsedSeconds() const {
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_ - start_time_);
    return duration.count() / 1e9;
  }
  double elapsedMilliseconds() const {
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_ - start_time_);
    return duration.count() / 1e6;
  }

private:
  std::chrono::high_resolution_clock::time_point start_time_;
  std::chrono::high_resolution_clock::time_point end_time_;
};

double computeWithIntra(const ParsedRegion &region,
                        std::vector<std::vector<double>> &results) {
  HighPrecisionTimer timer;
  timer.start();

  const size_t M = region.haplotypes.size();
  const size_t N = region.reads.size();

  results.resize(M);

  for (size_t h = 0; h < M; ++h) {
    results[h].resize(N);
    for (size_t r = 0; r < N; ++r) {
      TestCaseData data;
      data.hap_bases = region.haplotypes[h];
      data.read_bases = region.reads[r].sequence;
      data.read_qual = region.reads[r].read_qual;
      data.read_ins_qual = region.reads[r].read_ins_qual;
      data.read_del_qual = region.reads[r].read_del_qual;
      data.gcp = region.reads[r].gcp;

      size_t read_len = data.read_bases.length();
      if (data.read_qual.size() != read_len || data.read_ins_qual.size() != read_len ||
          data.read_del_qual.size() != read_len || data.gcp.size() != read_len) {
        results[h][r] = 0.0;
        continue;
      }

      TestCaseWrapper<64> wrapper(data);
      const TestCase &tc = wrapper.getTestCase();

      if (CpuFeatures::hasAVX512Support()) {
        results[h][r] = computeLikelihoodsAVX512(tc, false);
      } else if (CpuFeatures::hasAVX2Support()) {
        results[h][r] = computeLikelihoodsAVX2(tc, false);
      } else {
        std::cerr << "Error: CPU does not support AVX2 or AVX512" << std::endl;
        results[h][r] = 0.0;
      }
    }
  }

  timer.stop();
  return timer.elapsedSeconds();
}

double computeWithSchedule(const ParsedRegion &region,
                           std::vector<std::vector<double>> &results,
                           double max_idle_ratio_float,
                           double max_idle_ratio_double,
                           bool verbose,
                           double &data_prep_time) {
  HighPrecisionTimer total_timer;
  HighPrecisionTimer prep_timer;

  total_timer.start();
  prep_timer.start();

  const size_t M = region.haplotypes.size();
  const size_t N = region.reads.size();

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

  prep_timer.stop();
  data_prep_time = prep_timer.elapsedSeconds();

  std::vector<std::vector<double>> schedule_results;
  bool success = schedule_pairhmm(hap_vecs, read_vecs, schedule_results, qual_vecs, ins_vecs,
                                   del_vecs, gcp_vecs, false, max_idle_ratio_float,
                                   max_idle_ratio_double, verbose);

  total_timer.stop();

  if (!success) {
    std::cerr << "Error: schedule_pairhmm failed" << std::endl;
    return -1.0;
  }

  // schedule_results 为 [read][hap]，转换为 [hap][read]
  results.assign(M, std::vector<double>(N, 0.0));
  for (size_t r = 0; r < schedule_results.size(); ++r) {
    for (size_t h = 0; h < schedule_results[r].size(); ++h) {
      results[h][r] = schedule_results[r][h];
    }
  }

  return total_timer.elapsedSeconds();
}

double computeWithScheduleAllocator(const ParsedRegion &region,
                                     std::vector<std::vector<double>> &results,
                                     bool use_double,
                                     double max_idle_ratio_float,
                                     double max_idle_ratio_double,
                                     bool verbose) {
  std::vector<uint8_t> pool_buffer(16 * 1024 * 1024);
  rovaca::RovacaMemoryPool pool(pool_buffer.data(), pool_buffer.size());
  rovaca::roDV_adapter::RovacaMemoryPoolAllocator allocator(&pool);

  auto hdr = makeFakeHeader();
  auto hap_vec = buildHaplotypes(region.haplotypes, pool);
  auto holders_pair = buildReads(region, pool, hdr.get());
  auto &read_vec = holders_pair.first;
  auto holders = std::move(holders_pair.second);
  (void)holders;

  rovaca::DoubleVector2D result_matrix{&pool};
  bool success = schedule_pairhmm_with_allocator(
      hap_vec, read_vec, &result_matrix, allocator,
      0, PcrIndelModel::NONE, use_double,
      max_idle_ratio_float, max_idle_ratio_double, verbose);

  if (!success) {
    return -1.0;
  }

  const size_t N = result_matrix.size();
  size_t M = 0;
  for (size_t r = 0; r < N; ++r) {
    M = std::max(M, result_matrix[r].size());
  }
  results.assign(M, std::vector<double>(N, 0.0));
  for (size_t r = 0; r < N; ++r) {
    for (size_t h = 0; h < result_matrix[r].size(); ++h) {
      results[h][r] = result_matrix[r][h];
    }
  }
  return 0.0;
}

bool containsN(const std::string &sequence) {
  for (char c : sequence) {
    if (c == 'N' || c == 'n') {
      return true;
    }
  }
  return false;
}

bool compareResults(const std::vector<std::vector<double>> &results1,
                    const std::vector<std::vector<double>> &results2,
                    const ParsedRegion &region, double tolerance,
                    bool ignore_n_base) {
  if (results1.size() != results2.size()) {
    std::cerr << "Error: Result size mismatch (M)" << std::endl;
    return false;
  }

  for (size_t h = 0; h < results1.size(); ++h) {
    if (results1[h].size() != results2[h].size()) {
      std::cerr << "Error: Result size mismatch (N) for hap " << h << std::endl;
      return false;
    }
    if (ignore_n_base && containsN(region.haplotypes[h])) {
      continue;
    }

    for (size_t r = 0; r < results1[h].size(); ++r) {
      if (ignore_n_base && r < region.reads.size() &&
          containsN(region.reads[r].sequence)) {
        continue;
      }

      if (std::isnan(results1[h][r]) || std::isnan(results2[h][r]) ||
          std::isinf(results1[h][r]) || std::isinf(results2[h][r])) {
        continue;
      }

      if (std::abs(results1[h][r]) < 1e-10) {
        continue;
      }

      double diff = std::abs(results1[h][r] - results2[h][r]);
      if (diff > tolerance) {
        std::cerr << "Error: Result mismatch for H" << h << "_R" << r
                  << "\n  Intra result: " << results1[h][r]
                  << "\n  Schedule result: " << results2[h][r]
                  << "\n  Difference: " << diff
                  << "\n  Tolerance: " << tolerance << std::endl;
        return false;
      }
    }
  }

  return true;
}

void printUsage(const char *program_name) {
  std::cout << "Usage: " << program_name
            << " <log_file> <max_idle_ratio_float> <max_idle_ratio_double> [OPTIONS]"
            << std::endl;
  std::cout << "Arguments:" << std::endl;
  std::cout << "  log_file: Path to the log file containing regions"
            << std::endl;
  std::cout << "  max_idle_ratio_float: Maximum idle ratio for float precision"
            << std::endl;
  std::cout << "  max_idle_ratio_double: Maximum idle ratio for double precision"
            << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  -v, --verbose: Enable verbose output for schedule_pairhmm"
            << std::endl;
  std::cout << "  -i, --ignore-N-base: Skip comparison for reads containing N bases"
            << std::endl;
  std::cout << "  -h, --help: Show this help message" << std::endl;
}

int main(int argc, char *argv[]) {
  static struct option long_options[] = {
      {"verbose", no_argument, 0, 'v'},
      {"ignore-N-base", no_argument, 0, 'i'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};

  bool verbose = false;
  bool ignore_n_base = false;
  std::string log_file;
  double max_idle_ratio_float = 0.0;
  double max_idle_ratio_double = 0.0;

  int opt;
  int option_index = 0;

  while ((opt = getopt_long(argc, argv, "vih", long_options, &option_index)) != -1) {
    switch (opt) {
    case 'v':
      verbose = true;
      break;
    case 'i':
      ignore_n_base = true;
      break;
    case 'h':
      printUsage(argv[0]);
      return 0;
    case '?':
      printUsage(argv[0]);
      return 1;
    default:
      printUsage(argv[0]);
      return 1;
    }
  }

  if (optind + 3 > argc) {
    std::cerr << "Error: Missing required arguments" << std::endl;
    printUsage(argv[0]);
    return 1;
  }

  log_file = argv[optind];
  try {
    max_idle_ratio_float = std::stod(argv[optind + 1]);
    max_idle_ratio_double = std::stod(argv[optind + 2]);
  } catch (const std::exception &e) {
    std::cerr << "Error: Invalid number format: " << e.what() << std::endl;
    printUsage(argv[0]);
    return 1;
  }

  if (optind + 3 < argc) {
    std::cerr << "Error: Unexpected argument: " << argv[optind + 3] << std::endl;
    printUsage(argv[0]);
    return 1;
  }

  init_native();
  ConvertChar::init();
  std::cout << "==========================================" << std::endl;
  std::cout << "PairHMM Performance Test" << std::endl;
  std::cout << "==========================================" << std::endl;
  std::cout << "Log file: " << log_file << std::endl;
  std::cout << "Max idle ratio (float): " << max_idle_ratio_float << std::endl;
  std::cout << "Max idle ratio (double): " << max_idle_ratio_double << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  Verbose: " << (verbose ? "Yes" : "No") << std::endl;
  std::cout << "  Ignore N-base: " << (ignore_n_base ? "Yes" : "No") << std::endl;
  std::cout << "CPU Features:" << std::endl;
  std::cout << "  AVX2: " << (CpuFeatures::hasAVX2Support() ? "Yes" : "No")
            << std::endl;
  std::cout << "  AVX512: " << (CpuFeatures::hasAVX512Support() ? "Yes" : "No")
            << std::endl;
  std::cout << "==========================================" << std::endl;

  if (!CpuFeatures::hasAVX2Support() && !CpuFeatures::hasAVX512Support()) {
    std::cerr << "Error: CPU does not support AVX2 or AVX512" << std::endl;
    return 1;
  }

  std::vector<ParsedRegion> regions;
  try {
    regions = AssembleLogParser::parseLogFile(log_file);
  } catch (const std::exception &e) {
    std::cerr << "Error: Cannot parse log file: " << e.what() << std::endl;
    return 1;
  }

  if (regions.empty()) {
    std::cerr << "Error: No regions found in log file" << std::endl;
    return 1;
  }

  std::cout << "Parsed " << regions.size() << " region(s)" << std::endl;
  std::cout << "==========================================" << std::endl;

  double total_intra_time = 0.0;
  double total_schedule_time = 0.0;
  double total_schedule_data_prep_time = 0.0;
  size_t total_pairs = 0;
  bool all_passed = true;

  for (size_t region_idx = 0; region_idx < regions.size(); ++region_idx) {
    const auto &region = regions[region_idx];
    std::cout << "\nRegion " << (region_idx + 1) << "/" << regions.size()
              << ": " << region.region_str << std::endl;
    std::cout << "  Haplotypes: " << region.haplotypes.size() << std::endl;
    double avg_hap_length = 0.0;
    for (const auto &hap : region.haplotypes) {
      avg_hap_length += hap.length();
    }
    avg_hap_length /= region.haplotypes.size();
    std::cout << "  Haplotype  AVG length: " << avg_hap_length << std::endl;
    std::cout << "  Reads: " << region.reads.size() << std::endl;
    double avg_read_length = 0.0;
    for (const auto &read : region.reads) {
      avg_read_length += read.sequence.length();
    }
    avg_read_length /= region.reads.size();
    std::cout << "  Read AVG length: " << avg_read_length << std::endl;

    const size_t M = region.haplotypes.size();
    const size_t N = region.reads.size();
    total_pairs += M * N;

    std::vector<std::vector<double>> intra_results;
    double intra_time = computeWithIntra(region, intra_results);
    total_intra_time += intra_time;

    std::cout << "  Intra time: " << std::fixed << std::setprecision(6)
              << intra_time << " seconds" << std::endl;

    std::vector<std::vector<double>> schedule_results;
    double schedule_data_prep_time = 0.0;
    double schedule_time = computeWithSchedule(region, schedule_results,
                                               max_idle_ratio_float,
                                               max_idle_ratio_double, verbose,
                                               schedule_data_prep_time);

    if (schedule_time < 0.0) {
      std::cerr << "  Error: schedule_pairhmm failed" << std::endl;
      all_passed = false;
      continue;
    }

    total_schedule_time += schedule_time;
    total_schedule_data_prep_time += schedule_data_prep_time;

    std::cout << "  Schedule time: " << std::fixed << std::setprecision(6)
              << schedule_time << " seconds" << std::endl;
    std::cout << "    - Data prep time: " << std::fixed << std::setprecision(6)
              << schedule_data_prep_time << " seconds" << std::endl;
    std::cout << "    - Compute time: " << std::fixed << std::setprecision(6)
              << (schedule_time - schedule_data_prep_time) << " seconds"
              << std::endl;

    const double tolerance = 1e-5;
    bool passed = compareResults(intra_results, schedule_results, region,
                                 tolerance, ignore_n_base);

    if (passed) {
      std::cout << "  Result check: PASSED (tolerance: " << tolerance << ")"
                << std::endl;
    } else {
      std::cout << "  Result check: FAILED (tolerance: " << tolerance << ")"
                << std::endl;
      all_passed = false;
      break;
    }

    std::vector<std::vector<double>> allocator_results;
    double alloc_status = computeWithScheduleAllocator(
        region, allocator_results, false,
        max_idle_ratio_float, max_idle_ratio_double, verbose);
    if (alloc_status < 0.0) {
      std::cout << "  Allocator API check: FAILED (call failed)" << std::endl;
      all_passed = false;
      break;
    }
    bool alloc_pass = compareResults(intra_results, allocator_results, region,
                                     tolerance, ignore_n_base);
    if (alloc_pass) {
      std::cout << "  Allocator API check: PASSED" << std::endl;
    } else {
      std::cout << "  Allocator API check: FAILED" << std::endl;
      all_passed = false;
      break;
    }

    if (schedule_time > 0.0) {
      double speedup = intra_time / schedule_time;
      std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup
                << "x" << std::endl;
    }
  }

  std::cout << "\n==========================================" << std::endl;
  std::cout << "Summary" << std::endl;
  std::cout << "==========================================" << std::endl;
  std::cout << "Total regions: " << regions.size() << std::endl;
  std::cout << "Total pairs (M*N): " << total_pairs << std::endl;
  std::cout << "Total intra time: " << std::fixed << std::setprecision(6)
            << total_intra_time << " seconds" << std::endl;
  std::cout << "Total schedule time: " << std::fixed << std::setprecision(6)
            << total_schedule_time << " seconds" << std::endl;
  std::cout << "  - Total data prep time: " << std::fixed << std::setprecision(6)
            << total_schedule_data_prep_time << " seconds" << std::endl;
  std::cout << "  - Total compute time: " << std::fixed << std::setprecision(6)
            << (total_schedule_time - total_schedule_data_prep_time) << " seconds"
            << std::endl;

  if (total_schedule_time > 0.0) {
    double overall_speedup = total_intra_time / total_schedule_time;
    std::cout << "Overall speedup: " << std::fixed << std::setprecision(2)
              << overall_speedup << "x" << std::endl;

    double data_prep_ratio =
        (total_schedule_data_prep_time / total_schedule_time) * 100.0;
    std::cout << "Data prep time ratio: " << std::fixed << std::setprecision(2)
              << data_prep_ratio << "%" << std::endl;
  }

  std::cout << "Result check: " << (all_passed ? "PASSED" : "FAILED")
            << std::endl;
  std::cout << "==========================================" << std::endl;

  return all_passed ? 0 : 1;
}
