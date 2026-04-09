#ifndef ASSEMBLE_LOG_PARSER_H_
#define ASSEMBLE_LOG_PARSER_H_

#include <cstdint>
#include <string>
#include <vector>

namespace pairhmm {
namespace test {

struct ParsedRead {
  std::string sequence;
  std::vector<uint8_t> read_qual;
  std::vector<uint8_t> read_ins_qual;
  std::vector<uint8_t> read_del_qual;
  std::vector<uint8_t> gcp;
};

struct ParsedRegion {
  std::string region_str;
  std::vector<std::string> haplotypes;
  std::vector<ParsedRead> reads;
};

class AssembleLogParser {
public:
  static std::vector<ParsedRegion> parseLogFile(const std::string &filename);
  static std::vector<std::string> findLogFiles(
      const std::string &directory = ".",
      const std::string &pattern = "pairhmm_debug_t");
  static std::vector<uint8_t> parseQualityLine(const std::string &line);
};

} // namespace test
} // namespace pairhmm

#endif // ASSEMBLE_LOG_PARSER_H_
