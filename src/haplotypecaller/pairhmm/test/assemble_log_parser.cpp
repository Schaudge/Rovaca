#include "assemble_log_parser.h"

#include <algorithm>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <regex>
#include <sstream>
#include <sys/stat.h>

namespace pairhmm {
namespace test {

std::vector<ParsedRegion> AssembleLogParser::parseLogFile(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open log file: " + filename);
  }

  std::vector<ParsedRegion> regions;
  std::string line;
  bool line_available = false;

  while (line_available || std::getline(file, line)) {
    line_available = false;

    if (line.find("=== Region:") != std::string::npos) {
      ParsedRegion region;

      std::regex region_regex(R"(=== Region: (.+) ==)");
      std::smatch match;
      if (std::regex_search(line, match, region_regex)) {
        region.region_str = match[1].str();
      }

      int haplotype_count = 0;
      int read_count = 0;

      if (std::getline(file, line)) {
        std::regex hap_regex(R"(Haplotypes: (\d+))");
        if (std::regex_search(line, match, hap_regex)) {
          haplotype_count = std::stoi(match[1].str());
        }
      }

      if (std::getline(file, line)) {
        std::regex read_regex(R"(Reads: (\d+))");
        if (std::regex_search(line, match, read_regex)) {
          read_count = std::stoi(match[1].str());
        }
      }

      std::getline(file, line);

      for (int i = 0; i < haplotype_count; ++i) {
        if (std::getline(file, line)) {
          size_t colon_pos = line.find(':');
          if (colon_pos != std::string::npos) {
            std::string hap_seq = line.substr(colon_pos + 1);
            hap_seq.erase(0, hap_seq.find_first_not_of(" \t"));
            region.haplotypes.push_back(hap_seq);
          }
        }
      }

      std::getline(file, line);

      for (int i = 0; i < read_count; ++i) {
        ParsedRead read;

        if (!std::getline(file, line)) {
          break;
        }

        size_t colon_pos = line.find(':');
        if (colon_pos != std::string::npos) {
          read.sequence = line.substr(colon_pos + 1);
          read.sequence.erase(0, read.sequence.find_first_not_of(" \t"));
        }

        int quality_lines_read = 0;
        while (quality_lines_read < 4 && std::getline(file, line)) {
          if (line.find("=== Region:") != std::string::npos) {
            break;
          }

          if (line.empty()) {
            continue;
          }

          if (line.find("read-qual:") != std::string::npos) {
            read.read_qual = parseQualityLine(line);
            quality_lines_read++;
          } else if (line.find("read-ins-qual:") != std::string::npos) {
            read.read_ins_qual = parseQualityLine(line);
            quality_lines_read++;
          } else if (line.find("read-del-qual:") != std::string::npos) {
            read.read_del_qual = parseQualityLine(line);
            quality_lines_read++;
          } else if (line.find("gcp:") != std::string::npos) {
            read.gcp = parseQualityLine(line);
            quality_lines_read++;
            break;
          }
        }

        region.reads.push_back(std::move(read));

        if (line.find("=== Region:") != std::string::npos) {
          line_available = true;
          break;
        }
      }

      regions.push_back(std::move(region));

      if (line_available) {
        continue;
      }
    }
  }

  return regions;
}

std::vector<std::string> AssembleLogParser::findLogFiles(const std::string &directory,
                                                         const std::string &pattern) {
  std::vector<std::string> files;

  DIR *dir = opendir(directory.c_str());
  if (!dir) {
    throw std::runtime_error("Error opening directory: " + directory);
  }

  struct dirent *entry;
  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_name[0] == '.') {
      continue;
    }

    std::string filename(entry->d_name);
    if (filename.find(pattern) != std::string::npos && filename.find(".log") != std::string::npos) {
      std::string full_path = directory;
      if (!directory.empty() && directory.back() != '/') {
        full_path += "/";
      }
      full_path += filename;

      struct stat st;
      if (stat(full_path.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
        files.push_back(full_path);
      }
    }
  }

  closedir(dir);
  std::sort(files.begin(), files.end());
  return files;
}

std::vector<uint8_t> AssembleLogParser::parseQualityLine(const std::string &line) {
  std::vector<uint8_t> qualities;

  size_t colon_pos = line.find(':');
  if (colon_pos == std::string::npos) {
    return qualities;
  }

  std::string qual_str = line.substr(colon_pos + 1);
  std::istringstream iss(qual_str);
  int qual_value;
  while (iss >> qual_value) {
    if (qual_value >= 0 && qual_value <= 255) {
      qualities.push_back(static_cast<uint8_t>(qual_value));
    }
  }

  return qualities;
}

} // namespace test
} // namespace pairhmm
