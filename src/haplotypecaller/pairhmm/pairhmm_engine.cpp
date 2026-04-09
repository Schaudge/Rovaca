
#include "rovaca_logger.h"
#include "pairhmm_engine.h"
#include "pairhmm_internal.h"
#include "roDV_common/common.h"
#include "roDV_common/context.h"
#include "roDV_common/cpu_features.h"

namespace rovaca
{

void init_pairhmm_ptr(bool use_old);

DoubleVector2D (*call_pairhmm)(const HaplotypeVector& hs, ReadVector& rs, int32_t min_quality_threshold, PcrIndelModel pcr_option,
                               pMemoryPool pool);

bool avx512_supported() { return pairhmm::common::CpuFeatures::hasAVX512Support(); }

void init_pairhmm_ptr(bool use_old)
{
    bool avx512_supported = pairhmm::common::CpuFeatures::hasAVX512Support();
    bool avx2_supported = pairhmm::common::CpuFeatures::hasAVX2Support();

    if (avx512_supported) {
        RovacaLogger::info("pairhmm using avx512 instruction");
    }
    else if (avx2_supported) {
        RovacaLogger::info("pairhmm using avx2 instruction");
    }
    else {
        RovacaLogger::error("the machine must support avx2 or avx512 instruction");
        exit(EXIT_FAILURE);
    }
    pairhmm::common::ConvertChar::init();
    pairhmm::common::init_native();
    // 使用 roDV 优化版本（使用调度器优化）
    call_pairhmm = call_roDV_pairhmm_scheduled;
}

}  // namespace rovaca