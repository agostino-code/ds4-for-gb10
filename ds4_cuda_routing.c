#include "ds4.h"

bool ds4_cuda_is_spark_fastpath_enabled(int model_arch, int compute_cap) {
    // Only GB10 (SM 12.1 / compute capability 12.1 = 1210)
    if (compute_cap != 1210) return false;
    // Fast-path currently specialized for DeepSeek-V4-Flash
    if (model_arch != DS4_MODEL_FAMILY_DEEPSEEK4) return false;
    return true;
}
