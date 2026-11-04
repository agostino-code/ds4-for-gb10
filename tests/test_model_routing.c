#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include "ds4.h"

int main(void) {
    // GB10 compute capability is 1210 (SM 12.1)
    assert(ds4_cuda_is_spark_fastpath_enabled(MODEL_TYPE_DEEPSEEK4, 1210) == true);
    // Other models on GB10 must NOT engage DeepSeek fast-path
    assert(ds4_cuda_is_spark_fastpath_enabled(MODEL_TYPE_QWEN38, 1210) == false);
    assert(ds4_cuda_is_spark_fastpath_enabled(MODEL_TYPE_GLM53, 1210) == false);
    // Other architectures (e.g. Ada sm_89, Hopper sm_90) must not engage Spark fast-path
    assert(ds4_cuda_is_spark_fastpath_enabled(MODEL_TYPE_DEEPSEEK4, 890) == false);
    assert(ds4_cuda_is_spark_fastpath_enabled(MODEL_TYPE_DEEPSEEK4, 900) == false);

    // Device-resident routing on GB10
    assert(ds4_cuda_is_device_resident_routing_enabled(1210, false) == true);
    assert(ds4_cuda_is_device_resident_routing_enabled(1210, true) == false);
    assert(ds4_cuda_is_device_resident_routing_enabled(900, false) == false);

    printf("test_model_routing: PASS\n");
    return 0;
}
