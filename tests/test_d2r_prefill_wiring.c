#include <assert.h>
#include <stdio.h>
#include <stdbool.h>
#include "cuda/mmq/ds4_mmq_d2r.cuh"

int main(void) {
    // GB10 compute capability is 1210 (SM 12.1)
    int cc = 1210;
    assert(ds4_mmq_q2_K_moe_d2r_available(cc) == true);
    assert(ds4_mmq_iq2_xxs_moe_d2r_available(cc) == true);

    // Verify scratchpad capacities
    size_t q2k_bytes = ds4_mmq_q2_K_moe_d2r_scratch_bytes(2048, 256);
    size_t iq2_bytes = ds4_mmq_iq2_xxs_moe_d2r_fused_scratch_bytes(2048, 256);
    assert(q2k_bytes > 0);
    assert(iq2_bytes > 0);

    printf("test_d2r_prefill_wiring: PASS (q2k_scratch=%zu, iq2_scratch=%zu)\n",
           q2k_bytes, iq2_bytes);
    return 0;
}
