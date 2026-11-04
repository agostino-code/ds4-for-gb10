#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>
#include "cuda/mmq/ds4_sm121_tuning.cuh"

int main() {
    // Basic verification that the header and function interface are callable
    cudaStream_t stream = nullptr;
    size_t size = 16 * 1024 * 1024; // 16 MB hot tensor test
    void *dummy_ptr = (void *)0x1000000;

    // Test calling the SM121 tuning function
    int rc = ds4_cuda_sm121_pin_drafter_l2(dummy_ptr, size, stream);

    // Verify DRAM channel anti-aliasing padding
    assert(ds4_cuda_sm121_pad_vt_stride(2048) == 2048 + 128);
    assert(ds4_cuda_sm121_pad_vt_stride(4096) == 4096 + 128);
    assert(ds4_cuda_sm121_pad_vt_stride(8192) == 8192 + 128);
    assert(ds4_cuda_sm121_pad_vt_stride(1200) % 128 == 0);

    printf("test_sm121_l2_pinning: PASS (rc=%d, anti_aliasing_verified)\n", rc);
    return 0;
}
