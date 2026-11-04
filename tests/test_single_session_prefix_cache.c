#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include "ds4.h"

int main(void) {
    int prev_turn[] = {1, 1024, 2048, 4096, 8192};
    int next_turn[] = {1, 1024, 2048, 4096, 8192, 9999, 1234};
    size_t shared = ds4_find_matching_prefix_length(prev_turn, 5, next_turn, 7);
    assert(shared == 5);

    int altered[] = {1, 1024, 9999, 4096};
    assert(ds4_find_matching_prefix_length(prev_turn, 5, altered, 4) == 2);

    assert(ds4_find_matching_prefix_length(NULL, 0, next_turn, 7) == 0);
    assert(ds4_find_matching_prefix_length(prev_turn, 5, NULL, 0) == 0);

    printf("test_single_session_prefix_cache: PASS\n");
    return 0;
}
