#include "ds4.h"

size_t ds4_find_matching_prefix_length(const int *old_tokens, size_t old_len,
                                       const int *new_tokens, size_t new_len) {
    if (!old_tokens || !new_tokens || old_len == 0 || new_len == 0) {
        return 0;
    }
    size_t max_match = (old_len < new_len) ? old_len : new_len;
    size_t match = 0;
    while (match < max_match && old_tokens[match] == new_tokens[match]) {
        match++;
    }
    return match;
}
