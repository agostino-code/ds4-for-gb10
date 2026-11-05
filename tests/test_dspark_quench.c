#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include "ds4.h"

int main(void) {
    struct dspark_controller c;
    // Guard 2.16 for 0731 weights, k=4 draft tokens
    dspark_init(&c, 2.16f, 4);
    assert(dspark_step_should_speculate(&c) == true);
    assert(c.quenched == false);

    // Scenario A: High acceptance (4 accepted tokens -> tok/step > guard -> regret stays 0)
    dspark_update(&c, 4);
    assert(dspark_step_should_speculate(&c) == true);
    assert(c.quenched == false);

    // Scenario B: Repeated low acceptance (0 accepted tokens -> regret accumulates)
    for (int i = 0; i < 5; i++) {
        dspark_update(&c, 0);
    }
    // Controller must terminally quench after cumulative regret threshold exceeded (~8.0f)
    assert(dspark_step_should_speculate(&c) == false);
    assert(c.quenched == true);

    // Further updates should keep it quenched
    dspark_update(&c, 4);
    assert(dspark_step_should_speculate(&c) == false);

    // Scenario C: Re-initialization for next request
    dspark_init(&c, 2.16f, 4);
    assert(dspark_step_should_speculate(&c) == true);
    assert(c.quenched == false);

    // Scenario D: Proportional updates on truncated blocks (attempt 2, accept 2 -> regret 0)
    dspark_update_proportional(&c, 2, 2);
    assert(dspark_step_should_speculate(&c) == true);
    assert(c.regret_accum == 0.0f);

    // Attempt 2, accept 0 -> regret accumulates proportionally (~1.08 per step, not 2.16)
    dspark_update_proportional(&c, 0, 2);
    assert(c.regret_accum > 1.0f && c.regret_accum < 1.2f);
    assert(dspark_step_should_speculate(&c) == true);

    printf("test_dspark_quench: PASS\n");
    return 0;
}
