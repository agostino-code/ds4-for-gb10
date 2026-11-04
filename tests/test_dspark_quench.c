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

    printf("test_dspark_quench: PASS\n");
    return 0;
}
