#include "ds4.h"

void dspark_init(struct dspark_controller *c, float guard, int k_draft) {
    if (!c) return;
    c->guard = (guard > 0.0f) ? guard : 2.16f;
    c->regret_accum = 0.0f;
    c->regret_threshold = 8.0f; // ~4 plain-step penalty
    c->k_draft = (k_draft > 0) ? k_draft : 4;
    c->quenched = false;
}

bool dspark_step_should_speculate(struct dspark_controller *c) {
    if (!c) return false;
    return !c->quenched;
}

void dspark_update(struct dspark_controller *c, int accepted_tokens) {
    if (!c || c->quenched) return;
    float delta = c->guard - (float)accepted_tokens;
    c->regret_accum += delta;
    if (c->regret_accum < 0.0f) {
        c->regret_accum = 0.0f;
    }
    if (c->regret_accum >= c->regret_threshold) {
        c->quenched = true;
    }
}
