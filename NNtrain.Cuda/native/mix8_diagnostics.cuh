#pragma once

#include <cstddef>

// Device-resident, per-replica aggregate.  The diagnostic kernels reduce
// within a CTA before touching this structure, so the persistent cost is
// constant (32 bytes) regardless of model size.
struct alignas(8) nntrain_mix8_diagnostic_accumulator {
    double update_step_ratio_squared_sum;
    double residual_step_ratio_squared_sum;
    unsigned long long changed_code_count;
    unsigned long long element_count;
};

static_assert(sizeof(nntrain_mix8_diagnostic_accumulator) == 32,
    "mix8 diagnostic ABI must remain 32 bytes");
static_assert(offsetof(
    nntrain_mix8_diagnostic_accumulator,
    residual_step_ratio_squared_sum) == 8,
    "mix8 diagnostic residual offset changed");
static_assert(offsetof(
    nntrain_mix8_diagnostic_accumulator,
    changed_code_count) == 16,
    "mix8 diagnostic changed-code offset changed");
static_assert(offsetof(
    nntrain_mix8_diagnostic_accumulator,
    element_count) == 24,
    "mix8 diagnostic element-count offset changed");
