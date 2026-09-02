#include "cuda_internal.cuh"

#include <cuda_bf16.h>
#include <cmath>

namespace {

constexpr int kWarpSize = 32;
constexpr int kNekoMuonThreads = 256;
constexpr int kNekoMuonMaxBlocks = 128;

template <bool store_corrected_moments, bool check_finite, bool nesterov = false>
__global__ void nekomuon_moments_stats_block(
    const float* __restrict__ gradient,
    float* __restrict__ fast,
    float* __restrict__ slow,
    float* __restrict__ fast_hat,
    float* __restrict__ slow_hat,
    float* __restrict__ stats,
    int length,
    float beta_fast,
    float beta_slow,
    float fast_correction,
    float slow_correction,
    int* __restrict__ finite_status) {
    __shared__ float warp_stats[4][kNekoMuonThreads / kWarpSize];
    float dot = 0.f;
    float fast_norm = 0.f;
    float slow_norm = 0.f;
    float residual_norm = 0.f;
    bool values_are_finite = true;
    const int stride = blockDim.x * gridDim.x;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < length;
         index += stride) {
        const float gradient_value = gradient[index];
        const float previous_fast = fast[index];
        const float previous_slow = slow[index];
        const float next_fast = fmaf(
            beta_fast, previous_fast, (1.f - beta_fast) * gradient_value);
        const float next_slow = nesterov
            ? fmaf(beta_fast, next_fast, (1.f - beta_fast) * gradient_value)
            : fmaf(beta_slow, previous_slow, (1.f - beta_slow) * gradient_value);
        const float corrected_fast = nesterov ? next_slow : next_fast / fast_correction;
        const float corrected_slow = nesterov ? next_slow : next_slow / slow_correction;
        const float residual = corrected_fast - corrected_slow;
        fast[index] = next_fast;
        slow[index] = next_slow;
        if constexpr (store_corrected_moments) {
            fast_hat[index] = corrected_fast;
            slow_hat[index] = corrected_slow;
        }
        dot = fmaf(corrected_fast, corrected_slow, dot);
        fast_norm = fmaf(corrected_fast, corrected_fast, fast_norm);
        slow_norm = fmaf(corrected_slow, corrected_slow, slow_norm);
        residual_norm = fmaf(residual, residual, residual_norm);
        if constexpr (check_finite) {
            values_are_finite = values_are_finite
                && isfinite(gradient_value)
                && isfinite(previous_fast)
                && isfinite(previous_slow)
                && isfinite(next_fast)
                && isfinite(next_slow)
                && isfinite(corrected_fast)
                && isfinite(corrected_slow)
                && isfinite(residual)
                && isfinite(dot)
                && isfinite(fast_norm)
                && isfinite(slow_norm)
                && isfinite(residual_norm);
        }
    }
    if constexpr (check_finite) {
        if (!values_are_finite)
            atomicExch(finite_status, 1);
    }

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xffffffffu, dot, offset);
        fast_norm += __shfl_down_sync(0xffffffffu, fast_norm, offset);
        slow_norm += __shfl_down_sync(0xffffffffu, slow_norm, offset);
        residual_norm += __shfl_down_sync(
            0xffffffffu, residual_norm, offset);
    }
    if (lane == 0) {
        if constexpr (check_finite) {
            if (!isfinite(dot) || !isfinite(fast_norm)
                || !isfinite(slow_norm) || !isfinite(residual_norm)) {
                atomicExch(finite_status, 1);
            }
        }
        warp_stats[0][warp] = dot;
        warp_stats[1][warp] = fast_norm;
        warp_stats[2][warp] = slow_norm;
        warp_stats[3][warp] = residual_norm;
    }
    __syncthreads();

    if (warp == 0) {
        const int warp_count = blockDim.x / kWarpSize;
        dot = lane < warp_count ? warp_stats[0][lane] : 0.f;
        fast_norm = lane < warp_count ? warp_stats[1][lane] : 0.f;
        slow_norm = lane < warp_count ? warp_stats[2][lane] : 0.f;
        residual_norm = lane < warp_count ? warp_stats[3][lane] : 0.f;
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
            fast_norm += __shfl_down_sync(
                0xffffffffu, fast_norm, offset);
            slow_norm += __shfl_down_sync(
                0xffffffffu, slow_norm, offset);
            residual_norm += __shfl_down_sync(
                0xffffffffu, residual_norm, offset);
        }
        if (lane == 0) {
            if constexpr (check_finite) {
                if (!isfinite(dot) || !isfinite(fast_norm)
                    || !isfinite(slow_norm) || !isfinite(residual_norm)) {
                    atomicExch(finite_status, 1);
                }
            }
            atomicAdd(stats + 0, dot);
            atomicAdd(stats + 1, fast_norm);
            atomicAdd(stats + 2, slow_norm);
            atomicAdd(stats + 3, residual_norm);
        }
    }
}

// Pure BF16 persists both moments in BF16 and consumes a BF16 gradient.
// Statistics are stable FP32 reductions, but corrected values are computed
// from the just-published BF16 moments so no hidden FP32 state influences the
// next optimizer step.
template <bool check_finite, bool nesterov = false>
__global__ void nekomuon_moments_stats_bf16_block(
    const __nv_bfloat16* __restrict__ gradient,
    __nv_bfloat16* __restrict__ fast,
    __nv_bfloat16* __restrict__ slow,
    float* __restrict__ stats,
    int length,
    float beta_fast,
    float beta_slow,
    float fast_correction,
    float slow_correction,
    int* __restrict__ finite_status) {
    __shared__ float warp_stats[4][kNekoMuonThreads / kWarpSize];
    float dot = 0.f;
    float fast_norm = 0.f;
    float slow_norm = 0.f;
    float residual_norm = 0.f;
    bool values_are_finite = true;
    const int stride = blockDim.x * gridDim.x;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < length;
         index += stride) {
        const float gradient_value = __bfloat162float(gradient[index]);
        const float previous_fast = __bfloat162float(fast[index]);
        const float previous_slow = __bfloat162float(slow[index]);
        const __nv_bfloat16 next_fast_bf16 = __float2bfloat16_rn(fmaf(
            beta_fast, previous_fast, (1.f - beta_fast) * gradient_value));
        const __nv_bfloat16 next_slow_bf16 = __float2bfloat16_rn(nesterov
            ? fmaf(beta_fast, __bfloat162float(next_fast_bf16),
                (1.f - beta_fast) * gradient_value)
            : fmaf(beta_slow, previous_slow,
                (1.f - beta_slow) * gradient_value));
        fast[index] = next_fast_bf16;
        slow[index] = next_slow_bf16;
        const float next_fast = __bfloat162float(next_fast_bf16);
        const float next_slow = __bfloat162float(next_slow_bf16);
        const float corrected_fast = nesterov ? next_slow : next_fast / fast_correction;
        const float corrected_slow = nesterov ? next_slow : next_slow / slow_correction;
        const float residual = corrected_fast - corrected_slow;
        dot = fmaf(corrected_fast, corrected_slow, dot);
        fast_norm = fmaf(corrected_fast, corrected_fast, fast_norm);
        slow_norm = fmaf(corrected_slow, corrected_slow, slow_norm);
        residual_norm = fmaf(residual, residual, residual_norm);
        if constexpr (check_finite) {
            values_are_finite = values_are_finite
                && isfinite(gradient_value)
                && isfinite(next_fast)
                && isfinite(next_slow)
                && isfinite(dot)
                && isfinite(fast_norm)
                && isfinite(slow_norm)
                && isfinite(residual_norm);
        }
    }
    if constexpr (check_finite) {
        if (!values_are_finite)
            atomicExch(finite_status, 1);
    }

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xffffffffu, dot, offset);
        fast_norm += __shfl_down_sync(0xffffffffu, fast_norm, offset);
        slow_norm += __shfl_down_sync(0xffffffffu, slow_norm, offset);
        residual_norm += __shfl_down_sync(
            0xffffffffu, residual_norm, offset);
    }
    if (lane == 0) {
        warp_stats[0][warp] = dot;
        warp_stats[1][warp] = fast_norm;
        warp_stats[2][warp] = slow_norm;
        warp_stats[3][warp] = residual_norm;
    }
    __syncthreads();
    if (warp == 0) {
        const int warp_count = blockDim.x / kWarpSize;
        dot = lane < warp_count ? warp_stats[0][lane] : 0.f;
        fast_norm = lane < warp_count ? warp_stats[1][lane] : 0.f;
        slow_norm = lane < warp_count ? warp_stats[2][lane] : 0.f;
        residual_norm = lane < warp_count ? warp_stats[3][lane] : 0.f;
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
            fast_norm += __shfl_down_sync(0xffffffffu, fast_norm, offset);
            slow_norm += __shfl_down_sync(0xffffffffu, slow_norm, offset);
            residual_norm += __shfl_down_sync(
                0xffffffffu, residual_norm, offset);
        }
        if (lane == 0) {
            if constexpr (check_finite) {
                if (!isfinite(dot) || !isfinite(fast_norm)
                    || !isfinite(slow_norm) || !isfinite(residual_norm)) {
                    atomicExch(finite_status, 1);
                }
            }
            atomicAdd(stats + 0, dot);
            atomicAdd(stats + 1, fast_norm);
            atomicAdd(stats + 2, slow_norm);
            atomicAdd(stats + 3, residual_norm);
        }
    }
}

template <bool store_corrected_moments, bool check_finite, bool nesterov = false>
int launch_nekomuon_moments_stats(
    const float* gradient,
    float* fast,
    float* slow,
    float* fast_hat,
    float* slow_hat,
    float* stats,
    int length,
    float beta_fast,
    float beta_slow,
    float fast_correction,
    float slow_correction,
    int* finite_status,
    cudaStream_t stream) {
    if (!gradient || !fast || !slow || !stats
        || (store_corrected_moments && (!fast_hat || !slow_hat))
        || (check_finite && !finite_status)
        || length <= 0 || fast_correction <= 0.f
        || slow_correction <= 0.f) {
        return (int)cudaErrorInvalidValue;
    }
    const int blocks = min(
        (length + kNekoMuonThreads - 1) / kNekoMuonThreads,
        kNekoMuonMaxBlocks);
    nekomuon_moments_stats_block<
        store_corrected_moments, check_finite, nesterov><<<
        blocks, kNekoMuonThreads, 0, stream>>>(
        gradient, fast, slow, fast_hat, slow_hat, stats, length,
        beta_fast, beta_slow, fast_correction, slow_correction,
        finite_status);
    return (int)cudaPeekAtLastError();
}

template <bool check_finite, bool nesterov = false>
int launch_nekomuon_moments_stats_bf16(
    const __nv_bfloat16* gradient,
    __nv_bfloat16* fast,
    __nv_bfloat16* slow,
    float* stats,
    int length,
    float beta_fast,
    float beta_slow,
    float fast_correction,
    float slow_correction,
    int* finite_status,
    cudaStream_t stream) {
    if (!gradient || !fast || !slow || !stats
        || (check_finite && !finite_status) || length <= 0
        || fast_correction <= 0.f || slow_correction <= 0.f) {
        return (int)cudaErrorInvalidValue;
    }
    const int blocks = min(
        (length + kNekoMuonThreads - 1) / kNekoMuonThreads,
        kNekoMuonMaxBlocks);
    nekomuon_moments_stats_bf16_block<check_finite, nesterov><<<
        blocks, kNekoMuonThreads, 0, stream>>>(
        gradient, fast, slow, stats, length, beta_fast, beta_slow,
        fast_correction, slow_correction, finite_status);
    return (int)cudaPeekAtLastError();
}

}  // namespace

NNTRAIN_EXPORT int nntrain_nekomuon_moments_stats(
    const float* gradient, float* fast, float* slow,
    float* fast_hat, float* slow_hat, float* stats, int length,
    float beta_fast, float beta_slow, float fast_correction,
    float slow_correction, cudaStream_t stream) {
    return launch_nekomuon_moments_stats<true, false>(
        gradient, fast, slow, fast_hat, slow_hat, stats, length,
        beta_fast, beta_slow, fast_correction, slow_correction,
        nullptr, stream);
}

NNTRAIN_EXPORT int nntrain_nekomuon_moments_stats_compact(
    const float* gradient, float* fast, float* slow, float* stats,
    int length, float beta_fast, float beta_slow, float fast_correction,
    float slow_correction, cudaStream_t stream) {
    return launch_nekomuon_moments_stats<false, false>(
        gradient, fast, slow, nullptr, nullptr, stats, length,
        beta_fast, beta_slow, fast_correction, slow_correction,
        nullptr, stream);
}

NNTRAIN_EXPORT int nntrain_nekomuon_moments_stats_compact_finite(
    const float* gradient, float* fast, float* slow, float* stats,
    int length, float beta_fast, float beta_slow, float fast_correction,
    float slow_correction, int* finite_status, cudaStream_t stream) {
    return launch_nekomuon_moments_stats<false, true>(
        gradient, fast, slow, nullptr, nullptr, stats, length,
        beta_fast, beta_slow, fast_correction, slow_correction,
        finite_status, stream);
}

NNTRAIN_EXPORT int nntrain_nekomuon_moments_stats_bf16_compact(
    const __nv_bfloat16* gradient, __nv_bfloat16* fast,
    __nv_bfloat16* slow, float* stats, int length, float beta_fast,
    float beta_slow, float fast_correction, float slow_correction,
    cudaStream_t stream) {
    return launch_nekomuon_moments_stats_bf16<false>(
        gradient, fast, slow, stats, length, beta_fast, beta_slow,
        fast_correction, slow_correction, nullptr, stream);
}

NNTRAIN_EXPORT int nntrain_nekomuon_moments_stats_bf16_compact_finite(
    const __nv_bfloat16* gradient, __nv_bfloat16* fast,
    __nv_bfloat16* slow, float* stats, int length, float beta_fast,
    float beta_slow, float fast_correction, float slow_correction,
    int* finite_status, cudaStream_t stream) {
    return launch_nekomuon_moments_stats_bf16<true>(
        gradient, fast, slow, stats, length, beta_fast, beta_slow,
        fast_correction, slow_correction, finite_status, stream);
}

// ABI 1.29: reference Muon momentum. `fast` persists m_t while `slow`
// persists the Nesterov direction u_t, allowing the existing fixed-NS5
// pipeline to consume u_t without another allocation or launch.
NNTRAIN_EXPORT int nntrain_muon_moments_stats_compact(
    const float* gradient, float* fast, float* slow, float* stats,
    int length, float beta, cudaStream_t stream) {
    return launch_nekomuon_moments_stats<false, false, true>(
        gradient, fast, slow, nullptr, nullptr, stats, length,
        beta, beta, 1.f, 1.f, nullptr, stream);
}

NNTRAIN_EXPORT int nntrain_muon_moments_stats_compact_finite(
    const float* gradient, float* fast, float* slow, float* stats,
    int length, float beta, int* finite_status, cudaStream_t stream) {
    return launch_nekomuon_moments_stats<false, true, true>(
        gradient, fast, slow, nullptr, nullptr, stats, length,
        beta, beta, 1.f, 1.f, finite_status, stream);
}

NNTRAIN_EXPORT int nntrain_muon_moments_stats_bf16_compact(
    const __nv_bfloat16* gradient, __nv_bfloat16* fast,
    __nv_bfloat16* slow, float* stats, int length, float beta,
    cudaStream_t stream) {
    return launch_nekomuon_moments_stats_bf16<false, true>(
        gradient, fast, slow, stats, length, beta, beta,
        1.f, 1.f, nullptr, stream);
}

NNTRAIN_EXPORT int nntrain_muon_moments_stats_bf16_compact_finite(
    const __nv_bfloat16* gradient, __nv_bfloat16* fast,
    __nv_bfloat16* slow, float* stats, int length, float beta,
    int* finite_status, cudaStream_t stream) {
    return launch_nekomuon_moments_stats_bf16<true, true>(
        gradient, fast, slow, stats, length, beta, beta,
        1.f, 1.f, finite_status, stream);
}
