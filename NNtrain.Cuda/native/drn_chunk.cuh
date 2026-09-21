// Included in flash_attention.cu's anonymous namespace. Exact forward replay
// uses the same WMMA operands/order as drn_forward_warp_tensor_core_bf16.
constexpr int kDrnChunk = 16;

__device__ __forceinline__ float drn_chunk_sum(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        x += __shfl_xor_sync(0xffffffffu, x, offset);
    return x;
}

__device__ __forceinline__ float drn_chunk_sigmoid(float x) {
    if (x >= 0.f) return 1.f / (1.f + expf(-x));
    const float e = expf(x);
    return e / (1.f + e);
}

// The recurrent adjoint depends on Q/K/gate/beta/dY, not on forward state.
// A small separate reverse pass supplies exact terminal adjoints to chunks.
// Only chunk boundaries are written; no B*T*K*V history is materialized.
template<int K>
__global__ void drn_chunk_adjoint_boundaries(const __nv_bfloat16* p,
    const float* dy, float* state_gradient, float* boundaries,
    int sequence, int width, int value_width, float retention_floor) {
    const int b = blockIdx.x / value_width, v = blockIdx.x % value_width;
    const int lane = threadIdx.x;
    const int chunks = (sequence + kDrnChunk - 1) / kDrnChunk;
    const int matrix = K * value_width;
    float d = lane < K ? state_gradient[b * matrix + v * K + lane] : 0.f;
    if (lane < K) boundaries[(b * (chunks + 1) + chunks) * matrix + v * K + lane] = d;
    for (int t = sequence - 1; t >= 0; --t) {
        const int base = (b * sequence + t) * width;
        const float q = lane < K ? tanhf(__bfloat162float(p[base + lane])) : 0.f;
        const float k = lane < K ? tanhf(__bfloat162float(p[base + K + lane])) : 0.f;
        const float qs = rsqrtf(drn_chunk_sum(q * q) + 1e-8f);
        const float ks = 1.f / sqrtf(drn_chunk_sum(k * k) + 1e-8f);
        const float gate = drn_chunk_sigmoid(__bfloat162float(p[base + 2*K + v + value_width]));
        const float retention = retention_floor + (1.f - retention_floor) * gate;
        const float beta = drn_chunk_sigmoid(__bfloat162float(p[base + 2*K + v + 2*value_width]));
        const float nk = k * ks;
        const float de = beta * drn_chunk_sum(d * nk);
        d = q * qs * dy[(b * sequence + t) * value_width + v] + (d * retention - nk * de);
        if (lane < K && t % kDrnChunk == 0)
            boundaries[(b * (chunks + 1) + t / kDrnChunk) * matrix + v * K + lane] = d;
    }
    if (lane < K) state_gradient[b * matrix + v * K + lane] = d;
}

// One CTA owns 16 value rows of one time chunk. Warp 0 replays forward into
// shared FP32 state; then all 16 warps differentiate independent value rows.
// Chunk/time blocks are independent because both boundary states are known.
template<int K>
__global__ void drn_chunk_replay_backward(const __nv_bfloat16* projected,
    float* dp, const float* dy, const float* checkpoints, const float* adjoints,
    int sequence, int width, int value_width, float retention_floor) {
    constexpr int rows = 16, per_lane = rows * K / 32;
    __shared__ float history[kDrnChunk * rows * K];
    __shared__ __align__(32) __nv_bfloat16 state_tile[rows * K];
    __shared__ __align__(32) __nv_bfloat16 vectors[K * 16];
    __shared__ __align__(32) float product[16 * 16];
    const int b = blockIdx.x, row_base = blockIdx.y * rows, chunk = blockIdx.z;
    const int chunks = (sequence + kDrnChunk - 1) / kDrnChunk;
    const int start = chunk * kDrnChunk, count = min(kDrnChunk, sequence - start);
    const int matrix = K * value_width;
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    if (warp == 0) {
        float recurrent[per_lane];
        #pragma unroll
        for (int i = 0; i < per_lane; ++i)
            recurrent[i] = checkpoints[(b * chunks + chunk) * matrix + row_base * K + i * 32 + lane];
        for (int local = 0; local < count; ++local) {
            #pragma unroll
            for (int i = 0; i < per_lane; ++i)
                history[local * rows * K + i * 32 + lane] = recurrent[i];
            // Last state update is unused: backward needs M[t-1], not M[t].
            if (local + 1 == count) break;
            const int p = (b * sequence + start + local) * width;
            float k = lane < K ? tanhf(__bfloat162float(projected[p + K + lane])) : 0.f;
            float q = lane < K ? tanhf(__bfloat162float(projected[p + lane])) : 0.f;
            k *= rsqrtf(drn_chunk_sum(k * k) + 1e-8f);
            q *= rsqrtf(drn_chunk_sum(q * q) + 1e-8f);
            float gate = 0.f, beta = 0.f, value = 0.f;
            if (lane < rows) {
                const int v = p + 2*K + row_base + lane;
                gate = 1.f / (1.f + expf(-__bfloat162float(projected[v + value_width])));
                gate = retention_floor + (1.f - retention_floor) * gate;
                beta = 1.f / (1.f + expf(-__bfloat162float(projected[v + 2*value_width])));
                value = tanhf(__bfloat162float(projected[v]));
            }
            #pragma unroll
            for (int i = 0; i < K * 16 / 32; ++i) {
                const int index = i * 32 + lane;
                const float vk = __shfl_sync(0xffffffffu, k, index / 16);
                const float vq = __shfl_sync(0xffffffffu, q, index / 16);
                vectors[index] = __float2bfloat16_rn(index % 16 == 0 ? vk : index % 16 == 1 ? vq : 0.f);
            }
            #pragma unroll
            for (int i = 0; i < per_lane; ++i)
                state_tile[i * 32 + lane] = __float2bfloat16_rn(recurrent[i]);
            __syncwarp();
            forget_memory_tensor_core_matvec(state_tile, vectors, product, K / 16, K);
            __syncwarp();
            const float delta = lane < rows ? beta * (value - product[lane * 16]) : 0.f;
            #pragma unroll
            for (int i = 0; i < per_lane; ++i) {
                const int index = i * 32 + lane;
                const float g = __shfl_sync(0xffffffffu, gate, index / K);
                const float d = __shfl_sync(0xffffffffu, delta, index / K);
                const float nk = __shfl_sync(0xffffffffu, k, index % K);
                recurrent[i] = g * recurrent[i] + d * nk;
            }
            __syncwarp();
        }
    }
    __syncthreads();
    const int value_index = row_base + warp;
    float gradient = lane < K
        ? adjoints[(b * (chunks + 1) + chunk + 1) * matrix + value_index * K + lane] : 0.f;
    for (int local = count - 1; local >= 0; --local) {
        const int t = start + local, p = (b * sequence + t) * width;
        const int v = p + 2*K + value_index;
        const float q = lane < K ? tanhf(__bfloat162float(projected[p + lane])) : 0.f;
        const float k = lane < K ? tanhf(__bfloat162float(projected[p + K + lane])) : 0.f;
        const float qs = rsqrtf(drn_chunk_sum(q*q) + 1e-8f);
        const float ks = 1.f / sqrtf(drn_chunk_sum(k*k) + 1e-8f);
        const float previous = lane < K ? history[local * rows*K + warp*K + lane] : 0.f;
        const float upstream = dy[(b * sequence + t) * value_width + value_index];
        const float gate = drn_chunk_sigmoid(__bfloat162float(projected[v + value_width]));
        const float retention = retention_floor + (1.f - retention_floor) * gate;
        const float beta = drn_chunk_sigmoid(__bfloat162float(projected[v + 2*value_width]));
        const float value = tanhf(__bfloat162float(projected[v]));
        const float nk = k * ks;
        const float predicted = drn_chunk_sum(previous * nk);
        const float dot_key = drn_chunk_sum(gradient * nk);
        const float dgate = drn_chunk_sum(gradient * previous);
        const float error = value - predicted;
        const float dwrite = error * dot_key, derror = beta * dot_key;
        const float dq = previous * upstream;
        const float qdot = drn_chunk_sum(q * dq);
        const float dk = gradient * beta * error - previous * derror;
        const float kdot = drn_chunk_sum(k * dk);
        if (lane < K) {
            atomicAdd(dp + p + lane, (dq*qs - q*qdot*qs*qs*qs) * (1.f-q*q));
            atomicAdd(dp + p + K + lane, (dk*ks - k*kdot*ks*ks*ks) * (1.f-k*k));
        }
        if (lane == 0) {
            dp[v] += derror * (1.f-value*value);
            dp[v + value_width] += dgate * (1.f-retention_floor) * gate * (1.f-gate);
            dp[v + 2*value_width] += dwrite * beta * (1.f-beta);
        }
        gradient = q*qs*upstream + (gradient*retention - nk*derror);
    }
}
