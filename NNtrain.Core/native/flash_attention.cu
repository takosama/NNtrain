#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

#if defined(_WIN32)
#define NNTRAIN_EXPORT extern "C" __declspec(dllexport)
#else
#define NNTRAIN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return __shfl_sync(0xffffffffu, value, 0);
}

__global__ void attention_forward(const float* __restrict__ qkv,
    float* __restrict__ output, int batch, int sequence, int model_width,
    int heads, int causal) {
    const int work = blockIdx.x, query = work % sequence;
    const int batch_head = work / sequence, head = batch_head % heads;
    const int batch_index = batch_head / heads, lane = threadIdx.x;
    const int head_width = model_width / heads, projected_width = 3 * model_width;
    const int batch_base = batch_index * sequence * projected_width;
    const int head_base = head * head_width;
    const int query_base = batch_base + query * projected_width + head_base;
    const int output_base = (batch_index * sequence + query) * model_width + head_base;
    const int last_key = causal ? query : sequence - 1;
    const float scale = rsqrtf((float)head_width);
    float accum[4] = {0.f, 0.f, 0.f, 0.f}, maximum = -3.402823466e+38F, denominator = 0.f;
    for (int key = 0; key <= last_key; ++key) {
        const int key_base = batch_base + key * projected_width + model_width + head_base;
        float partial = 0.f;
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int column = lane + item * 32;
            if (column < head_width)
                partial = fmaf(qkv[query_base + column], qkv[key_base + column], partial);
        }
        const float score = warp_sum(partial) * scale;
        const float next_maximum = fmaxf(maximum, score);
        const float old_scale = expf(maximum - next_maximum);
        const float new_scale = expf(score - next_maximum);
        denominator = denominator * old_scale + new_scale;
        const int value_base = key_base + model_width;
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int column = lane + item * 32;
            if (column < head_width)
                accum[item] = accum[item] * old_scale + new_scale * qkv[value_base + column];
        }
        maximum = next_maximum;
    }
    const float inverse = 1.f / denominator;
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int column = lane + item * 32;
        if (column < head_width) output[output_base + column] = accum[item] * inverse;
    }
}

__global__ void attention_backward(const float* __restrict__ qkv,
    const float* __restrict__ output, const float* __restrict__ output_gradient,
    float* __restrict__ qkv_gradient, int batch, int sequence,
    int model_width, int heads, int causal) {
    const int work = blockIdx.x, query = work % sequence;
    const int batch_head = work / sequence, head = batch_head % heads;
    const int batch_index = batch_head / heads, lane = threadIdx.x;
    const int head_width = model_width / heads, projected_width = 3 * model_width;
    const int batch_base = batch_index * sequence * projected_width;
    const int head_base = head * head_width;
    const int query_base = batch_base + query * projected_width + head_base;
    const int output_base = (batch_index * sequence + query) * model_width + head_base;
    const int last_key = causal ? query : sequence - 1;
    const float scale = rsqrtf((float)head_width);
    float maximum = -3.402823466e+38F;
    for (int key = 0; key <= last_key; ++key) {
        const int key_base = batch_base + key * projected_width + model_width + head_base;
        float partial = 0.f;
        #pragma unroll
        for (int item = 0; item < 4; ++item) { const int c = lane + item * 32;
            if (c < head_width) partial = fmaf(qkv[query_base+c], qkv[key_base+c], partial); }
        maximum = fmaxf(maximum, warp_sum(partial) * scale);
    }
    float denominator = 0.f, delta_partial = 0.f;
    #pragma unroll
    for (int item = 0; item < 4; ++item) { const int c = lane + item * 32;
        if (c < head_width) delta_partial = fmaf(output_gradient[output_base+c], output[output_base+c], delta_partial); }
    const float delta = warp_sum(delta_partial);
    for (int key = 0; key <= last_key; ++key) {
        const int key_base = batch_base + key * projected_width + model_width + head_base;
        float partial = 0.f;
        #pragma unroll
        for (int item = 0; item < 4; ++item) { const int c = lane + item * 32;
            if (c < head_width) partial = fmaf(qkv[query_base+c], qkv[key_base+c], partial); }
        denominator += expf(warp_sum(partial) * scale - maximum);
    }
    float dq[4] = {0.f, 0.f, 0.f, 0.f};
    for (int key = 0; key <= last_key; ++key) {
        const int key_base = batch_base + key * projected_width + model_width + head_base;
        const int value_base = key_base + model_width;
        float score_partial = 0.f, dp_partial = 0.f;
        #pragma unroll
        for (int item = 0; item < 4; ++item) { const int c = lane + item * 32;
            if (c < head_width) { score_partial = fmaf(qkv[query_base+c], qkv[key_base+c], score_partial);
                dp_partial = fmaf(output_gradient[output_base+c], qkv[value_base+c], dp_partial); } }
        const float probability = expf(warp_sum(score_partial) * scale - maximum) / denominator;
        const float ds = probability * (warp_sum(dp_partial) - delta) * scale;
        #pragma unroll
        for (int item = 0; item < 4; ++item) { const int c = lane + item * 32;
            if (c < head_width) { dq[item] = fmaf(ds, qkv[key_base+c], dq[item]);
                atomicAdd(qkv_gradient + key_base+c, ds*qkv[query_base+c]);
                atomicAdd(qkv_gradient + value_base+c, probability*output_gradient[output_base+c]); } }
    }
    #pragma unroll
    for (int item = 0; item < 4; ++item) { const int c = lane + item * 32;
        if (c < head_width) atomicAdd(qkv_gradient + query_base+c, dq[item]); }
}

__global__ void attention_forward_bf16(const __nv_bfloat16* __restrict__ qkv,
    __nv_bfloat16* __restrict__ output, int batch, int sequence,
    int model_width, int heads, int causal) {
    const int work = blockIdx.x, query = work % sequence;
    const int batch_head = work / sequence, head = batch_head % heads;
    const int batch_index = batch_head / heads, lane = threadIdx.x;
    const int head_width = model_width / heads, projected_width = 3 * model_width;
    const int batch_base = batch_index * sequence * projected_width;
    const int head_base = head * head_width;
    const int query_base = batch_base + query * projected_width + head_base;
    const int output_base = (batch_index * sequence + query) * model_width + head_base;
    const int last_key = causal ? query : sequence - 1;
    const float scale = rsqrtf((float)head_width);
    float accum[4] = {0.f, 0.f, 0.f, 0.f};
    float maximum = -3.402823466e+38F, denominator = 0.f;
    for (int key = 0; key <= last_key; ++key) {
        const int key_base = batch_base + key * projected_width + model_width + head_base;
        float partial = 0.f;
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int column = lane + item * 32;
            if (column < head_width)
                partial = fmaf(__bfloat162float(qkv[query_base + column]),
                    __bfloat162float(qkv[key_base + column]), partial);
        }
        const float score = warp_sum(partial) * scale;
        const float next_maximum = fmaxf(maximum, score);
        const float old_scale = expf(maximum - next_maximum);
        const float new_scale = expf(score - next_maximum);
        denominator = denominator * old_scale + new_scale;
        const int value_base = key_base + model_width;
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int column = lane + item * 32;
            if (column < head_width)
                accum[item] = accum[item] * old_scale
                    + new_scale * __bfloat162float(qkv[value_base + column]);
        }
        maximum = next_maximum;
    }
    const float inverse = 1.f / denominator;
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int column = lane + item * 32;
        if (column < head_width)
            output[output_base + column] = __float2bfloat16_rn(accum[item] * inverse);
    }
}

__global__ void attention_backward_bf16(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ output,
    const float* __restrict__ output_gradient,
    float* __restrict__ qkv_gradient, int batch, int sequence,
    int model_width, int heads, int causal) {
    const int work = blockIdx.x, query = work % sequence;
    const int batch_head = work / sequence, head = batch_head % heads;
    const int batch_index = batch_head / heads, lane = threadIdx.x;
    const int head_width = model_width / heads, projected_width = 3 * model_width;
    const int batch_base = batch_index * sequence * projected_width;
    const int head_base = head * head_width;
    const int query_base = batch_base + query * projected_width + head_base;
    const int output_base = (batch_index * sequence + query) * model_width + head_base;
    const int last_key = causal ? query : sequence - 1;
    const float scale = rsqrtf((float)head_width);
    float maximum = -3.402823466e+38F;
    for (int key = 0; key <= last_key; ++key) {
        const int key_base = batch_base + key * projected_width + model_width + head_base;
        float partial = 0.f;
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int c = lane + item * 32;
            if (c < head_width)
                partial = fmaf(__bfloat162float(qkv[query_base+c]),
                    __bfloat162float(qkv[key_base+c]), partial);
        }
        maximum = fmaxf(maximum, warp_sum(partial) * scale);
    }
    float denominator = 0.f, delta_partial = 0.f;
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int c = lane + item * 32;
        if (c < head_width)
            delta_partial = fmaf(output_gradient[output_base+c],
                __bfloat162float(output[output_base+c]), delta_partial);
    }
    const float delta = warp_sum(delta_partial);
    for (int key = 0; key <= last_key; ++key) {
        const int key_base = batch_base + key * projected_width + model_width + head_base;
        float partial = 0.f;
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int c = lane + item * 32;
            if (c < head_width)
                partial = fmaf(__bfloat162float(qkv[query_base+c]),
                    __bfloat162float(qkv[key_base+c]), partial);
        }
        denominator += expf(warp_sum(partial) * scale - maximum);
    }
    float dq[4] = {0.f, 0.f, 0.f, 0.f};
    for (int key = 0; key <= last_key; ++key) {
        const int key_base = batch_base + key * projected_width + model_width + head_base;
        const int value_base = key_base + model_width;
        float score_partial = 0.f, dp_partial = 0.f;
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int c = lane + item * 32;
            if (c < head_width) {
                score_partial = fmaf(__bfloat162float(qkv[query_base+c]),
                    __bfloat162float(qkv[key_base+c]), score_partial);
                dp_partial = fmaf(output_gradient[output_base+c],
                    __bfloat162float(qkv[value_base+c]), dp_partial);
            }
        }
        const float probability = expf(warp_sum(score_partial) * scale - maximum)
            / denominator;
        const float ds = probability * (warp_sum(dp_partial) - delta) * scale;
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int c = lane + item * 32;
            if (c < head_width) {
                dq[item] = fmaf(ds, __bfloat162float(qkv[key_base+c]), dq[item]);
                atomicAdd(qkv_gradient + key_base+c,
                    ds * __bfloat162float(qkv[query_base+c]));
                atomicAdd(qkv_gradient + value_base+c,
                    probability * output_gradient[output_base+c]);
            }
        }
    }
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int c = lane + item * 32;
        if (c < head_width)
            atomicAdd(qkv_gradient + query_base+c, dq[item]);
    }
}
}

NNTRAIN_EXPORT int nntrain_flash_attention_forward(const float* qkv, float* output,
    int batch, int sequence, int model_width, int heads, int causal, cudaStream_t stream) {
    if (!qkv || !output || heads <= 0 || model_width % heads || model_width / heads > 128)
        return (int)cudaErrorInvalidValue;
    attention_forward<<<batch*heads*sequence, 32, 0, stream>>>(qkv, output, batch, sequence, model_width, heads, causal);
    return (int)cudaPeekAtLastError();
}

NNTRAIN_EXPORT int nntrain_flash_attention_backward(const float* qkv,
    const float* output, const float* output_gradient, float* qkv_gradient,
    int batch, int sequence, int model_width, int heads, int causal, cudaStream_t stream) {
    if (!qkv || !output || !output_gradient || !qkv_gradient || heads <= 0 || model_width / heads > 128)
        return (int)cudaErrorInvalidValue;
    attention_backward<<<batch*heads*sequence, 32, 0, stream>>>(qkv, output, output_gradient,
        qkv_gradient, batch, sequence, model_width, heads, causal);
    return (int)cudaPeekAtLastError();
}

NNTRAIN_EXPORT int nntrain_flash_attention_forward_bf16(
    const __nv_bfloat16* qkv, __nv_bfloat16* output, int batch,
    int sequence, int model_width, int heads, int causal,
    cudaStream_t stream) {
    if (!qkv || !output || heads <= 0 || model_width % heads
        || model_width / heads > 128)
        return (int)cudaErrorInvalidValue;
    attention_forward_bf16<<<batch*heads*sequence, 32, 0, stream>>>(
        qkv, output, batch, sequence, model_width, heads, causal);
    return (int)cudaPeekAtLastError();
}

NNTRAIN_EXPORT int nntrain_flash_attention_backward_bf16(
    const __nv_bfloat16* qkv, const __nv_bfloat16* output,
    const float* output_gradient, float* qkv_gradient, int batch,
    int sequence, int model_width, int heads, int causal,
    cudaStream_t stream) {
    if (!qkv || !output || !output_gradient || !qkv_gradient
        || heads <= 0 || model_width % heads || model_width / heads > 128)
        return (int)cudaErrorInvalidValue;
    attention_backward_bf16<<<batch*heads*sequence, 32, 0, stream>>>(
        qkv, output, output_gradient, qkv_gradient, batch, sequence,
        model_width, heads, causal);
    return (int)cudaPeekAtLastError();
}
