// Shared helpers may appear later in the concatenated OpenCL resources.
inline float q35d_sigmoid(float x);
inline float q35d_silu(float x);
inline float q35d_softplus(float x);

// Fused update for the 128-component Gated DeltaNet heads in Qwen3.5 27B.
// One workgroup owns one head; each lane exclusively owns one state column.
// q35d activation helpers are defined in qwen35_delta.cl.
__kernel void q35d_recurrent_gated_rmsnorm_fused128(
    __global const float* mixed, __global const float* alpha,
    __global const float* beta, __global const float* dt,
    __global const float* a, __global float* state,
    __global const float* norm, __global const float* gate,
    __global float* output, int key_heads, float eps)
{
    int h = get_group_id(0), v = get_local_id(0);
    int item = h * 128 + v, key_size = key_heads * 128;
    int q_offset = (h % key_heads) * 128, k_offset = key_size + q_offset;
    int state_offset = h * 128 * 128;
    __local float shared_decay, shared_update_rate, inverse;
    __local float values[128];
    if (v == 0)
    {
        shared_decay = exp(a[h] * q35d_softplus(alpha[h] + dt[h]));
        shared_update_rate = q35d_sigmoid(beta[h]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float decay = shared_decay, update_rate = shared_update_rate;
    float predicted = 0.0f;
    for (int k = 0; k < 128; ++k)
    {
        int index = state_offset + k * 128 + v;
        state[index] *= decay;
        predicted += mixed[k_offset + k] * state[index];
    }
    float delta = (mixed[2 * key_size + item] - predicted) * update_rate;
    float value = 0.0f;
    for (int k = 0; k < 128; ++k)
    {
        int index = state_offset + k * 128 + v;
        state[index] += mixed[k_offset + k] * delta;
        value += mixed[q_offset + k] * state[index];
    }
    values[v] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (v == 0)
    {
        // Keep the reference kernel's component order for this RMS sum.
        float sum = 0.0f;
        for (int i = 0; i < 128; ++i)
        {
            float x = values[i];
            sum += x * x;
        }
        inverse = 1.0f / sqrt(sum / 128 + eps);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    output[item] = values[v] * inverse * norm[v] * q35d_silu(gate[item]);
}

// A workgroup retains ownership of one value head throughout a prompt chunk.
// State and RMS sums keep exactly the single-token component and row order.
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void q35d_recurrent_gated_rmsnorm_rows128(
    __global const float* mixed, __global const float* alpha,
    __global const float* beta, __global const float* dt,
    __global const float* a, __global float* state,
    __global const float* norm, __global const float* gate,
    __global float* output, int key_heads, int value_heads, int channels,
    int start_row, int rows, float eps)
{
    int h = get_group_id(0), v = get_local_id(0);
    int item = h * 128 + v, key_size = key_heads * 128;
    int state_offset = h * 128 * 128;
    __local float shared_decay, shared_update_rate, inverse, values[128];
    for (int row = start_row; row < start_row + rows; ++row)
    {
        int q_offset = row * channels + (h % key_heads) * 128;
        int k_offset = key_size + q_offset;
        if (v == 0)
        {
            shared_decay = exp(a[h] * q35d_softplus(alpha[row * value_heads + h] + dt[h]));
            shared_update_rate = q35d_sigmoid(beta[row * value_heads + h]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float decay = shared_decay, update_rate = shared_update_rate;
        float predicted = 0.0f;
        for (int k = 0; k < 128; ++k)
        {
            int index = state_offset + k * 128 + v;
            state[index] *= decay;
            predicted += mixed[k_offset + k] * state[index];
        }
        float delta = (mixed[row * channels + 2 * key_size + item] - predicted) * update_rate;
        float value = 0.0f;
        for (int k = 0; k < 128; ++k)
        {
            int index = state_offset + k * 128 + v;
            state[index] += mixed[k_offset + k] * delta;
            value += mixed[q_offset + k] * state[index];
        }
        values[v] = value;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (v == 0)
        {
            float sum = 0.0f;
            for (int i = 0; i < 128; ++i)
            {
                float x = values[i];
                sum += x * x;
            }
            inverse = 1.0f / sqrt(sum / 128 + eps);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        int output_item = row * value_heads * 128 + item;
        output[output_item] = values[v] * inverse * norm[v] * q35d_silu(gate[output_item]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// Load one complete state column into private FP32 storage. Each lane owns
// all of its components across every row and publishes the final state once.
// This removes repeated global state traffic without changing FMA order.
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void q35d_recurrent_gated_rmsnorm_rows128_cached(
    __global const float* mixed, __global const float* alpha,
    __global const float* beta, __global const float* dt,
    __global const float* a, __global float* state,
    __global const float* norm, __global const float* gate,
    __global float* output, int key_heads, int value_heads, int channels,
    int start_row, int rows, float eps)
{
    int h = get_group_id(0), v = get_local_id(0);
    int item = h * 128 + v, key_size = key_heads * 128;
    int state_offset = h * 128 * 128;
    float state_column[128];
    for (int k = 0; k < 128; ++k)
        state_column[k] = state[state_offset + k * 128 + v];
    __local float shared_decay, shared_update_rate, inverse, values[128];
    for (int row = start_row; row < start_row + rows; ++row)
    {
        int q_offset = row * channels + (h % key_heads) * 128;
        int k_offset = key_size + q_offset;
        if (v == 0)
        {
            shared_decay = exp(a[h] * q35d_softplus(alpha[row * value_heads + h] + dt[h]));
            shared_update_rate = q35d_sigmoid(beta[row * value_heads + h]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float decay = shared_decay, update_rate = shared_update_rate;
        float predicted = 0.0f;
        for (int k = 0; k < 128; ++k)
        {
            state_column[k] *= decay;
            predicted += mixed[k_offset + k] * state_column[k];
        }
        float delta = (mixed[row * channels + 2 * key_size + item] - predicted) * update_rate;
        float value = 0.0f;
        for (int k = 0; k < 128; ++k)
        {
            state_column[k] += mixed[k_offset + k] * delta;
            value += mixed[q_offset + k] * state_column[k];
        }
        values[v] = value;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (v == 0)
        {
            float sum = 0.0f;
            for (int i = 0; i < 128; ++i)
            {
                float x = values[i];
                sum += x * x;
            }
            inverse = 1.0f / sqrt(sum / 128 + eps);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        int output_item = row * value_heads * 128 + item;
        output[output_item] = values[v] * inverse * norm[v] * q35d_silu(gate[output_item]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int k = 0; k < 128; ++k)
        state[state_offset + k * 128 + v] = state_column[k];
}

__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void q35d_recurrent_gated_rmsnorm_rows128_unrolled(
    __global const float* mixed, __global const float* alpha,
    __global const float* beta, __global const float* dt,
    __global const float* a, __global float* state,
    __global const float* norm, __global const float* gate,
    __global float* output, int key_heads, int value_heads, int channels,
    int start_row, int rows, float eps)
{
    int h = get_group_id(0), v = get_local_id(0);
    int item = h * 128 + v, key_size = key_heads * 128;
    int state_offset = h * 128 * 128;
    float state_column[128];
    #pragma unroll
        for (int k = 0; k < 128; ++k)
        state_column[k] = state[state_offset + k * 128 + v];
    __local float shared_decay, shared_update_rate, inverse, values[128];
    for (int row = start_row; row < start_row + rows; ++row)
    {
        int q_offset = row * channels + (h % key_heads) * 128;
        int k_offset = key_size + q_offset;
        if (v == 0)
        {
            shared_decay = exp(a[h] * q35d_softplus(alpha[row * value_heads + h] + dt[h]));
            shared_update_rate = q35d_sigmoid(beta[row * value_heads + h]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float decay = shared_decay, update_rate = shared_update_rate;
        float predicted = 0.0f;
        #pragma unroll
        for (int k = 0; k < 128; ++k)
        {
            state_column[k] *= decay;
            predicted += mixed[k_offset + k] * state_column[k];
        }
        float delta = (mixed[row * channels + 2 * key_size + item] - predicted) * update_rate;
        float value = 0.0f;
        #pragma unroll
        for (int k = 0; k < 128; ++k)
        {
            state_column[k] += mixed[k_offset + k] * delta;
            value += mixed[q_offset + k] * state_column[k];
        }
        values[v] = value;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (v == 0)
        {
            float sum = 0.0f;
            for (int i = 0; i < 128; ++i)
            {
                float x = values[i];
                sum += x * x;
            }
            inverse = 1.0f / sqrt(sum / 128 + eps);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        int output_item = row * value_heads * 128 + item;
        output[output_item] = values[v] * inverse * norm[v] * q35d_silu(gate[output_item]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
        for (int k = 0; k < 128; ++k)
        state[state_offset + k * 128 + v] = state_column[k];
}

#if defined(cl_intel_subgroups) && defined(ARC_DELTA_SUBGROUP_RMS)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void q35d_recurrent_gated_rmsnorm_rows128_subgroup(
    __global const float* mixed, __global const float* alpha,
    __global const float* beta, __global const float* dt,
    __global const float* a, __global float* state,
    __global const float* norm, __global const float* gate,
    __global float* output, int key_heads, int value_heads, int channels,
    int start_row, int rows, float eps)
{
    int h = get_group_id(0), v = get_local_id(0);
    int item = h * 128 + v, key_size = key_heads * 128;
    int state_offset = h * 128 * 128;
    float state_column[128];
    #pragma unroll
        for (int k = 0; k < 128; ++k)
        state_column[k] = state[state_offset + k * 128 + v];
    __local float shared_decay, shared_update_rate, values[128];
    for (int row = start_row; row < start_row + rows; ++row)
    {
        int q_offset = row * channels + (h % key_heads) * 128;
        int k_offset = key_size + q_offset;
        if (v == 0)
        {
            shared_decay = exp(a[h] * q35d_softplus(alpha[row * value_heads + h] + dt[h]));
            shared_update_rate = q35d_sigmoid(beta[row * value_heads + h]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float decay = shared_decay, update_rate = shared_update_rate;
        float predicted = 0.0f;
        #pragma unroll
        for (int k = 0; k < 128; ++k)
        {
            state_column[k] *= decay;
            predicted += mixed[k_offset + k] * state_column[k];
        }
        float delta = (mixed[row * channels + 2 * key_size + item] - predicted) * update_rate;
        float value = 0.0f;
        #pragma unroll
        for (int k = 0; k < 128; ++k)
        {
            state_column[k] += mixed[k_offset + k] * delta;
            value += mixed[q_offset + k] * state_column[k];
        }
        values[v] = value;
        barrier(CLK_LOCAL_MEM_FENCE);
        float inverse = 0.0f;
        // Each SG16 leader keeps the reference component order.
        // The next row starts with a workgroup barrier before values reuse.
        if ((v & 15) == 0)
        {
            float sum = 0.0f;
            for (int i = 0; i < 128; ++i)
            {
                float x = values[i];
                sum += x * x;
            }
            inverse = 1.0f / sqrt(sum / 128 + eps);
        }
        inverse = intel_sub_group_shuffle(inverse, 0);
        int output_item = row * value_heads * 128 + item;
        output[output_item] = value * inverse * norm[v] * q35d_silu(gate[output_item]);
    }
    #pragma unroll
        for (int k = 0; k < 128; ++k)
        state[state_offset + k * 128 + v] = state_column[k];
}

#endif
