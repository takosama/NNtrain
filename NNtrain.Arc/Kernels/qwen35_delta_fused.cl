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
