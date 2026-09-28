// Full-sequence Gated DeltaNet with exact reverse-time adjoints. No host state.
// State layout: [time, value head, key component, value component].
inline float q35t_d_silu_prime(float x)
{
    float s = q35d_sigmoid(x);
    return s * (1.0f + x * (1.0f - s));
}

__kernel void q35t_delta_conv(__global const float* x, __global const float* weights,
    __global float* pre, __global float* mixed, int sequence, int channels, int conv_kernel)
{
    int i = get_global_id(0);
    if (i >= sequence * channels) return;
    int t = i / channels, c = i % channels;
    float sum = 0.0f;
    for (int tap = 0; tap < conv_kernel; ++tap)
    {
        int source = t - conv_kernel + 1 + tap;
        if (source >= 0) sum += x[source * channels + c] * weights[c * conv_kernel + tap];
    }
    pre[i] = sum;
    mixed[i] = q35d_silu(sum);
}

__kernel void q35t_delta_qk(__global float* mixed,
    int sequence, int key_heads, int value_heads, int width, float eps)
{
    int item = get_global_id(0);
    if (item >= sequence * key_heads) return;
    int t = item / key_heads, h = item % key_heads;
    int channels = (2 * key_heads + value_heads) * width;
    int qo = t * channels + h * width, ko = qo + key_heads * width;
    float qs = 0.0f, ks = 0.0f;
    for (int k = 0; k < width; ++k)
    {
        qs += mixed[qo + k] * mixed[qo + k];
        ks += mixed[ko + k] * mixed[ko + k];
    }
    float qi = (1.0f / sqrt((float)width)) / sqrt(qs + eps);
    float ki = 1.0f / sqrt(ks + eps);
    for (int k = 0; k < width; ++k)
    {
        mixed[qo + k] *= qi;
        mixed[ko + k] *= ki;
    }
}

// Each work item owns a value column throughout time. keep_tape=0 needs only
// one state, keep_tape=1 writes every state for a single layer's backward pass.
// Optional chunk checkpoints are copied after the same state update used for
// the forward output, so backward need not replay the recurrence a third time.
__kernel void q35t_delta_recur(__global const float* mixed,
    __global const float* alpha, __global const float* beta,
    __global const float* dt, __global const float* a,
    __global float* states, __global float* checkpoints, __global float* raw,
    int sequence, int chunk_size, int key_heads, int value_heads, int width,
    int keep_tape, int capture_checkpoints)
{
    int item = get_global_id(0), values = value_heads * width;
    if (item >= values) return;
    int h = item / width, v = item % width, keys = key_heads * width;
    int channels = 2 * keys + values, stride = values * width;
    int so = h * width * width + v, qo = (h % key_heads) * width;
    for (int k = 0; k < width; ++k)
    {
        int index = so + k * width;
        states[index] = 0.0f;
        if (capture_checkpoints) checkpoints[index] = 0.0f;
    }
    for (int t = 0; t < sequence; ++t)
    {
        int prev = keep_tape ? t * stride : 0;
        int next = keep_tape ? (t + 1) * stride : 0;
        int mo = t * channels;
        float decay = exp(a[h] * q35d_softplus(alpha[t * value_heads + h] + dt[h]));
        float rate = q35d_sigmoid(beta[t * value_heads + h]);
        float prediction = 0.0f;
        for (int k = 0; k < width; ++k)
        {
            float s = states[prev + so + k * width] * decay;
            states[next + so + k * width] = s;
            prediction += mixed[mo + keys + qo + k] * s;
        }
        float delta = (mixed[mo + 2 * keys + item] - prediction) * rate;
        float output = 0.0f;
        for (int k = 0; k < width; ++k)
        {
            int index = next + so + k * width;
            states[index] += mixed[mo + keys + qo + k] * delta;
            output += mixed[mo + qo + k] * states[index];
        }
        raw[t * values + item] = output;
        if (capture_checkpoints && (t + 1) % chunk_size == 0 && t + 1 < sequence)
        {
            int offset = ((t + 1) / chunk_size) * stride;
            for (int k = 0; k < width; ++k)
            {
                int index = so + k * width;
                checkpoints[offset + index] = states[next + index];
            }
        }
    }
}

// Recompute the recurrence once and retain only states at chunk boundaries.
// Work-item ownership and operation order match q35t_delta_recur.
__kernel void q35t_delta_checkpoints(__global const float* mixed,
    __global const float* alpha, __global const float* beta,
    __global const float* dt, __global const float* a,
    __global float* state, __global float* checkpoints,
    int sequence, int chunk_size, int key_heads, int value_heads, int width)
{
    int item = get_global_id(0), values = value_heads * width;
    if (item >= values) return;
    int h = item / width, keys = key_heads * width;
    int channels = 2 * keys + values, stride = values * width;
    int so = h * width * width + item % width, qo = (h % key_heads) * width;
    for (int k = 0; k < width; ++k)
    {
        int index = so + k * width;
        state[index] = 0.0f;
        checkpoints[index] = 0.0f;
    }
    for (int t = 0; t < sequence; ++t)
    {
        int mo = t * channels;
        float decay = exp(a[h] * q35d_softplus(alpha[t * value_heads + h] + dt[h]));
        float rate = q35d_sigmoid(beta[t * value_heads + h]);
        float prediction = 0.0f;
        for (int k = 0; k < width; ++k)
        {
            int index = so + k * width;
            float s = state[index] * decay;
            state[index] = s;
            prediction += mixed[mo + keys + qo + k] * s;
        }
        float delta = (mixed[mo + 2 * keys + item] - prediction) * rate;
        for (int k = 0; k < width; ++k)
        {
            int index = so + k * width;
            state[index] += mixed[mo + keys + qo + k] * delta;
        }
        if ((t + 1) % chunk_size == 0 && t + 1 < sequence)
        {
            int offset = ((t + 1) / chunk_size) * stride;
            for (int k = 0; k < width; ++k)
            {
                int index = so + k * width;
                checkpoints[offset + index] = state[index];
            }
        }
    }
}

// Restore one boundary state and reconstruct at most chunk_size + 1 states.
// Global token indices still address mixed/alpha/beta and their gradients.
__kernel void q35t_delta_recur_chunk(__global const float* mixed,
    __global const float* alpha, __global const float* beta,
    __global const float* dt, __global const float* a,
    __global const float* checkpoints, __global float* states,
    int first, int count, int chunk, int key_heads, int value_heads, int width)
{
    int item = get_global_id(0), values = value_heads * width;
    if (item >= values) return;
    int h = item / width, keys = key_heads * width;
    int channels = 2 * keys + values, stride = values * width;
    int so = h * width * width + item % width, qo = (h % key_heads) * width;
    for (int k = 0; k < width; ++k)
    {
        int index = so + k * width;
        states[index] = checkpoints[chunk * stride + index];
    }
    for (int step = 0; step < count; ++step)
    {
        int t = first + step, mo = t * channels;
        int prev = step * stride, next = (step + 1) * stride;
        float decay = exp(a[h] * q35d_softplus(alpha[t * value_heads + h] + dt[h]));
        float rate = q35d_sigmoid(beta[t * value_heads + h]);
        float prediction = 0.0f;
        for (int k = 0; k < width; ++k)
        {
            int index = so + k * width;
            float s = states[prev + index] * decay;
            states[next + index] = s;
            prediction += mixed[mo + keys + qo + k] * s;
        }
        float delta = (mixed[mo + 2 * keys + item] - prediction) * rate;
        for (int k = 0; k < width; ++k)
        {
            int index = next + so + k * width;
            states[index] += mixed[mo + keys + qo + k] * delta;
        }
    }
}

__kernel void q35t_delta_gate(__global const float* raw, __global const float* gate,
    __global const float* norm, __global float* output,
    int sequence, int heads, int width, float eps)
{
    int h = get_global_id(0);
    if (h >= sequence * heads) return;
    int offset = h * width;
    float sum = 0.0f;
    for (int v = 0; v < width; ++v) sum += raw[offset + v] * raw[offset + v];
    float inverse = 1.0f / sqrt(sum / width + eps);
    for (int v = 0; v < width; ++v)
        output[offset + v] = raw[offset + v] * inverse * norm[v] * q35d_silu(gate[offset + v]);
}

__kernel void q35t_delta_gate_backward(__global const float* raw,
    __global const float* gate, __global const float* norm,
    __global const float* output, __global const float* dy,
    __global float* draw, __global float* dgate,
    int sequence, int heads, int width, float eps)
{
    int h = get_global_id(0);
    if (h >= sequence * heads) return;
    int offset = h * width;
    float ss = 0.0f, output_dot = 0.0f;
    for (int v = 0; v < width; ++v)
    {
        int i = offset + v;
        ss += raw[i] * raw[i];
        // output = raw * inv * norm * silu(gate). Reuse that saved forward
        // result instead of evaluating sigmoid again in this reduction.
        output_dot += dy[i] * output[i];
    }
    float inv = 1.0f / sqrt(ss / width + eps);
    float correction = output_dot * inv / width;
    for (int v = 0; v < width; ++v)
    {
        int i = offset + v;
        float g = gate[i], sigmoid = q35d_sigmoid(g);
        float silu = g * sigmoid;
        float silu_prime = sigmoid * (1.0f + g * (1.0f - sigmoid));
        draw[i] = inv * (dy[i] * norm[v] * silu - raw[i] * correction);
        dgate[i] += dy[i] * raw[i] * inv * norm[v] * silu_prime;
    }
}

// adj initially contains dL/dS from future tokens. Add the current readout,
// then form per-value intermediates. Scratch is [value, delta, dp, db, dr].
__kernel void q35t_delta_prepare(__global const float* mixed,
    __global const float* alpha, __global const float* beta,
    __global const float* dt, __global const float* a,
    __global const float* states, __global const float* draw,
    __global float* adj, __global float* scratch, __global float* dmixed,
    int time, int state_first, int key_heads, int value_heads, int width)
{
    int item = get_global_id(0), values = value_heads * width;
    if (item >= values) return;
    int h = item / width, v = item % width, keys = key_heads * width;
    int channels = 2 * keys + values, stride = values * width;
    int state_time = time - state_first;
    int so = h * width * width + v, qo = time * channels + (h % key_heads) * width;
    float decay = exp(a[h] * q35d_softplus(alpha[time * value_heads + h] + dt[h]));
    float rate = q35d_sigmoid(beta[time * value_heads + h]);
    float prediction = 0.0f, du = 0.0f;
    for (int k = 0; k < width; ++k)
    {
        int index = so + k * width;
        float key = mixed[qo + keys + k];
        adj[index] += mixed[qo + k] * draw[time * values + item];
        du += key * adj[index];
        prediction += key * (decay * states[state_time * stride + index]);
    }
    float difference = mixed[time * channels + 2 * keys + item] - prediction;
    float dp = -rate * du, dr = 0.0f;
    for (int k = 0; k < width; ++k)
    {
        int index = so + k * width;
        dr += states[state_time * stride + index] * (adj[index] + mixed[qo + keys + k] * dp);
    }
    scratch[item * 4] = difference * rate;
    scratch[item * 4 + 1] = dp;
    scratch[item * 4 + 2] = difference * du;
    scratch[item * 4 + 3] = dr;
    dmixed[time * channels + 2 * keys + item] = rate * du;
}

// Sum over values and all tiled value heads without atomics.
__kernel void q35t_delta_qk_backward_step(__global const float* alpha,
    __global const float* dt, __global const float* a,
    __global const float* states, __global const float* draw,
    __global const float* adj, __global const float* scratch, __global float* dmixed,
    int time, int state_first, int key_heads, int value_heads, int width)
{
    int item = get_global_id(0), keys = key_heads * width;
    if (item >= keys) return;
    int kh = item / width, k = item % width, values = value_heads * width;
    int stride = values * width, channels = 2 * keys + values;
    int state_time = time - state_first;
    float dq = 0.0f, dk = 0.0f;
    for (int h = kh; h < value_heads; h += key_heads)
    {
        float decay = exp(a[h] * q35d_softplus(alpha[time * value_heads + h] + dt[h]));
        for (int v = 0; v < width; ++v)
        {
            int vi = h * width + v, si = h * width * width + k * width + v;
            dq += states[(state_time + 1) * stride + si] * draw[time * values + vi];
            dk += scratch[vi * 4] * adj[si]
                + (decay * states[state_time * stride + si]) * scratch[vi * 4 + 1];
        }
    }
    dmixed[time * channels + item] = dq;
    dmixed[time * channels + keys + item] = dk;
}

// One group owns a key component. Adjacent lanes walk adjacent value columns,
// matching the state layout rather than loading a different strided row per lane.
// The serial variant above remains the small-width path and a numerical oracle.
__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void q35t_delta_qk_backward_step_cooperative(__global const float* alpha,
    __global const float* dt, __global const float* a,
    __global const float* states, __global const float* draw,
    __global const float* adj, __global const float* scratch, __global float* dmixed,
    int time, int state_first, int key_heads, int value_heads, int width)
{
    int item = get_group_id(0), lane = get_local_id(0), keys = key_heads * width;
    if (item >= keys) return;
    int kh = item / width, k = item % width, values = value_heads * width;
    int stride = values * width, channels = 2 * keys + values;
    int state_time = time - state_first;
    float dq = 0.0f, dk = 0.0f;
    for (int h = kh; h < value_heads; h += key_heads)
    {
        float decay = exp(a[h] * q35d_softplus(alpha[time * value_heads + h] + dt[h]));
        for (int v = lane; v < width; v += 32)
        {
            int vi = h * width + v, si = h * width * width + k * width + v;
            dq += states[(state_time + 1) * stride + si] * draw[time * values + vi];
            dk += scratch[vi * 4] * adj[si]
                + (decay * states[state_time * stride + si]) * scratch[vi * 4 + 1];
        }
    }
    __local float sum_q[32], sum_k[32];
    sum_q[lane] = dq;
    sum_k[lane] = dk;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        if (lane < offset)
        {
            sum_q[lane] += sum_q[lane + offset];
            sum_k[lane] += sum_k[lane + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0)
    {
        dmixed[time * channels + item] = sum_q[0];
        dmixed[time * channels + keys + item] = sum_k[0];
    }
}

__kernel void q35t_delta_finish(__global const float* mixed,
    __global const float* alpha, __global const float* beta,
    __global const float* dt, __global const float* a,
    __global const float* scratch, __global float* adj,
    __global float* dalpha, __global float* dbeta,
    int time, int key_heads, int value_heads, int width)
{
    int item = get_global_id(0), values = value_heads * width;
    if (item >= values) return;
    int h = item / width, v = item % width, keys = key_heads * width;
    int channels = 2 * keys + values, qo = time * channels + (h % key_heads) * width;
    float x = alpha[time * value_heads + h] + dt[h];
    float decay = exp(a[h] * q35d_softplus(x));
    for (int k = 0; k < width; ++k)
    {
        int si = h * width * width + k * width + v;
        adj[si] = decay * (adj[si] + mixed[qo + keys + k] * scratch[item * 4 + 1]);
    }
    if (v == 0)
    {
        float db = 0.0f, dr = 0.0f;
        for (int j = 0; j < width; ++j)
        {
            db += scratch[(h * width + j) * 4 + 2];
            dr += scratch[(h * width + j) * 4 + 3];
        }
        float rate = q35d_sigmoid(beta[time * value_heads + h]);
        float softplus_prime = x > 20.0f ? 1.0f : x < -20.0f ? exp(x) : q35d_sigmoid(x);
        dalpha[time * value_heads + h] += dr * decay * a[h] * softplus_prime;
        dbeta[time * value_heads + h] += db * rate * (1.0f - rate);
    }
}

// A work item owns one value column, so its recurrent adjoint needs no
// inter-work-item barrier. Save the adjoint after the old prepare operation
// and before the old finish operation; q/k gradients read that exact state.
// The scratch and adjoint snapshots cover only one bounded chunk.
__kernel void q35t_delta_reverse_chunk(__global const float* mixed,
    __global const float* alpha, __global const float* beta,
    __global const float* dt, __global const float* a,
    __global const float* states, __global const float* draw,
    __global float* adj, __global float* adj_snapshots,
    __global float* scratch, __global float* dmixed,
    int first, int count, int key_heads, int value_heads, int width)
{
    int item = get_global_id(0), values = value_heads * width;
    if (item >= values) return;
    int h = item / width, v = item % width, keys = key_heads * width;
    int channels = 2 * keys + values, stride = values * width;
    int so = h * width * width + v;
    for (int step = count - 1; step >= 0; --step)
    {
        int t = first + step, qo = t * channels + (h % key_heads) * width;
        float decay = exp(a[h] * q35d_softplus(alpha[t * value_heads + h] + dt[h]));
        float rate = q35d_sigmoid(beta[t * value_heads + h]);
        float prediction = 0.0f, du = 0.0f;
        for (int k = 0; k < width; ++k)
        {
            int index = so + k * width;
            float key = mixed[qo + keys + k];
            adj[index] += mixed[qo + k] * draw[t * values + item];
            adj_snapshots[step * stride + index] = adj[index];
            du += key * adj[index];
            prediction += key * (decay * states[step * stride + index]);
        }
        float difference = mixed[t * channels + 2 * keys + item] - prediction;
        float dp = -rate * du, dr = 0.0f;
        for (int k = 0; k < width; ++k)
        {
            int index = so + k * width;
            float updated = adj[index] + mixed[qo + keys + k] * dp;
            dr += states[step * stride + index] * updated;
            adj[index] = decay * updated;
        }
        int slot = (step * values + item) * 4;
        scratch[slot] = difference * rate;
        scratch[slot + 1] = dp;
        scratch[slot + 2] = difference * du;
        scratch[slot + 3] = dr;
        dmixed[t * channels + 2 * keys + item] = rate * du;
    }
}

// One launch replaces the per-token q/k and alpha/beta launches. The q/k
// groups use the same lane traversal and reduction as the cooperative oracle.
// Alpha/beta sums remain sequential on lane zero to preserve their order.
__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void q35t_delta_chunk_grad(__global const float* alpha,
    __global const float* beta, __global const float* dt, __global const float* a,
    __global const float* states, __global const float* draw,
    __global const float* adj_snapshots, __global const float* scratch,
    __global float* dmixed, __global float* dalpha, __global float* dbeta,
    int first, int count, int key_heads, int value_heads, int width)
{
    int group = get_group_id(0), lane = get_local_id(0);
    __local float sum_q[32], sum_k[32];
    int keys = key_heads * width, values = value_heads * width;
    int qk_groups = count * keys, stride = values * width;
    int channels = 2 * keys + values;
    if (group < qk_groups)
    {
        int step = group / keys, item = group % keys, t = first + step;
        int kh = item / width, k = item % width;
        float dq = 0.0f, dk = 0.0f;
        for (int h = kh; h < value_heads; h += key_heads)
        {
            float decay = exp(a[h] * q35d_softplus(alpha[t * value_heads + h] + dt[h]));
            for (int v = lane; v < width; v += 32)
            {
                int vi = h * width + v, si = h * width * width + k * width + v;
                dq += states[(step + 1) * stride + si] * draw[t * values + vi];
                dk += scratch[(step * values + vi) * 4] * adj_snapshots[step * stride + si]
                    + (decay * states[step * stride + si]) * scratch[(step * values + vi) * 4 + 1];
            }
        }
        sum_q[lane] = dq;
        sum_k[lane] = dk;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            if (lane < offset)
            {
                sum_q[lane] += sum_q[lane + offset];
                sum_k[lane] += sum_k[lane + offset];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lane == 0)
        {
            dmixed[t * channels + item] = sum_q[0];
            dmixed[t * channels + keys + item] = sum_k[0];
        }
        return;
    }

    int alpha_group = group - qk_groups;
    if (alpha_group >= count * value_heads || lane != 0) return;
    int step = alpha_group / value_heads, h = alpha_group % value_heads;
    int t = first + step;
    float db = 0.0f, dr = 0.0f;
    for (int j = 0; j < width; ++j)
    {
        int slot = (step * values + h * width + j) * 4;
        db += scratch[slot + 2];
        dr += scratch[slot + 3];
    }
    float x = alpha[t * value_heads + h] + dt[h];
    float decay = exp(a[h] * q35d_softplus(x));
    float rate = q35d_sigmoid(beta[t * value_heads + h]);
    float softplus_prime = x > 20.0f ? 1.0f : x < -20.0f ? exp(x) : q35d_sigmoid(x);
    dalpha[t * value_heads + h] += dr * decay * a[h] * softplus_prime;
    dbeta[t * value_heads + h] += db * rate * (1.0f - rate);
}

__kernel void q35t_delta_qk_backward(__global const float* pre, __global float* dmixed,
    int sequence, int key_heads, int value_heads, int width, float eps)
{
    int item = get_global_id(0);
    if (item >= sequence * key_heads * 2) return;
    int t = item / (2 * key_heads), h = item % (2 * key_heads);
    int channels = (2 * key_heads + value_heads) * width, offset = t * channels + h * width;
    float sum = 0.0f, dot = 0.0f;
    for (int k = 0; k < width; ++k)
    {
        float x = q35d_silu(pre[offset + k]);
        sum += x * x;
        dot += x * dmixed[offset + k];
    }
    float scale = h < key_heads ? 1.0f / sqrt((float)width) : 1.0f;
    float inv = scale / sqrt(sum + eps), correction = dot / (sum + eps);
    for (int k = 0; k < width; ++k)
        dmixed[offset + k] = inv * (dmixed[offset + k] - q35d_silu(pre[offset + k]) * correction);
}

__kernel void q35t_delta_conv_backward(__global const float* pre,
    __global const float* weights, __global const float* dmixed, __global float* dx,
    int sequence, int channels, int conv_kernel)
{
    int item = get_global_id(0);
    if (item >= sequence * channels) return;
    int t = item / channels, c = item % channels;
    float sum = 0.0f;
    for (int distance = 0; distance < conv_kernel && t + distance < sequence; ++distance)
    {
        int i = (t + distance) * channels + c;
        sum += dmixed[i] * q35t_d_silu_prime(pre[i]) * weights[c * conv_kernel + conv_kernel - 1 - distance];
    }
    dx[item] += sum;
}
