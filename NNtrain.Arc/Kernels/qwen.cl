// Correctness-first Qwen2 kernels.  These intentionally keep the same
// straightforward FP32 arithmetic style as training.cl; XMX-specific
// replacements can be introduced behind the same Tensor operations later.

inline float qwen_rope_inv_freq(int pair, int width, float theta) {
    return pow(theta, -2.0f * (float)pair / (float)width);
}

inline void qwen_rope_pair(
    __global const float* x, int base, int pair, int width, int position,
    float theta, float* first, float* second) {
    int half = width / 2;
    float angle = (float)position * qwen_rope_inv_freq(pair, width, theta);
    float c = cos(angle), s = sin(angle);
    float a = x[base + pair], b = x[base + pair + half];
    *first = a * c - b * s;
    *second = b * c + a * s;
}

inline void qwen_rope_pair_backward(
    float first, float second, int pair, int width, int position, float theta,
    float* rawFirst, float* rawSecond) {
    float angle = (float)position * qwen_rope_inv_freq(pair, width, theta);
    float c = cos(angle), s = sin(angle);
    *rawFirst = first * c + second * s;
    *rawSecond = -first * s + second * c;
}

__kernel void qwen_rmsnorm(
    __global const float* x, __global const float* weight,
    __global float* y, __global float* inv_rms,
    int rows, int width, float eps) {
    int r = get_global_id(0);
    if (r >= rows) return;
    float square = 0.0f;
    for (int c = 0; c < width; ++c) {
        float v = x[r * width + c];
        square = fma(v, v, square);
    }
    float inv = rsqrt(square / (float)width + eps);
    inv_rms[r] = inv;
    for (int c = 0; c < width; ++c) {
        int i = r * width + c;
        y[i] = x[i] * inv * weight[c];
    }
}

__kernel void qwen_rmsnorm_dx(
    __global const float* x, __global const float* weight,
    __global const float* dy, __global const float* inv_rms,
    __global float* dx, int rows, int width) {
    int r = get_global_id(0);
    if (r >= rows) return;
    float inv = inv_rms[r];
    float dot = 0.0f;
    for (int c = 0; c < width; ++c) {
        int i = r * width + c;
        dot = fma(dy[i] * weight[c], x[i], dot);
    }
    float correction = inv * inv * inv * dot / (float)width;
    for (int c = 0; c < width; ++c) {
        int i = r * width + c;
        dx[i] += dy[i] * weight[c] * inv - x[i] * correction;
    }
}

__kernel void qwen_rmsnorm_dw(
    __global const float* x, __global const float* dy,
    __global const float* inv_rms, __global float* dw,
    int rows, int width) {
    int c = get_global_id(0);
    if (c >= width) return;
    float sum = 0.0f;
    for (int r = 0; r < rows; ++r)
        sum = fma(dy[r * width + c], x[r * width + c] * inv_rms[r], sum);
    dw[c] += sum;
}

__kernel void qwen_silu_mul(
    __global const float* gate, __global const float* up,
    __global float* y, int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float x = gate[i];
    float sigmoid = 1.0f / (1.0f + exp(-x));
    y[i] = (x * sigmoid) * up[i];
}

__kernel void qwen_silu_mul_back(
    __global const float* gate, __global const float* up,
    __global const float* dy, __global float* dgate,
    __global float* dup, int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float x = gate[i];
    float sigmoid = 1.0f / (1.0f + exp(-x));
    float silu = x * sigmoid;
    dgate[i] += dy[i] * up[i] * sigmoid * (1.0f + x * (1.0f - sigmoid));
    dup[i] += dy[i] * silu;
}

// Query is [B,S,H*D], key/value are [B,S,KV*D].  RoPE follows the
// Qwen/Llama NEOX layout: first and second half of each head form pairs.
__kernel void qwen_gqa(
    __global const float* query, __global const float* key,
    __global const float* value, __global float* y, __global float* prob,
    int batch, int seq, int heads, int kv_heads, int head_width,
    int causal, float theta, __local float* scores) {
    int row = get_group_id(0);
    int lane = get_local_id(0);
    int local = get_local_size(0);
    int q = row % seq;
    int h = (row / seq) % heads;
    int b = row / (seq * heads);
    int kvh = h / (heads / kv_heads);
    int q_width = heads * head_width;
    int kv_width = kv_heads * head_width;
    int qbase = (b * seq + q) * q_width + h * head_width;
    int active = causal ? q + 1 : seq;
    float scale = rsqrt((float)head_width);

    for (int kpos = lane; kpos < seq; kpos += local) {
        float dot = 0.0f;
        if (kpos < active) {
            int kbase = (b * seq + kpos) * kv_width + kvh * head_width;
            for (int p = 0; p < head_width / 2; ++p) {
                float q1, q2, k1, k2;
                qwen_rope_pair(query, qbase, p, head_width, q, theta, &q1, &q2);
                qwen_rope_pair(key, kbase, p, head_width, kpos, theta, &k1, &k2);
                dot = fma(q1, k1, dot);
                dot = fma(q2, k2, dot);
            }
            scores[kpos] = dot * scale;
        } else scores[kpos] = -INFINITY;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lane == 0) {
        float maximum = -INFINITY, sum = 0.0f;
        for (int kpos = 0; kpos < active; ++kpos)
            maximum = fmax(maximum, scores[kpos]);
        for (int kpos = 0; kpos < active; ++kpos) {
            scores[kpos] = exp(scores[kpos] - maximum);
            sum += scores[kpos];
        }
        for (int kpos = 0; kpos < active; ++kpos) scores[kpos] /= sum;
        for (int kpos = active; kpos < seq; ++kpos) scores[kpos] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kpos = lane; kpos < seq; kpos += local)
        prob[row * seq + kpos] = scores[kpos];

    for (int c = lane; c < head_width; c += local) {
        float out = 0.0f;
        for (int kpos = 0; kpos < active; ++kpos) {
            int vi = (b * seq + kpos) * kv_width + kvh * head_width + c;
            out = fma(scores[kpos], value[vi], out);
        }
        y[(b * seq + q) * q_width + h * head_width + c] = out;
    }
}

__kernel void qwen_gqa_ds(
    __global const float* value, __global const float* dy,
    __global const float* prob, __global float* ds,
    int batch, int seq, int heads, int kv_heads, int head_width,
    int causal, __local float* scratch) {
    int row = get_group_id(0);
    int lane = get_local_id(0);
    int local = get_local_size(0);
    int q = row % seq;
    int h = (row / seq) % heads;
    int b = row / (seq * heads);
    int kvh = h / (heads / kv_heads);
    int q_width = heads * head_width;
    int kv_width = kv_heads * head_width;
    int active = causal ? q + 1 : seq;

    for (int kpos = lane; kpos < seq; kpos += local) {
        float dot = 0.0f;
        if (kpos < active) {
            int vbase = (b * seq + kpos) * kv_width + kvh * head_width;
            int dybase = (b * seq + q) * q_width + h * head_width;
            for (int c = 0; c < head_width; ++c)
                dot = fma(dy[dybase + c], value[vbase + c], dot);
        }
        scratch[kpos] = dot;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lane == 0) {
        float dot = 0.0f;
        for (int kpos = 0; kpos < active; ++kpos)
            dot = fma(prob[row * seq + kpos], scratch[kpos], dot);
        scratch[seq] = dot;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float scale = rsqrt((float)head_width);
    for (int kpos = lane; kpos < seq; kpos += local) {
        float p = prob[row * seq + kpos];
        ds[row * seq + kpos] =
            kpos < active ? p * (scratch[kpos] - scratch[seq]) * scale : 0.0f;
    }
}

__kernel void qwen_gqa_dq(
    __global const float* key, __global const float* ds,
    __global float* dq, int batch, int seq, int heads, int kv_heads,
    int head_width, int causal, float theta) {
    int half = head_width / 2;
    int i = get_global_id(0);
    int total = batch * seq * heads * half;
    if (i >= total) return;
    int pair = i % half;
    int h = (i / half) % heads;
    int q = (i / (half * heads)) % seq;
    int b = i / (half * heads * seq);
    int kvh = h / (heads / kv_heads);
    int kv_width = kv_heads * head_width;
    int q_width = heads * head_width;
    int active = causal ? q + 1 : seq;
    int row = (b * heads + h) * seq + q;

    float dr1 = 0.0f, dr2 = 0.0f;
    for (int kpos = 0; kpos < active; ++kpos) {
        int kbase = (b * seq + kpos) * kv_width + kvh * head_width;
        float k1, k2;
        qwen_rope_pair(key, kbase, pair, head_width, kpos, theta, &k1, &k2);
        float g = ds[row * seq + kpos];
        dr1 = fma(g, k1, dr1);
        dr2 = fma(g, k2, dr2);
    }

    float raw1, raw2;
    qwen_rope_pair_backward(dr1, dr2, pair, head_width, q, theta, &raw1, &raw2);
    int base = (b * seq + q) * q_width + h * head_width;
    dq[base + pair] += raw1;
    dq[base + pair + half] += raw2;
}

__kernel void qwen_gqa_dkv(
    __global const float* query, __global const float* dy,
    __global const float* prob, __global const float* ds,
    __global float* dk, __global float* dv,
    int batch, int seq, int heads, int kv_heads, int head_width,
    int causal, float theta) {
    int half = head_width / 2;
    int i = get_global_id(0);
    int total = batch * seq * kv_heads * half;
    if (i >= total) return;
    int pair = i % half;
    int kvh = (i / half) % kv_heads;
    int kpos = (i / (half * kv_heads)) % seq;
    int b = i / (half * kv_heads * seq);
    int group = heads / kv_heads;
    int q_width = heads * head_width;
    int kv_width = kv_heads * head_width;

    float dkr1 = 0.0f, dkr2 = 0.0f;
    float dv1 = 0.0f, dv2 = 0.0f;
    for (int gh = 0; gh < group; ++gh) {
        int h = kvh * group + gh;
        for (int q = 0; q < seq; ++q) {
            if (causal && kpos > q) continue;
            int row = (b * heads + h) * seq + q;
            float score_grad = ds[row * seq + kpos];
            int qbase = (b * seq + q) * q_width + h * head_width;
            float q1, q2;
            qwen_rope_pair(query, qbase, pair, head_width, q, theta, &q1, &q2);
            dkr1 = fma(score_grad, q1, dkr1);
            dkr2 = fma(score_grad, q2, dkr2);

            float p = prob[row * seq + kpos];
            int dybase = (b * seq + q) * q_width + h * head_width;
            dv1 = fma(p, dy[dybase + pair], dv1);
            dv2 = fma(p, dy[dybase + pair + half], dv2);
        }
    }

    float raw1, raw2;
    qwen_rope_pair_backward(
        dkr1, dkr2, pair, head_width, kpos, theta, &raw1, &raw2);
    int base = (b * seq + kpos) * kv_width + kvh * head_width;
    dk[base + pair] += raw1;
    dk[base + pair + half] += raw2;
    dv[base + pair] += dv1;
    dv[base + pair + half] += dv2;
}
