// Qwen3-VL vision encoder. All activations are F32; matrix weights stay in
// GGUF F16 storage. The spatial patch order is 2x2 merge-group major.

__kernel void q35v_zero(__global float * output, int count) {
    int i = get_global_id(0);
    if (i < count) output[i] = 0.0f;
}

inline float q35v_half_to_float(ushort encoded) {
    uint sign = ((uint)encoded & 0x8000u) << 16;
    uint exponent = ((uint)encoded >> 10) & 0x1fu;
    uint mantissa = (uint)encoded & 0x03ffu;
    uint bits;
    if (exponent == 0u) {
        if (mantissa == 0u) bits = sign;
        else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0u) { mantissa <<= 1; ++shift; }
            bits = sign | ((uint)(127 - 14 - shift) << 23)
                | ((mantissa & 0x03ffu) << 13);
        }
    } else if (exponent == 31u) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    return as_float(bits);
}

inline float q35v_gelu(float x) {
    const float c = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanh(c * (x + 0.044715f*x*x*x)));
}

// GGUF conv2d patch tensors have shape [patch_x,patch_y,channel,output].
// A static image repeats the same spatial patch in both temporal frames.
__kernel void q35v_patch_embed(__global const float * patches,
    __global const ushort * weight0, __global const ushort * weight1,
    __global const float * bias, __global float * output,
    int rows, int input_width, int width) {
    int id = get_global_id(0);
    if (id >= rows*width) return;
    int patch = id / width;
    int channel = id - patch*width;
    int base = channel*input_width;
    float sum = bias[channel];
    for (int k = 0; k < input_width; k++) {
        int wk = base + k;
        sum = fma(patches[patch*input_width + k],
            q35v_half_to_float(weight0[wk]) + q35v_half_to_float(weight1[wk]), sum);
    }
    output[id] = sum;
}

// Qwen3-VL resizes learned 48x48 positions with bilinear align-corners,
// then reorders them to the same 2x2 merge-group order as the patches.
__kernel void q35v_add_position(__global float * hidden,
    __global const float * positions, int rows, int width,
    int grid_h, int grid_w, int merge, int side) {
    int id = get_global_id(0);
    if (id >= rows*width) return;
    int patch = id / width;
    int feature = id - patch*width;
    int group = patch/(merge*merge);
    int local_patch = patch%(merge*merge);
    int gy = group/(grid_w/merge);
    int gx = group%(grid_w/merge);
    int y = gy*merge + local_patch/merge;
    int x = gx*merge + local_patch%merge;
    float src_y = grid_h == 1 ? 0.0f : (float)y*(float)(side-1)/(float)(grid_h-1);
    float src_x = grid_w == 1 ? 0.0f : (float)x*(float)(side-1)/(float)(grid_w-1);
    int y0 = (int)floor(src_y), x0 = (int)floor(src_x);
    int y1 = min(y0+1, side-1), x1 = min(x0+1, side-1);
    float fy = src_y-(float)y0, fx = src_x-(float)x0;
    float a = positions[(y0*side+x0)*width+feature];
    float b = positions[(y0*side+x1)*width+feature];
    float c = positions[(y1*side+x0)*width+feature];
    float d = positions[(y1*side+x1)*width+feature];
    hidden[id] += mix(mix(a,b,fx), mix(c,d,fx), fy);
}

__kernel void q35v_layer_norm(__global const float * input,
    __global const float * scale, __global const float * bias,
    __global float * output, int width, float epsilon) {
    int row = get_group_id(0);
    int lane = get_local_id(0);
    __local float scratch[256];
    float sum = 0.0f;
    for (int i = lane; i < width; i += 256) sum += input[row*width+i];
    scratch[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int step = 128; step > 0; step >>= 1) {
        if (lane < step) scratch[lane] += scratch[lane+step];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mean = scratch[0]/(float)width;
    float variance = 0.0f;
    for (int i = lane; i < width; i += 256) {
        float centered = input[row*width+i]-mean;
        variance = fma(centered,centered,variance);
    }
    scratch[lane] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int step = 128; step > 0; step >>= 1) {
        if (lane < step) scratch[lane] += scratch[lane+step];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = rsqrt(scratch[0]/(float)width+epsilon);
    for (int i = lane; i < width; i += 256)
        output[row*width+i] = (input[row*width+i]-mean)*inv*scale[i]+bias[i];
}

// Each 16x16 group computes a tile of [token, output channel].
// GGUF matrix storage has [input channel, output channel] dimensions.
__kernel void q35v_linear_f16(__global const float * input,
    __global const ushort * weight, __global const float * bias,
    __global float * output, int rows, int input_width, int output_width,
    int gelu) {
    int x = get_global_id(0), y = get_global_id(1);
    int lx = get_local_id(0), ly = get_local_id(1);
    __local float activations[16][16];
    __local float weights[16][16];
    float result = 0.0f;
    for (int base = 0; base < input_width; base += 16) {
        int ak = base+lx, bk = base+ly;
        activations[ly][lx] = y < rows && ak < input_width
            ? input[y*input_width+ak] : 0.0f;
        weights[ly][lx] = x < output_width && bk < input_width
            ? q35v_half_to_float(weight[x*input_width+bk]) : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < 16; k++)
            result = fma(activations[ly][k], weights[k][lx], result);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (x < output_width && y < rows) {
        result += bias[x];
        output[y*output_width+x] = gelu ? q35v_gelu(result) : result;
    }
}

// GGML_ROPE_TYPE_VISION rotates first/second head halves. Its two sectors
// select patch y/x and reset the frequency exponent at the sector boundary.
__kernel void q35v_rope(__global float * qkv, int rows, int width,
    int heads, int head_width, int grid_w, int merge, float theta) {
    int id = get_global_id(0);
    int pairs = head_width/2;
    if (id >= rows*heads*pairs) return;
    int pair = id%pairs;
    int head = (id/pairs)%heads;
    int patch = id/(pairs*heads);
    int group = patch/(merge*merge);
    int local_patch = patch%(merge*merge);
    int y = group/(grid_w/merge)*merge + local_patch/merge;
    int x = group%(grid_w/merge)*merge + local_patch%merge;
    int section = pairs/2;
    int coordinate = pair < section ? y : x;
    int frequency = pair < section ? pair : pair-section;
    float angle = (float)coordinate*pow(theta,-2.0f*(float)frequency/(float)pairs);
    float sine = sin(angle), cosine = cos(angle);
    int base = patch*(3*width)+head*head_width;
    for (int q_or_k = 0; q_or_k < 2; q_or_k++) {
        int offset = base+q_or_k*width;
        float a = qkv[offset+pair], b = qkv[offset+pairs+pair];
        qkv[offset+pair] = a*cosine-b*sine;
        qkv[offset+pairs+pair] = a*sine+b*cosine;
    }
}

__kernel void q35v_attention_scores(__global const float * qkv,
    __global float * scores, int rows, int heads, int head_width) {
    int id = get_global_id(0);
    int count = rows*heads*rows;
    if (id >= count) return;
    int key = id%rows;
    int head = (id/rows)%heads;
    int query = id/(rows*heads);
    int width = heads*head_width;
    int q = query*3*width+head*head_width;
    int k = key*3*width+width+head*head_width;
    float value = 0.0f;
    for (int d = 0; d < head_width; d++)
        value = fma(qkv[q+d],qkv[k+d],value);
    scores[id] = value*rsqrt((float)head_width);
}

__kernel void q35v_attention_softmax(__global float * scores,
    int rows) {
    int query_head = get_group_id(0);
    int lane = get_local_id(0);
    int base = query_head*rows;
    __local float scratch[256];
    float maximum = -INFINITY;
    for (int k = lane; k < rows; k += 256)
        maximum = fmax(maximum,scores[base+k]);
    scratch[lane] = maximum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int step = 128; step > 0; step >>= 1) {
        if (lane < step) scratch[lane] = fmax(scratch[lane],scratch[lane+step]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maximum = scratch[0];
    float sum = 0.0f;
    for (int k = lane; k < rows; k += 256) {
        float probability = exp(scores[base+k]-maximum);
        scores[base+k] = probability;
        sum += probability;
    }
    scratch[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int step = 128; step > 0; step >>= 1) {
        if (lane < step) scratch[lane] += scratch[lane+step];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inverse = 1.0f/scratch[0];
    for (int k = lane; k < rows; k += 256)
        scores[base+k] *= inverse;
}

__kernel void q35v_attention_context(__global const float * qkv,
    __global const float * scores, __global float * output,
    int rows, int heads, int head_width) {
    int id = get_global_id(0);
    int width = heads*head_width;
    if (id >= rows*width) return;
    int query = id/width;
    int feature = id-query*width;
    int head = feature/head_width;
    int score_base = (query*heads+head)*rows;
    float value = 0.0f;
    for (int key = 0; key < rows; key++)
        value = fma(scores[score_base+key],
            qkv[key*3*width+2*width+feature],value);
    output[id] = value;
}

__kernel void q35v_add_in_place(__global float * destination,
    __global const float * source, int count) {
    int i = get_global_id(0);
    if (i < count) destination[i] += source[i];
}
