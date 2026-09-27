// Low-rank branches are separate FP32 buffers; the GGUF is never rewritten.
__kernel void q35l_lora_a_coop(__global const float* x, __global const float* a,
    __global float* z, int rows, int width, int rank) {
    int item=get_group_id(0), tid=get_local_id(0);
    int t=item/rank, r=item%rank; float sum=0;
    __local float partial[128];
    for(int j=tid;j<width;j+=128) sum=fma(x[t*width+j],a[r*width+j],sum);
    partial[tid]=sum; barrier(CLK_LOCAL_MEM_FENCE);
    for(int s=64;s>0;s>>=1){if(tid<s)partial[tid]+=partial[tid+s];barrier(CLK_LOCAL_MEM_FENCE);}
    if(tid==0) z[item]=partial[0];
}

// Inference reduction variants. Keep the original WG128 training kernel intact.
#define Q35L_LORA_A_COOPERATIVE(NAME, WIDTH) \
__attribute__((reqd_work_group_size(WIDTH, 1, 1))) \
__kernel void NAME(__global const float* x, __global const float* a, \
    __global float* z, int rows, int width, int rank) { \
    int item = get_group_id(0), tid = get_local_id(0); \
    int t = item / rank, r = item % rank; \
    float sum = 0.0f; \
    __local float partial[WIDTH]; \
    for (int j = tid; j < width; j += WIDTH) sum = fma(x[t * width + j], a[r * width + j], sum); \
    partial[tid] = sum; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    for (int stride = WIDTH / 2; stride > 0; stride >>= 1) { \
        if (tid < stride) partial[tid] += partial[tid + stride]; \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    if (tid == 0) z[item] = partial[0]; \
}

Q35L_LORA_A_COOPERATIVE(q35l_lora_a_coop256, 256)
Q35L_LORA_A_COOPERATIVE(q35l_lora_a_coop512, 512)
Q35L_LORA_A_COOPERATIVE(q35l_lora_a_coop1024, 1024)
#undef Q35L_LORA_A_COOPERATIVE

#if defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
// Two rank outputs per work-group; each lane accumulates four contiguous values.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void q35l_lora_a_sg16(__global const float* x, __global const float* a,
    __global float* z, int rows, int width, int rank) {
    int item = get_group_id(0) * 2 + get_sub_group_id(), lane = get_sub_group_local_id();
    if (item >= rows * rank) return;
    int t = item / rank, r = item % rank;
    x += (size_t)t * width;
    a += (size_t)r * width;
    float4 sums = (float4)(0.0f);
    int vector_end = width & ~3;
    for (int j = lane * 4; j < vector_end; j += 64)
        sums = fma(vload4(0, x + j), vload4(0, a + j), sums);
    float sum = ((sums.x + sums.y) + sums.z) + sums.w;
    for (int j = vector_end + lane; j < width; j += 16)
        sum = fma(x[j], a[j], sum);
    float total = sub_group_reduce_add(sum);
    if (lane == 0) z[item] = total;
}
#endif

__kernel void q35l_lora_a(__global const float* x, __global const float* a,
    __global float* z, int rows, int width, int rank) {
    int i=get_global_id(0); if(i>=rows*rank) return;
    int t=i/rank, r=i%rank; float sum=0;
    for(int j=0;j<width;j++) sum=fma(x[t*width+j],a[r*width+j],sum);
    z[i]=sum;
}
__kernel void q35l_lora_b(__global const float* z, __global const float* b,
    __global float* y, int rows, int output, int rank, float scale) {
    int i=get_global_id(0); if(i>=rows*output) return;
    int t=i/output, o=i%output; float sum=0;
    for(int r=0;r<rank;r++) sum=fma(z[t*rank+r],b[o*rank+r],sum);
    y[i]+=scale*sum;
}
