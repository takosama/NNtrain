// Exact-order subgroup replacements for the two scalar attention reductions.
// ARC_SG currently follows the XMX capability build flag, so callers must gate
// on XmxMatrices && SupportsXmx && MinimumSubgroupSize==16 until an independent
// subgroup-only compilation capability is added. Keep the legacy kernels for
// all other devices/modes. Argument and launch contracts are unchanged.
#if defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

// Every subgroup evaluates the same first 16 nodes. This deliberately does
// NOT use sub_group_reduce_add: its implementation need not reproduce the old
// 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1 tree. The two local-memory levels below
// are (a[i]+a[i+32]) + (a[i+16]+a[i+48]), with explicit shuffle levels after.
inline float attention_subgroup_sum64_exact(__local const float* values) {
    const uint lane=get_sub_group_local_id();
    float value=values[lane]+values[lane+32];
    const float upper=values[lane+16]+values[lane+48];
    value+=upper;
    #pragma unroll
    for(uint stride=8;stride;stride/=2) {
        const float other=intel_sub_group_shuffle(value,(lane+stride)&15);
        if(lane<stride)value+=other;
    }
    return intel_sub_group_shuffle(value,0);
}

inline float attention_subgroup_max64_exact(__local const float* values) {
    const uint lane=get_sub_group_local_id();
    float value=fmax(values[lane],values[lane+32]);
    const float upper=fmax(values[lane+16],values[lane+48]);
    value=fmax(value,upper);
    #pragma unroll
    for(uint stride=8;stride;stride/=2) {
        const float other=intel_sub_group_shuffle(value,(lane+stride)&15);
        if(lane<stride)value=fmax(value,other);
    }
    return intel_sub_group_shuffle(value,0);
}

// global=count*seq*64, local=64. Fixed SLM: 256 bytes.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64,1,1)))
__kernel void attention_probabilities_subgroup_candidate(
    __global float* scores,__global float* stats,
    int seq,int width,int heads,int first,int causal,int saved) {
    const int row=get_group_id(0),q=row%seq,g=row/seq,l=get_local_id(0);
    const int n=causal?q+1:seq,stat=(first+g)*seq+q;
    const int limit=causal==2?min(seq,(q/64+1)*64):seq;
    const float scale=rsqrt((float)(width/heads));
    __local float tmp[64];
    float maximum=-INFINITY;
    if(!saved) {
        for(int key=l;key<n;key+=64)maximum=fmax(maximum,scores[row*seq+key]*scale);
        tmp[l]=maximum;
        barrier(CLK_LOCAL_MEM_FENCE);
        maximum=attention_subgroup_max64_exact(tmp);
        // All four subgroups must finish reading the max tree before any
        // work-item reuses tmp for the following sum tree.
        barrier(CLK_LOCAL_MEM_FENCE);
    } else maximum=stats[2*stat];
    float sum=0;
    for(int key=l;key<limit;key+=64) {
        // Preserve the reference expression, including compiler contraction.
        const float p=key<n?exp(scores[row*seq+key]*scale-maximum):0;
        scores[row*seq+key]=p;sum+=p;
    }
    float inverse;
    if(!saved) {
        tmp[l]=sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        inverse=1.0f/attention_subgroup_sum64_exact(tmp);
        if(l==0){stats[2*stat]=maximum;stats[2*stat+1]=inverse;}
    } else inverse=stats[2*stat+1];
    for(int key=l;key<limit;key+=64)scores[row*seq+key]*=inverse;
}

// global=count*seq*64, local=64. Fixed SLM: 256 bytes.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64,1,1)))
__kernel void attention_derivatives_subgroup_candidate(
    __global const float* p,__global float* dp,
    int seq,int width,int heads,int causal) {
    const int row=get_group_id(0),q=row%seq,l=get_local_id(0),count=causal?q+1:seq;
    const int limit=causal==2?min(seq,(q/64+1)*64):seq;
    __local float tmp[64];
    float sum=0;
    for(int key=l;key<count;key+=64)sum=fma(p[row*seq+key],dp[row*seq+key],sum);
    tmp[l]=sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    const float delta=attention_subgroup_sum64_exact(tmp),scale=rsqrt((float)(width/heads));
    for(int key=l;key<limit;key+=64) {
        const int i=row*seq+key;
        dp[i]=key<count?p[i]*(dp[i]-delta)*scale:0;
    }
}
#endif
