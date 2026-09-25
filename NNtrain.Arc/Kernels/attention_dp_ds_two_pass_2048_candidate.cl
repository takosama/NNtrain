// D=32, SG16 dP + softmax derivative. Each subgroup owns one query row.
//
// Pass 1 computes the original modulo-64 FP32 FMA partial sums. Pass 2
// recomputes one bounded 64-key dP tile at a time and immediately publishes
// dS. Keeping just a float4 dP tile per lane avoids the T=2048 register spill
// risk of the whole-row register candidate, and eliminates global dP storage.
// The two passes share only the read-only QKV, dY, and P inputs.
//
// global=(ceil(seq/16)*256,tileHeads), local=(256,1).
// QKV=[batch,seq,3*width], dY=[batch,seq,width], P/dS=[tileHeads,seq,seq].
// causal: 0=dense, 1=publish all masked keys, 2=publish through the enclosing
// 64-key tile (the region consumed by bounded causal dQ/dK/dV kernels).
#if defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(256,1,1)))
__kernel void attention_dp_ds_two_pass_d32_2048_candidate(
    __global const float* qkv, __global const float* dy,
    __global const float* p, __global float* ds,
    int seq, int width, int heads, int first, int causal) {
    const int lane=get_sub_group_local_id(), query=get_group_id(0)*16+get_sub_group_id();
    const int tid=get_local_id(0), g=get_group_id(1), h=first+g;
    const int batch=h/heads, channel=(h%heads)*32;
    const int qbase=batch*seq*3*width+channel+2*width;
    const int ybase=batch*seq*width+channel;
    const int scorebase=(g*seq+query)*seq;
    const int count=causal?min(seq,query+1):seq;
    // All 16 subgroups must execute the same number of SLM barriers.
    const int end=causal?min(seq,((int)get_group_id(0)+1)*16):seq;
    const int published=causal==2?min(seq,(query/64+1)*64):seq;
    const float dy0=query<seq?dy[ybase+query*width+lane]:0;
    const float dy1=query<seq?dy[ybase+query*width+lane+16]:0;
    __local float values[32][65];
    float4 partial=(float4)(0);

    for(int base=0;base<end;base+=64) {
        // The 256 work-items cooperatively stage 64 V rows, 32 channels.
        #pragma unroll
        for(int part=0;part<2;part++) {
            const int packed=(tid+part*256)*4;
            const int key=packed/32, c=packed%32;
            float4 v=(float4)(0);
            if(base+key<seq)v=vload4(0,qkv+qbase+(base+key)*3*width+c);
            values[c][key]=v.s0;values[c+1][key]=v.s1;
            values[c+2][key]=v.s2;values[c+3][key]=v.s3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float4 dot=(float4)(0);
        #pragma unroll
        for(int c=0;c<32;c++) {
            const float gradient=intel_sub_group_shuffle(c<16?dy0:dy1,c&15);
            const float4 v=(float4)(values[c][lane],values[c][lane+16],
                                    values[c][lane+32],values[c][lane+48]);
            dot=fma((float4)(gradient),v,dot);
        }
        #pragma unroll
        for(int segment=0;segment<4;segment++) {
            const int key=base+lane+16*segment;
            if(query<seq&&key<count)
                partial[segment]=fma(p[scorebase+key],dot[segment],partial[segment]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Same 64-lane reduction tree as attention_derivatives: strides
    // 32, 16, then 8, 4, 2, 1. No reassociation of the FMA streams.
    float delta=(partial.s0+partial.s2)+(partial.s1+partial.s3);
    #pragma unroll
    for(uint stride=8;stride;stride/=2) {
        const float other=intel_sub_group_shuffle(delta,(lane+stride)&15);
        if(lane<stride)delta+=other;
    }
    delta=intel_sub_group_shuffle(delta,0);
    const float scale=rsqrt((float)(width/heads));

    for(int base=0;base<end;base+=64) {
        #pragma unroll
        for(int part=0;part<2;part++) {
            const int packed=(tid+part*256)*4;
            const int key=packed/32, c=packed%32;
            float4 v=(float4)(0);
            if(base+key<seq)v=vload4(0,qkv+qbase+(base+key)*3*width+c);
            values[c][key]=v.s0;values[c+1][key]=v.s1;
            values[c+2][key]=v.s2;values[c+3][key]=v.s3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float4 dot=(float4)(0);
        #pragma unroll
        for(int c=0;c<32;c++) {
            const float gradient=intel_sub_group_shuffle(c<16?dy0:dy1,c&15);
            const float4 v=(float4)(values[c][lane],values[c][lane+16],
                                    values[c][lane+32],values[c][lane+48]);
            dot=fma((float4)(gradient),v,dot);
        }
        #pragma unroll
        for(int segment=0;segment<4;segment++) {
            const int key=base+lane+16*segment;
            if(query<seq&&key<published)
                ds[scorebase+key]=key<count?p[scorebase+key]*(dot[segment]-delta)*scale:0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // A causal row may need zeroes beyond its last staged V tile. In mode 2
    // this extends to the next 64-key boundary; in mode 1 it extends to seq.
    const int staged=(end+63)/64*64;
    if(query<seq)for(int key=staged+lane;key<published;key+=16)
        ds[scorebase+key]=0;
}
#endif
