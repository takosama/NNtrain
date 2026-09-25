// Experimental T2048/D32 PV and packed-dQ tiles. Two physical BF16 channels
// used by one work item share one SLM word; A and the FP32 FMA order are the
// same as the production BM64/K32 kernels.
#if defined(ARC_XMX) && ARC_SG == 16 && defined(ARC_SLM_BLOCK_IO)

__attribute__((reqd_work_group_size(16,16,1)))
__kernel void attention_mix8_16_bf16_pv_bpair_d32_2048(
    __global const float* p,__global const ushort* qkv,__global float* output,
    int seq,int width,int heads,int first) {
    const int lx=get_local_id(0),ly=get_local_id(1),tid=ly*16+lx;
    const int g=get_group_id(2),h=first+g,rb=get_group_id(1)*64;
    const int ap=g*seq*seq;
    const int bp=(h/heads)*seq*3*width+(h%heads)*32+2*width;
    const int cp=(h/heads)*seq*width+(h%heads)*32;
    __local float at[64][33];
    __local uint bt[32][17];
    float2 sums[4];
    #pragma unroll
    for(int r=0;r<4;r++)sums[r]=(float2)(0);
    for(int base=0;base<rb+64;base+=32){
        const int r0=tid/8,k0=(tid%8)*4,channel=(tid%8)*2;
        #pragma unroll
        for(int part=0;part<2;part++){
            const int row=rb+r0+32*part;
            float4 av=vload4(0,p+ap+row*seq+base+k0);
            vstore4(av,0,&at[r0+32*part][k0]);
        }
        const int source=bp+(base+r0)*3*width+channel;
        ushort2 lo=vload2(0,qkv+source);
        ushort2 hi=vload2(0,qkv+source+16);
        bt[r0][channel]=(uint)lo.s0|((uint)hi.s0<<16);
        bt[r0][channel+1]=(uint)lo.s1|((uint)hi.s1<<16);
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for(int inner=0;inner<32;inner++){
            uint pair=bt[inner][lx];
            float2 b=(float2)(as_float((pair&0xffffu)<<16),
                as_float(pair&0xffff0000u));
            #pragma unroll
            for(int r=0;r<4;r++)
                sums[r]=fma((float2)(at[ly+16*r][inner]),b,sums[r]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for(int r=0;r<4;r++){
        const int row=rb+ly+16*r;
        output[cp+row*width+lx]=sums[r].s0;
        output[cp+row*width+lx+16]=sums[r].s1;
    }
}

__attribute__((reqd_work_group_size(16,16,1)))
__kernel void attention_mix8_16_bf16_dq_packed_bpair_2048(
    __global const uint* packed,__global const ushort* qkv,__global float* dx,
    int seq,int width,int heads,int first) {
    const int lx=get_local_id(0),ly=get_local_id(1),tid=ly*16+lx;
    const int g=get_group_id(2),h=first+g,rb=get_group_id(1)*64;
    const int ap=g*seq*seq;
    const int bp=(h/heads)*seq*3*width+(h%heads)*32+width;
    const int cp=(h/heads)*seq*3*width+(h%heads)*32;
    __local float at[64][33];
    __local uint bt[32][17];
    float2 sums[4];
    #pragma unroll
    for(int r=0;r<4;r++){
        const int row=rb+ly+16*r;
        sums[r]=(float2)(dx[cp+row*3*width+lx],
            dx[cp+row*3*width+lx+16]);
    }
    for(int base=0;base<rb+64;base+=32){
        const int r0=tid/8,k0=(tid%8)*4,channel=(tid%8)*2;
        #pragma unroll
        for(int part=0;part<2;part++){
            const int row=rb+r0+32*part;
            uint4 bits=vload4(0,packed+ap+row*seq+base+k0);
            vstore4(as_float4(bits&(uint4)(0xffff0000u)),0,
                &at[r0+32*part][k0]);
        }
        const int source=bp+(base+r0)*3*width+channel;
        ushort2 lo=vload2(0,qkv+source);
        ushort2 hi=vload2(0,qkv+source+16);
        bt[r0][channel]=(uint)lo.s0|((uint)hi.s0<<16);
        bt[r0][channel+1]=(uint)lo.s1|((uint)hi.s1<<16);
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for(int inner=0;inner<32;inner++){
            uint pair=bt[inner][lx];
            float2 b=(float2)(as_float((pair&0xffffu)<<16),
                as_float(pair&0xffff0000u));
            #pragma unroll
            for(int r=0;r<4;r++)
                sums[r]=fma((float2)(at[ly+16*r][inner]),b,sums[r]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for(int r=0;r<4;r++){
        const int row=rb+ly+16*r;
        dx[cp+row*3*width+lx]=sums[r].s0;
        dx[cp+row*3*width+lx+16]=sums[r].s1;
    }
}

// The high half of packed P/dS is already BF16. Keep it as ushort in SLM as
// well, then reconstruct the identical FP32 operand just before each FMA.
// A uses a 36-half pitch for aligned vector stores and B remains one uint per
// channel pair. This reduces the BM64/K32 SLM footprint to 6,784 bytes.
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void attention_mix8_16_bf16_dq_packed_ab_2048(
    __global const uint* packed,__global const ushort* qkv,__global float* dx,
    int seq,int width,int heads,int first) {
    const int lx=get_local_id(0),ly=get_local_id(1),tid=ly*16+lx;
    const int g=get_group_id(2),h=first+g,rb=get_group_id(1)*64;
    const int ap=g*seq*seq;
    const int bp=(h/heads)*seq*3*width+(h%heads)*32+width;
    const int cp=(h/heads)*seq*3*width+(h%heads)*32;
    __local ushort at[64][36];
    __local uint bt[32][17];
    float2 sums[4];
    #pragma unroll
    for(int r=0;r<4;r++){
        const int row=rb+ly+16*r;
        sums[r]=(float2)(dx[cp+row*3*width+lx],
            dx[cp+row*3*width+lx+16]);
    }
    for(int base=0;base<rb+64;base+=32){
        const int r0=tid/8,k0=(tid%8)*4,channel=(tid%8)*2;
        #pragma unroll
        for(int part=0;part<2;part++){
            const int row=rb+r0+32*part;
            uint4 bits=vload4(0,packed+ap+row*seq+base+k0);
            ushort4 ds=convert_ushort4(bits>>16);
            vstore4(ds,0,&at[r0+32*part][k0]);
        }
        const int source=bp+(base+r0)*3*width+channel;
        ushort2 lo=vload2(0,qkv+source);
        ushort2 hi=vload2(0,qkv+source+16);
        bt[r0][channel]=(uint)lo.s0|((uint)hi.s0<<16);
        bt[r0][channel+1]=(uint)lo.s1|((uint)hi.s1<<16);
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for(int inner=0;inner<32;inner++){
            uint pair=bt[inner][lx];
            float2 b=(float2)(as_float((pair&0xffffu)<<16),
                as_float(pair&0xffff0000u));
            #pragma unroll
            for(int r=0;r<4;r++){
                float a=as_float((uint)at[ly+16*r][inner]<<16);
                sums[r]=fma((float2)(a),b,sums[r]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for(int r=0;r<4;r++){
        const int row=rb+ly+16*r;
        dx[cp+row*3*width+lx]=sums[r].s0;
        dx[cp+row*3*width+lx+16]=sums[r].s1;
    }
}
#endif
