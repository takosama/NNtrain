// Experimental T2048/D32 attention tiles. The products and epilogue keep the
// production FP32 FMA order; only panel ownership and barrier count change.
#if defined(ARC_XMX) && ARC_SG == 16 && defined(ARC_SLM_BLOCK_IO)
ushort xmx_bf16(float value);

// The production fused dP/dS tile loads K16 twice. Load the complete K32
// panel once so each workgroup uses one publication and one reuse barrier.
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void attention_mix8_16_bf16_dpds_k32_d32_2048(
    __global const float* dy, __global const ushort* qkv,
    __global uint* packed, __global const float* delta,
    int seq, int width, int heads, int first) {
    const int lx=get_local_id(0),ly=get_local_id(1),tid=ly*16+lx;
    const int g=get_group_id(2),h=first+g;
    const int rb=get_group_id(1)*64,nb=get_group_id(0)*64;
    if(nb>=rb+64)return;
    const int ap=(h/heads)*seq*width+(h%heads)*32;
    const int bp=(h/heads)*seq*3*width+(h%heads)*32+2*width;
    const int cp=g*seq*seq;
    __local float at[64][33],bt[32][65];
    float4 sums[4];
    #pragma unroll
    for(int r=0;r<4;r++)sums[r]=(float4)(0);

    const int row0=tid/8,channel=(tid%8)*4;
    #pragma unroll
    for(int part=0;part<2;part++){
        const int row=row0+32*part;
        float4 av=vload4(0,dy+ap+(rb+row)*width+channel);
        ushort4 bits=vload4(0,qkv+bp+(nb+row)*3*width+channel);
        float4 bv=as_float4(convert_uint4(bits)<<16);
        vstore4(av,0,&at[row][channel]);
        #pragma unroll
        for(int z=0;z<4;z++)bt[channel+z][row]=bv[z];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for(int inner=0;inner<32;inner++){
        float4 bv=(float4)(bt[inner][lx],bt[inner][lx+16],
            bt[inner][lx+32],bt[inner][lx+48]);
        #pragma unroll
        for(int r=0;r<4;r++)
            sums[r]=fma((float4)(at[ly+16*r][inner]),bv,sums[r]);
    }
    const float scale=rsqrt(32.0f);
    #pragma unroll
    for(int r=0;r<4;r++){
        const int row=rb+ly+16*r;
        const float rowDelta=delta[h*seq+row];
        #pragma unroll
        for(int col=0;col<4;col++){
            const int column=nb+lx+16*col;
            const int index=cp+row*seq+column;
            if(column<=row){
                float p=((__global const float*)packed)[index];
                float ds=p*(sums[r][col]-rowDelta)*scale;
                packed[index]=(uint)xmx_bf16(p)|((uint)xmx_bf16(ds)<<16);
            }else{
                packed[index]=0u;
            }
        }
    }
}

// Two adjacent 64-row causal blocks share one BF16 V panel. The upper block
// stops after its last key block; the lower block continues, so every output
// retains the production ascending-K accumulation order.
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void attention_mix8_16_bf16_pv_m128_d32_2048(
    __global const float* p, __global const ushort* qkv,__global float* output,
    int seq,int width,int heads,int first) {
    const int lx=get_local_id(0),ly=get_local_id(1),tid=ly*16+lx;
    const int g=get_group_id(2),h=first+g,rb=get_group_id(1)*128;
    const int ap=g*seq*seq;
    const int bp=(h/heads)*seq*3*width+(h%heads)*32+2*width;
    const int cp=(h/heads)*seq*width+(h%heads)*32;
    __local float at[128][33],bt[32][33];
    float2 sums[8];
    #pragma unroll
    for(int r=0;r<8;r++)sums[r]=(float2)(0);

    for(int base=0;base<rb+128;base+=32){
        const int r0=tid/8,k0=(tid%8)*4;
        #pragma unroll
        for(int part=0;part<4;part++){
            if(base<rb+64 || part>=2){
                const int row=rb+r0+32*part;
                float4 av=vload4(0,p+ap+row*seq+base+k0);
                vstore4(av,0,&at[r0+32*part][k0]);
            }
        }
        ushort4 bits=vload4(0,qkv+bp+(base+r0)*3*width+k0);
        vstore4(as_float4(convert_uint4(bits)<<16),0,&bt[r0][k0]);
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for(int inner=0;inner<32;inner++){
            float2 b=(float2)(bt[inner][lx],bt[inner][lx+16]);
            #pragma unroll
            for(int r=4;r<8;r++)
                sums[r]=fma((float2)(at[ly+16*r][inner]),b,sums[r]);
            if(base<rb+64){
                #pragma unroll
                for(int r=0;r<4;r++)
                    sums[r]=fma((float2)(at[ly+16*r][inner]),b,sums[r]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for(int r=0;r<8;r++){
        const int row=rb+ly+16*r;
        output[cp+row*width+lx]=sums[r].s0;
        output[cp+row*width+lx+16]=sums[r].s1;
    }
}

// Same two-block panel sharing for packed dS x BF16 K. The high half of each
// score word is dS; the Q-gradient seed is loaded once per output element.
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void attention_mix8_16_bf16_dq_packed_m128_2048(
    __global const uint* packed,__global const ushort* qkv,__global float* dx,
    int seq,int width,int heads,int first) {
    const int lx=get_local_id(0),ly=get_local_id(1),tid=ly*16+lx;
    const int g=get_group_id(2),h=first+g,rb=get_group_id(1)*128;
    const int ap=g*seq*seq;
    const int bp=(h/heads)*seq*3*width+(h%heads)*32+width;
    const int cp=(h/heads)*seq*3*width+(h%heads)*32;
    __local float at[128][33],bt[32][33];
    float2 sums[8];
    #pragma unroll
    for(int r=0;r<8;r++){
        const int row=rb+ly+16*r;
        sums[r]=(float2)(dx[cp+row*3*width+lx],
            dx[cp+row*3*width+lx+16]);
    }

    for(int base=0;base<rb+128;base+=32){
        const int r0=tid/8,k0=(tid%8)*4;
        #pragma unroll
        for(int part=0;part<4;part++){
            if(base<rb+64 || part>=2){
                const int row=rb+r0+32*part;
                uint4 bits=vload4(0,packed+ap+row*seq+base+k0);
                vstore4(as_float4(bits&(uint4)(0xffff0000u)),0,
                    &at[r0+32*part][k0]);
            }
        }
        ushort4 bits=vload4(0,qkv+bp+(base+r0)*3*width+k0);
        vstore4(as_float4(convert_uint4(bits)<<16),0,&bt[r0][k0]);
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for(int inner=0;inner<32;inner++){
            float2 b=(float2)(bt[inner][lx],bt[inner][lx+16]);
            #pragma unroll
            for(int r=4;r<8;r++)
                sums[r]=fma((float2)(at[ly+16*r][inner]),b,sums[r]);
            if(base<rb+64){
                #pragma unroll
                for(int r=0;r<4;r++)
                    sums[r]=fma((float2)(at[ly+16*r][inner]),b,sums[r]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for(int r=0;r<8;r++){
        const int row=rb+ly+16*r;
        dx[cp+row*3*width+lx]=sums[r].s0;
        dx[cp+row*3*width+lx+16]=sums[r].s1;
    }
}
#endif
