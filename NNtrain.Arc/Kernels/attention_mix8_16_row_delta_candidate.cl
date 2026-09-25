// Experimental T2048/D32 row-delta path. A saved BF16 copy of the unquantized
// attention output supplies delta=dY dot O. dP and the softmax derivatives stay
// FP32 until the existing optional packed P/dS publication.
#if defined(ARC_XMX) && ARC_SG == 16
ushort xmx_bf16(float x);

__kernel void attention_mix8_16_store_raw_bf16_2048(
    __global const float* raw, __global ushort* saved, int count) {
    int i=get_global_id(0);
    if(i<count)saved[i]=xmx_bf16(raw[i]);
}

__kernel void attention_mix8_16_row_delta_bf16_2048(
    __global const float* dy, __global const ushort* saved,
    __global float* delta, int seq, int width, int heads, int first, int count) {
    int i=get_global_id(0);
    if(i>=count*seq)return;
    int h=first+i/seq,t=i%seq,d=width/heads;
    int offset=((h/heads)*seq+t)*width+(h%heads)*d;
    float sum=0.0f;
    for(int c=0;c<d;c++)
        sum=fma(dy[offset+c],as_float((uint)saved[offset+c]<<16),sum);
    delta[h*seq+t]=sum;
}

// The score buffers are head tiles: g=0 starts at the passed first head.
// A work item owns one score, so in-place P→packed(P,dS) is race-free.
__kernel void attention_mix8_16_row_derivatives_fp32_2048(
    __global const float* probabilities, __global float* derivatives,
    __global const float* delta,
    int seq, int width, int heads, int first, int count, int causal) {
    int i=get_global_id(0),n=count*seq*seq;
    if(i>=n)return;
    int key=i%seq,row=i/seq,query=row%seq,g=row/seq;
    int limit=causal==2?min(seq,(query/64+1)*64):seq;
    if(key>=limit)return;
    float value=0.0f;
    if(causal==0||key<=query) {
        float p=probabilities[i];
        value=p*(derivatives[i]-delta[(first+g)*seq+query])
            *rsqrt((float)(width/heads));
    }
    derivatives[i]=value;
}

__kernel void attention_mix8_16_row_derivatives_packed_2048(
    __global uint* packed, __global const float* derivatives,
    __global const float* delta,
    int seq, int width, int heads, int first, int count, int causal) {
    int i=get_global_id(0),n=count*seq*seq;
    if(i>=n)return;
    int key=i%seq,row=i/seq,query=row%seq,g=row/seq;
    int limit=causal==2?min(seq,(query/64+1)*64):seq;
    if(key>=limit)return;
    float p=0.0f,ds=0.0f;
    if(causal==0||key<=query) {
        p=((__global const float*)packed)[i];
        ds=p*(derivatives[i]-delta[(first+g)*seq+query])
            *rsqrt((float)(width/heads));
    }
    packed[i]=(uint)xmx_bf16(p)|((uint)xmx_bf16(ds)<<16);
}

// The existing T2048/D32 dP 64x64x32 tile with the pointwise softmax
// derivative in its epilogue. P starts as FP32 and becomes packed BF16 P/dS
// in the same score slot. No FP32 dP/dS matrix is materialized.
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void attention_mix8_16_bf16_dpds_d32_2048(
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
    __local float at[64][17],bt[16][65];float4 sums[4];
    #pragma unroll
    for(int r=0;r<4;r++)sums[r]=(float4)(0);
    for(int base=0;base<32;base+=16){
        int row=tid/4,inner=(tid%4)*4;
        float4 av=vload4(0,dy+ap+(rb+row)*width+base+inner);
        ushort4 bits=vload4(0,qkv+bp+(nb+row)*3*width+base+inner);
        float4 bv=as_float4(convert_uint4(bits)<<16);
        vstore4(av,0,&at[row][inner]);
        #pragma unroll
        for(int z=0;z<4;z++)bt[inner+z][row]=bv[z];
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for(int inner=0;inner<16;inner++){
            float4 bv=(float4)(bt[inner][lx],bt[inner][lx+16],
                bt[inner][lx+32],bt[inner][lx+48]);
            #pragma unroll
            for(int r=0;r<4;r++)
                sums[r]=fma((float4)(at[ly+16*r][inner]),bv,sums[r]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float scale=rsqrt(32.0f);
    #pragma unroll
    for(int r=0;r<4;r++){
        int row=rb+ly+16*r;
        float rowDelta=delta[h*seq+row];
        #pragma unroll
        for(int col=0;col<4;col++){
            int column=nb+lx+16*col;
            int index=cp+row*seq+column;
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

#if defined(ARC_SLM_BLOCK_IO)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_local_block_io : enable

// Same ABI and packed epilogue as the FP32 fused tile above. dY is rounded
// once to BF16 in SLM; V is already physical BF16. Two K16 DPAS products
// produce dP. The 64x64 tile needs no separate operand-pack launch.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16,8,1)))
__kernel void attention_mix8_16_bf16_dpds_xmx_d32_2048(
    __global const float* dy, __global const ushort* qkv,
    __global uint* packed, __global const float* delta,
    int seq, int width, int heads, int first) {
    const int lane=get_local_id(0),sg=get_local_id(1),tid=sg*16+lane;
    const int g=get_group_id(2),h=first+g;
    const int rb=get_group_id(1)*64,nb=get_group_id(0)*64;
    if(nb>=rb+64)return;
    const int ap=(h/heads)*seq*width+(h%heads)*32;
    const int bp=(h/heads)*seq*3*width+(h%heads)*32+2*width;
    const int cp=g*seq*seq;
    __local ushort ahigh[64][33];
    __local uint bhigh[4*2*128];
    for(int i=tid;i<64*32;i+=128){
        int row=i/32,channel=i%32;
        ahigh[row][channel]=xmx_bf16(dy[ap+(rb+row)*width+channel]);
    }
    for(int i=tid;i<16*64;i+=128){
        int pair=i%16,col=i/16,channel=pair*2;
        int source=bp+(nb+col)*3*width+channel;
        ushort lo=qkv[source],hi=qkv[source+1];
        int destination=((col/16)*2+pair/8)*128+(pair%8)*16+col%16;
        bhigh[destination]=(uint)lo|((uint)hi<<16);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float8 sums[4];
    #pragma unroll
    for(int t=0;t<4;t++)sums[t]=0;
    #pragma unroll
    for(int part=0;part<2;part++){
        short8 av;
        #pragma unroll
        for(int r=0;r<8;r++)
            av[r]=as_short(ahigh[sg*8+r][part*16+lane]);
        #pragma unroll
        for(int t=0;t<4;t++){
            int8 bv=as_int8(intel_sub_group_block_read8(
                bhigh+(t*2+part)*128));
            sums[t]=intel_sub_group_bf16_bf16_matrix_mad_k16(av,bv,sums[t]);
        }
    }
    const float scale=rsqrt(32.0f);
    #pragma unroll
    for(int t=0;t<4;t++){
        int column=nb+t*16+lane;
        #pragma unroll
        for(int r=0;r<8;r++){
            int row=rb+sg*8+r,index=cp+row*seq+column;
            if(column<=row){
                float p=((__global const float*)packed)[index];
                float ds=p*(sums[t][r]-delta[h*seq+row])*scale;
                packed[index]=(uint)xmx_bf16(p)|((uint)xmx_bf16(ds)<<16);
            }else{
                packed[index]=0u;
            }
        }
    }
}
#endif
#endif
