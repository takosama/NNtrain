// The BFP8 QKV decoder already rounds its values to BF16. Retain those exact
// values as two-byte activations for the T2048/D32 training attention path.
// Each consumer expands to FP32 before the established ordered FMA loops.
#if defined(ARC_XMX) && ARC_SG == 16 && defined(ARC_SLM_BLOCK_IO)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_local_block_io : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

ushort xmx_bf16(float x);

inline float mix8_16_qkv_value(__global const ushort* values, int index) {
    return as_float((uint)values[index] << 16);
}

inline float4 mix8_16_qkv_values4(__global const ushort* values, int index) {
    return as_float4(convert_uint4(vload4(0, values + index)) << 16);
}

__kernel void attention_bfp8_decode_bf16_qk_2048(
    __global const char* payload, __global const float* scales,
    __global ushort* decoded, __global uint* packedQ, __global uint* packedK,
    int batch, int seq, int width, int heads, int block) {
    int i = get_global_id(0) * 2;
    int n = batch * seq * 3 * width;
    if (i >= n) return;
    char2 encoded = vload2(0, payload + i);
    float scale = scales[i / block];
    ushort x = xmx_bf16((float)encoded.s0 * scale);
    ushort y = xmx_bf16((float)encoded.s1 * scale);
    vstore2((ushort2)(x, y), 0, decoded + i);

    int tokenGlobal = i / (3 * width);
    int token = tokenGlobal % seq;
    int component = (i % (3 * width)) / width;
    if (component == 2) return;
    int feature = i % width;
    int head = feature / 32;
    int channel = feature % 32;
    int g = (tokenGlobal / seq) * heads + head;
    uint bits = (uint)x | ((uint)y << 16);
    if (component == 0) {
        int address = g * seq * 32
            + ((channel / 16) * (seq / 8) + token / 8) * 128
            + (token % 8) * 16 + channel % 16;
        packedQ[address / 2] = bits;
    } else {
        int pair = g * seq * 16
            + ((channel / 16) * (seq / 16) + token / 16) * 128
            + ((channel % 16) / 2) * 16 + token % 16;
        packedK[pair] = bits;
    }
}

// The T2048/D32 PV and dQ kernels use the same 64x32x32 tile, score order,
// and FP32 local-memory layout as attention_fp32_{pv,dq}_d32_aligned.
#define MIX8_16_QKV_N(NAME, COMPONENT, OUTPUT_COMPONENTS, ACCUMULATE) \
__attribute__((reqd_work_group_size(16,16,1))) \
__kernel void NAME(__global const float* a,__global const ushort* b,__global float* c, \
 int seq,int width,int heads,int first){ \
 const int lx=get_local_id(0),ly=get_local_id(1),tid=ly*16+lx,g=get_group_id(2),h=first+g; \
 const int rb=get_group_id(1)*64,ap=g*seq*seq; \
 const int bp=(h/heads)*seq*3*width+(h%heads)*32+COMPONENT*width; \
 const int cp=(h/heads)*seq*OUTPUT_COMPONENTS*width+(h%heads)*32; \
 __local float at[64][33],bt[32][33];float2 sums[4]; \
 _Pragma("unroll") \
 for(int r=0;r<4;r++){int row=rb+ly+16*r;sums[r]=(float2)(0); \
  if(ACCUMULATE)sums[r]=(float2)(c[cp+row*OUTPUT_COMPONENTS*width+lx],c[cp+row*OUTPUT_COMPONENTS*width+lx+16]);} \
 for(int base=0;base<rb+64;base+=32){ \
  const int r0=tid/8,k0=(tid%8)*4; \
  _Pragma("unroll") \
  for(int part=0;part<2;part++){int row=rb+r0+32*part; \
   float4 av=vload4(0,a+ap+row*seq+base+k0); \
   vstore4(av,0,&at[r0+32*part][k0]);} \
  float4 bv=mix8_16_qkv_values4(b,bp+(base+r0)*3*width+k0); \
  vstore4(bv,0,&bt[r0][k0]); \
  barrier(CLK_LOCAL_MEM_FENCE); \
  _Pragma("unroll") \
  for(int inner=0;inner<32;inner++){ \
   const float2 bv=(float2)(bt[inner][lx],bt[inner][lx+16]); \
   _Pragma("unroll") \
   for(int r=0;r<4;r++)sums[r]=fma((float2)(at[ly+16*r][inner]),bv,sums[r]);} \
  barrier(CLK_LOCAL_MEM_FENCE); \
 } \
 _Pragma("unroll") \
 for(int r=0;r<4;r++){int row=rb+ly+16*r; \
  c[cp+row*OUTPUT_COMPONENTS*width+lx]=sums[r].s0; \
  c[cp+row*OUTPUT_COMPONENTS*width+lx+16]=sums[r].s1;} \
}
MIX8_16_QKV_N(attention_mix8_16_bf16_pv_d32_2048,2,1,0)
MIX8_16_QKV_N(attention_mix8_16_bf16_dq_d32_2048,1,3,1)
#undef MIX8_16_QKV_N

// As in attention_fp32_dp_d32_aligned, each score tile uses 32 ordered
// products. Only the V loader and the physical QKV stride differ.
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void attention_mix8_16_bf16_dp_d32_2048(
    __global const float* dy, __global const ushort* qkv, __global float* ds,
    int seq, int width, int heads, int first) {
    const int lx=get_local_id(0),ly=get_local_id(1),tid=ly*16+lx,g=get_group_id(2),h=first+g;
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
        float4 bv=mix8_16_qkv_values4(qkv,bp+(nb+row)*3*width+base+inner);
        vstore4(av,0,&at[row][inner]);
        #pragma unroll
        for(int z=0;z<4;z++)bt[inner+z][row]=bv[z];
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for(int inner=0;inner<16;inner++){
            const float4 bv=(float4)(bt[inner][lx],bt[inner][lx+16],bt[inner][lx+32],bt[inner][lx+48]);
            #pragma unroll
            for(int r=0;r<4;r++)sums[r]=fma((float4)(at[ly+16*r][inner]),bv,sums[r]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for(int r=0;r<4;r++){
        int row=rb+ly+16*r;
        #pragma unroll
        for(int col=0;col<4;col++){
            int column=nb+lx+16*col;
            if(column/32<=row/32)ds[cp+row*seq+column]=sums[r][col];
        }
    }
}

// Causal K32/Q32 dK/dV SLM tile. Scores and gradients remain FP32;
// BF16 Q expands once into the local-memory publication for each query.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void attention_mix8_16_bf16_dkv_d32_2048(
    __global const ushort* qkv,__global const float* dy,
    __global const float* p,__global const float* ds,__global float* dx,
    int seq,int width,int heads,int first) {
    int lx=get_local_id(0),ly=get_local_id(1),tid=ly*16+lx;
    int g=get_group_id(2),h=first+g,kb=get_group_id(1)*32;
    int qb=(h/heads)*seq*3*width+(h%heads)*32,yb=(h/heads)*seq*width+(h%heads)*32;
    __local float2 pairs[32][33];__local uint qy[32][64];
    float4 sums[2];
    #pragma unroll
    for(int r=0;r<2;r++){
        int off=qb+(kb+ly+16*r)*3*width+width+lx;
        sums[r]=(float4)(dx[off],dx[off+16],dx[off+width],dx[off+width+16]);
    }
    for(int base=kb;base<seq;base+=32){
        #pragma unroll
        for(int i=tid;i<1024;i+=256){
            int query=i/32,key=i%32,score=(g*seq+base+query)*seq+kb+key;
            pairs[key][query]=(float2)(p[score],ds[score]);
            int qi=base+query;
            qy[query][key]=as_uint(mix8_16_qkv_value(qkv,qb+qi*3*width+key));
            qy[query][key+32]=as_uint(dy[yb+qi*width+key]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for(int inner=0;inner<32;inner++){
            float4 xy=as_float4(intel_sub_group_block_read4(&qy[inner][0]));
            #pragma unroll
            for(int r=0;r<2;r++){
                float2 weights=pairs[ly+16*r][inner];
                sums[r]=fma(weights.yyxx,xy,sums[r]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for(int r=0;r<2;r++){
        int off=qb+(kb+ly+16*r)*3*width+width+lx;
        dx[off]=sums[r].s0;dx[off+16]=sums[r].s1;
        dx[off+width]=sums[r].s2;dx[off+width+16]=sums[r].s3;
    }
}

// Combined with the packed P/dS backward path: read packed dS from P's high
// half and BF16 K from the decoded QKV buffer. The FP32 FMA order is unchanged.
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void attention_mix8_16_bf16_dq_packed_2048(
    __global const uint* packed, __global const ushort* qkv, __global float* dx,
    int seq, int width, int heads, int first) {
    const int lx=get_local_id(0),ly=get_local_id(1),tid=ly*16+lx;
    const int g=get_group_id(2),h=first+g,rb=get_group_id(1)*64;
    const int ap=g*seq*seq;
    const int bp=(h/heads)*seq*3*width+(h%heads)*32+width;
    const int cp=(h/heads)*seq*3*width+(h%heads)*32;
    __local float at[64][33],bt[32][33];float2 sums[4];
    #pragma unroll
    for(int r=0;r<4;r++){
        int row=rb+ly+16*r;
        sums[r]=(float2)(dx[cp+row*3*width+lx],dx[cp+row*3*width+lx+16]);
    }
    for(int base=0;base<rb+64;base+=32){
        int r0=tid/8,k0=(tid%8)*4;
        #pragma unroll
        for(int part=0;part<2;part++){
            int row=rb+r0+32*part;
            uint4 bits=vload4(0,packed+ap+row*seq+base+k0);
            float4 av=as_float4(bits&(uint4)(0xffff0000u));
            vstore4(av,0,&at[r0+32*part][k0]);
        }
        float4 bv=mix8_16_qkv_values4(qkv,bp+(base+r0)*3*width+k0);
        vstore4(bv,0,&bt[r0][k0]);
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for(int inner=0;inner<32;inner++){
            float2 b=(float2)(bt[inner][lx],bt[inner][lx+16]);
            #pragma unroll
            for(int r=0;r<4;r++)sums[r]=fma((float2)(at[ly+16*r][inner]),b,sums[r]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for(int r=0;r<4;r++){
        int row=rb+ly+16*r;
        dx[cp+row*3*width+lx]=sums[r].s0;
        dx[cp+row*3*width+lx+16]=sums[r].s1;
    }
}

// Packed score pairs remain one global word. Q is the same rounded BF16 value
// as the packed-backward kernel's FP32 Q view, with half the physical storage.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void attention_mix8_16_bf16_dkv_packed_2048(
    __global const ushort* qkv,__global const float* dy,
    __global const uint* packed,__global float* dx,
    int seq,int width,int heads,int first) {
    const int lx=get_local_id(0),ly=get_local_id(1),tid=ly*16+lx;
    const int g=get_group_id(2),h=first+g,kb=get_group_id(1)*32;
    const int qb=(h/heads)*seq*3*width+(h%heads)*32;
    const int yb=(h/heads)*seq*width+(h%heads)*32;
    __local float2 pairs[32][33];__local uint qy[32][64];
    float4 sums[2];
    #pragma unroll
    for(int r=0;r<2;r++){
        int off=qb+(kb+ly+16*r)*3*width+width+lx;
        sums[r]=(float4)(dx[off],dx[off+16],dx[off+width],dx[off+width+16]);
    }
    for(int base=kb;base<seq;base+=32){
        #pragma unroll
        for(int i=tid;i<1024;i+=256){
            int query=i/32,key=i%32;
            int score=(g*seq+base+query)*seq+kb+key;
            uint bits=packed[score];
            pairs[key][query]=(float2)(as_float((bits&0xffffu)<<16),
                as_float(bits&0xffff0000u));
            int qi=base+query;
            qy[query][key]=as_uint(mix8_16_qkv_value(qkv,qb+qi*3*width+key));
            qy[query][key+32]=as_uint(dy[yb+qi*width+key]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for(int inner=0;inner<32;inner++){
            float4 xy=as_float4(intel_sub_group_block_read4(&qy[inner][0]));
            #pragma unroll
            for(int r=0;r<2;r++){
                float2 weights=pairs[ly+16*r][inner];
                sums[r]=fma(weights.yyxx,xy,sums[r]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for(int r=0;r<2;r++){
        int off=qb+(kb+ly+16*r)*3*width+width+lx;
        dx[off]=sums[r].s0;dx[off+16]=sums[r].s1;
        dx[off+width]=sums[r].s2;dx[off+width+16]=sums[r].s3;
    }
}
#endif
