// SG16 direct-panel GEMM experiments. There is no workgroup cooperation, so
// subgroup count per workgroup can be tuned independently from the register
// tile. Packed operands use the xmx_next_pack_a/b layouts and BF16 RNE.
#if defined(ARC_XMX) && ARC_SG == 16
float round_bf16(float x);
#ifdef cl_intel_subgroups_short
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#endif
short8 xmx_direct_tune_a(__global const ushort* p,int lane,int blockRead){
#ifdef cl_intel_subgroups_short
 if(blockRead)return as_short8(intel_sub_group_block_read_us8(p));
#endif
 short8 value;
 #pragma unroll
 for(int r=0;r<8;r++)value[r]=as_short(p[r*16+lane]);
 return value;
}
#define XMX_DIRECT_TUNE(NAME,ROW_BLOCKS,BN,SG_COUNT,BLOCK_A) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(16,SG_COUNT,1))) \
__kernel void NAME(__global const ushort* a,__global const uint* b,__global float* c, \
 __global const float* bias,__global const float* gate, \
 int m,int n,int k,int runtimeTa,int runtimeTb,int bf16Operands,int accumulate,int addBias,int relu,int gateOperand,int kStart,int kCount){ \
 int slice=get_group_id(2);kStart+=slice*kCount;c+=slice*m*n; \
 int lane=get_local_id(0),sg=get_local_id(1); \
 int row=get_group_id(1)*(SG_COUNT*ROW_BLOCKS*8)+sg*(ROW_BLOCKS*8),colBase=get_group_id(0)*BN; \
 int rows=(m+7)/8,cols=(n+15)/16,end=min(k,kStart+kCount); \
 float8 sums[ROW_BLOCKS][BN/16]; \
 _Pragma("unroll") \
 for(int rb=0;rb<ROW_BLOCKS;rb++){ \
  _Pragma("unroll") \
  for(int tile=0;tile<BN/16;tile++){ \
   sums[rb][tile]=(float8)(0);int col=colBase+tile*16+lane; \
   _Pragma("unroll") \
   for(int r=0;r<8;r++)if(row+rb*8+r<m&&col<n){if(accumulate)sums[rb][tile][r]=c[(row+rb*8+r)*n+col];if(addBias)sums[rb][tile][r]+=bias[col];} \
  } \
 } \
 for(int base=kStart;base<end;base+=16){ \
  short8 av[ROW_BLOCKS]; \
  _Pragma("unroll") \
  for(int rb=0;rb<ROW_BLOCKS;rb++){ \
   av[rb]=(short8)(0); \
   if(row+rb*8<m){av[rb]=xmx_direct_tune_a(a+((base/16)*rows+row/8+rb)*128,lane,BLOCK_A);if(base+lane>=end)av[rb]=(short8)(0);} \
  } \
  _Pragma("unroll") \
  for(int tile=0;tile<BN/16;tile++){ \
   int cb=colBase/16+tile;int8 bv=(int8)(0); \
   if(cb<cols)bv=as_int8(intel_sub_group_block_read8(b+((base/16)*cols+cb)*128)); \
   _Pragma("unroll") \
   for(int rb=0;rb<ROW_BLOCKS;rb++)sums[rb][tile]=intel_sub_group_bf16_bf16_matrix_mad_k16(av[rb],bv,sums[rb][tile]); \
  } \
 } \
 _Pragma("unroll") \
 for(int rb=0;rb<ROW_BLOCKS;rb++){ \
  _Pragma("unroll") \
  for(int tile=0;tile<BN/16;tile++){int col=colBase+tile*16+lane; \
   _Pragma("unroll") \
   for(int r=0;r<8;r++)if(row+rb*8+r<m&&col<n){float value=sums[rb][tile][r];c[(row+rb*8+r)*n+col]=relu==2?round_bf16(value):(relu?fmax(0.0f,value):value);} \
  } \
 } \
}
XMX_DIRECT_TUNE(gemm_xmx_direct_block_8x64_wg4,1,64,4,1)
XMX_DIRECT_TUNE(gemm_xmx_direct_block_8x64_wg8,1,64,8,1)
XMX_DIRECT_TUNE(gemm_xmx_direct_block_8x64_wg16,1,64,16,1)
XMX_DIRECT_TUNE(gemm_xmx_direct_scalar_8x64_wg8,1,64,8,0)
XMX_DIRECT_TUNE(gemm_xmx_direct_block_8x32_wg8,1,32,8,1)
XMX_DIRECT_TUNE(gemm_xmx_direct_block_8x32_wg16,1,32,16,1)
XMX_DIRECT_TUNE(gemm_xmx_direct_block_16x32_wg8,2,32,8,1)
XMX_DIRECT_TUNE(gemm_xmx_direct_block_16x32_wg16,2,32,16,1)
XMX_DIRECT_TUNE(gemm_xmx_direct_block_16x64_wg4,2,64,4,1)
XMX_DIRECT_TUNE(gemm_xmx_direct_block_16x64_wg8,2,64,8,1)
XMX_DIRECT_TUNE(gemm_xmx_direct_block_16x64_wg16,2,64,16,1)
XMX_DIRECT_TUNE(gemm_xmx_direct_block_32x32_wg4,4,32,4,1)
XMX_DIRECT_TUNE(gemm_xmx_direct_block_32x32_wg8,4,32,8,1)
XMX_DIRECT_TUNE(gemm_xmx_direct_block_32x32_wg16,4,32,16,1)
#undef XMX_DIRECT_TUNE
#endif
