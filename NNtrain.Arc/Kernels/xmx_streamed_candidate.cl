// Identical DPAS/FMA order to direct-panel GEMM, with an explicit output
// element offset. Row tiling writes into the original output/gradient, avoiding
// a separate add that would change FP32 accumulation order. K streaming keeps
// exactly the original 2048-element split boundaries and split-finish policy.
#if defined(ARC_XMX) && ARC_SG == 16
short8 xmx_direct_tune_a(__global const ushort* p,int lane,int blockRead);
#define XMX_STREAMED(NAME,ROW_BLOCKS,COL_TILES) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(16,16,1))) \
__kernel void NAME(__global const ushort* a,__global const uint* b,__global float* c, \
 __global const float* bias,int m,int n,int k,int accumulate,int addBias,int relu,int kStart,int kCount,int outputOffset){ \
 int slice=get_group_id(2);kStart+=slice*kCount;c+=outputOffset+slice*m*n; \
 int lane=get_local_id(0),sg=get_local_id(1); \
 int row=get_group_id(1)*(16*ROW_BLOCKS*8)+sg*(ROW_BLOCKS*8),colBase=get_group_id(0)*(16*COL_TILES); \
 int rows=(m+7)/8,cols=(n+15)/16,end=min(k,kStart+kCount); \
 float8 sums[ROW_BLOCKS][COL_TILES]; \
 _Pragma("unroll") \
 for(int rb=0;rb<ROW_BLOCKS;rb++){ \
  _Pragma("unroll") \
  for(int tile=0;tile<COL_TILES;tile++){ \
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
   if(row+rb*8<m){av[rb]=xmx_direct_tune_a(a+((base/16)*rows+row/8+rb)*128,lane,1);if(base+lane>=end)av[rb]=(short8)(0);} \
  } \
  _Pragma("unroll") \
  for(int tile=0;tile<COL_TILES;tile++){ \
   int cb=colBase/16+tile;int8 bv=(int8)(0); \
   if(cb<cols)bv=as_int8(intel_sub_group_block_read8(b+((base/16)*cols+cb)*128)); \
   _Pragma("unroll") \
   for(int rb=0;rb<ROW_BLOCKS;rb++)sums[rb][tile]=intel_sub_group_bf16_bf16_matrix_mad_k16(av[rb],bv,sums[rb][tile]); \
  } \
 } \
 _Pragma("unroll") \
 for(int rb=0;rb<ROW_BLOCKS;rb++){ \
  _Pragma("unroll") \
  for(int tile=0;tile<COL_TILES;tile++){int col=colBase+tile*16+lane; \
   _Pragma("unroll") \
   for(int r=0;r<8;r++)if(row+rb*8+r<m&&col<n)c[(row+rb*8+r)*n+col]=relu?fmax(0.0f,sums[rb][tile][r]):sums[rb][tile][r]; \
  } \
 } \
}
XMX_STREAMED(gemm_xmx_streamed_block_8x32_wg16,1,2)
XMX_STREAMED(gemm_xmx_streamed_block_16x32_wg16,2,2)
// Candidate for streamed dW: four row blocks share each loaded B panel.
// Keep the production tile selection unchanged until its wall-time A/B passes.
XMX_STREAMED(gemm_xmx_streamed_block_32x32_wg16,4,2)
// Candidate for streamed dX: four column tiles share each loaded A panel.
XMX_STREAMED(gemm_xmx_streamed_block_16x64_wg16,2,4)
#undef XMX_STREAMED
#endif
