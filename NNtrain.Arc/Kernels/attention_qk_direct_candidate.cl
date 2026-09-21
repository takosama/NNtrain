// Per-head-tile Q/K panels. Source is FP32 [batch,T,3*width]; conversion is the
// exact xmx_bf16 used by attention_xmx_panel. No host packing or persistent cache.
// D is padded to 32 (not merely 16): this preserves the reference's final pair
// of DPAS instructions, including signed-zero behavior for D<=16 and D%32 tails.
#if defined(ARC_XMX) && ARC_SG == 16
ushort xmx_bf16(float x);
short8 xmx_direct_tune_a(__global const ushort* p,int lane,int blockRead);

__kernel void attention_qk_pack_combined_candidate(__global const float* qkv,
 __global ushort* q,__global uint* k,int seq,int width,int heads,int first,int count){
 int i=get_global_id(0),d=width/heads,rows=((seq+15)/16)*16,depth=((d+31)/32)*32;
 int pairs=rows*depth/2,g=i/pairs;if(g>=count)return;
 int element=i-g*pairs,token=element/(depth/2),channel=(element%(depth/2))*2;
 int h=first+g,origin=(h/heads)*seq*3*width+(h%heads)*d+token*3*width+channel;
 ushort2 qv=(ushort2)(0),kv=(ushort2)(0);
 if(token<seq&&channel+1<d){
  float2 x=vload2(0,qkv+origin),y=vload2(0,qkv+origin+width);
  qv=(ushort2)(xmx_bf16(x.s0),xmx_bf16(x.s1));kv=(ushort2)(xmx_bf16(y.s0),xmx_bf16(y.s1));
 }else if(token<seq&&channel<d){qv.s0=xmx_bf16(qkv[origin]);kv.s0=xmx_bf16(qkv[origin+width]);}
 int qi=g*rows*depth+((channel/16)*(rows/8)+token/8)*128+(token%8)*16+channel%16;
 int ki=g*pairs+((channel/16)*(rows/16)+token/16)*128+((channel%16)/2)*16+token%16;
 vstore2(qv,0,q+qi);k[ki]=(uint)kv.s0|((uint)kv.s1<<16);
}

#define ARC_QK_DIRECT(NAME,ROW_BLOCKS,BN,EXTRA,HEAD_OFFSET) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(16,16,1))) \
__kernel void NAME(__global const ushort* q,__global const uint* k,__global float* scores, \
 int seq,int d,int add,int causal EXTRA){ \
 int lane=get_local_id(0),sg=get_local_id(1),g=get_group_id(2); \
 int paddedRows=((seq+15)/16)*16,depth=((d+31)/32)*32; \
 int row=get_group_id(1)*(128*ROW_BLOCKS)+sg*(8*ROW_BLOCKS),colBase=get_group_id(0)*BN; \
 /* Whole-subgroup branch: retain precisely the reference 128x64 published band, even with 256-row workgroups. */ \
 if(row>=seq||(causal&&colBase>=((row/128)+1)*128))return; \
 q+=(g+HEAD_OFFSET)*paddedRows*depth;k+=(g+HEAD_OFFSET)*paddedRows*depth/2;scores+=g*seq*seq; \
 int rows=paddedRows/8,cols=paddedRows/16;float8 sums[ROW_BLOCKS][BN/16]; \
 _Pragma("unroll") \
 for(int rb=0;rb<ROW_BLOCKS;rb++){ \
  _Pragma("unroll") \
  for(int tile=0;tile<BN/16;tile++){ \
   sums[rb][tile]=(float8)(0);int col=colBase+tile*16+lane; \
   _Pragma("unroll") \
   for(int r=0;r<8;r++)if(add&&row+rb*8+r<seq&&col<seq)sums[rb][tile][r]=scores[(row+rb*8+r)*seq+col]; \
  } \
 } \
 for(int base=0;base<depth;base+=16){ \
  short8 av[ROW_BLOCKS]; \
  _Pragma("unroll") \
  for(int rb=0;rb<ROW_BLOCKS;rb++){ \
   av[rb]=(short8)(0);if(row+rb*8<seq)av[rb]=xmx_direct_tune_a(q+((base/16)*rows+row/8+rb)*128,lane,1); \
  } \
  _Pragma("unroll") \
  for(int tile=0;tile<BN/16;tile++){ \
   int colBlock=colBase/16+tile;int8 bv=(int8)(0); \
   if(colBlock<cols)bv=as_int8(intel_sub_group_block_read8(k+((base/16)*cols+colBlock)*128)); \
   _Pragma("unroll") \
   for(int rb=0;rb<ROW_BLOCKS;rb++)sums[rb][tile]=intel_sub_group_bf16_bf16_matrix_mad_k16(av[rb],bv,sums[rb][tile]); \
  } \
 } \
 _Pragma("unroll") \
 for(int rb=0;rb<ROW_BLOCKS;rb++){ \
  _Pragma("unroll") \
  for(int tile=0;tile<BN/16;tile++){ \
   int col=colBase+tile*16+lane; \
   _Pragma("unroll") \
   for(int r=0;r<8;r++)if(row+rb*8+r<seq&&col<seq)scores[(row+rb*8+r)*seq+col]=sums[rb][tile][r]; \
  } \
 } \
}
ARC_QK_DIRECT(attention_qk_direct_128x64_candidate,1,64,,0)
ARC_QK_DIRECT(attention_qk_direct_128x32_candidate,1,32,,0)
ARC_QK_DIRECT(attention_qk_direct_256x32_candidate,2,32,,0)
#define ARC_QK_FIRST ,int firstHead
ARC_QK_DIRECT(attention_qk_direct_128x32_offset,1,32,ARC_QK_FIRST,firstHead)
ARC_QK_DIRECT(attention_qk_direct_256x32_offset,2,32,ARC_QK_FIRST,firstHead)
#undef ARC_QK_FIRST
#undef ARC_QK_DIRECT
#endif
