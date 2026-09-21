// Experimental candidates. Not selected by normal dispatch until measured.
// Candidate 1: 16-row x 64-column register tiles, same 64 FP32 accumulator
// scalars/thread as the shipped 8x128 tile. Reuses each loaded B across two As.
// Candidate 2: device-only BF16 panels and direct global subgroup block reads.
// It removes SLM barriers at the cost of redundant cache reads; packing must
// be included in benchmarks. No persistent cache/coherence assumptions here.
#if defined(ARC_XMX) && ARC_SG == 16
ushort xmx_bf16(float x);
int8 xmx_next_local_b(const __local uint* p,int lane){
#ifdef ARC_SLM_BLOCK_IO
 return as_int8(intel_sub_group_block_read8(p));
#else
 int8 x;for(int i=0;i<8;i++)x[i]=as_int(p[i*16+lane]);return x;
#endif
}
#define XMX_NEXT_RETILE(NAME,TA,TB) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(16,32,1))) \
__kernel void NAME(__global const float* a,__global const float* b,__global float* c, \
 __global const float* bias,__global const float* gate, \
 int m,int n,int k,int runtimeTa,int runtimeTb,int bf16Operands,int accumulate,int addBias,int relu,int gateOperand,int kStart,int kCount){ \
 const int ta=TA,tb=TB;int slice=get_group_id(2);kStart+=slice*kCount;c+=slice*m*n; \
 int lane=get_local_id(0),sg=get_local_id(1),tid=sg*16+lane,rs=sg/2,cs=sg%2; \
 int rowBase=get_group_id(1)*256,colBase=get_group_id(0)*128,row=rowBase+rs*16,end=min(k,kStart+kCount); \
 __local ushort at[256][33];__local uint bt[16*128]; \
 float8 sums[2][4]; \
 _Pragma("unroll") \
 for(int rb=0;rb<2;rb++) { \
  _Pragma("unroll") \
  for(int tile=0;tile<4;tile++){ \
   sums[rb][tile]=(float8)(0);int col=colBase+cs*64+tile*16+lane; \
   _Pragma("unroll") \
   for(int r=0;r<8;r++)if(row+rb*8+r<m&&col<n){if(accumulate)sums[rb][tile][r]=c[(row+rb*8+r)*n+col];if(addBias)sums[rb][tile][r]+=bias[col];} \
  } \
 } \
 for(int base=kStart;base<end;base+=32){ \
  for(int i=tid;i<256*32;i+=512){ \
   int rr=ta?i%256:i/32,kk=ta?i/256:i%32,r=rowBase+rr,q=base+kk;float x=0; \
   if(r<m&&q<end){int idx=ta?q*m+r:r*k+q;x=gateOperand==1&&gate[idx]<=0?0:a[idx];}at[rr][kk]=xmx_bf16(x); \
  } \
  for(int i=tid;i<16*128;i+=512){ \
   int kp=tb?i%16:i/128,nn=tb?i/16:i%128,q=base+kp*2,col=colBase+nn;float x=0,y=0; \
   if(col<n&&q<end){int idx=tb?col*k+q:q*n+col;x=gateOperand==2&&gate[idx]<=0?0:b[idx];} \
   if(col<n&&q+1<end){int idx=tb?col*k+q+1:(q+1)*n+col;y=gateOperand==2&&gate[idx]<=0?0:b[idx];} \
   bt[((nn/16)*2+kp/8)*128+(kp%8)*16+nn%16]=(uint)xmx_bf16(x)|((uint)xmx_bf16(y)<<16); \
  } \
  barrier(CLK_LOCAL_MEM_FENCE); \
  _Pragma("unroll") \
  for(int part=0;part<2;part++){ \
   short8 av0,av1; \
   _Pragma("unroll") \
   for(int r=0;r<8;r++){av0[r]=as_short(at[rs*16+r][part*16+lane]);av1[r]=as_short(at[rs*16+8+r][part*16+lane]);} \
   _Pragma("unroll") \
   for(int tile=0;tile<4;tile++){ \
    int8 bv=xmx_next_local_b(bt+((cs*4+tile)*2+part)*128,lane); \
    sums[0][tile]=intel_sub_group_bf16_bf16_matrix_mad_k16(av0,bv,sums[0][tile]); \
    sums[1][tile]=intel_sub_group_bf16_bf16_matrix_mad_k16(av1,bv,sums[1][tile]); \
   } \
  } \
  barrier(CLK_LOCAL_MEM_FENCE); \
 } \
 _Pragma("unroll") \
 for(int rb=0;rb<2;rb++){ \
  _Pragma("unroll") \
  for(int tile=0;tile<4;tile++){int col=colBase+cs*64+tile*16+lane; \
   _Pragma("unroll") \
   for(int r=0;r<8;r++)if(row+rb*8+r<m&&col<n)c[(row+rb*8+r)*n+col]=relu?fmax(0.0f,sums[rb][tile][r]):sums[rb][tile][r]; \
  } \
 } \
}
XMX_NEXT_RETILE(gemm_xmx_retile_nn,0,0)
XMX_NEXT_RETILE(gemm_xmx_retile_nt,0,1)
XMX_NEXT_RETILE(gemm_xmx_retile_tn,1,0)
XMX_NEXT_RETILE(gemm_xmx_retile_tt,1,1)
#undef XMX_NEXT_RETILE

// A: [K16][M8][row8][K16] ushort. B: [K16][N16][Kpair8][N16] uint.
// Rounding is exactly the same xmx_bf16 RNE used by the existing kernel.
__kernel void xmx_next_pack_a(__global const float* source,__global const float* gate,
 __global ushort* panels,int m,int k,int transpose,int gated){
 int i=get_global_id(0),rowGroups=(m+7)/8,total=((k+15)/16)*rowGroups*128;
 if(i>=total)return;int kk=i%16,rr=(i/16)%8,rb=(i/128)%rowGroups,kb=i/(128*rowGroups);
 int row=rb*8+rr,q=kb*16+kk;float x=0;
 if(row<m&&q<k){int idx=transpose?q*m+row:row*k+q;x=gated&&gate[idx]<=0?0:source[idx];}
 panels[i]=xmx_bf16(x);
}
__kernel void xmx_next_pack_b(__global const float* source,__global const float* gate,
 __global uint* panels,int n,int k,int transpose,int gated){
 int i=get_global_id(0),colGroups=(n+15)/16,total=((k+15)/16)*colGroups*128;
 if(i>=total)return;int cc=i%16,kp=(i/16)%8,cb=(i/128)%colGroups,kb=i/(128*colGroups);
 int col=cb*16+cc,q=kb*16+kp*2;float x=0,y=0;
 if(col<n&&q<k){int idx=transpose?col*k+q:q*n+col;x=gated&&gate[idx]<=0?0:source[idx];}
 if(col<n&&q+1<k){int idx=transpose?col*k+q+1:(q+1)*n+col;y=gated&&gate[idx]<=0?0:source[idx];}
 panels[i]=(uint)xmx_bf16(x)|((uint)xmx_bf16(y)<<16);
}
#define XMX_NEXT_DIRECT(NAME,BM,BN) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(16,BM/8,1))) \
__kernel void NAME(__global const ushort* a,__global const uint* b,__global float* c, \
 __global const float* bias,__global const float* gate, \
 int m,int n,int k,int runtimeTa,int runtimeTb,int bf16Operands,int accumulate,int addBias,int relu,int gateOperand,int kStart,int kCount){ \
 int slice=get_group_id(2);kStart+=slice*kCount;c+=slice*m*n; \
 int lane=get_local_id(0),sg=get_local_id(1); \
 int row=get_group_id(1)*BM+sg*8,colBase=get_group_id(0)*BN; \
 int rows=(m+7)/8,cols=(n+15)/16,end=min(k,kStart+kCount); \
 float8 sums[BN/16]; \
 _Pragma("unroll") \
 for(int tile=0;tile<BN/16;tile++){ \
  sums[tile]=(float8)(0);int col=colBase+tile*16+lane; \
  _Pragma("unroll") \
  for(int r=0;r<8;r++)if(row+r<m&&col<n){if(accumulate)sums[tile][r]=c[(row+r)*n+col];if(addBias)sums[tile][r]+=bias[col];} \
 } \
 for(int base=kStart;base<end;base+=16){ \
  short8 av=(short8)(0); \
  if(row<m){ \
   __global const ushort* ap=a+((base/16)*rows+row/8)*128; \
   _Pragma("unroll") \
   for(int r=0;r<8;r++)av[r]=base+lane<end?as_short(ap[r*16+lane]):0; \
  } \
  _Pragma("unroll") \
  for(int tile=0;tile<BN/16;tile++){ \
   int cb=colBase/16+tile;int8 bv=(int8)(0); \
   if(cb<cols)bv=as_int8(intel_sub_group_block_read8(b+((base/16)*cols+cb)*128)); \
   sums[tile]=intel_sub_group_bf16_bf16_matrix_mad_k16(av,bv,sums[tile]); \
  } \
 } \
 _Pragma("unroll") \
 for(int tile=0;tile<BN/16;tile++){int col=colBase+tile*16+lane; \
  _Pragma("unroll") \
  for(int r=0;r<8;r++)if(row+r<m&&col<n)c[(row+r)*n+col]=relu?fmax(0.0f,sums[tile][r]):sums[tile][r]; \
 } \
}
XMX_NEXT_DIRECT(gemm_xmx_direct_wide,256,128)
XMX_NEXT_DIRECT(gemm_xmx_direct_narrow,128,64)
#undef XMX_NEXT_DIRECT
#endif
