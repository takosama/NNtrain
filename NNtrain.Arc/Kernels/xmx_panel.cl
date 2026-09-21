// BF16 XMX panels, FP32 accumulators. 256x128 and 128x64 output tiles.
// Pack adjacent BF16 K values once, then read a complete DPAS B operand with
// subgroup local block I/O. Only on-chip layout changes; no extra device buffer.
// Compile-time transpose directions keep global load addressing coalesced.
// Barriers separate producers/consumers; every out-of-range operand is zeroed.
#ifdef ARC_XMX
ushort xmx_bf16(float x);
int8 xmx_panel_b(const __local uint* p,int lane){
#ifdef ARC_SLM_BLOCK_IO
 return as_int8(intel_sub_group_block_read8(p));
#else
 int8 value;for(int r=0;r<8;r++)value[r]=as_int(p[r*ARC_SG+lane]);return value;
#endif
}
#if ARC_SG == 16
#define XMX_PANEL_AV short8
#define XMX_PANEL_LOAD_A(at,sg,lane,part,av) \
    _Pragma("unroll") \
    for(int r=0;r<8;r++) av[r]=as_short(at[sg*8+r][part*16+lane]);
#else
#define XMX_PANEL_AV int8
#define XMX_PANEL_LOAD_A(at,sg,lane,part,av) \
    _Pragma("unroll") \
    for(int r=0;r<8;r++) av[r]=as_int((uint)at[sg*8+r][part*16+lane*2] | ((uint)at[sg*8+r][part*16+lane*2+1]<<16));
#endif
#define XMX_PANEL(NAME,BM,BN,BK,TA,TB) \
__attribute__((intel_reqd_sub_group_size(ARC_SG))) \
__attribute__((reqd_work_group_size(ARC_SG,BM/8,1))) \
__kernel void NAME(__global const float* a,__global const float* b,__global float* c, \
 __global const float* bias,__global const float* gate, \
 int m,int n,int k,int runtimeTa,int runtimeTb,int bf16Operands,int accumulate,int addBias,int relu,int gateOperand,int kStart,int kCount){ \
 const int ta=TA,tb=TB;int slice=get_group_id(2);kStart+=slice*kCount;c+=slice*m*n; \
 int lane=get_local_id(0),sg=get_local_id(1),tid=sg*ARC_SG+lane; \
 int rowBase=get_group_id(1)*BM,colBase=get_group_id(0)*BN,row=rowBase+sg*8,end=min(k,kStart+kCount); \
 __local ushort at[BM][BK+1];__local uint bt[(BK/2)*BN]; \
 float8 sums[BN/ARC_SG]; \
 _Pragma("unroll") \
 for(int tile=0;tile<BN/ARC_SG;tile++){ \
  sums[tile]=(float8)(0);int col=colBase+tile*ARC_SG+lane; \
  _Pragma("unroll") \
  for(int r=0;r<8;r++)if(row+r<m&&col<n){if(accumulate)sums[tile][r]=c[(row+r)*n+col];if(addBias)sums[tile][r]+=bias[col];} \
 } \
 for(int base=kStart;base<end;base+=BK){ \
  for(int i=tid;i<BM*BK;i+=ARC_SG*(BM/8)){ \
   int rr=ta?i%BM:i/BK,kk=ta?i/BM:i%BK,r=rowBase+rr,q=base+kk;float x=0; \
   if(r<m&&q<end){int idx=ta?q*m+r:r*k+q;x=gateOperand==1&&gate[idx]<=0?0:a[idx];}at[rr][kk]=xmx_bf16(x); \
  } \
  for(int i=tid;i<(BK/2)*BN;i+=ARC_SG*(BM/8)){ \
   int kp=tb?i%(BK/2):i/BN,nn=tb?i/(BK/2):i%BN,q=base+kp*2,col=colBase+nn;float x=0,y=0; \
   if(col<n&&q<end){int idx=tb?col*k+q:q*n+col;x=gateOperand==2&&gate[idx]<=0?0:b[idx];} \
   if(col<n&&q+1<end){int idx=tb?col*k+q+1:(q+1)*n+col;y=gateOperand==2&&gate[idx]<=0?0:b[idx];} \
   bt[((nn/ARC_SG)*(BK/16)+kp/8)*8*ARC_SG+(kp%8)*ARC_SG+nn%ARC_SG]=(uint)xmx_bf16(x)|((uint)xmx_bf16(y)<<16); \
  } \
  barrier(CLK_LOCAL_MEM_FENCE); \
  _Pragma("unroll") \
  for(int part=0;part<BK/16;part++){ \
   XMX_PANEL_AV av;XMX_PANEL_LOAD_A(at,sg,lane,part,av) \
   _Pragma("unroll") \
   for(int tile=0;tile<BN/ARC_SG;tile++){ \
    int8 bv=xmx_panel_b(bt+(tile*(BK/16)+part)*8*ARC_SG,lane); \
    sums[tile]=intel_sub_group_bf16_bf16_matrix_mad_k16(av,bv,sums[tile]); \
   } \
  } \
  barrier(CLK_LOCAL_MEM_FENCE); \
 } \
 _Pragma("unroll") \
 for(int tile=0;tile<BN/ARC_SG;tile++){int col=colBase+tile*ARC_SG+lane; \
  _Pragma("unroll") \
  for(int r=0;r<8;r++)if(row+r<m&&col<n)c[(row+r)*n+col]=relu?fmax(0.0f,sums[tile][r]):sums[tile][r]; \
 } \
}
XMX_PANEL(gemm_xmx_wide_nn,256,128,32,0,0)
XMX_PANEL(gemm_xmx_wide_nt,256,128,32,0,1)
XMX_PANEL(gemm_xmx_wide_tn,256,128,32,1,0)
XMX_PANEL(gemm_xmx_wide_tt,256,128,32,1,1)
XMX_PANEL(gemm_xmx_narrow_nn,128,64,32,0,0)
XMX_PANEL(gemm_xmx_narrow_nt,128,64,32,0,1)
XMX_PANEL(gemm_xmx_narrow_tn,128,64,32,1,0)
XMX_PANEL(gemm_xmx_narrow_tt,128,64,32,1,1)
#undef XMX_PANEL
#undef XMX_PANEL_LOAD_A
#undef XMX_PANEL_AV
#endif
