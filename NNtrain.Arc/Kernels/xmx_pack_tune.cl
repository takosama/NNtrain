// Device-only transpose/rounding experiments for the direct XMX panel layout.
// Both conversions preserve infinity and quiet every NaN, including FP32 NaN
// payloads whose only nonzero bits would otherwise be discarded by BF16.
#if defined(ARC_XMX) && ARC_SG == 16
ushort xmx_bf16(float x);
ushort xmx_pack_tune_bf16(float x){
 uint u=as_uint(x),magnitude=u&0x7fffffffu;
 uint finiteMask=0u-(uint)(magnitude<0x7f800000u);
 uint nanMask=0u-(uint)(magnitude>0x7f800000u);
 uint rounded=u+((0x7fffu+((u>>16)&1u))&finiteMask);
 return (ushort)((rounded>>16)|(nanMask&0x40u));
}
__kernel void xmx_pack_tune_roundtrip(__global const uint* input,__global ushort* original,__global ushort* branchless,int length){
 int i=get_global_id(0);if(i>=length)return;
 original[i]=xmx_bf16(as_float(input[i]));branchless[i]=xmx_pack_tune_bf16(as_float(input[i]));
}
__kernel void xmx_pack_tune_a(__global const float* source,__global const float* gate,
 __global ushort* panels,int m,int k,int transpose,int gated){
 int i=get_global_id(0),rowGroups=(m+7)/8,total=((k+15)/16)*rowGroups*128;
 if(i>=total)return;int kk=i%16,rr=(i/16)%8,rb=(i/128)%rowGroups,kb=i/(128*rowGroups);
 int row=rb*8+rr,q=kb*16+kk;float x=0;
 if(row<m&&q<k){int idx=transpose?q*m+row:row*k+q;x=gated&&gate[idx]<=0?0:source[idx];}
 panels[i]=xmx_pack_tune_bf16(x);
}
__kernel void xmx_pack_tune_b(__global const float* source,__global const float* gate,
 __global uint* panels,int n,int k,int transpose,int gated){
 int i=get_global_id(0),colGroups=(n+15)/16,total=((k+15)/16)*colGroups*128;
 if(i>=total)return;int cc=i%16,kp=(i/16)%8,cb=(i/128)%colGroups,kb=i/(128*colGroups);
 int col=cb*16+cc,q=kb*16+kp*2;float x=0,y=0;
 if(col<n&&q<k){int idx=transpose?col*k+q:q*n+col;x=gated&&gate[idx]<=0?0:source[idx];}
 if(col<n&&q+1<k){int idx=transpose?col*k+q+1:(q+1)*n+col;y=gated&&gate[idx]<=0?0:source[idx];}
 panels[i]=(uint)xmx_pack_tune_bf16(x)|((uint)xmx_pack_tune_bf16(y)<<16);
}
#define XMX_PACK_TUNE_AT(NAME,ROUND) \
__attribute__((reqd_work_group_size(256,1,1))) \
__kernel void NAME(__global const float* source,__global const float* gate, \
 __global ushort* panels,int m,int k,int transpose,int gated){ \
 int tid=get_local_id(0),rowBase=get_group_id(0)*32,kBase=get_group_id(1)*32; \
 int rowGroups=(m+7)/8,kGroups=(k+15)/16; \
 __local ushort values[32][33]; \
 for(int i=tid;i<1024;i+=256){ \
  int rr=i%32,kk=i/32,row=rowBase+rr,q=kBase+kk;float x=0; \
  if(row<m&&q<k){int idx=q*m+row;x=gated&&gate[idx]<=0?0:source[idx];} \
  values[rr][kk]=ROUND(x); \
 } \
 barrier(CLK_LOCAL_MEM_FENCE); \
 for(int i=tid;i<1024;i+=256){ \
  int rr=((i/128)%4)*8+(i/16)%8,kk=(i/512)*16+i%16; \
  int rb=(rowBase+rr)/8,kb=(kBase+kk)/16; \
  if(rb<rowGroups&&kb<kGroups)panels[(kb*rowGroups+rb)*128+(rr%8)*16+kk%16]=values[rr][kk]; \
 } \
}
XMX_PACK_TUNE_AT(xmx_pack_tune_a_transpose,xmx_bf16)
XMX_PACK_TUNE_AT(xmx_pack_tune_a_transpose_fast,xmx_pack_tune_bf16)
#undef XMX_PACK_TUNE_AT
#define XMX_PACK_TUNE_BT(NAME,ROUND) \
__attribute__((reqd_work_group_size(256,1,1))) \
__kernel void NAME(__global const float* source,__global const float* gate, \
 __global uint* panels,int n,int k,int transpose,int gated){ \
 int tid=get_local_id(0),colBase=get_group_id(0)*32,kBase=get_group_id(1)*32; \
 int colGroups=(n+15)/16,kGroups=(k+15)/16; \
 __local ushort values[32][33]; \
 for(int i=tid;i<1024;i+=256){ \
  int nn=i/32,kk=i%32,col=colBase+nn,q=kBase+kk;float x=0; \
  if(col<n&&q<k){int idx=col*k+q;x=gated&&gate[idx]<=0?0:source[idx];} \
  values[nn][kk]=ROUND(x); \
 } \
 barrier(CLK_LOCAL_MEM_FENCE); \
 for(int i=tid;i<512;i+=256){ \
  int nn=((i/128)%2)*16+i%16,kk=(i/256)*16+((i/16)%8)*2; \
  int cb=(colBase+nn)/16,kb=(kBase+kk)/16; \
  if(cb<colGroups&&kb<kGroups)panels[(kb*colGroups+cb)*128+((kk%16)/2)*16+nn%16]=(uint)values[nn][kk]|((uint)values[nn][kk+1]<<16); \
 } \
}
XMX_PACK_TUNE_BT(xmx_pack_tune_b_transpose,xmx_bf16)
XMX_PACK_TUNE_BT(xmx_pack_tune_b_transpose_fast,xmx_pack_tune_bf16)
#undef XMX_PACK_TUNE_BT
#endif
