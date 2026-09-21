// Same ascending FP32 FMA order as K32/Q32. Fewer work-items own more keys,
// reusing each Q/DY vector across register-resident output rows. No atomics,
// precision conversion or extra global workspace. SG16-only dispatch.
#if defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define ARC_DKV_REGISTER_TILE(NAME,ROWS,CAUSAL,COMPACT,INNER_UNROLL) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(16,ROWS,1))) \
__kernel void NAME(__global const float* qkv,__global const float* dy, \
 __global const float* p,__global const float* ds,__global float* dx, \
 int seq,int width,int heads,int first,int runtimeCausal){ \
 const int lx=get_local_id(0),ly=get_local_id(1),tid=ly*16+lx; \
 const int g=get_group_id(2),h=first+g,b=h/heads,hd=(h%heads)*32; \
 const int kb=get_group_id(1)*32,pbase=g*seq*seq; \
 const int qbase=b*seq*3*width+hd,ybase=b*seq*width+hd; \
 __local float ps[32][33-COMPACT],ss[32][33-COMPACT],qs[32][33-COMPACT],ys[32][33-COMPACT]; \
 float2 dk[32/ROWS],dv[32/ROWS]; \
 _Pragma("unroll") \
 for(int r=0;r<32/ROWS;r++){int off=qbase+(kb+ly+ROWS*r)*3*width+width+lx; \
  dk[r]=(float2)(dx[off],dx[off+16]);dv[r]=(float2)(dx[off+width],dx[off+width+16]);} \
 for(int base=CAUSAL?kb:0;base<seq;base+=32){ \
  _Pragma("unroll") \
  for(int i=tid;i<1024;i+=16*ROWS){int query=i/32,key=i%32,qi=base+query,score=pbase+qi*seq+kb+key; \
   ps[key][COMPACT?(query^key):query]=p[score];ss[key][COMPACT?(query^key):query]=ds[score]; \
   qs[query][key]=qkv[qbase+qi*3*width+key];ys[query][key]=dy[ybase+qi*width+key];} \
  barrier(CLK_LOCAL_MEM_FENCE); \
  _Pragma(INNER_UNROLL) \
  for(int inner=0;inner<32;inner++){ \
   float2 qv=(float2)(qs[inner][lx],qs[inner][lx+16]),yv=(float2)(ys[inner][lx],ys[inner][lx+16]); \
   _Pragma("unroll") \
   for(int r=0;r<32/ROWS;r++){ \
    const int key=ly+ROWS*r,qi=COMPACT?(inner^key):inner; \
    dk[r]=fma((float2)(ss[key][qi]),qv,dk[r]); \
    dv[r]=fma((float2)(ps[key][qi]),yv,dv[r]);} \
  } \
  barrier(CLK_LOCAL_MEM_FENCE); \
 } \
 _Pragma("unroll") \
 for(int r=0;r<32/ROWS;r++){int off=qbase+(kb+ly+ROWS*r)*3*width+width+lx; \
  dx[off]=dk[r].s0;dx[off+16]=dk[r].s1;dx[off+width]=dv[r].s0;dx[off+width+16]=dv[r].s1;} \
}
ARC_DKV_REGISTER_TILE(attention_dkv_reg8_causal,8,1,0,"unroll")
ARC_DKV_REGISTER_TILE(attention_dkv_reg8_dense,8,0,0,"unroll")
ARC_DKV_REGISTER_TILE(attention_dkv_reg4_causal,4,1,0,"unroll")
ARC_DKV_REGISTER_TILE(attention_dkv_reg4_dense,4,0,0,"unroll")
ARC_DKV_REGISTER_TILE(attention_dkv_reg2_causal,2,1,0,"unroll")
ARC_DKV_REGISTER_TILE(attention_dkv_reg2_dense,2,0,0,"unroll")
ARC_DKV_REGISTER_TILE(attention_dkv_swizzle16_causal,16,1,1,"unroll")
ARC_DKV_REGISTER_TILE(attention_dkv_swizzle16_dense,16,0,1,"unroll")
ARC_DKV_REGISTER_TILE(attention_dkv_swizzle8_causal,8,1,1,"unroll")
ARC_DKV_REGISTER_TILE(attention_dkv_swizzle8_dense,8,0,1,"unroll")
ARC_DKV_REGISTER_TILE(attention_dkv_swizzle4_causal,4,1,1,"unroll")
ARC_DKV_REGISTER_TILE(attention_dkv_swizzle4_dense,4,0,1,"unroll")
ARC_DKV_REGISTER_TILE(attention_dkv_unroll1_causal,16,1,0,"unroll 1")
ARC_DKV_REGISTER_TILE(attention_dkv_unroll1_dense,16,0,0,"unroll 1")
ARC_DKV_REGISTER_TILE(attention_dkv_unroll4_causal,16,1,0,"unroll 4")
ARC_DKV_REGISTER_TILE(attention_dkv_unroll4_dense,16,0,0,"unroll 4")
ARC_DKV_REGISTER_TILE(attention_dkv_unroll8_causal,16,1,0,"unroll 8")
ARC_DKV_REGISTER_TILE(attention_dkv_unroll8_dense,16,0,0,"unroll 8")
ARC_DKV_REGISTER_TILE(attention_dkv_unroll16_causal,16,1,0,"unroll 16")
ARC_DKV_REGISTER_TILE(attention_dkv_unroll16_dense,16,0,0,"unroll 16")
#undef ARC_DKV_REGISTER_TILE
#endif
