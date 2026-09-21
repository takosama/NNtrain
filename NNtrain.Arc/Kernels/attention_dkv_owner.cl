// Register-only key-owner experiment: one SG owns 16 keys and two channels.
// Each output accumulates in the original ascending query/FMA order.
#if defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define ARC_DKV_OWNER(NAME,CAUSAL) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(16,16,1))) \
__kernel void NAME(__global const float* qkv,__global const float* dy,__global const float* p,__global const float* ds,__global float* dx, \
 int seq,int width,int heads,int first,int runtimeCausal){ \
 int key=get_group_id(1)*16+get_local_id(0),ch=get_local_id(1),g=get_group_id(2),h=first+g; \
 int qb=(h/heads)*seq*3*width+(h%heads)*32,yb=(h/heads)*seq*width+(h%heads)*32,score=g*seq*seq+key; \
 int off=qb+key*3*width+width+ch; \
 float2 dk=(float2)(dx[off],dx[off+16]),dv=(float2)(dx[off+width],dx[off+width+16]); \
 for(int query=CAUSAL?(key/32)*32:0;query<seq;query++){ \
  float pr=p[score+query*seq],der=ds[score+query*seq]; \
  float2 q=(float2)(qkv[qb+query*3*width+ch],qkv[qb+query*3*width+ch+16]); \
  float2 y=(float2)(dy[yb+query*width+ch],dy[yb+query*width+ch+16]); \
  dk=fma((float2)(der),q,dk);dv=fma((float2)(pr),y,dv); \
 } \
 dx[off]=dk.s0;dx[off+16]=dk.s1;dx[off+width]=dv.s0;dx[off+width+16]=dv.s1; \
}
ARC_DKV_OWNER(attention_dkv_owner_dense,0)
ARC_DKV_OWNER(attention_dkv_owner_causal,1)
#undef ARC_DKV_OWNER
#endif
