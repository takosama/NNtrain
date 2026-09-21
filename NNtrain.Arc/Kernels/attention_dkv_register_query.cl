// D32 ordered-FP32 DKV: stage score tiles only; keep Q/DY in registers and
// broadcast P/DS within SG16. Each output retains the reference ascending
// query FMA order, initial accumulator and enclosing K32 causal start.
#if defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#ifdef ARC_OPTIMIZATION_PROBES
#define ARC_DKV_RQ(NAME,Q,ROWS,CAUSAL) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(16,ROWS,1))) \
__kernel void NAME(__global const float* qkv,__global const float* dy, \
 __global const float* p,__global const float* ds,__global float* dx, \
 int seq,int width,int heads,int first,int runtimeCausal){ \
 const int lx=get_local_id(0),ly=get_local_id(1),tid=ly*16+lx; \
 const int g=get_group_id(2),h=first+g,kb=get_group_id(1)*32; \
 const int qb=(h/heads)*seq*3*width+(h%heads)*32,yb=(h/heads)*seq*width+(h%heads)*32; \
 __local float ps[32][Q+1],ss[32][Q+1]; \
 float2 dk[32/ROWS],dv[32/ROWS]; \
 _Pragma("unroll") \
 for(int r=0;r<32/ROWS;r++){int off=qb+(kb+ly+ROWS*r)*3*width+width+lx; \
  dk[r]=(float2)(dx[off],dx[off+16]);dv[r]=(float2)(dx[off+width],dx[off+width+16]);} \
 for(int base=CAUSAL?kb:0;base<seq;base+=Q){ \
  float2 qv[Q],yv[Q];float pv[32/ROWS][(Q+15)/16],sv[32/ROWS][(Q+15)/16]; \
  _Pragma("unroll") \
  for(int i=tid;i<32*Q;i+=16*ROWS){int query=i/32,key=i%32,score=(g*seq+base+query)*seq+kb+key; \
   ps[key][query]=p[score];ss[key][query]=ds[score];} \
  _Pragma("unroll") \
  for(int q=0;q<Q;q++){int qi=base+q; \
   qv[q]=(float2)(qkv[qb+qi*3*width+lx],qkv[qb+qi*3*width+lx+16]); \
   yv[q]=(float2)(dy[yb+qi*width+lx],dy[yb+qi*width+lx+16]);} \
  barrier(CLK_LOCAL_MEM_FENCE); \
  _Pragma("unroll") \
  for(int r=0;r<32/ROWS;r++){ \
   _Pragma("unroll") \
   for(int part=0;part<(Q+15)/16;part++){int q=part*16+(lx%min(Q,16)); \
    pv[r][part]=ps[ly+ROWS*r][q];sv[r][part]=ss[ly+ROWS*r][q];}} \
  barrier(CLK_LOCAL_MEM_FENCE); \
  _Pragma("unroll") \
  for(int q=0;q<Q;q++){ \
   _Pragma("unroll") \
   for(int r=0;r<32/ROWS;r++){ \
    float pr=intel_sub_group_shuffle(pv[r][q/16],q%16),der=intel_sub_group_shuffle(sv[r][q/16],q%16); \
    dk[r]=fma((float2)(der),qv[q],dk[r]);dv[r]=fma((float2)(pr),yv[q],dv[r]);}} \
 } \
 _Pragma("unroll") \
 for(int r=0;r<32/ROWS;r++){int off=qb+(kb+ly+ROWS*r)*3*width+width+lx; \
  dx[off]=dk[r].s0;dx[off+16]=dk[r].s1;dx[off+width]=dv[r].s0;dx[off+width+16]=dv[r].s1;} \
}
ARC_DKV_RQ(attention_dkv_rq8_causal,8,16,1)
ARC_DKV_RQ(attention_dkv_rq8_dense,8,16,0)
ARC_DKV_RQ(attention_dkv_rq16_causal,16,16,1)
ARC_DKV_RQ(attention_dkv_rq16_dense,16,16,0)
ARC_DKV_RQ(attention_dkv_rq32_causal,32,16,1)
ARC_DKV_RQ(attention_dkv_rq32_dense,32,16,0)
ARC_DKV_RQ(attention_dkv_rq16_w128_causal,16,8,1)
ARC_DKV_RQ(attention_dkv_rq16_w128_dense,16,8,0)
#undef ARC_DKV_RQ
#endif
#ifdef ARC_SLM_BLOCK_IO
// Block SLM reads issue one coalesced transaction for Q0/Q16/dY0/dY16.
// Paired P/DS are broadcast together. No arithmetic association changes.
#pragma OPENCL EXTENSION cl_intel_subgroup_local_block_io : enable
#define ARC_DKV_BLOCK_SLM(NAME,KEYS,QUERIES,CAUSAL) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(16,16,1))) \
__kernel void NAME(__global const float* qkv,__global const float* dy, \
 __global const float* p,__global const float* ds,__global float* dx, \
 int seq,int width,int heads,int first,int runtimeCausal){ \
 int lx=get_local_id(0),ly=get_local_id(1),tid=ly*16+lx; \
 int g=get_group_id(2),h=first+g,kb=get_group_id(1)*KEYS; \
 int qb=(h/heads)*seq*3*width+(h%heads)*32,yb=(h/heads)*seq*width+(h%heads)*32; \
 __local float2 pairs[KEYS][QUERIES+1];__local uint qy[QUERIES][64]; \
 float4 sums[KEYS/16]; \
 _Pragma("unroll") \
 for(int r=0;r<KEYS/16;r++){int off=qb+(kb+ly+16*r)*3*width+width+lx; \
  sums[r]=(float4)(dx[off],dx[off+16],dx[off+width],dx[off+width+16]);} \
 for(int base=CAUSAL?kb:0;base<seq;base+=QUERIES){ \
  _Pragma("unroll") \
  for(int i=tid;i<KEYS*QUERIES;i+=256){int query=i/KEYS,key=i%KEYS,score=(g*seq+base+query)*seq+kb+key; \
   pairs[key][query]=(float2)(p[score],ds[score]); \
   if(KEYS==32){int qi=base+query;qy[query][key]=as_uint(qkv[qb+qi*3*width+key]);qy[query][key+32]=as_uint(dy[yb+qi*width+key]);}} \
  if(KEYS!=32) \
  for(int i=tid;i<QUERIES*32;i+=256){int query=i/32,key=i%32,qi=base+query; \
   qy[query][key]=as_uint(qkv[qb+qi*3*width+key]);qy[query][key+32]=as_uint(dy[yb+qi*width+key]);} \
  barrier(CLK_LOCAL_MEM_FENCE); \
  _Pragma("unroll") \
  for(int inner=0;inner<QUERIES;inner++){ \
   float4 xy=as_float4(intel_sub_group_block_read4(&qy[inner][0])); \
   _Pragma("unroll") \
   for(int r=0;r<KEYS/16;r++)if(!CAUSAL||r<2||base+inner>=kb+(r/2)*32){float2 weights=pairs[ly+16*r][inner]; \
    sums[r]=fma(weights.yyxx,xy,sums[r]);}} \
  barrier(CLK_LOCAL_MEM_FENCE); \
 } \
 _Pragma("unroll") \
 for(int r=0;r<KEYS/16;r++){int off=qb+(kb+ly+16*r)*3*width+width+lx; \
  dx[off]=sums[r].s0;dx[off+16]=sums[r].s1;dx[off+width]=sums[r].s2;dx[off+width+16]=sums[r].s3;} \
}
ARC_DKV_BLOCK_SLM(attention_dkv_block_slm_causal,32,32,1)
ARC_DKV_BLOCK_SLM(attention_dkv_block_slm_dense,32,32,0)
#ifdef ARC_OPTIMIZATION_PROBES
ARC_DKV_BLOCK_SLM(attention_dkv_block_slm_k32q16_causal,32,16,1)
ARC_DKV_BLOCK_SLM(attention_dkv_block_slm_k32q16_dense,32,16,0)
ARC_DKV_BLOCK_SLM(attention_dkv_block_slm_k64q16_causal,64,16,1)
ARC_DKV_BLOCK_SLM(attention_dkv_block_slm_k64q16_dense,64,16,0)
ARC_DKV_BLOCK_SLM(attention_dkv_block_slm_k64q32_causal,64,32,1)
ARC_DKV_BLOCK_SLM(attention_dkv_block_slm_k64q32_dense,64,32,0)
ARC_DKV_BLOCK_SLM(attention_dkv_block_slm_k128q16_causal,128,16,1)
ARC_DKV_BLOCK_SLM(attention_dkv_block_slm_k128q16_dense,128,16,0)
#endif
#undef ARC_DKV_BLOCK_SLM
#endif
#endif
