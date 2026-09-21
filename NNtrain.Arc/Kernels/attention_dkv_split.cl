// Query-partitioned DKV: each partial uses ordered FP32 FMA; a second kernel
// sums partials in increasing slice order. No atomics or reduced precision.
// T must be divisible by 32*splits. Scratch is count*splits*T*64 floats.
#if defined(ARC_XMX) && ARC_SG == 16 && defined(ARC_SLM_BLOCK_IO) && defined(ARC_OPTIMIZATION_PROBES)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_local_block_io : enable
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void attention_dkv_split_partial(__global const float* qkv,__global const float* dy,
 __global const float* p,__global const float* ds,__global float* parts,
 int seq,int width,int heads,int first,int count,int splits,int causal){
 int lx=get_local_id(0),ly=get_local_id(1),tid=ly*16+lx;
 int group=get_group_id(2),g=group/splits,slice=group%splits,h=first+g,kb=get_group_id(1)*32;
 int qb=(h/heads)*seq*3*width+(h%heads)*32,yb=(h/heads)*seq*width+(h%heads)*32;
 int start=max(causal?kb:0,slice*(seq/splits)),end=(slice+1)*(seq/splits);
 __local float2 pairs[32][33];__local uint qy[32][64];
 float4 sums[2]={(float4)(0),(float4)(0)};
 for(int base=start;base<end;base+=32){
  #pragma unroll
  for(int i=tid;i<1024;i+=256){int query=i/32,key=i%32,qi=base+query,score=(g*seq+qi)*seq+kb+key;
   pairs[key][query]=(float2)(p[score],ds[score]);
   qy[query][key]=as_uint(qkv[qb+qi*3*width+key]);qy[query][key+32]=as_uint(dy[yb+qi*width+key]);}
  barrier(CLK_LOCAL_MEM_FENCE);
  #pragma unroll
  for(int inner=0;inner<32;inner++){
   float4 xy=as_float4(intel_sub_group_block_read4(&qy[inner][0]));
   #pragma unroll
   for(int r=0;r<2;r++){float2 weights=pairs[ly+16*r][inner];sums[r]=fma(weights.yyxx,xy,sums[r]);}}
  barrier(CLK_LOCAL_MEM_FENCE);
 }
 #pragma unroll
 for(int r=0;r<2;r++){int off=(group*seq+kb+ly+16*r)*64+lx;
  parts[off]=sums[r].s0;parts[off+16]=sums[r].s1;parts[off+32]=sums[r].s2;parts[off+48]=sums[r].s3;}
}
__kernel void attention_dkv_split_finish(__global const float* parts,__global float* dx,
 int seq,int width,int heads,int first,int count,int splits){
 int i=get_global_id(0);if(i>=count*seq*32)return;
 int g=i/(seq*32),key=(i/32)%seq,ch=i%32,h=first+g;
 int off=((h/heads)*seq+key)*3*width+width+(h%heads)*32+ch;
 float dk=dx[off],dv=dx[off+width];
 for(int slice=0;slice<splits;slice++){int index=((g*splits+slice)*seq+key)*64+ch;
  dk+=parts[index];dv+=parts[index+32];}
 dx[off]=dk;dx[off+width]=dv;
}
#endif
