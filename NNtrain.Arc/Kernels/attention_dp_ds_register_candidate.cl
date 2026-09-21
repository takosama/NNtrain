// Bounded FP32 dP + dS experiment: D=32, 1 <= sequence <= 1024, SG16.
// One subgroup owns a query; sixteen queries share each coalesced V tile.
// dP remains in registers until the complete softmax-gradient delta is known.
// Every dot product retains increasing-channel FP32 FMA. Four independent
// key-modulo-64 sums reproduce the original 64-lane binary reduction tree.
// No global dP temporary, atomic, approximate math, or precision conversion.
// global=(ceil(sequence/16)*256,tileHeads), local=(256,1).
// QKV=[batch,sequence,3*width], DY=[batch,sequence,width], P/DS=[tileHeads,T,T].
// Causal modes match attention_derivatives: 0 dense, 1 full zero tail,
// 2 zero only through the enclosing 64-key tile (remaining tail untouched).
#if defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(256,1,1)))
__kernel void attention_dp_ds_register_d32_candidate(
 __global const float* qkv,__global const float* dy,__global const float* p,
 __global float* ds,int seq,int width,int heads,int first,int causal){
 const int lane=get_sub_group_local_id(),r=get_sub_group_id(),tid=get_local_id(0);
 const int g=get_group_id(1),h=first+g,b=h/heads,hd=(h%heads)*32;
 const int qb=get_group_id(0)*16,query=qb+r;
 const int ybase=b*seq*width+hd,qbase=b*seq*3*width+hd+2*width;
 const int count=causal?min(seq,query+1):seq;
 const int end=causal?min(seq,qb+16):seq;
 const int published=causal==2?min(seq,(query/64+1)*64):seq;
 const int scorebase=(g*seq+query)*seq;
 __local float values[32][65];
 const float dy0=query<seq?dy[ybase+query*width+lane]:0;
 const float dy1=query<seq?dy[ybase+query*width+lane+16]:0;
 float4 partial=(float4)(0),cache[16];
 // Constant indexing permits scalar register allocation rather than an
 // addressable private-memory array. Resource/spill cost is probe-visible.
 #pragma unroll
 for(int tile=0;tile<16;tile++){
  const int base=tile*64;
  if(base<end){
   // Each of 64 source keys is fetched in contiguous float4 transactions.
   #pragma unroll
   for(int part=0;part<2;part++){
    const int packed=(tid+part*256)*4,key=packed/32,channel=packed%32;
    float4 v=(float4)(0);
    if(base+key<seq)v=vload4(0,qkv+qbase+(base+key)*3*width+channel);
    values[channel][key]=v.s0;values[channel+1][key]=v.s1;
    values[channel+2][key]=v.s2;values[channel+3][key]=v.s3;
   }
   barrier(CLK_LOCAL_MEM_FENCE);
   float4 dot=(float4)(0);
   #pragma unroll
   for(uint c=0;c<32;c++){
    const float gradient=intel_sub_group_shuffle(c<16?dy0:dy1,c&15);
    const float4 value=(float4)(values[c][lane],values[c][lane+16],values[c][lane+32],values[c][lane+48]);
    dot=fma((float4)(gradient),value,dot);
   }
   cache[tile]=dot;
   #pragma unroll
   for(int segment=0;segment<4;segment++){
    const int key=base+lane+16*segment;
    if(query<seq&&key<count)partial[segment]=fma(p[scorebase+key],dot[segment],partial[segment]);
   }
   barrier(CLK_LOCAL_MEM_FENCE);
  }
 }
 // Original strides 32,16 then 8,4,2,1. Do not reassociate these sums.
 float delta=(partial.s0+partial.s2)+(partial.s1+partial.s3);
 #pragma unroll
 for(uint stride=8;stride;stride/=2){
  const float other=intel_sub_group_shuffle(delta,(lane+stride)&15);
  if(lane<stride)delta+=other;
 }
 delta=intel_sub_group_shuffle(delta,0);
 const float scale=rsqrt((float)(width/heads));
 #pragma unroll
 for(int tile=0;tile<16;tile++){
  #pragma unroll
  for(int segment=0;segment<4;segment++){
   const int key=tile*64+lane+16*segment;
   if(query<seq&&key<published)
    ds[scorebase+key]=key<count?p[scorebase+key]*(cache[tile][segment]-delta)*scale:0;
  }
 }
}
#endif
