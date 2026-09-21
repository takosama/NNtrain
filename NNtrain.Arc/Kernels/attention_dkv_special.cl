// D=32, sequence divisible by 64. All workgroups are complete, so there are
// no channel/key/query tail predicates. Causal and dense variants are compiled
// separately. QKV, DY, P/DS and DX layouts match attention_dkv_fused_candidate.
// The 64-key variants keep the reference 32-key causal FMA start for each half;
// skipping the extra zero products also preserves non-finite classification.
// global=(16,(sequence/KEYS)*16,tileHeads), local=(16,16,1).
#define ARC_DKV_SPECIAL(NAME,KEYS,QUERIES,CAUSAL,FIXED_HEADS) \
__attribute__((reqd_work_group_size(16,16,1))) \
__kernel void NAME(__global const float* qkv,__global const float* dy, \
 __global const float* p,__global const float* ds,__global float* dx, \
 int seq,int runtimeWidth,int runtimeHeads,int first,int runtimeCausal){ \
 const int heads=FIXED_HEADS?FIXED_HEADS:runtimeHeads; \
 const int width=FIXED_HEADS?FIXED_HEADS*32:runtimeWidth; \
 const int lx=get_local_id(0),ly=get_local_id(1),tid=ly*16+lx; \
 const int g=get_group_id(2),h=first+g,b=h/heads,hd=(h%heads)*32; \
 const int kb=get_group_id(1)*KEYS,pbase=g*seq*seq; \
 const int qbase=b*seq*3*width+hd,ybase=b*seq*width+hd; \
 __local float ps[KEYS][QUERIES+1],ss[KEYS][QUERIES+1],qs[QUERIES][33],ys[QUERIES][33]; \
 float2 dk[KEYS/16],dv[KEYS/16]; \
 _Pragma("unroll") \
 for(int r=0;r<KEYS/16;r++){ \
  const int off=qbase+(kb+ly+16*r)*3*width+width+lx; \
  dk[r]=(float2)(dx[off],dx[off+16]);dv[r]=(float2)(dx[off+width],dx[off+width+16]); \
 } \
 for(int base=CAUSAL?(kb/QUERIES)*QUERIES:0;base<seq;base+=QUERIES){ \
  _Pragma("unroll") \
  for(int i=tid;i<KEYS*QUERIES;i+=256){ \
   const int query=i/KEYS,key=i%KEYS,qi=base+query; \
   const int score=pbase+qi*seq+kb+key; \
   ps[key][query]=p[score];ss[key][query]=ds[score]; \
   if(KEYS==32){qs[query][key]=qkv[qbase+qi*3*width+key];ys[query][key]=dy[ybase+qi*width+key];} \
  } \
  if(KEYS!=32){ \
   _Pragma("unroll") \
   for(int i=tid;i<QUERIES*32;i+=256){ \
    const int query=i/32,channel=i%32,qi=base+query; \
    qs[query][channel]=qkv[qbase+qi*3*width+channel];ys[query][channel]=dy[ybase+qi*width+channel]; \
   } \
  } \
  barrier(CLK_LOCAL_MEM_FENCE); \
  _Pragma("unroll") \
  for(int inner=0;inner<QUERIES;inner++){ \
   const float2 qv=(float2)(qs[inner][lx],qs[inner][lx+16]); \
   const float2 yv=(float2)(ys[inner][lx],ys[inner][lx+16]); \
   _Pragma("unroll") \
   for(int r=0;r<KEYS/16;r++){ \
    if(!CAUSAL||(KEYS>=QUERIES&&r<2)||base+inner>=kb+(r/2)*32){ \
     dk[r]=fma((float2)(ss[ly+16*r][inner]),qv,dk[r]); \
     dv[r]=fma((float2)(ps[ly+16*r][inner]),yv,dv[r]); \
    } \
   } \
  } \
  barrier(CLK_LOCAL_MEM_FENCE); \
 } \
 _Pragma("unroll") \
 for(int r=0;r<KEYS/16;r++){ \
  const int off=qbase+(kb+ly+16*r)*3*width+width+lx; \
  dx[off]=dk[r].s0;dx[off+16]=dk[r].s1;dx[off+width]=dv[r].s0;dx[off+width+16]=dv[r].s1; \
 } \
}
ARC_DKV_SPECIAL(attention_dkv_d32_k32_q32_dense,32,32,0,0)
ARC_DKV_SPECIAL(attention_dkv_d32_k32_q32_causal,32,32,1,0)
ARC_DKV_SPECIAL(attention_dkv_d32_k64_q32_dense,64,32,0,0)
ARC_DKV_SPECIAL(attention_dkv_d32_k64_q32_causal,64,32,1,0)
ARC_DKV_SPECIAL(attention_dkv_d32_k32_q64_dense,32,64,0,0)
ARC_DKV_SPECIAL(attention_dkv_d32_k32_q64_causal,32,64,1,0)
ARC_DKV_SPECIAL(attention_dkv_d32_k64_q16_dense,64,16,0,0)
ARC_DKV_SPECIAL(attention_dkv_d32_k64_q16_causal,64,16,1,0)
ARC_DKV_SPECIAL(attention_dkv_d32_h16_k64_q32_dense,64,32,0,16)
ARC_DKV_SPECIAL(attention_dkv_d32_h16_k64_q32_causal,64,32,1,16)
#undef ARC_DKV_SPECIAL
