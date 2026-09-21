// One SG16 owns a complete attention row. Four private partial streams stand
// for the old lanes [lane,lane+16,lane+32,lane+48]; each advances by64 keys.
// Combining (p0+p2)+(p1+p3), then shuffle offsets8/4/2/1, reproduces the old
// 64-lane reduction tree exactly. There are no workgroup barriers or SLM.
#if defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

inline float attention_row_sg16_sum64(float4 partial) {
    const uint lane=get_sub_group_local_id();
    float value=(partial.s0+partial.s2)+(partial.s1+partial.s3);
    #pragma unroll
    for(uint stride=8;stride;stride/=2) {
        const float other=intel_sub_group_shuffle(value,(lane+stride)&15);
        if(lane<stride)value+=other;
    }
    return intel_sub_group_shuffle(value,0);
}
inline float attention_row_sg16_max64(float4 partial) {
    const uint lane=get_sub_group_local_id();
    float value=fmax(fmax(partial.s0,partial.s2),fmax(partial.s1,partial.s3));
    #pragma unroll
    for(uint stride=8;stride;stride/=2) {
        const float other=intel_sub_group_shuffle(value,(lane+stride)&15);
        if(lane<stride)value=fmax(value,other);
    }
    return intel_sub_group_shuffle(value,0);
}

// Old probability ABI + final totalRows argument for partial last workgroups.
// global=ceil(totalRows/(WG/16))*WG, local=WG.
#define ATT_ROW_SG16_PROB(NAME,WG) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(WG,1,1))) \
__kernel void NAME(__global float* scores,__global float* stats, \
 int seq,int width,int heads,int first,int causal,int saved,int totalRows){ \
 const int lane=get_sub_group_local_id(),row=get_group_id(0)*(WG/16)+get_sub_group_id(); \
 if(row>=totalRows)return; \
 const int q=row%seq,g=row/seq,n=causal?q+1:seq,stat=(first+g)*seq+q; \
 const int limit=causal==2?min(seq,(q/64+1)*64):seq; \
 const float scale=rsqrt((float)(width/heads));float maximum; \
 if(!saved){ \
  float4 partial=(float4)(-INFINITY); \
  for(int base=0;base<n;base+=64){ \
   _Pragma("unroll") \
   for(int part=0;part<4;part++){int key=base+lane+16*part; \
    if(key<n)partial[part]=fmax(partial[part],scores[row*seq+key]*scale); \
   } \
  } \
  maximum=attention_row_sg16_max64(partial); \
 }else maximum=stats[2*stat]; \
 float4 partial=(float4)(0); \
 for(int base=0;base<limit;base+=64){ \
  _Pragma("unroll") \
  for(int part=0;part<4;part++){int key=base+lane+16*part; \
   if(key<limit){float p=key<n?exp(scores[row*seq+key]*scale-maximum):0;scores[row*seq+key]=p;partial[part]+=p;} \
  } \
 } \
 float inverse; \
 if(!saved){inverse=1.0f/attention_row_sg16_sum64(partial);if(lane==0){stats[2*stat]=maximum;stats[2*stat+1]=inverse;}} \
 else inverse=stats[2*stat+1]; \
 for(int base=0;base<limit;base+=64){ \
  _Pragma("unroll") \
  for(int part=0;part<4;part++){int key=base+lane+16*part;if(key<limit)scores[row*seq+key]*=inverse;} \
 } \
}

// Old derivative ABI + final totalRows argument. Same launch as probability.
#define ATT_ROW_SG16_DERIV(NAME,WG) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(WG,1,1))) \
__kernel void NAME(__global const float* p,__global float* dp, \
 int seq,int width,int heads,int causal,int totalRows){ \
 const int lane=get_sub_group_local_id(),row=get_group_id(0)*(WG/16)+get_sub_group_id(); \
 if(row>=totalRows)return; \
 const int q=row%seq,count=causal?q+1:seq,limit=causal==2?min(seq,(q/64+1)*64):seq; \
 float4 partial=(float4)(0); \
 for(int base=0;base<count;base+=64){ \
  _Pragma("unroll") \
  for(int part=0;part<4;part++){int key=base+lane+16*part; \
   if(key<count)partial[part]=fma(p[row*seq+key],dp[row*seq+key],partial[part]); \
  } \
 } \
 const float delta=attention_row_sg16_sum64(partial),scale=rsqrt((float)(width/heads)); \
 for(int base=0;base<limit;base+=64){ \
  _Pragma("unroll") \
  for(int part=0;part<4;part++){int key=base+lane+16*part; \
   if(key<limit){int i=row*seq+key;dp[i]=key<count?p[i]*(dp[i]-delta)*scale:0;} \
  } \
 } \
}

ATT_ROW_SG16_PROB(attention_probabilities_row_sg16_w64,64)
ATT_ROW_SG16_PROB(attention_probabilities_row_sg16_w128,128)
ATT_ROW_SG16_DERIV(attention_derivatives_row_sg16_w64,64)
ATT_ROW_SG16_DERIV(attention_derivatives_row_sg16_w128,128)
#undef ATT_ROW_SG16_PROB
#undef ATT_ROW_SG16_DERIV
#endif
