// Bounded row-register caching, not a different reduction algorithm.
// The 64 work-items, modulo64 FP32 streams, FMA order, max/sum tree, saved
// statistics and causal publication region are identical to the reference.
// Exact old ABIs. Dispatch only the variant matching sequence (512 or 1024).
#if defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
// Each SG redundantly combines the original 64 partials. The two cross-SG
// levels use SLM; the remaining four levels use shuffles in exactly the same
// tree. Callers fence the initial publication and any subsequent SLM reuse.
float arc_row_exact_reduce64(__local float* tmp,int l,int maximum){
 int lane=get_sub_group_local_id(),sg=get_sub_group_size();
 float a=tmp[lane],c=tmp[lane+32];
 float value=maximum?fmax(a,c):a+c;
 if(sg==16){float b=tmp[lane+16],d=tmp[lane+48];value=maximum?fmax(value,fmax(b,d)):value+(b+d);}
 #pragma unroll
 for(int stride=sg/2;stride;stride/=2){
  float other=intel_sub_group_shuffle(value,(lane+stride)&(sg-1));
  if(lane<stride)value=maximum?fmax(value,other):value+other;
 }
 return intel_sub_group_shuffle(value,0);
}
#define ARC_ROW_EXACT_SG_1 __attribute__((intel_reqd_sub_group_size(32)))
#else
float arc_row_exact_reduce64(__local float* tmp,int l,int maximum){
 for(int s=32;s;s/=2){if(l<s)tmp[l]=maximum?fmax(tmp[l],tmp[l+s]):tmp[l]+tmp[l+s];barrier(CLK_LOCAL_MEM_FENCE);}
 return tmp[0];
}
#define ARC_ROW_EXACT_SG_1
#endif
#define ARC_ROW_EXACT_SG_0
#define ARC_ROW_REGISTER_PROB(NAME,VALUES,FAST) \
ARC_ROW_EXACT_SG_##FAST \
__attribute__((reqd_work_group_size(64,1,1))) \
__kernel void NAME(__global float* scores,__global float* stats,int seq,int width,int heads,int first,int causal,int saved){ \
 int row=get_group_id(0),q=row%seq,g=row/seq,l=get_local_id(0),n=causal?q+1:seq,stat=(first+g)*seq+q; \
 int limit=causal==2?min(seq,(q/64+1)*64):seq; \
 float scale=rsqrt((float)(width/heads));__local float tmp[64];float cache[VALUES]; \
 /* Keep raw scores: pre-scaling could change exp's contracted multiply/subtract rounding. */ \
 _Pragma("unroll") \
 for(int slot=0;slot<VALUES;slot++){int key=l+slot*64;cache[slot]=key<n?scores[row*seq+key]:0;} \
 float maximum=-INFINITY; \
 if(!saved){ \
  _Pragma("unroll") \
  for(int slot=0;slot<VALUES;slot++){int key=l+slot*64;if(key<n)maximum=fmax(maximum,cache[slot]*scale);} \
  tmp[l]=maximum;barrier(CLK_LOCAL_MEM_FENCE); \
  if(FAST)maximum=arc_row_exact_reduce64(tmp,l,1); \
  else{for(int s=32;s;s/=2){if(l<s)tmp[l]=fmax(tmp[l],tmp[l+s]);barrier(CLK_LOCAL_MEM_FENCE);}maximum=tmp[0];} \
 }else maximum=stats[2*stat]; \
 barrier(CLK_LOCAL_MEM_FENCE); \
 float sum=0; \
 _Pragma("unroll") \
 for(int slot=0;slot<VALUES;slot++){int key=l+slot*64;if(key<limit){float p=key<n?exp(cache[slot]*scale-maximum):0;cache[slot]=p;sum+=p;}} \
 float inv; \
 if(!saved){ \
  tmp[l]=sum;barrier(CLK_LOCAL_MEM_FENCE); \
  float total; \
  if(FAST)total=arc_row_exact_reduce64(tmp,l,0); \
  else{for(int s=32;s;s/=2){if(l<s)tmp[l]+=tmp[l+s];barrier(CLK_LOCAL_MEM_FENCE);}total=tmp[0];} \
  inv=1.0f/total;if(l==0){stats[2*stat]=maximum;stats[2*stat+1]=inv;} \
 }else inv=stats[2*stat+1]; \
 _Pragma("unroll") \
 for(int slot=0;slot<VALUES;slot++){int key=l+slot*64;if(key<limit)scores[row*seq+key]=cache[slot]*inv;} \
}

#define ARC_ROW_REGISTER_DERIV(NAME,VALUES,FAST) \
ARC_ROW_EXACT_SG_##FAST \
__attribute__((reqd_work_group_size(64,1,1))) \
__kernel void NAME(__global const float* p,__global float* dp,int seq,int width,int heads,int causal){ \
 int row=get_group_id(0),q=row%seq,l=get_local_id(0),count=causal?q+1:seq;__local float tmp[64]; \
 int limit=causal==2?min(seq,(q/64+1)*64):seq;float sum=0,probability[VALUES],gradient[VALUES]; \
 _Pragma("unroll") \
 for(int slot=0;slot<VALUES;slot++){int key=l+slot*64;probability[slot]=0;gradient[slot]=0; \
  if(key<count){probability[slot]=p[row*seq+key];gradient[slot]=dp[row*seq+key];sum=fma(probability[slot],gradient[slot],sum);} \
 } \
 tmp[l]=sum;barrier(CLK_LOCAL_MEM_FENCE); \
 float delta; \
 if(FAST)delta=arc_row_exact_reduce64(tmp,l,0); \
 else{for(int s=32;s;s/=2){if(l<s)tmp[l]+=tmp[l+s];barrier(CLK_LOCAL_MEM_FENCE);}delta=tmp[0];} \
 float scale=rsqrt((float)(width/heads)); \
 _Pragma("unroll") \
 for(int slot=0;slot<VALUES;slot++){int key=l+slot*64;if(key<limit)dp[row*seq+key]=key<count?probability[slot]*(gradient[slot]-delta)*scale:0;} \
}

ARC_ROW_REGISTER_PROB(attention_probabilities_register_512,8,0)
ARC_ROW_REGISTER_PROB(attention_probabilities_register_1024,16,0)
ARC_ROW_REGISTER_DERIV(attention_derivatives_register_512,8,0)
ARC_ROW_REGISTER_DERIV(attention_derivatives_register_1024,16,0)
#ifdef ARC_OPTIMIZATION_PROBES
ARC_ROW_REGISTER_PROB(attention_probabilities_exact_sg_512,8,1)
ARC_ROW_REGISTER_PROB(attention_probabilities_exact_sg_1024,16,1)
ARC_ROW_REGISTER_DERIV(attention_derivatives_exact_sg_512,8,1)
ARC_ROW_REGISTER_DERIV(attention_derivatives_exact_sg_1024,16,1)
#endif
#undef ARC_ROW_REGISTER_PROB
#undef ARC_ROW_REGISTER_DERIV
#undef ARC_ROW_EXACT_SG_0
#undef ARC_ROW_EXACT_SG_1
