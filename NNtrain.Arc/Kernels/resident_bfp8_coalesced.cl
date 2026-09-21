// One SG16 owns one quantization block. Max/finite reductions are exact;
// division and nearest-even rounding remain identical to resident_bfp8.
#if defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define ARC_BFP8_COALESCED(NAME,VALUES) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(256,1,1))) \
__kernel void NAME(__global const float* x,__global char* y,__global float* scales,__global int* status,int n,int block){ \
 int b=get_group_id(0)*16+get_sub_group_id(),l=get_sub_group_local_id(),start=b*block; \
 if(start>=n)return; \
 float cache[VALUES],maximum=0;int valid=1; \
 _Pragma("unroll") \
 for(int j=0;j<VALUES;j++){int i=start+l+j*16;float v=i<n?x[i]:0;cache[j]=v;maximum=fmax(maximum,fabs(v));valid&=isfinite(v);} \
 _Pragma("unroll") \
 for(int stride=8;stride;stride/=2){maximum=fmax(maximum,intel_sub_group_shuffle(maximum,l^stride));valid&=intel_sub_group_shuffle(valid,l^stride);} \
 float scale=maximum==0?1:maximum/127.0f; \
 valid&=scale>0&&isfinite(scale); \
 if(l==0){scales[b]=valid?scale:NAN;if(!valid)atomic_or(status,1);} \
 _Pragma("unroll") \
 for(int j=0;j<VALUES;j++){int i=start+l+j*16;if(i<n)y[i]=valid?convert_char_sat_rte(clamp(cache[j]/scale,-127.0f,127.0f)):0;} \
}
ARC_BFP8_COALESCED(resident_bfp8_sg16_32,2)
ARC_BFP8_COALESCED(resident_bfp8_sg16_64,4)
ARC_BFP8_COALESCED(resident_bfp8_sg16_128,8)
ARC_BFP8_COALESCED(resident_bfp8_sg16_256,16)
#undef ARC_BFP8_COALESCED

#define ARC_BFP8_QUAD(NAME,LANES) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(256,1,1))) \
__kernel void NAME(__global const float* x,__global char* y,__global float* scales,__global int* status,int n,int block){ \
 int b=get_global_id(0)/LANES,l=get_global_id(0)%LANES,start=b*32; \
 /* Do not exit part of a subgroup before its shuffle collectives. */ \
 float cache[32/LANES],maximum=0;int valid=1; \
 _Pragma("unroll") \
 for(int j=0;j<32/LANES;j++){int i=start+l+j*LANES;float v=i<n?x[i]:0;cache[j]=v;maximum=fmax(maximum,fabs(v));valid&=isfinite(v);} \
 _Pragma("unroll") \
 for(int stride=LANES/2;stride;stride/=2){int other=get_sub_group_local_id()^stride;maximum=fmax(maximum,intel_sub_group_shuffle(maximum,other));valid&=intel_sub_group_shuffle(valid,other);} \
 float scale=maximum==0?1:maximum/127.0f;valid&=scale>0&&isfinite(scale); \
 if(l==0&&start<n){scales[b]=valid?scale:NAN;if(!valid)atomic_or(status,1);} \
 _Pragma("unroll") \
 for(int j=0;j<32/LANES;j++){int i=start+l+j*LANES;if(i<n)y[i]=valid?convert_char_sat_rte(clamp(cache[j]/scale,-127.0f,127.0f)):0;} \
}
ARC_BFP8_QUAD(resident_bfp8_quad2_32,2)
ARC_BFP8_QUAD(resident_bfp8_quad4_32,4)
ARC_BFP8_QUAD(resident_bfp8_quad8_32,8)
#undef ARC_BFP8_QUAD
#endif

// Fixed block32: retain the block in private registers instead of reading it
// twice with a runtime loop. No inter-lane communication or changed rounding.
__kernel void resident_bfp8_private32(__global const float* x,__global char* y,__global float* scales,__global int* status,int n,int block){
 int b=get_global_id(0),start=b*32;if(start>=n)return;
 float cache[32],maximum=0;int valid=1;
 #pragma unroll
 for(int j=0;j<32;j++){float v=start+j<n?x[start+j]:0;cache[j]=v;maximum=fmax(maximum,fabs(v));valid&=isfinite(v);}
 float scale=maximum==0?1:maximum/127.0f;valid&=scale>0&&isfinite(scale);
 scales[b]=valid?scale:NAN;if(!valid)atomic_or(status,1);
 #pragma unroll
 for(int j=0;j<32;j++)if(start+j<n)y[start+j]=valid?convert_char_sat_rte(clamp(cache[j]/scale,-127.0f,127.0f)):0;
}
