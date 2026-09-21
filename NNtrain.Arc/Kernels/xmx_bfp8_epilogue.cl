// Exact SG16 DPAS sequence, followed by block32 publication in registers.
// No FP32 output allocation or second global read. n is divisible by BN;
// 64 <= k <= 2048. Each subgroup owns complete quantization blocks.
#if defined(ARC_XMX) && ARC_SG == 16
#ifdef ARC_EPILOGUE_STANDALONE
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
short8 xmx_direct_tune_a(__global const ushort* p,int lane,int blockRead){
 return as_short8(intel_sub_group_block_read_us8(p));
}
#else
short8 xmx_direct_tune_a(__global const ushort* p,int lane,int blockRead);
#endif
#define ARC_XMX_BFP8_EPILOGUE(NAME,RB,BN,SG) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(16,SG,1))) \
__kernel void NAME(__global const ushort* a,__global const uint* b,__global char* output, \
 __global float* scales,__global int* status,__global const float* bias, \
 int m,int n,int k,int relu,int outputOffset){ \
 const int lane=get_local_id(0),sg=get_local_id(1); \
 const int row=get_group_id(1)*(SG*RB*8)+sg*(RB*8),colBase=get_group_id(0)*BN; \
 const int rows=(m+7)/8,cols=n/16; \
 float8 sums[RB][BN/16]; \
 _Pragma("unroll") \
 for(int rb=0;rb<RB;rb++){ \
  _Pragma("unroll") \
  for(int tile=0;tile<BN/16;tile++){ \
   sums[rb][tile]=(float8)(0); \
   _Pragma("unroll") \
   for(int r=0;r<8;r++)if(row+rb*8+r<m)sums[rb][tile][r]+=bias[colBase+tile*16+lane]; \
  } \
 } \
 for(int base=0;base<k;base+=16){ \
  short8 av[RB]; \
  _Pragma("unroll") \
  for(int rb=0;rb<RB;rb++){ \
   av[rb]=(short8)(0); \
   if(row+rb*8<m){av[rb]=xmx_direct_tune_a(a+((base/16)*rows+row/8+rb)*128,lane,1);if(base+lane>=k)av[rb]=(short8)(0);} \
  } \
  _Pragma("unroll") \
  for(int tile=0;tile<BN/16;tile++){ \
   int8 bv=as_int8(intel_sub_group_block_read8(b+((base/16)*cols+colBase/16+tile)*128)); \
   _Pragma("unroll") \
   for(int rb=0;rb<RB;rb++)sums[rb][tile]=intel_sub_group_bf16_bf16_matrix_mad_k16(av[rb],bv,sums[rb][tile]); \
  } \
 } \
 _Pragma("unroll") \
 for(int rb=0;rb<RB;rb++){ \
  _Pragma("unroll") \
  for(int pair=0;pair<BN/32;pair++){ \
   _Pragma("unroll") \
   for(int r=0;r<8;r++){ \
    const int globalRow=row+rb*8+r; \
    if(globalRow<m){ \
     const float x=relu?fmax(0.0f,sums[rb][pair*2][r]):sums[rb][pair*2][r]; \
     const float y=relu?fmax(0.0f,sums[rb][pair*2+1][r]):sums[rb][pair*2+1][r]; \
     float maximum=fmax(fabs(x),fabs(y));int valid=isfinite(x)&&isfinite(y); \
     _Pragma("unroll") \
     for(int stride=8;stride;stride/=2){maximum=fmax(maximum,intel_sub_group_shuffle(maximum,lane^stride));valid&=intel_sub_group_shuffle(valid,lane^stride);} \
     const float scale=maximum==0?1:maximum/127.0f;valid&=scale>0&&isfinite(scale); \
     const int offset=outputOffset+globalRow*n+colBase+pair*32; \
     if(lane==0){scales[offset/32]=valid?scale:NAN;if(!valid)atomic_or(status,1);} \
     output[offset+lane]=valid?convert_char_sat_rte(clamp(x/scale,-127.0f,127.0f)):0; \
     output[offset+lane+16]=valid?convert_char_sat_rte(clamp(y/scale,-127.0f,127.0f)):0; \
    } \
   } \
  } \
 } \
}
ARC_XMX_BFP8_EPILOGUE(gemm_xmx_bfp8_epilogue_16x32,2,32,16)
ARC_XMX_BFP8_EPILOGUE(gemm_xmx_bfp8_epilogue_16x64,2,64,16)
#ifdef ARC_EPILOGUE_STANDALONE
ARC_XMX_BFP8_EPILOGUE(gemm_xmx_bfp8_epilogue_32x64_wg4,4,64,4)
ARC_XMX_BFP8_EPILOGUE(gemm_xmx_bfp8_epilogue_32x64_wg8,4,64,8)
ARC_XMX_BFP8_EPILOGUE(gemm_xmx_bfp8_epilogue_16x128_wg4,2,128,4)
ARC_XMX_BFP8_EPILOGUE(gemm_xmx_bfp8_epilogue_16x128_wg8,2,128,8)
#endif
#undef ARC_XMX_BFP8_EPILOGUE
#endif
