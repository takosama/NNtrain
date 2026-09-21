// Non-transposed A packing experiments. Native source storage is read in
// contiguous four-element vectors; optional SLM reorders wide source rows to
// the existing [K16 panel][M8 panel][M8][K16] BF16 DPAS layout.
// ABI is identical to xmx_storage_pack_a_*; transpose MUST be zero.
#if defined(ARC_XMX) && ARC_SG == 16
ushort xmx_bf16(float x);
int xmx_storage_scale_index(int index,int block);
ushort xmx_storage_read_f32(__global const uchar*,__global const float*,int,int);
ushort xmx_storage_read_bf16(__global const uchar*,__global const float*,int,int);
ushort xmx_storage_read_bfp8(__global const uchar*,__global const float*,int,int);

ushort4 xmx_storage_linear_read4_f32(__global const uchar* source,__global const float* scales,int index,int block){
 const float4 value=vload4(0,((__global const float*)source)+index);
 return (ushort4)(xmx_bf16(value.s0),xmx_bf16(value.s1),xmx_bf16(value.s2),xmx_bf16(value.s3));
}
ushort4 xmx_storage_linear_read4_bf16(__global const uchar* source,__global const float* scales,int index,int block){
 const ushort4 bits=vload4(0,((__global const ushort*)source)+index);
 const ushort4 nanMask=convert_ushort4((bits&(ushort4)(0x7fff))>(ushort4)(0x7f80));
 return bits|(nanMask&(ushort4)(0x40));
}
ushort4 xmx_storage_linear_read4_bfp8(__global const uchar* source,__global const float* scales,int index,int block){
 const float4 quantized=convert_float4(vload4(0,((__global const char*)source)+index));
 const int first=xmx_storage_scale_index(index,block);
 float4 scale;
 if(xmx_storage_scale_index(index+3,block)==first)scale=(float4)(scales[first]);
 else scale=(float4)(scales[first],scales[xmx_storage_scale_index(index+1,block)],scales[xmx_storage_scale_index(index+2,block)],scales[xmx_storage_scale_index(index+3,block)]);
 const float4 value=quantized*scale;
 return (ushort4)(xmx_bf16(value.s0),xmx_bf16(value.s1),xmx_bf16(value.s2),xmx_bf16(value.s3));
}

#define XMX_LINEAR_LOAD4(READ,READ4) \
 ushort4 value=(ushort4)(0); \
 if(row<m&&q+3<k){ \
  int idx=row*k+q;value=READ4(source,scales,offset+idx,block); \
  if(gated){float4 mask=vload4(0,gate+gateOffset+idx); \
   _Pragma("unroll") \
   for(int z=0;z<4;z++)if(mask[z]<=0)value[z]=0; \
  } \
 }else if(row<m){ \
  _Pragma("unroll") \
  for(int z=0;z<4;z++)if(q+z<k){int idx=row*k+q+z;if(!gated||!(gate[gateOffset+idx]<=0))value[z]=READ(source,scales,offset+idx,block);} \
 }

#define XMX_STORAGE_PACK_LINEAR(NAME,READ,READ4,BM,BK,WG) \
__attribute__((reqd_work_group_size(WG,1,1))) \
__kernel void NAME(__global const uchar* source,__global const float* scales,__global const float* gate, \
 __global ushort* panels,int m,int k,int transpose,int block,int gated,int offset,int gateOffset){ \
 int tid=get_local_id(0),rowBase=get_group_id(0)*BM,kBase=get_group_id(1)*BK; \
 int rowGroups=(m+7)/8,kGroups=(k+15)/16;__local ushort values[BM][BK+1]; \
 for(int i=tid*4;i<BM*BK;i+=WG*4){ \
  int rr=i/BK,kk=i%BK,row=rowBase+rr,q=kBase+kk; \
  XMX_LINEAR_LOAD4(READ,READ4) \
  vstore4(value,0,&values[rr][kk]); \
 } \
 barrier(CLK_LOCAL_MEM_FENCE); \
 for(int i=tid*4;i<BM*BK;i+=WG*4){ \
  int rr=((i/128)%(BM/8))*8+(i/16)%8,kk=(i/(BM*16))*16+i%16; \
  int rb=(rowBase+rr)/8,kb=(kBase+kk)/16; \
  if(rb<rowGroups&&kb<kGroups)vstore4(vload4(0,&values[rr][kk]),0,panels+(kb*rowGroups+rb)*128+(rr%8)*16+kk%16); \
 } \
}

#define XMX_STORAGE_PACK_LINEAR_VECTOR(NAME,READ,READ4) \
__kernel void NAME(__global const uchar* source,__global const float* scales,__global const float* gate, \
 __global ushort* panels,int m,int k,int transpose,int block,int gated,int offset,int gateOffset){ \
 int i=get_global_id(0)*4,rowGroups=(m+7)/8,kGroups=(k+15)/16,kPadded=kGroups*16; \
 int total=rowGroups*8*kPadded;if(i>=total)return; \
 int row=i/kPadded,q=i%kPadded; \
 XMX_LINEAR_LOAD4(READ,READ4) \
 vstore4(value,0,panels+((q/16)*rowGroups+row/8)*128+(row%8)*16+q%16); \
}

#define XMX_LINEAR_VARIANTS(SUFFIX,READ,READ4) \
 XMX_STORAGE_PACK_LINEAR(xmx_storage_pack_a_linear_c32_##SUFFIX,READ,READ4,32,32,256) \
 XMX_STORAGE_PACK_LINEAR(xmx_storage_pack_a_linear_c64_##SUFFIX,READ,READ4,32,64,256) \
 XMX_STORAGE_PACK_LINEAR(xmx_storage_pack_a_linear_c128_##SUFFIX,READ,READ4,32,128,256) \
 XMX_STORAGE_PACK_LINEAR(xmx_storage_pack_a_linear_r16c128_##SUFFIX,READ,READ4,16,128,256) \
 XMX_STORAGE_PACK_LINEAR(xmx_storage_pack_a_linear_c64w128_##SUFFIX,READ,READ4,32,64,128) \
 XMX_STORAGE_PACK_LINEAR_VECTOR(xmx_storage_pack_a_linear_vec4_##SUFFIX,READ,READ4)

XMX_LINEAR_VARIANTS(f32,xmx_storage_read_f32,xmx_storage_linear_read4_f32)
XMX_LINEAR_VARIANTS(bf16,xmx_storage_read_bf16,xmx_storage_linear_read4_bf16)
XMX_LINEAR_VARIANTS(bfp8,xmx_storage_read_bfp8,xmx_storage_linear_read4_bfp8)
#undef XMX_LINEAR_VARIANTS
#undef XMX_STORAGE_PACK_LINEAR_VECTOR
#undef XMX_STORAGE_PACK_LINEAR
#undef XMX_LINEAR_LOAD4
#endif
