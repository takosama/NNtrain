// Resident storage -> BF16 DPAS panels. No FP32 decode allocation or roundtrip.
// The BFP8 multiply is evaluated in FP32 before the exact existing BF16 RNE.
// Source offsets address both payload and its original block-scale index.
#if defined(ARC_XMX) && ARC_SG == 16
ushort xmx_bf16(float x);
// Packing-only ABI; negative block encodes ~log2(power-of-two block).
int xmx_storage_scale_index(int index,int block){return block<0?index>>(~block):index/block;}
ushort xmx_storage_read_f32(__global const uchar* source,__global const float* scales,int index,int block){
 return xmx_bf16(((__global const float*)source)[index]);
}
ushort xmx_storage_read_bf16(__global const uchar* source,__global const float* scales,int index,int block){
 ushort bits=((__global const ushort*)source)[index];
 // Expanding BF16 then passing through the existing XMX converter only changes
 // signaling NaNs. Retain that behavior even though ordinary BF16 is copied.
 return (ushort)(bits|(((uint)(bits&0x7fffu)>0x7f80u)?0x40u:0u));
}
ushort xmx_storage_read_bfp8(__global const uchar* source,__global const float* scales,int index,int block){
 float value=(float)((__global const char*)source)[index]*scales[xmx_storage_scale_index(index,block)];
 return xmx_bf16(value);
}
#define XMX_STORAGE_PACK_A(NAME,READ) \
__kernel void NAME(__global const uchar* source,__global const float* scales,__global const float* gate, \
 __global ushort* panels,int m,int k,int transpose,int block,int gated,int offset,int gateOffset){ \
 int i=get_global_id(0),rowGroups=(m+7)/8,total=((k+15)/16)*rowGroups*128; \
 if(i>=total)return;int kk=i%16,rr=(i/16)%8,rb=(i/128)%rowGroups,kb=i/(128*rowGroups); \
 int row=rb*8+rr,q=kb*16+kk;ushort value=0; \
 if(row<m&&q<k){int idx=transpose?q*m+row:row*k+q;if(!gated||!(gate[gateOffset+idx]<=0))value=READ(source,scales,offset+idx,block);} \
 panels[i]=value; \
}
XMX_STORAGE_PACK_A(xmx_storage_pack_a_f32,xmx_storage_read_f32)
XMX_STORAGE_PACK_A(xmx_storage_pack_a_bf16,xmx_storage_read_bf16)
XMX_STORAGE_PACK_A(xmx_storage_pack_a_bfp8,xmx_storage_read_bfp8)
#undef XMX_STORAGE_PACK_A
#define XMX_STORAGE_PACK_B(NAME,READ) \
__kernel void NAME(__global const uchar* source,__global const float* scales,__global const float* gate, \
 __global uint* panels,int n,int k,int transpose,int block,int gated,int offset,int gateOffset){ \
 int i=get_global_id(0),colGroups=(n+15)/16,total=((k+15)/16)*colGroups*128; \
 if(i>=total)return;int cc=i%16,kp=(i/16)%8,cb=(i/128)%colGroups,kb=i/(128*colGroups); \
 int col=cb*16+cc,q=kb*16+kp*2;ushort x=0,y=0; \
 if(col<n&&q<k){int idx=transpose?col*k+q:q*n+col;if(!gated||!(gate[gateOffset+idx]<=0))x=READ(source,scales,offset+idx,block);} \
 if(col<n&&q+1<k){int idx=transpose?col*k+q+1:(q+1)*n+col;if(!gated||!(gate[gateOffset+idx]<=0))y=READ(source,scales,offset+idx,block);} \
 panels[i]=(uint)x|((uint)y<<16); \
}
XMX_STORAGE_PACK_B(xmx_storage_pack_b_f32,xmx_storage_read_f32)
XMX_STORAGE_PACK_B(xmx_storage_pack_b_bf16,xmx_storage_read_bf16)
XMX_STORAGE_PACK_B(xmx_storage_pack_b_bfp8,xmx_storage_read_bfp8)
#undef XMX_STORAGE_PACK_B

// Transposed operands otherwise gather along the leading dimension while
// writing panel order. Read a coalesced 32x32 source tile, then use padded SLM
// to permute its BF16 values into the identical DPAS layout (2,112 bytes/WG).
#define XMX_STORAGE_PACK_AT(NAME,READ) \
__attribute__((reqd_work_group_size(256,1,1))) \
__kernel void NAME(__global const uchar* source,__global const float* scales,__global const float* gate, \
 __global ushort* panels,int m,int k,int transpose,int block,int gated,int offset,int gateOffset){ \
 int tid=get_local_id(0),rowBase=get_group_id(0)*32,kBase=get_group_id(1)*32; \
 int rowGroups=(m+7)/8,kGroups=(k+15)/16; \
 __local ushort values[32][33]; \
 for(int i=tid;i<1024;i+=256){ \
  int rr=i%32,kk=i/32,row=rowBase+rr,q=kBase+kk;ushort value=0; \
  if(row<m&&q<k){int idx=q*m+row;if(!gated||!(gate[gateOffset+idx]<=0))value=READ(source,scales,offset+idx,block);} \
  values[rr][kk]=value; \
 } \
 barrier(CLK_LOCAL_MEM_FENCE); \
 for(int i=tid;i<1024;i+=256){ \
  int rr=((i/128)%4)*8+(i/16)%8,kk=(i/512)*16+i%16; \
  int rb=(rowBase+rr)/8,kb=(kBase+kk)/16; \
  if(rb<rowGroups&&kb<kGroups)panels[(kb*rowGroups+rb)*128+(rr%8)*16+kk%16]=values[rr][kk]; \
 } \
}
XMX_STORAGE_PACK_AT(xmx_storage_pack_a_transpose_f32,xmx_storage_read_f32)
XMX_STORAGE_PACK_AT(xmx_storage_pack_a_transpose_bf16,xmx_storage_read_bf16)
XMX_STORAGE_PACK_AT(xmx_storage_pack_a_transpose_bfp8,xmx_storage_read_bfp8)
#undef XMX_STORAGE_PACK_AT
#define XMX_STORAGE_PACK_BT(NAME,READ) \
__attribute__((reqd_work_group_size(256,1,1))) \
__kernel void NAME(__global const uchar* source,__global const float* scales,__global const float* gate, \
 __global uint* panels,int n,int k,int transpose,int block,int gated,int offset,int gateOffset){ \
 int tid=get_local_id(0),colBase=get_group_id(0)*32,kBase=get_group_id(1)*32; \
 int colGroups=(n+15)/16,kGroups=(k+15)/16; \
 __local ushort values[32][33]; \
 for(int i=tid;i<1024;i+=256){ \
  int nn=i/32,kk=i%32,col=colBase+nn,q=kBase+kk;ushort value=0; \
  if(col<n&&q<k){int idx=col*k+q;if(!gated||!(gate[gateOffset+idx]<=0))value=READ(source,scales,offset+idx,block);} \
  values[nn][kk]=value; \
 } \
 barrier(CLK_LOCAL_MEM_FENCE); \
 for(int i=tid;i<512;i+=256){ \
  int nn=((i/128)%2)*16+i%16,kk=(i/256)*16+((i/16)%8)*2; \
  int cb=(colBase+nn)/16,kb=(kBase+kk)/16; \
  if(cb<colGroups&&kb<kGroups)panels[(kb*colGroups+cb)*128+((kk%16)/2)*16+nn%16]=(uint)values[nn][kk]|((uint)values[nn][kk+1]<<16); \
 } \
}
XMX_STORAGE_PACK_BT(xmx_storage_pack_b_transpose_f32,xmx_storage_read_f32)
XMX_STORAGE_PACK_BT(xmx_storage_pack_b_transpose_bf16,xmx_storage_read_bf16)
XMX_STORAGE_PACK_BT(xmx_storage_pack_b_transpose_bfp8,xmx_storage_read_bfp8)
#undef XMX_STORAGE_PACK_BT
#endif
