// Compensated XMX products for the bounded, materialized attention path.
// Preserve FP32 P/dY/dS with hi+residual BF16, never just truncate them to BF16.
// 64-row tiles match the probability kernel's causal zero-publication band.
#if defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#ifdef cl_intel_subgroups_short
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#endif
ushort xmx_bf16(float x) {
    uint u=as_uint(x);
    if((u&0x7f800000u)!=0x7f800000u)u+=0x7fffu+((u>>16)&1u);
    else if((u&0x007fffffu)!=0)u|=0x00400000u;
    return (ushort)(u>>16);
}
int att_address(int g,int first,int heads,int gs,int bs,int hs,int off) {
    int h=first+g;return gs?g*gs+off:(h/heads)*bs+(h%heads)*hs+off;
}
int8 xmx_panel_b(const __local uint* p,int lane) {
#ifdef ARC_SLM_BLOCK_IO
    return as_int8(intel_sub_group_block_read8(p));
#else
    int8 v;for(int r=0;r<8;r++)v[r]=as_int(p[r*16+lane]);return v;
#endif
}
inline ushort ap_low(float x,ushort hi) { return xmx_bf16(x-as_float((uint)hi<<16)); }
#define ATT_PRODUCTS(NAME,BN,LOW_B) \
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16,8,1))) \
__kernel void NAME(__global const float* a,__global const float* b,__global float* c, \
 int m,int n,int k,int ar,int ac,int ag,int ab,int ah,int ao,int br,int bc,int bg,int bb,int bh,int bo, \
 int cr,int cc,int cg,int cb,int ch,int co,int heads,int first,int add,int causalMode) { \
 int rowBase=get_group_id(1)*64,colBase=get_group_id(0)*BN; \
 if(causalMode==1&&colBase>=rowBase+64)return; \
 int start=causalMode==3?rowBase:0,end=causalMode==2?min(k,rowBase+64):k; \
 int group=get_group_id(2),lane=get_local_id(0),sg=get_local_id(1),tid=sg*16+lane,row=rowBase+sg*8; \
 int ap=att_address(group,first,heads,ag,ab,ah,ao),bp=att_address(group,first,heads,bg,bb,bh,bo),cp=att_address(group,first,heads,cg,cb,ch,co); \
 __local ushort ahigh[64][33],alow[64][33];__local uint bhigh[16*BN],blow[LOW_B?16*BN:1]; \
 float8 sum[BN/16]; \
 _Pragma("unroll") \
 for(int t=0;t<BN/16;t++){sum[t]=0;int col=colBase+t*16+lane; \
   _Pragma("unroll") \
   for(int r=0;r<8;r++)if(add&&row+r<m&&col<n)sum[t][r]=c[cp+(row+r)*cr+col*cc];} \
 for(int base=start;base<end;base+=32) { \
   for(int i=tid;i<64*32;i+=128) { \
     int rr=ac!=1?i%64:i/32,kk=ac!=1?i/64:i%32,r=rowBase+rr,q=base+kk; \
     float x=r<m&&q<end?a[ap+r*ar+q*ac]:0; \
     ushort hi=xmx_bf16(x);ahigh[rr][kk]=hi;alow[rr][kk]=ap_low(x,hi); \
   } \
   for(int i=tid;i<16*BN;i+=128) { \
     int kp=bc!=1?i%16:i/BN,nn=bc!=1?i/16:i%BN,q=base+2*kp,col=colBase+nn; \
     float x=col<n&&q<end?b[bp+q*br+col*bc]:0,y=col<n&&q+1<end?b[bp+(q+1)*br+col*bc]:0; \
     ushort hx=xmx_bf16(x),hy=xmx_bf16(y); \
     int dst=((nn/16)*2+kp/8)*128+(kp%8)*16+nn%16; \
     bhigh[dst]=(uint)hx|((uint)hy<<16); \
     if(LOW_B)blow[dst]=(uint)ap_low(x,hx)|((uint)ap_low(y,hy)<<16); \
   } \
   barrier(CLK_LOCAL_MEM_FENCE); \
   _Pragma("unroll") \
   for(int part=0;part<2;part++) { \
     short8 av,al; \
     _Pragma("unroll") \
     for(int r=0;r<8;r++){av[r]=as_short(ahigh[sg*8+r][part*16+lane]);al[r]=as_short(alow[sg*8+r][part*16+lane]);} \
     _Pragma("unroll") \
     for(int t=0;t<BN/16;t++) { \
       int8 bv=xmx_panel_b(bhigh+(t*2+part)*128,lane); \
       if(LOW_B){int8 bl=xmx_panel_b(blow+(t*2+part)*128,lane); \
         sum[t]=intel_sub_group_bf16_bf16_matrix_mad_k16(al,bl,sum[t]); \
         sum[t]=intel_sub_group_bf16_bf16_matrix_mad_k16(av,bl,sum[t]);} \
       sum[t]=intel_sub_group_bf16_bf16_matrix_mad_k16(al,bv,sum[t]); \
       sum[t]=intel_sub_group_bf16_bf16_matrix_mad_k16(av,bv,sum[t]); \
     } \
   } \
   barrier(CLK_LOCAL_MEM_FENCE); \
 } \
 _Pragma("unroll") \
 for(int t=0;t<BN/16;t++){int col=colBase+t*16+lane; \
   _Pragma("unroll") \
   for(int r=0;r<8;r++)if(row+r<m&&col<n)c[cp+(row+r)*cr+col*cc]=sum[t][r];} \
}
#if !ARC_ATTN_DIRECT
ATT_PRODUCTS(attention_xmx_products_n32_b0,32,0)
ATT_PRODUCTS(attention_xmx_products_n32_b1,32,1)
ATT_PRODUCTS(attention_xmx_products_n64_b0,64,0)
ATT_PRODUCTS(attention_xmx_products_n64_b1,64,1)
#else

// Direct panels avoid repeating FP32->hi/low conversion and SLM barriers for
// every output tile. Scratch is scoped to one bounded head tile and one GEMM.
short8 xmx_direct_tune_a(__global const ushort* p,int lane,int blockRead) {
#ifdef cl_intel_subgroups_short
    if(blockRead)return as_short8(intel_sub_group_block_read_us8(p));
#endif
    short8 v;for(int r=0;r<8;r++)v[r]=as_short(p[r*16+lane]);return v;
}
__kernel void attention_products_pack_a(__global const float* src,__global ushort* dst,
 int m,int k,int ar,int ac,int ag,int ab,int ah,int ao,int heads,int first,int count,int causalMode) {
 int rows=(m+7)/8*8,depth=(k+15)/16*16,size=rows*depth;
 int i=get_global_id(0),g=i/size;if(g>=count)return;
 int e=i%size,r=e/depth,q=e%depth;
 float f=0;
 if(r<m&&q<k&&!(causalMode==2&&q>r)&&!(causalMode==3&&r>q))
   f=src[att_address(g,first,heads,ag,ab,ah,ao)+r*ar+q*ac];
 int offset=g*size*2+((q/16)*(rows/8)+r/8)*128+(r%8)*16+q%16;
 ushort hi=xmx_bf16(f);dst[offset]=hi;dst[offset+size]=ap_low(f,hi);
}
__kernel void attention_products_pack_b(__global const float* src,__global uint* dst,
 int n,int k,int br,int bc,int bg,int bb,int bh,int bo,int heads,int first,int count,int lowB) {
 int cols=(n+15)/16*16,depth=(k+15)/16*16,size=cols*depth/2;
 int i=get_global_id(0),g=i/size;if(g>=count)return;
 int e=i%size,c=e%cols,q=(e/cols)*2;
 int base=att_address(g,first,heads,bg,bb,bh,bo);
 float x=c<n&&q<k?src[base+q*br+c*bc]:0,y=c<n&&q+1<k?src[base+(q+1)*br+c*bc]:0;
 ushort hx=xmx_bf16(x),hy=xmx_bf16(y);
 int offset=g*size*(lowB?2:1)+((q/16)*(cols/16)+c/16)*128+(q%16/2)*16+c%16;
 dst[offset]=(uint)hx|((uint)hy<<16);
 if(lowB)dst[offset+size]=(uint)ap_low(x,hx)|((uint)ap_low(y,hy)<<16);
}
#define ATT_DIRECT(NAME,BN,LOW_B) \
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16,16,1))) \
__kernel void NAME(__global const ushort* a,__global const uint* b,__global float* c,int m,int n,int k, \
 int cr,int cc,int cg,int cb,int ch,int co,int heads,int first,int add,int causalMode) { \
 int lane=get_local_id(0),sg=get_local_id(1),g=get_group_id(2),row=get_group_id(1)*128+sg*8,colBase=get_group_id(0)*BN; \
 if(row>=m||(causalMode==1&&colBase>=((row/64)+1)*64))return; \
 int mr=(m+7)/8,nr=(n+15)/16,depth=(k+15)/16*16,asize=mr*8*depth,bsize=nr*16*depth/2; \
 a+=g*asize*2;b+=g*bsize*(LOW_B?2:1);c+=att_address(g,first,heads,cg,cb,ch,co); \
 int start=causalMode==3?row/64*64:0,end=causalMode==2?min(k,(row/64+1)*64):k; \
 float8 sum[BN/16]; \
 _Pragma("unroll") \
 for(int t=0;t<BN/16;t++){sum[t]=0;int col=colBase+t*16+lane; \
   _Pragma("unroll") \
   for(int r=0;r<8;r++)if(add&&row+r<m&&col<n)sum[t][r]=c[(row+r)*cr+col*cc];} \
 for(int q=start;q<end;q+=16) { \
   int offset=((q/16)*mr+row/8)*128; \
   short8 hi=xmx_direct_tune_a(a+offset,lane,1),lo=xmx_direct_tune_a(a+asize+offset,lane,1); \
   _Pragma("unroll") \
   for(int t=0;t<BN/16;t++)if(colBase/16+t<nr) { \
     int bo=((q/16)*nr+colBase/16+t)*128; \
     int8 bh=as_int8(intel_sub_group_block_read8(b+bo)); \
     if(LOW_B){int8 bl=as_int8(intel_sub_group_block_read8(b+bsize+bo)); \
       sum[t]=intel_sub_group_bf16_bf16_matrix_mad_k16(lo,bl,sum[t]); \
       sum[t]=intel_sub_group_bf16_bf16_matrix_mad_k16(hi,bl,sum[t]);} \
     sum[t]=intel_sub_group_bf16_bf16_matrix_mad_k16(lo,bh,sum[t]); \
     sum[t]=intel_sub_group_bf16_bf16_matrix_mad_k16(hi,bh,sum[t]); \
   } \
 } \
 _Pragma("unroll") \
 for(int t=0;t<BN/16;t++){int col=colBase+t*16+lane; \
   _Pragma("unroll") \
   for(int r=0;r<8;r++)if(row+r<m&&col<n)c[(row+r)*cr+col*cc]=sum[t][r];} \
}
ATT_DIRECT(attention_products_direct_n32_b0,32,0)
ATT_DIRECT(attention_products_direct_n32_b1,32,1)
ATT_DIRECT(attention_products_direct_n64_b0,64,0)
ATT_DIRECT(attention_products_direct_n64_b1,64,1)
#endif
#endif
