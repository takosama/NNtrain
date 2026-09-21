// FP32-only attention tile experiments. Same strided ABI as attention_gemm.
// No packing, reduced-precision operands, approximate functions or atomics.
// Each kernel preserves increasing-K FMA order. Four-wide loads follow the
// contiguous input axis, including transposed score matrices in dK/dV.
// Modes 2/3 explicitly mask causal A values outside the diagonal, because a
// larger query tile must not read uninitialised portions of the P workspace.

#define ATT_FP32_TILE(NAME,BM,BN,BK,LY,VEC) \
__attribute__((reqd_work_group_size(16,LY,1))) \
__kernel void NAME(__global const float* a,__global const float* b,__global float* c, \
 int m,int n,int k,int ar,int ac,int ag,int ab,int ah,int ao,int br,int bc,int bg,int bb,int bh,int bo, \
 int cr,int cc,int cg,int cb,int ch,int co,int heads,int first,int add,int causalMode){ \
 const int lx=get_local_id(0),ly=get_local_id(1),tid=ly*16+lx,g=get_group_id(2); \
 const int rb=get_group_id(1)*BM,nb=get_group_id(0)*BN; \
 if(causalMode==1&&nb>=rb+BM)return; \
 const int begin=causalMode==3?rb:0,end=causalMode==2?min(k,rb+BM):k; \
 const int ap=att_address(g,first,heads,ag,ab,ah,ao),bp=att_address(g,first,heads,bg,bb,bh,bo),cp=att_address(g,first,heads,cg,cb,ch,co); \
 __local float at[BM][BK+1],bt[BK][BN+1];VEC sums[BM/LY]; \
 _Pragma("unroll") \
 for(int r=0;r<BM/LY;r++){sums[r]=(VEC)(0);int row=rb+ly+LY*r; \
  _Pragma("unroll") \
  for(int col=0;col<BN/16;col++)if(add&&row<m&&nb+lx+16*col<n)sums[r][col]=c[cp+row*cr+(nb+lx+16*col)*cc];} \
 for(int base=begin;base<end;base+=BK){ \
  for(int i=tid*4;i<BM*BK;i+=16*LY*4){ \
   int rr=ac==1?i/BK:i%BM,kk=ac==1?i%BK:i/BM; \
   int row=rb+rr,inner=base+kk;float4 av=(float4)(0); \
   int full=ac==1?(row<m&&inner+3<end):(row+3<m&&inner<end&&ar==1); \
   if(causalMode==2)full=full&&(ac==1?inner+3<=row:inner<=row); \
   if(causalMode==3)full=full&&(ac==1?inner>=row:inner>=row+3); \
   if(full)av=vload4(0,a+ap+row*ar+inner*ac); \
   else{ \
    _Pragma("unroll") \
    for(int z=0;z<4;z++){int r=row+(ac==1?0:z),q=inner+(ac==1?z:0); \
     if(r<m&&q<end&&(causalMode!=2||q<=r)&&(causalMode!=3||q>=r))av[z]=a[ap+r*ar+q*ac];} \
   } \
   if(ac==1)vstore4(av,0,&at[rr][kk]);else{ \
    _Pragma("unroll") \
    for(int z=0;z<4;z++)at[rr+z][kk]=av[z]; \
   } \
  } \
  for(int i=tid*4;i<BK*BN;i+=16*LY*4){ \
   int kk=bc==1?i/BN:i%BK,nn=bc==1?i%BN:i/BK; \
   int inner=base+kk,col=nb+nn;float4 bv=(float4)(0); \
   int full=bc==1?(inner<end&&col+3<n):(inner+3<end&&col<n&&br==1); \
   if(full)bv=vload4(0,b+bp+inner*br+col*bc); \
   else{ \
    _Pragma("unroll") \
    for(int z=0;z<4;z++){int q=inner+(bc==1?0:z),c=col+(bc==1?z:0);if(q<end&&c<n)bv[z]=b[bp+q*br+c*bc];} \
   } \
   if(bc==1)vstore4(bv,0,&bt[kk][nn]);else{ \
    _Pragma("unroll") \
    for(int z=0;z<4;z++)bt[kk+z][nn]=bv[z]; \
   } \
  } \
  barrier(CLK_LOCAL_MEM_FENCE); \
  _Pragma("unroll") \
  for(int inner=0;inner<BK;inner++)if(base+inner<end){ \
   VEC bv; \
   _Pragma("unroll") \
   for(int col=0;col<BN/16;col++)bv[col]=bt[inner][lx+16*col]; \
   _Pragma("unroll") \
   for(int r=0;r<BM/LY;r++)sums[r]=fma((VEC)(at[ly+LY*r][inner]),bv,sums[r]); \
  } \
  barrier(CLK_LOCAL_MEM_FENCE); \
 } \
 _Pragma("unroll") \
 for(int r=0;r<BM/LY;r++){int row=rb+ly+LY*r; \
  _Pragma("unroll") \
  for(int col=0;col<BN/16;col++){int column=nb+lx+16*col; \
   if(row<m&&column<n&&(causalMode!=1||column/32<=row/32))c[cp+row*cr+column*cc]=sums[r][col]; \
  } \
 } \
}

ATT_FP32_TILE(attention_fp32_dp_64x64x32,64,64,32,16,float4)
ATT_FP32_TILE(attention_fp32_dp_64x64x16,64,64,16,16,float4)
ATT_FP32_TILE(attention_fp32_dp_32x64x32,32,64,32,16,float4)
ATT_FP32_TILE(attention_fp32_dp_64x64x32_w128,64,64,32,8,float4)
ATT_FP32_TILE(attention_fp32_n_32x32x64,32,32,64,16,float2)
ATT_FP32_TILE(attention_fp32_n_64x32x32,64,32,32,16,float2)
ATT_FP32_TILE(attention_fp32_n_64x32x64,64,32,64,16,float2)
ATT_FP32_TILE(attention_fp32_n_128x32x32,128,32,32,16,float2)
ATT_FP32_TILE(attention_fp32_n_128x32x64,128,32,64,16,float2)
ATT_FP32_TILE(attention_fp32_n_256x32x16,256,32,16,16,float2)
ATT_FP32_TILE(attention_fp32_n_64x32x32_w128,64,32,32,8,float2)
ATT_FP32_TILE(attention_fp32_n_32x32x64_w128,32,32,64,8,float2)
#undef ATT_FP32_TILE

// Constant-layout D=32 specializations. The unused generic ABI arguments are
// retained for dispatch compatibility, but these kernels are only valid for
// the production interleaved DY/QKV/score/output views stated below. "aligned"
// additionally requires sequence % 64 == 0; all other shapes use guarded code.
#define ATT_DP_D32(NAME,ALIGNED) \
__attribute__((reqd_work_group_size(16,16,1))) \
__kernel void NAME(__global const float* a,__global const float* b,__global float* c, \
 int m,int n,int k,int ar,int ac,int ag,int ab,int ah,int ao,int br,int bc,int bg,int bb,int bh,int bo, \
 int cr,int cc,int cg,int cb,int ch,int co,int heads,int first,int add,int causalMode){ \
 const int lx=get_local_id(0),ly=get_local_id(1),tid=ly*16+lx,g=get_group_id(2),h=first+g; \
 const int width=heads*32,rb=get_group_id(1)*64,nb=get_group_id(0)*64; \
 if(causalMode==1&&nb>=rb+64)return; \
 const int ap=(h/heads)*m*width+(h%heads)*32,bp=(h/heads)*m*3*width+(h%heads)*32+2*width,cp=g*m*m; \
 __local float at[64][17],bt[16][65];float4 sums[4]; \
 _Pragma("unroll") \
 for(int r=0;r<4;r++)sums[r]=(float4)(0); \
 _Pragma("unroll") \
 for(int base=0;base<32;base+=16){ \
  int row=tid/4,inner=(tid%4)*4; \
  float4 av=(float4)(0),bv=(float4)(0); \
  if(ALIGNED||rb+row<m)av=vload4(0,a+ap+(rb+row)*width+base+inner); \
  if(ALIGNED||nb+row<m)bv=vload4(0,b+bp+(nb+row)*3*width+base+inner); \
  vstore4(av,0,&at[row][inner]); \
  _Pragma("unroll") \
  for(int z=0;z<4;z++)bt[inner+z][row]=bv[z]; \
  barrier(CLK_LOCAL_MEM_FENCE); \
  _Pragma("unroll") \
  for(int inner=0;inner<16;inner++){ \
   const float4 bv=(float4)(bt[inner][lx],bt[inner][lx+16],bt[inner][lx+32],bt[inner][lx+48]); \
   _Pragma("unroll") \
   for(int r=0;r<4;r++)sums[r]=fma((float4)(at[ly+16*r][inner]),bv,sums[r]); \
  } \
  barrier(CLK_LOCAL_MEM_FENCE); \
 } \
 _Pragma("unroll") \
 for(int r=0;r<4;r++){int row=rb+ly+16*r; \
  _Pragma("unroll") \
  for(int col=0;col<4;col++){int column=nb+lx+16*col; \
   if((ALIGNED||(row<m&&column<m))&&(causalMode!=1||column/32<=row/32))c[cp+row*m+column]=sums[r][col]; \
  } \
 } \
}

ATT_DP_D32(attention_fp32_dp_d32_special,0)
ATT_DP_D32(attention_fp32_dp_d32_aligned,1)
#undef ATT_DP_D32

// A=contiguous P/DS; B=interleaved QKV component; C=Y or dQ, respectively.
// BM64 exactly matches the causal probability zero-publication boundary, so
// no elementwise diagonal masks or transpose-dependent loader branches remain.
#define ATT_N_D32(NAME,COMPONENT,OUTPUT_COMPONENTS,ACCUMULATE,ALIGNED) \
__attribute__((reqd_work_group_size(16,16,1))) \
__kernel void NAME(__global const float* a,__global const float* b,__global float* c, \
 int m,int n,int k,int ar,int ac,int ag,int ab,int ah,int ao,int br,int bc,int bg,int bb,int bh,int bo, \
 int cr,int cc,int cg,int cb,int ch,int co,int heads,int first,int add,int causalMode){ \
 const int lx=get_local_id(0),ly=get_local_id(1),tid=ly*16+lx,g=get_group_id(2),h=first+g; \
 const int width=heads*32,rb=get_group_id(1)*64; \
 const int ap=g*m*m,bp=(h/heads)*m*3*width+(h%heads)*32+COMPONENT*width; \
 const int cp=(h/heads)*m*OUTPUT_COMPONENTS*width+(h%heads)*32; \
 const int end=causalMode==2?(ALIGNED?rb+64:min(m,rb+64)):m; \
 __local float at[64][33],bt[32][33];float2 sums[4]; \
 _Pragma("unroll") \
 for(int r=0;r<4;r++){int row=rb+ly+16*r;sums[r]=(float2)(0); \
  if(ACCUMULATE&&(ALIGNED||row<m))sums[r]=(float2)(c[cp+row*OUTPUT_COMPONENTS*width+lx],c[cp+row*OUTPUT_COMPONENTS*width+lx+16]); \
 } \
 for(int base=0;base<end;base+=32){ \
  const int r0=tid/8,k0=(tid%8)*4; \
  _Pragma("unroll") \
  for(int part=0;part<2;part++){ \
   int row=rb+r0+32*part,inner=base+k0;float4 av=(float4)(0); \
   if(ALIGNED||(row<m&&inner+3<end))av=vload4(0,a+ap+row*m+inner); \
   else{ \
    _Pragma("unroll") \
    for(int z=0;z<4;z++)if(row<m&&inner+z<end)av[z]=a[ap+row*m+inner+z]; \
   } \
   vstore4(av,0,&at[r0+32*part][k0]); \
  } \
  float4 bv=(float4)(0); \
  if(ALIGNED||base+r0<end)bv=vload4(0,b+bp+(base+r0)*3*width+k0); \
  vstore4(bv,0,&bt[r0][k0]); \
  barrier(CLK_LOCAL_MEM_FENCE); \
  _Pragma("unroll") \
  for(int inner=0;inner<32;inner++)if(ALIGNED||base+inner<end){ \
   const float2 bv=(float2)(bt[inner][lx],bt[inner][lx+16]); \
   _Pragma("unroll") \
   for(int r=0;r<4;r++)sums[r]=fma((float2)(at[ly+16*r][inner]),bv,sums[r]); \
  } \
  barrier(CLK_LOCAL_MEM_FENCE); \
 } \
 _Pragma("unroll") \
 for(int r=0;r<4;r++){int row=rb+ly+16*r; \
  if(ALIGNED||row<m){c[cp+row*OUTPUT_COMPONENTS*width+lx]=sums[r].s0;c[cp+row*OUTPUT_COMPONENTS*width+lx+16]=sums[r].s1;} \
 } \
}

ATT_N_D32(attention_fp32_pv_d32_special,2,1,0,0)
ATT_N_D32(attention_fp32_pv_d32_aligned,2,1,0,1)
ATT_N_D32(attention_fp32_dq_d32_special,1,3,1,0)
ATT_N_D32(attention_fp32_dq_d32_aligned,1,3,1,1)
#undef ATT_N_D32
