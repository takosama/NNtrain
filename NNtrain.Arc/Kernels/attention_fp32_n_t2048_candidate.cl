// Opt-in PV/dQ tile alternatives for the production D32, T2048 layout.
// Both retain FP32 operands, increasing-K FMA order, interleaved QKV and
// the 64-key causal publication boundary of attention_probabilities.
// No score/gradient packing, extra global workspace or atomics are needed.

#define ATT_N_T2048(NAME,BM,BK,COMPONENT,OUTPUT_COMPONENTS,ACCUMULATE) \
__attribute__((reqd_work_group_size(16,16,1))) \
__kernel void NAME(__global const float* a,__global const float* b,__global float* c, \
 int m,int n,int k,int ar,int ac,int ag,int ab,int ah,int ao,int br,int bc,int bg,int bb,int bh,int bo, \
 int cr,int cc,int cg,int cb,int ch,int co,int heads,int first,int add,int causalMode) { \
 const int lx=get_local_id(0),ly=get_local_id(1),tid=ly*16+lx,g=get_group_id(2),h=first+g; \
 const int width=heads*32,rb=get_group_id(1)*BM; \
 const int ap=g*m*m,bp=(h/heads)*m*3*width+(h%heads)*32+COMPONENT*width; \
 const int cp=(h/heads)*m*OUTPUT_COMPONENTS*width+(h%heads)*32; \
 const int end=causalMode==2?min(m,((rb/64)+1)*64):m; \
 __local float at[BM][BK+1],bt[BK][33];float2 sums[BM/16]; \
 _Pragma("unroll") \
 for(int r=0;r<BM/16;r++){sums[r]=(float2)(0); \
  if(ACCUMULATE){int row=rb+ly+16*r; \
   sums[r]=(float2)(c[cp+row*OUTPUT_COMPONENTS*width+lx], \
                    c[cp+row*OUTPUT_COMPONENTS*width+lx+16]);} \
 } \
 for(int base=0;base<end;base+=BK){ \
  for(int i=tid*4;i<BM*BK;i+=1024){ \
   const int row=i/BK,inner=i%BK; \
   vstore4(vload4(0,a+ap+(rb+row)*m+base+inner),0,&at[row][inner]); \
  } \
  for(int i=tid*4;i<BK*32;i+=1024){ \
   const int row=i/32,col=i%32; \
   vstore4(vload4(0,b+bp+(base+row)*3*width+col),0,&bt[row][col]); \
  } \
  barrier(CLK_LOCAL_MEM_FENCE); \
  for(int inner=0;inner<BK;inner++){ \
   const float2 bv=(float2)(bt[inner][lx],bt[inner][lx+16]); \
   _Pragma("unroll") \
   for(int r=0;r<BM/16;r++)sums[r]=fma((float2)(at[ly+16*r][inner]),bv,sums[r]); \
  } \
  barrier(CLK_LOCAL_MEM_FENCE); \
 } \
 _Pragma("unroll") \
 for(int r=0;r<BM/16;r++){const int row=rb+ly+16*r; \
  c[cp+row*OUTPUT_COMPONENTS*width+lx]=sums[r].s0; \
  c[cp+row*OUTPUT_COMPONENTS*width+lx+16]=sums[r].s1; \
 } \
}

ATT_N_T2048(attention_fp32_pv_d32_t2048_m32k32,32,32,2,1,0)
ATT_N_T2048(attention_fp32_dq_d32_t2048_m32k32,32,32,1,3,1)
ATT_N_T2048(attention_fp32_pv_d32_t2048_m64k64,64,64,2,1,0)
ATT_N_T2048(attention_fp32_dq_d32_t2048_m64k64,64,64,1,3,1)
#undef ATT_N_T2048
