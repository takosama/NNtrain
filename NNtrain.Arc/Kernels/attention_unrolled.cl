// Unrolled FP32 inner tile; tail predicates retain the exact original FMA sequence.
// Reduction order matches attention_gemm_narrow; only independent rows split.
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void attention_gemm_compact_unrolled(__global const float* a,__global const float* b,__global float* c,
 int m,int n,int k,int ar,int ac,int ag,int ab,int ah,int ao,int br,int bc,int bg,int bb,int bh,int bo,
 int cr,int cc,int cg,int cb,int ch,int co,int heads,int first,int add,int causalMode){
 int lx=get_local_id(0),ly=get_local_id(1),tid=ly*16+lx,g=get_group_id(2),rb=get_group_id(1)*32,colBase=get_group_id(0)*32;
 if(causalMode==1&&colBase>=rb+32)return;
 int begin=causalMode==3?rb:0,end=causalMode==2?min(k,rb+32):k;
 int ap=att_address(g,first,heads,ag,ab,ah,ao),bp=att_address(g,first,heads,bg,bb,bh,bo),cp=att_address(g,first,heads,cg,cb,ch,co);
 __local float at[32][33],bt[32][33];float2 sums[2];
 #pragma unroll
 for(int i=0;i<2;i++){sums[i]=(float2)(0);int row=rb+ly+16*i;
  if(add&&row<m){if(colBase+lx<n)sums[i].s0=c[cp+row*cr+(colBase+lx)*cc];if(colBase+lx+16<n)sums[i].s1=c[cp+row*cr+(colBase+lx+16)*cc];}}
 for(int base=begin;base<end;base+=32){
  for(int i=tid;i<1024;i+=256){int r=ac==1?i/32:i%32,j=ac==1?i%32:i/32;at[r][j]=(rb+r<m&&base+j<k)?a[ap+(rb+r)*ar+(base+j)*ac]:0;}
  for(int i=tid;i<1024;i+=256){int j=bc==1?i/32:i%32,col=bc==1?i%32:i/32;bt[j][col]=(base+j<k&&colBase+col<n)?b[bp+(base+j)*br+(colBase+col)*bc]:0;}
  barrier(CLK_LOCAL_MEM_FENCE);
  #pragma unroll
  for(int inner=0;inner<32;inner++)if(inner<k-base){
   float2 bv=(float2)(bt[inner][lx],bt[inner][lx+16]);
   #pragma unroll
   for(int i=0;i<2;i++)sums[i]=fma((float2)(at[ly+i*16][inner]),bv,sums[i]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
 }
 #pragma unroll
 for(int i=0;i<2;i++){int row=rb+ly+16*i;if(row<m){if(colBase+lx<n)c[cp+row*cr+(colBase+lx)*cc]=sums[i].s0;if(colBase+lx+16<n)c[cp+row*cr+(colBase+lx+16)*cc]=sums[i].s1;}}
}
