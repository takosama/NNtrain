// Batched, strided attention products. One workgroup owns its output tile;
// disjoint heads and dQ/dK/dV destinations never need floating-point atomics.
int att_address(int g,int first,int heads,int gs,int bs,int hs,int off){int h=first+g;return gs?g*gs+off:(h/heads)*bs+(h%heads)*hs+off;}
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void attention_gemm(__global const float* a,__global const float* b,__global float* c,
 int m,int n,int k,int ar,int ac,int ag,int ab,int ah,int ao,int br,int bc,int bg,int bb,int bh,int bo,
 int cr,int cc,int cg,int cb,int ch,int co,int heads,int first,int add,int causalMode){
 int lx=get_local_id(0),ly=get_local_id(1),g=get_group_id(2),r0=get_group_id(1)*32+ly,r1=r0+16,c0=get_group_id(0)*32+lx,c1=c0+16;
 if(causalMode==1&&get_group_id(0)*32>=get_group_id(1)*32+32)return;
 int begin=causalMode==3?get_group_id(1)*32:0,end=causalMode==2?min(k,(int)get_group_id(1)*32+32):k;
 int ap=att_address(g,first,heads,ag,ab,ah,ao),bp=att_address(g,first,heads,bg,bb,bh,bo),cp=att_address(g,first,heads,cg,cb,ch,co);
 __local float at[32][17],bt[16][33];
 float s00=0,s01=0,s10=0,s11=0;
 if(add){if(r0<m&&c0<n)s00=c[cp+r0*cr+c0*cc];if(r0<m&&c1<n)s01=c[cp+r0*cr+c1*cc];if(r1<m&&c0<n)s10=c[cp+r1*cr+c0*cc];if(r1<m&&c1<n)s11=c[cp+r1*cr+c1*cc];}
 for(int base=begin;base<end;base+=16){
  // Match contiguous source axis for both normal and transposed matrices.
  int al=ac==1?ly:lx,ak=ac==1?lx:ly,bk=bc==1?ly:lx,bl=bc==1?lx:ly;
  for(int part=0;part<2;part++){
   int r=get_group_id(1)*32+al+part*16,j=base+ak;
   at[al+part*16][ak]=(r<m&&j<k)?a[ap+r*ar+j*ac]:0;
   int col=get_group_id(0)*32+bl+part*16,l=base+bk;
   bt[bk][bl+part*16]=(col<n&&l<k)?b[bp+l*br+col*bc]:0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int j=0;j<min(16,k-base);j++){float a0=at[ly][j],a1=at[ly+16][j],b0=bt[j][lx],b1=bt[j][lx+16];s00=fma(a0,b0,s00);s01=fma(a0,b1,s01);s10=fma(a1,b0,s10);s11=fma(a1,b1,s11);}
  barrier(CLK_LOCAL_MEM_FENCE);
 }
 if(r0<m&&c0<n)c[cp+r0*cr+c0*cc]=s00;if(r0<m&&c1<n)c[cp+r0*cr+c1*cc]=s01;
 if(r1<m&&c0<n)c[cp+r1*cr+c0*cc]=s10;if(r1<m&&c1<n)c[cp+r1*cr+c1*cc]=s11;
}

// Narrow head outputs reuse a K/V tile across 64 queries, with twice the K
// depth per barrier. Ordered FP32 FMA is unchanged, including dP/dS operands.
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void attention_gemm_narrow(__global const float* a,__global const float* b,__global float* c,
 int m,int n,int k,int ar,int ac,int ag,int ab,int ah,int ao,int br,int bc,int bg,int bb,int bh,int bo,
 int cr,int cc,int cg,int cb,int ch,int co,int heads,int first,int add,int causalMode){
 int lx=get_local_id(0),ly=get_local_id(1),tid=ly*16+lx,g=get_group_id(2),rb=get_group_id(1)*64,colBase=get_group_id(0)*32;
 if(causalMode==1&&colBase>=rb+64)return;
 int begin=causalMode==3?rb:0,end=causalMode==2?min(k,rb+64):k;
 int ap=att_address(g,first,heads,ag,ab,ah,ao),bp=att_address(g,first,heads,bg,bb,bh,bo),cp=att_address(g,first,heads,cg,cb,ch,co);
 __local float at[64][33],bt[32][33];float2 sums[4];
 #pragma unroll
 for(int i=0;i<4;i++){sums[i]=(float2)(0);int row=rb+ly+16*i;
  if(add&&row<m){if(colBase+lx<n)sums[i].s0=c[cp+row*cr+(colBase+lx)*cc];if(colBase+lx+16<n)sums[i].s1=c[cp+row*cr+(colBase+lx+16)*cc];}}
 for(int base=begin;base<end;base+=32){
  for(int i=tid;i<2048;i+=256){int r=ac==1?i/32:i%64,j=ac==1?i%32:i/64;at[r][j]=(rb+r<m&&base+j<k)?a[ap+(rb+r)*ar+(base+j)*ac]:0;}
  for(int i=tid;i<1024;i+=256){int j=bc==1?i/32:i%32,col=bc==1?i%32:i/32;bt[j][col]=(base+j<k&&colBase+col<n)?b[bp+(base+j)*br+(colBase+col)*bc]:0;}
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int inner=0;inner<min(32,k-base);inner++){
   float2 bv=(float2)(bt[inner][lx],bt[inner][lx+16]);
   #pragma unroll
   for(int i=0;i<4;i++)sums[i]=fma((float2)(at[ly+i*16][inner]),bv,sums[i]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
 }
 #pragma unroll
 for(int i=0;i<4;i++){int row=rb+ly+16*i;if(row<m){if(colBase+lx<n)c[cp+row*cr+(colBase+lx)*cc]=sums[i].s0;if(colBase+lx+16<n)c[cp+row*cr+(colBase+lx+16)*cc]=sums[i].s1;}}
}

__attribute__((reqd_work_group_size(64,1,1)))
__kernel void attention_probabilities(__global float* scores,__global float* stats,int seq,int width,int heads,int first,int causal,int saved){
 int row=get_group_id(0),q=row%seq,g=row/seq,l=get_local_id(0),n=causal?q+1:seq,stat=(first+g)*seq+q;
 // All consumers have query/key row tiles <=64. Beyond this diagonal tile
 // the entries are never read when causal GEMM bounds are active.
 int limit=causal==2?min(seq,(q/64+1)*64):seq;
 float scale=rsqrt((float)(width/heads));__local float tmp[64];
 float maximum=-INFINITY;
 if(!saved){for(int key=l;key<n;key+=64)maximum=fmax(maximum,scores[row*seq+key]*scale);tmp[l]=maximum;barrier(CLK_LOCAL_MEM_FENCE);for(int s=32;s;s/=2){if(l<s)tmp[l]=fmax(tmp[l],tmp[l+s]);barrier(CLK_LOCAL_MEM_FENCE);}maximum=tmp[0];}
 else maximum=stats[2*stat];
 barrier(CLK_LOCAL_MEM_FENCE);
 float sum=0;
 for(int key=l;key<limit;key+=64){float p=key<n?exp(scores[row*seq+key]*scale-maximum):0;scores[row*seq+key]=p;sum+=p;}
 float inv;
 if(!saved){tmp[l]=sum;barrier(CLK_LOCAL_MEM_FENCE);for(int s=32;s;s/=2){if(l<s)tmp[l]+=tmp[l+s];barrier(CLK_LOCAL_MEM_FENCE);}inv=1.0f/tmp[0];if(l==0){stats[2*stat]=maximum;stats[2*stat+1]=inv;}}
 else inv=stats[2*stat+1];
 for(int key=l;key<limit;key+=64)scores[row*seq+key]*=inv;
}

__attribute__((reqd_work_group_size(64,1,1)))
__kernel void attention_derivatives(__global const float* p,__global float* dp,int seq,int width,int heads,int causal){
 int row=get_group_id(0),q=row%seq,l=get_local_id(0),count=causal?q+1:seq;__local float tmp[64];float sum=0;
 int limit=causal==2?min(seq,(q/64+1)*64):seq;
 for(int key=l;key<count;key+=64)sum=fma(p[row*seq+key],dp[row*seq+key],sum);
 tmp[l]=sum;barrier(CLK_LOCAL_MEM_FENCE);for(int s=32;s;s/=2){if(l<s)tmp[l]+=tmp[l+s];barrier(CLK_LOCAL_MEM_FENCE);}
 float delta=tmp[0],scale=rsqrt((float)(width/heads));
 for(int key=l;key<limit;key+=64){int i=row*seq+key;dp[i]=key<count?p[i]*(dp[i]-delta)*scale:0;}
}
