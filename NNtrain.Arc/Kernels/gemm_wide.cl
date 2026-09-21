// FP32 64x64 register tile: each thread accumulates 4x4 outputs.
// Keep the same ordered K loop and RNE operand policy as gemm_tiled.
float wide_bf16(float x){uint u=as_uint(x);if((u&0x7f800000u)==0x7f800000u)return x;return as_float((u+0x7fffu+((u>>16)&1u))&0xffff0000u);}
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void gemm_wide(__global const float* a,__global const float* b,__global float* c,
 __global const float* bias,__global const float* gate,int m,int n,int k,int ta,int tb,int bf16Operands,
 int accumulate,int addBias,int relu,int gateOperand,int kStart,int kCount){
 int lx=get_local_id(0),ly=get_local_id(1),rb=get_group_id(1)*64,cb=get_group_id(0)*64,end=min(k,kStart+kCount);
 __local float at[64][17],bt[16][65];float4 sums[4];
 #pragma unroll
 for(int i=0;i<4;i++){
  sums[i]=(float4)(0);int row=rb+ly+16*i;
  #pragma unroll
  for(int j=0;j<4;j++){int col=cb+lx+16*j;if(row<m&&col<n){if(accumulate)sums[i][j]=c[row*n+col];if(addBias)sums[i][j]+=bias[col];}}
 }
 for(int base=kStart;base<end;base+=16){
  int ar=ta?lx:ly,ak=ta?ly:lx,bk=tb?lx:ly,bc=tb?ly:lx;
  #pragma unroll
  for(int part=0;part<4;part++){
   int r=rb+ar+16*part,q=base+ak;float av=0;
   if(r<m&&q<end){int idx=ta?q*m+r:r*k+q;av=gateOperand==1&&gate[idx]<=0?0:a[idx];if(bf16Operands&1)av=wide_bf16(av);}
   at[ar+16*part][ak]=av;
   int col=cb+bc+16*part,s=base+bk;float bv=0;
   if(col<n&&s<end){int idx=tb?col*k+s:s*n+col;bv=gateOperand==2&&gate[idx]<=0?0:b[idx];if(bf16Operands&2)bv=wide_bf16(bv);}
   bt[bk][bc+16*part]=bv;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int inner=0;inner<min(16,end-base);inner++){
   float4 bv=(float4)(bt[inner][lx],bt[inner][lx+16],bt[inner][lx+32],bt[inner][lx+48]);
   #pragma unroll
   for(int i=0;i<4;i++)sums[i]=fma((float4)(at[ly+16*i][inner]),bv,sums[i]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
 }
 #pragma unroll
 for(int i=0;i<4;i++){
  int row=rb+ly+16*i;
  #pragma unroll
  for(int j=0;j<4;j++){int col=cb+lx+16*j;if(row<m&&col<n)c[row*n+col]=relu?fmax(0.0f,sums[i][j]):sums[i][j];}
 }
}
