// Strict fused residual normalization: one work-item per row, as in norm and
// norm_dx_set. Reordering mean/variance sums moved low-precision rounding
// boundaries enough to change training updates. Keep the old operation order
// and remove only the residual scratch and separate gradient-copy passes.
float mask(uint seed,int i,uint threshold,float scale);

// The serial normalization gradient has already been rounded into scratch.
// Fuse only its two coalesced consumers, retaining their update order when
// residual and branch are the same Tensor (dx and dbranch alias).
__kernel void norm_residual_back_accumulate(
 __global const float* scratch,__global float* dx,__global float* dbranch,
 int length,uint seed,uint threshold,float scale){
 int i=get_global_id(0);if(i>=length)return;
 float value=scratch[i];
 dx[i]+=value;
 dbranch[i]+=value*mask(seed,i,threshold,scale);
}

float arc_norm_candidate_input(__global const float* x,__global const float* branch,
 int i,uint seed,uint threshold,float scale,int residual){
 if(!residual)return x[i];
 // The legacy dropout expression is a contracted multiply/add followed by
 // an FP32 store. An explicit fma keeps that one-rounding boundary without
 // volatile private-memory spills after this helper is inlined.
 return fma(branch[i],mask(seed,i,threshold,scale),x[i]);
}

__kernel void norm_residual_serial(
 __global const float* x,__global const float* branch,
 __global const float* gamma,__global const float* beta,
 __global float* y,__global float* stats,int rows,int width,float eps,
 uint seed,uint threshold,float scale,int residual){
 int row=get_global_id(0);if(row>=rows)return;
 float mean=0;
 for(int c=0;c<width;c++)mean+=arc_norm_candidate_input(x,branch,row*width+c,seed,threshold,scale,residual);
 mean/=width;
 float variance=0;
 for(int c=0;c<width;c++){
  float d=arc_norm_candidate_input(x,branch,row*width+c,seed,threshold,scale,residual)-mean;
  variance+=d*d;
 }
 float inv=rsqrt(variance/width+eps);
 stats[2*row]=mean;stats[2*row+1]=inv;
 for(int c=0;c<width;c++){
  float value=arc_norm_candidate_input(x,branch,row*width+c,seed,threshold,scale,residual);
  y[row*width+c]=(value-mean)*inv*gamma[c]+beta[c];
 }
}

__kernel void norm_residual_dx_serial(
 __global const float* x,__global const float* branch,__global const float* gamma,
 __global const float* dy,__global const float* stats,
 __global float* dx,__global float* dbranch,int rows,int width,
 uint seed,uint threshold,float scale,int residual){
 int row=get_global_id(0);if(row>=rows)return;
 float mean=stats[2*row],inv=stats[2*row+1],sum=0,weighted=0;
 for(int c=0;c<width;c++){
  int i=row*width+c;
  float value=arc_norm_candidate_input(x,branch,i,seed,threshold,scale,residual);
  float g=dy[i]*gamma[c];sum+=g;weighted+=g*(value-mean)*inv;
 }
 for(int c=0;c<width;c++){
  int i=row*width+c;
  float value=arc_norm_candidate_input(x,branch,i,seed,threshold,scale,residual);
  // norm_dx_set stored this product before copy_scale/dropout_back. Explicit
  // fma keeps that rounding boundary without a volatile private allocation.
  // Give its zero addend the product's sign: unlike a literal +0, this also
  // preserves a negative zero product (including negative underflow).
  float inner=dy[i]*gamma[c]-(sum+(value-mean)*inv*weighted)/width;
  float productZero=as_float((as_uint(inv)^as_uint(inner))&0x80000000u);
  float grad=fma(inv,inner,productZero);
  dx[i]+=grad;
  if(residual)dbranch[i]+=grad*mask(seed,i,threshold,scale);
 }
}

// Same row partition, serial accumulation and local reduction as gradient_rows;
// only the residual input is reconstructed instead of read from a scratch array.
__attribute__((reqd_work_group_size(32,8,1)))
__kernel void norm_residual_parameter_parts(
 __global const float* dy,__global const float* x,__global const float* branch,
 __global const float* stats,__global float* parts,int rows,int width,
 uint seed,uint threshold,float scale,int residual){
 int col=get_group_id(0)*32+get_local_id(0),lane=get_local_id(1),group=get_group_id(1);
 int end=min(rows,(group+1)*256);float g=0,b=0;
 for(int row=group*256+lane;row<end;row+=8)if(col<width){
  int i=row*width+col;float d=dy[i];
  float value=arc_norm_candidate_input(x,branch,i,seed,threshold,scale,residual);
  b+=d;g+=d*(value-stats[row*2])*stats[row*2+1];
 }
 __local float gs[8][32],bs[8][32];int c=get_local_id(0);
 gs[lane][c]=g;bs[lane][c]=b;barrier(CLK_LOCAL_MEM_FENCE);
 for(int stride=4;stride;stride/=2){
  if(lane<stride){gs[lane][c]+=gs[lane+stride][c];bs[lane][c]+=bs[lane+stride][c];}
  barrier(CLK_LOCAL_MEM_FENCE);
 }
 if(lane==0&&col<width){
  parts[group*width+col]=bs[0][c];
  parts[((rows+255)/256+group)*width+col]=gs[0][c];
 }
}
