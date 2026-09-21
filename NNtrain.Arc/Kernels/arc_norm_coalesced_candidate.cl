// Strict-order LayerNorm dX experiments. The scalar arithmetic expressions and
// increasing-channel sums match norm_dx_set; only memory access changes.

// Row-major X/DY -> column-major scratch, sharing one transpose barrier.
// global=(ceil(width/32)*16,ceil(rows/32)*16), local=(16,16).
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void norm_dx_pack_columns_candidate(
 __global const float* x,__global const float* dy,__global float* xc,__global float* dyc,int rows,int width){
 int lx=get_local_id(0),ly=get_local_id(1),rb=get_group_id(1)*32,cb=get_group_id(0)*32;
 __local float xs[32][33],ds[32][33];
 #pragma unroll
 for(int i=0;i<2;i++){
  #pragma unroll
  for(int j=0;j<2;j++){
   int r=rb+ly+16*i,c=cb+lx+16*j;
   xs[ly+16*i][lx+16*j]=(r<rows&&c<width)?x[r*width+c]:0;
   ds[ly+16*i][lx+16*j]=(r<rows&&c<width)?dy[r*width+c]:0;
  }
 }
 barrier(CLK_LOCAL_MEM_FENCE);
 #pragma unroll
 for(int i=0;i<2;i++){
  #pragma unroll
  for(int j=0;j<2;j++){
   int r=rb+lx+16*j,c=cb+ly+16*i;
   if(r<rows&&c<width){xc[c*rows+r]=xs[lx+16*j][ly+16*i];dyc[c*rows+r]=ds[lx+16*j][ly+16*i];}
  }
 }
}

// In-place XC -> dXC is safe: after its complete serial reduction each row
// owner reads only the individual channel it is about to replace. Different
// work-items own disjoint rows, so no cross-work-item value is overwritten.
__kernel void norm_dx_columns_serial_candidate(
 __global float* xc,__global const float* gamma,__global const float* dyc,
 __global const float* stats,int rows,int width){
 int r=get_global_id(0);if(r>=rows)return;
 float m=stats[2*r],iv=stats[2*r+1],s=0,t=0;
 for(int c=0;c<width;c++){float g=dyc[c*rows+r]*gamma[c];s+=g;t+=g*(xc[c*rows+r]-m)*iv;}
 for(int c=0;c<width;c++){int i=c*rows+r;xc[i]=iv*(dyc[i]*gamma[c]-(s+(xc[i]-m)*iv*t)/width);}
}

// Forward variant consumes and replaces one column-major X scratch. Statistics
// and all three channel traversals are identical to the original norm kernel.
__kernel void norm_columns_serial_candidate(
 __global float* xc,__global const float* gamma,__global const float* beta,
 __global float* stats,int rows,int width,float eps){
 int r=get_global_id(0);if(r>=rows)return;
 float mean=0;for(int c=0;c<width;c++)mean+=xc[c*rows+r];mean/=width;
 float var=0;for(int c=0;c<width;c++){float d=xc[c*rows+r]-mean;var+=d*d;}float inv=rsqrt(var/width+eps);
 stats[2*r]=mean;stats[2*r+1]=inv;
 for(int c=0;c<width;c++){int i=c*rows+r;xc[i]=(xc[i]-mean)*inv*gamma[c]+beta[c];}
}

// General FP32 tiled transpose. For restoring dX call with inputRows=width,
// inputColumns=rows, global=(ceil(rows/32)*16,ceil(width/32)*16).
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void norm_dx_unpack_columns_candidate(
 __global const float* input,__global float* output,int inputRows,int inputColumns){
 int lx=get_local_id(0),ly=get_local_id(1),rb=get_group_id(1)*32,cb=get_group_id(0)*32;
 __local float tile[32][33];
 #pragma unroll
 for(int i=0;i<2;i++){
  #pragma unroll
  for(int j=0;j<2;j++){
   int r=rb+ly+16*i,c=cb+lx+16*j;
   tile[ly+16*i][lx+16*j]=(r<inputRows&&c<inputColumns)?input[r*inputColumns+c]:0;
  }
 }
 barrier(CLK_LOCAL_MEM_FENCE);
 #pragma unroll
 for(int i=0;i<2;i++){
  #pragma unroll
  for(int j=0;j<2;j++){
   int r=rb+lx+16*j,c=cb+ly+16*i;
   if(r<inputRows&&c<inputColumns)output[c*inputRows+r]=tile[lx+16*j][ly+16*i];
  }
 }
}

#if defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

// SG16 coalesces 16 channels, then supplies them in the original c=0..width-1
// order. Broadcast X, DY and gamma separately so the original g/s/t expression
// retains its contraction opportunities; do not pre-round a weighted product.
// Same ABI as norm_dx_set. global=ceil(rows/(WG/16))*WG, local=WG.
#define NORM_DX_ROW_SG(NAME,WG) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(WG,1,1))) \
__kernel void NAME(__global const float* x,__global const float* gamma,__global const float* dy, \
 __global const float* stats,__global float* dx,int rows,int width){ \
 int lane=get_sub_group_local_id(),r=get_group_id(0)*(WG/16)+get_sub_group_id();if(r>=rows)return; \
 float m=stats[2*r],iv=stats[2*r+1],s=0,t=0; \
 for(int base=0;base<width;base+=16){ \
  int c=base+lane;float xv=0,dv=0,gv=0;if(c<width){xv=x[r*width+c];dv=dy[r*width+c];gv=gamma[c];} \
  _Pragma("unroll") \
  for(uint j=0;j<16;j++){ \
   float value=intel_sub_group_shuffle(xv,j),gradient=intel_sub_group_shuffle(dv,j),weight=intel_sub_group_shuffle(gv,j); \
   if(base+j<width){float g=gradient*weight;s+=g;t+=g*(value-m)*iv;} \
  } \
 } \
 for(int c=lane;c<width;c+=16){int i=r*width+c;dx[i]=iv*(dy[i]*gamma[c]-(s+(x[i]-m)*iv*t)/width);} \
}
NORM_DX_ROW_SG(norm_dx_row_sg16_w64_candidate,64)
NORM_DX_ROW_SG(norm_dx_row_sg16_w128_candidate,128)
#undef NORM_DX_ROW_SG

// Forward SG rows retain the original serial mean/variance reduction order.
// Same ABI as norm, same subgroup launch mapping as the dX variants.
#define NORM_ROW_SG(NAME,WG) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(WG,1,1))) \
__kernel void NAME(__global const float* x,__global const float* gamma,__global const float* beta, \
 __global float* y,__global float* stats,int rows,int width,float eps){ \
 int lane=get_sub_group_local_id(),r=get_group_id(0)*(WG/16)+get_sub_group_id();if(r>=rows)return; \
 float mean=0; \
 for(int base=0;base<width;base+=16){int c=base+lane;float xv=c<width?x[r*width+c]:0; \
  _Pragma("unroll") \
  for(uint j=0;j<16;j++){float value=intel_sub_group_shuffle(xv,j);if(base+j<width)mean+=value;} \
 } \
 mean/=width;float var=0; \
 for(int base=0;base<width;base+=16){int c=base+lane;float xv=c<width?x[r*width+c]:0; \
  _Pragma("unroll") \
  for(uint j=0;j<16;j++){float value=intel_sub_group_shuffle(xv,j);if(base+j<width){float d=value-mean;var+=d*d;}} \
 } \
 float inv=rsqrt(var/width+eps);if(lane==0){stats[2*r]=mean;stats[2*r+1]=inv;} \
 for(int c=lane;c<width;c+=16){int i=r*width+c;y[i]=(x[i]-mean)*inv*gamma[c]+beta[c];} \
}
NORM_ROW_SG(norm_row_sg16_w64_candidate,64)
NORM_ROW_SG(norm_row_sg16_w128_candidate,128)
#undef NORM_ROW_SG
#endif
