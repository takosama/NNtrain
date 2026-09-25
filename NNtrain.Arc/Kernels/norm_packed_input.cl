float mask(uint seed,int i,uint threshold,float scale);
float norm_storage_read(__global const uchar* data,__global const float* scales,int i,int type,int block) {
    if(type==0)return ((__global const float*)data)[i];
    if(type==1)return as_float((uint)((__global const ushort*)data)[i]<<16);
    return (float)((__global const char*)data)[i]*scales[i/block];
}
__kernel void norm_packed_residual_input(
    __global const uchar* x,__global const float* xs,
    __global const uchar* b,__global const float* bs,__global float* output,
    int n,int xt,int bt,int xb,int bb,uint seed,uint threshold,float scale) {
    int i=get_global_id(0);if(i>=n)return;
    // Match separately materialized FP32 decode before the existing dropout
    // expression. Do not contract BFP8 decode into the subsequent arithmetic.
    volatile float residual=norm_storage_read(x,xs,i,xt,xb);
    volatile float branch=norm_storage_read(b,bs,i,bt,bb);
    output[i]=branch*mask(seed,i,threshold,scale)+residual;
}

// Experimental SG16 path for the production width-512 residual LayerNorm.
// Recreate the FP32 residual sum at every consumer instead of materializing
// a rows*width FP32 buffer. The serial channel order of the selected SG16
// normalization kernels is kept, including their reduction trees.
#if defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

inline float norm_packed_residual_value(
    __global const uchar* x,__global const float* xs,
    __global const uchar* branch,__global const float* bs,
    int index,int xt,int bt,int xb,int bb,
    uint seed,uint threshold,float scale) {
    volatile float residual=norm_storage_read(x,xs,index,xt,xb);
    volatile float dropped=norm_storage_read(branch,bs,index,bt,bb);
    // The old packed-input kernel stores this rounded FP32 value before norm.
    volatile float value=fma(dropped,mask(seed,index,threshold,scale),residual);
    return value;
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64,1,1)))
__kernel void norm_packed_residual_row_sg16_w64(
    __global const uchar* x,__global const float* xs,
    __global const uchar* branch,__global const float* bs,
    __global const float* gamma,__global const float* beta,
    __global float* y,__global float* stats,
    int rows,int width,float eps,int xt,int bt,int xb,int bb,
    uint seed,uint threshold,float scale) {
    int lane=get_sub_group_local_id(),subgroup=get_sub_group_id();
    int r=get_group_id(0)*4+subgroup;
    // One rounded FP32 residual value per element, as in the old global
    // scratch kernel. Four rows occupy 8 KiB of SLM (plus row padding).
    __local float values[4][513];
    for(int c=lane;c<width;c+=16)if(r<rows) {
        int i=r*width+c;
        values[subgroup][c]=norm_packed_residual_value(x,xs,branch,bs,i,
            xt,bt,xb,bb,seed,threshold,scale);
    }
    // All four subgroups must reach this barrier, including a tail row.
    barrier(CLK_LOCAL_MEM_FENCE);
    if(r>=rows)return;
    float mean=0;
    for(int base=0;base<width;base+=16) {
        int c=base+lane;
        float xv=c<width?values[subgroup][c]:0;
        #pragma unroll
        for(uint j=0;j<16;j++) {
            float value=intel_sub_group_shuffle(xv,j);
            if(base+j<width)mean+=value;
        }
    }
    mean/=width;
    float variance=0;
    for(int base=0;base<width;base+=16) {
        int c=base+lane;
        float xv=c<width?values[subgroup][c]:0;
        #pragma unroll
        for(uint j=0;j<16;j++) {
            float value=intel_sub_group_shuffle(xv,j);
            if(base+j<width) {float delta=value-mean;variance+=delta*delta;}
        }
    }
    float inv=rsqrt(variance/width+eps);
    if(lane==0) {stats[2*r]=mean;stats[2*r+1]=inv;}
    for(int c=lane;c<width;c+=16) {
        int i=r*width+c;
        float value=values[subgroup][c];
        y[i]=(value-mean)*inv*gamma[c]+beta[c];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64,1,1)))
__kernel void norm_packed_residual_dx_row_sg16_w64(
    __global const uchar* x,__global const float* xs,
    __global const uchar* branch,__global const float* bs,
    __global const float* gamma,__global const float* dy,__global const float* stats,
    __global float* dx,__global float* dbranch,
    int rows,int width,int xt,int bt,int xb,int bb,
    uint seed,uint threshold,float scale) {
    int lane=get_sub_group_local_id(),subgroup=get_sub_group_id();
    int r=get_group_id(0)*4+subgroup;
    __local float values[4][513];
    for(int c=lane;c<width;c+=16)if(r<rows) {
        int i=r*width+c;
        values[subgroup][c]=norm_packed_residual_value(x,xs,branch,bs,i,
            xt,bt,xb,bb,seed,threshold,scale);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(r>=rows)return;
    float mean=stats[2*r],inv=stats[2*r+1],sum=0,weighted=0;
    for(int base=0;base<width;base+=16) {
        int c=base+lane;
        float value=c<width?values[subgroup][c]:0;
        float gradient=c<width?dy[r*width+c]:0;
        float weight=c<width?gamma[c]:0;
        #pragma unroll
        for(uint j=0;j<16;j++) {
            float xv=intel_sub_group_shuffle(value,j);
            float dv=intel_sub_group_shuffle(gradient,j);
            float gv=intel_sub_group_shuffle(weight,j);
            if(base+j<width) {float g=dv*gv;sum+=g;weighted+=g*(xv-mean)*inv;}
        }
    }
    for(int c=lane;c<width;c+=16) {
        int i=r*width+c;
        float value=values[subgroup][c];
        // The old norm_dx kernel first stores a rounded FP32 scratch value.
        volatile float gradient=inv*(dy[i]*gamma[c]
            -(sum+(value-mean)*inv*weighted)/width);
        dx[i]+=gradient;
        dbranch[i]+=gradient*mask(seed,i,threshold,scale);
    }
}

__attribute__((reqd_work_group_size(32,8,1)))
__kernel void norm_packed_residual_parameter_parts(
    __global const float* dy,
    __global const uchar* x,__global const float* xs,
    __global const uchar* branch,__global const float* bs,
    __global const float* stats,__global float* parts,
    int rows,int width,int xt,int bt,int xb,int bb,
    uint seed,uint threshold,float scale) {
    int col=get_group_id(0)*32+get_local_id(0),lane=get_local_id(1),group=get_group_id(1);
    int end=min(rows,(group+1)*256);
    float g=0,b=0;
    for(int row=group*256+lane;row<end;row+=8)if(col<width) {
        int i=row*width+col;
        float d=dy[i];
        float value=norm_packed_residual_value(x,xs,branch,bs,i,
            xt,bt,xb,bb,seed,threshold,scale);
        b+=d;
        g+=d*(value-stats[row*2])*stats[row*2+1];
    }
    __local float gs[8][32],bsums[8][32];int c=get_local_id(0);
    gs[lane][c]=g;bsums[lane][c]=b;barrier(CLK_LOCAL_MEM_FENCE);
    for(int stride=4;stride;stride/=2) {
        if(lane<stride) {gs[lane][c]+=gs[lane+stride][c];bsums[lane][c]+=bsums[lane+stride][c];}
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lane==0&&col<width) {
        parts[group*width+col]=bsums[0][c];
        parts[((rows+255)/256+group)*width+col]=gs[0][c];
    }
}
#endif
