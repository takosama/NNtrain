// Mix8_16 normalization boundaries for A/B. Keep the established SG16 row
// traversal and FP32 arithmetic; change only the output ownership boundary.
#if defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

float mask(uint seed,int i,uint threshold,float scale);

inline ushort norm_mix8_16_pack_bf16(float value) {
    uint bits=as_uint(value);
    if((bits&0x7f800000u)!=0x7f800000u)
        bits+=0x7fffu+((bits>>16)&1u);
    else if((bits&0x007fffffu)!=0)
        bits|=0x00400000u;
    return (ushort)(bits>>16);
}

// Same operation order and launch shape as norm_row_sg16_w64_candidate.
// Its final FP32 value is converted by the same nearest-even codec as
// resident_bf16, without a full-sized FP32 result or separate publish pass.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64,1,1)))
__kernel void norm_row_sg16_w64_direct_bf16(
    __global const float* x,__global const float* gamma,__global const float* beta,
    __global ushort* y,__global float* stats,int rows,int width,float eps) {
    int lane=get_sub_group_local_id(),r=get_group_id(0)*4+get_sub_group_id();
    if(r>=rows)return;
    float mean=0;
    for(int base=0;base<width;base+=16){
        int c=base+lane;float xv=c<width?x[r*width+c]:0;
        #pragma unroll
        for(uint j=0;j<16;j++){
            float value=intel_sub_group_shuffle(xv,j);
            if(base+j<width)mean+=value;
        }
    }
    mean/=width;float var=0;
    for(int base=0;base<width;base+=16){
        int c=base+lane;float xv=c<width?x[r*width+c]:0;
        #pragma unroll
        for(uint j=0;j<16;j++){
            float value=intel_sub_group_shuffle(xv,j);
            if(base+j<width){float d=value-mean;var+=d*d;}
        }
    }
    float inv=rsqrt(var/width+eps);
    if(lane==0){stats[2*r]=mean;stats[2*r+1]=inv;}
    for(int c=lane;c<width;c+=16){
        int i=r*width+c;
        float value=(x[i]-mean)*inv*gamma[c]+beta[c];
        y[i]=norm_mix8_16_pack_bf16(value);
    }
}

// Same reduction as norm_dx_row_sg16_w64_candidate. The old FP32 scratch
// store is retained as a private volatile value before the two ordered adds.
// One work-item owns each element, including when both destinations alias.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64,1,1)))
__kernel void norm_dx_row_sg16_w64_residual_accumulate_bf16(
    __global const float* x,__global const float* gamma,__global const float* dy,
    __global const float* stats,__global float* dx,__global float* dbranch,
    int rows,int width,uint seed,uint threshold,float scale) {
    int lane=get_sub_group_local_id(),r=get_group_id(0)*4+get_sub_group_id();
    if(r>=rows)return;
    float m=stats[2*r],iv=stats[2*r+1],s=0,t=0;
    for(int base=0;base<width;base+=16){
        int c=base+lane;float xv=0,dv=0,gv=0;
        if(c<width){xv=x[r*width+c];dv=dy[r*width+c];gv=gamma[c];}
        #pragma unroll
        for(uint j=0;j<16;j++){
            float value=intel_sub_group_shuffle(xv,j);
            float gradient=intel_sub_group_shuffle(dv,j);
            float weight=intel_sub_group_shuffle(gv,j);
            if(base+j<width){float g=gradient*weight;s+=g;t+=g*(value-m)*iv;}
        }
    }
    for(int c=lane;c<width;c+=16){
        int i=r*width+c;
        volatile float gradient=iv*(dy[i]*gamma[c]-(s+(x[i]-m)*iv*t)/width);
        dx[i]+=gradient;
        dbranch[i]+=gradient*mask(seed,i,threshold,scale);
    }
}

// Mix8_16 speed-policy candidate: each SG16 lane sums differences from the
// first value, then the subgroup combines 16 partials. Anchoring prevents a
// large common offset from swallowing small residual differences in FP32.
// Variance remains a separate centered pass.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64,1,1)))
__kernel void norm_row_sg16_w64_parallel_bf16(
    __global const float* x,__global const float* gamma,__global const float* beta,
    __global ushort* y,__global float* stats,int rows,int width,float eps) {
    int lane=get_sub_group_local_id(),r=get_group_id(0)*4+get_sub_group_id();
    if(r>=rows)return;
    float anchor=x[r*width],sum=0;
    for(int c=lane;c<width;c+=16)sum+=x[r*width+c]-anchor;
    float mean=anchor+sub_group_reduce_add(sum)/width;
    float variance=0;
    for(int c=lane;c<width;c+=16){
        float delta=x[r*width+c]-mean;
        variance+=delta*delta;
    }
    float inv=rsqrt(sub_group_reduce_add(variance)/width+eps);
    if(lane==0){stats[2*r]=mean;stats[2*r+1]=inv;}
    for(int c=lane;c<width;c+=16){
        int i=r*width+c;
        float value=(x[i]-mean)*inv*gamma[c]+beta[c];
        y[i]=norm_mix8_16_pack_bf16(value);
    }
}

// The same lane-local/subgroup reduction computes the two backward sums.
// Final writes remain ordered for the aliased residual/branch case.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64,1,1)))
__kernel void norm_dx_row_sg16_w64_parallel_residual_bf16(
    __global const float* x,__global const float* gamma,__global const float* dy,
    __global const float* stats,__global float* dx,__global float* dbranch,
    int rows,int width,uint seed,uint threshold,float scale) {
    int lane=get_sub_group_local_id(),r=get_group_id(0)*4+get_sub_group_id();
    if(r>=rows)return;
    float mean=stats[2*r],inv=stats[2*r+1],sum=0,weighted=0;
    for(int c=lane;c<width;c+=16){
        int i=r*width+c;
        float gradient=dy[i]*gamma[c];
        sum+=gradient;
        weighted+=gradient*(x[i]-mean)*inv;
    }
    sum=sub_group_reduce_add(sum);
    weighted=sub_group_reduce_add(weighted);
    for(int c=lane;c<width;c+=16){
        int i=r*width+c;
        volatile float gradient=inv*(dy[i]*gamma[c]
            -(sum+(x[i]-mean)*inv*weighted)/width);
        dx[i]+=gradient;
        dbranch[i]+=gradient*mask(seed,i,threshold,scale);
    }
}

// The parallel row derivative already reads dy, x and the saved statistics.
// Publish four-row beta/gamma partials from those reads for a later 256-row
// reduction. The four subgroups retain the established one-row ownership of
// dx and dbranch, including their ordered writes when the destinations alias.
// This candidate is dispatched only for width 512, keeping local storage at
// 16 KiB per workgroup.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64,1,1)))
__kernel void norm_dx_row_sg16_w64_parallel_residual_parameter_bf16(
    __global const float* x,__global const float* gamma,__global const float* dy,
    __global const float* stats,__global float* dx,__global float* dbranch,
    __global float* partials,int rows,int width,uint seed,uint threshold,float scale) {
    int lane=get_sub_group_local_id(),subgroup=get_sub_group_id();
    int group=get_group_id(0),r=group*4+subgroup;
    int fourRowGroups=(rows+3)/4;
    float mean=0,inv=0,sum=0,weighted=0;
    __local float betaPartials[4][512],gammaPartials[4][512];
    if(r<rows){
        mean=stats[2*r];inv=stats[2*r+1];
        for(int c=lane;c<width;c+=16){
            int i=r*width+c;
            float gradient=dy[i]*gamma[c];
            sum+=gradient;
            weighted+=gradient*(x[i]-mean)*inv;
        }
        sum=sub_group_reduce_add(sum);
        weighted=sub_group_reduce_add(weighted);
    }
    for(int c=lane;c<width;c+=16){
        float d=0,xv=0;
        if(r<rows){
            int i=r*width+c;
            d=dy[i];xv=x[i];
            volatile float gradient=inv*(d*gamma[c]
                -(sum+(xv-mean)*inv*weighted)/width);
            dx[i]+=gradient;
            dbranch[i]+=gradient*mask(seed,i,threshold,scale);
        }
        betaPartials[subgroup][c]=d;
        gammaPartials[subgroup][c]=d*(xv-mean)*inv;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(subgroup==0)for(int c=lane;c<width;c+=16){
        partials[group*width+c]=betaPartials[0][c]+betaPartials[1][c]
            +betaPartials[2][c]+betaPartials[3][c];
        partials[(fourRowGroups+group)*width+c]=gammaPartials[0][c]
            +gammaPartials[1][c]+gammaPartials[2][c]+gammaPartials[3][c];
    }
}

// Collapse 64 consecutive four-row partials into the same two-panel layout
// expected by gradient_rows_finish. No x, dy or statistics are read again.
__attribute__((reqd_work_group_size(32,8,1)))
__kernel void norm_parameter_4row_partials_256row(
    __global const float* fourRowPartials,__global float* tilePartials,
    int rows,int width) {
    int col=get_group_id(0)*32+get_local_id(0);
    int lane=get_local_id(1),group=get_group_id(1);
    int fourRowGroups=(rows+3)/4,groups=(rows+255)/256;
    int end=min(fourRowGroups,(group+1)*64);
    float b=0,g=0;
    for(int part=group*64+lane;part<end;part+=8)if(col<width){
        b+=fourRowPartials[part*width+col];
        g+=fourRowPartials[(fourRowGroups+part)*width+col];
    }
    __local float bs[8][32],gs[8][32];int c=get_local_id(0);
    bs[lane][c]=b;gs[lane][c]=g;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int stride=4;stride;stride/=2){
        if(lane<stride){bs[lane][c]+=bs[lane+stride][c];gs[lane][c]+=gs[lane+stride][c];}
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lane==0&&col<width){
        tilePartials[group*width+col]=bs[0][c];
        tilePartials[(groups+group)*width+col]=gs[0][c];
    }
}
#endif

float arc_gradient_bf16(float v);

// Bias-only form of gradient_rows(norm=2). Keep the same 256-row tile,
// per-lane row order, and 8->4->2->1 tree so repeated accumulation has
// identical rounding. The gamma partials and their local reduction are absent.
__attribute__((reqd_work_group_size(32,8,1)))
__kernel void gradient_rows_bias_bf16(
    __global const float* dy,__global const float* gate,__global float* parts,
    int rows,int width,int relu) {
    int col=get_group_id(0)*32+get_local_id(0);
    int lane=get_local_id(1),group=get_group_id(1);
    int end=min(rows,(group+1)*256);
    float b=0;
    for(int row=group*256+lane;row<end;row+=8)if(col<width){
        int idx=row*width+col;
        float d=relu&&gate[idx]<=0?0:dy[idx];
        d=arc_gradient_bf16(d);
        b+=d;
    }
    __local float bs[8][32];int c=get_local_id(0);
    bs[lane][c]=b;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int stride=4;stride;stride/=2){
        if(lane<stride)bs[lane][c]+=bs[lane+stride][c];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lane==0&&col<width)parts[group*width+col]=bs[0][c];
}
