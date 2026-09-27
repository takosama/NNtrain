// Full sequence FP32 adjoints. Each output element has one writer.
__kernel void q35t_zero_range(__global float* result,int offset,int length) {
    int i=get_global_id(0);if(i<length)result[offset+i]=0.0f;
}
__kernel void q35t_norm2_split(__global const float* x,__global float* result,int n,int offset,int splits) {
    int group=get_group_id(0),tid=get_local_id(0);__local float v[128];float sum=0;
    for(int i=group*128+tid;i<n;i+=128*splits){float g=x[i];sum+=isfinite(g)?g*g:INFINITY;}
    v[tid]=sum;barrier(CLK_LOCAL_MEM_FENCE);
    for(int s=64;s>0;s>>=1){if(tid<s)v[tid]+=v[tid+s];barrier(CLK_LOCAL_MEM_FENCE);}if(tid==0)result[offset+group]=v[0];
}
__kernel void q35t_add_offset(__global const float* x,__global float* y,int n,int offset) {
    int i=get_global_id(0);if(i<n)y[offset+i]+=x[i];
}
__kernel void q35t_lora_dz_coop(__global const float* dy,__global const float* b,__global float* dz,
    int rows,int output,int rank,float scale) {
    int item=get_group_id(0),tid=get_local_id(0),t=item/rank,r=item%rank;
    __local float partial[128];float sum=0;
    for(int o=tid;o<output;o+=128)sum=fma(dy[t*output+o],b[o*rank+r],sum);
    partial[tid]=sum;barrier(CLK_LOCAL_MEM_FENCE);
    for(int s=64;s>0;s>>=1){if(tid<s)partial[tid]+=partial[tid+s];barrier(CLK_LOCAL_MEM_FENCE);}
    if(tid==0)dz[item]=partial[0]*scale;
}
__kernel void q35t_add(__global const float* x,__global const float* y,__global float* z,int n) {
    int i=get_global_id(0); if(i<n) z[i]=x[i]+y[i];
}
__kernel void q35t_rms_dx(__global const float* x,__global const float* w,
    __global const float* dy,__global float* dx,int rows,int width,float eps) {
    int row=get_group_id(0), tid=get_local_id(0);
    __local float a[128],b[128]; float xx=0,dot=0;
    for(int j=tid;j<width;j+=128) {float v=x[row*width+j];xx+=v*v;dot+=v*w[j]*dy[row*width+j];}
    a[tid]=xx;b[tid]=dot;barrier(CLK_LOCAL_MEM_FENCE);
    for(int s=64;s>0;s>>=1) {if(tid<s){a[tid]+=a[tid+s];b[tid]+=b[tid+s];}barrier(CLK_LOCAL_MEM_FENCE);}
    float inv=1/sqrt(a[0]/width+eps),c=b[0]*inv*inv/width;
    for(int j=tid;j<width;j+=128) dx[row*width+j]+=inv*(dy[row*width+j]*w[j]-x[row*width+j]*c);
}
__kernel void q35t_silu_dx(__global const float* gate,__global const float* up,__global const float* dy,
    __global float* dg,__global float* du,int n) {
    int i=get_global_id(0);if(i>=n)return;
    float x=gate[i],s=1/(1+exp(-x));
    dg[i]+=dy[i]*up[i]*s*(1+x*(1-s));du[i]+=dy[i]*x*s;
}
__kernel void q35t_lora_dz(__global const float* dy,__global const float* b,__global float* dz,
    int rows,int output,int rank,float scale) {
    int i=get_global_id(0);if(i>=rows*rank)return;int t=i/rank,r=i%rank;float sum=0;
    for(int o=0;o<output;o++)sum=fma(dy[t*output+o],b[o*rank+r],sum);dz[i]=sum*scale;
}
__kernel void q35t_lora_db(__global const float* dy,__global const float* z,__global float* db,
    int rows,int output,int rank,float scale) {
    int i=get_global_id(0);if(i>=output*rank)return;int o=i/rank,r=i%rank;float sum=0;
    for(int t=0;t<rows;t++)sum=fma(dy[t*output+o],z[t*rank+r],sum);db[i]+=sum*scale;
}
__kernel void q35t_lora_da(__global const float* dz,__global const float* x,__global float* da,
    int rows,int width,int rank) {
    int i=get_global_id(0);if(i>=rank*width)return;int r=i/width,j=i%width;float sum=0;
    for(int t=0;t<rows;t++)sum=fma(dz[t*rank+r],x[t*width+j],sum);da[i]+=sum;
}
// First contribution after ZeroGrad: replace stale storage without a separate
// clear launch. Later backward calls retain the additive kernels above.
__kernel void q35t_lora_db_write(__global const float* dy,__global const float* z,__global float* db,
    int rows,int output,int rank,float scale) {
    int i=get_global_id(0);if(i>=output*rank)return;int o=i/rank,r=i%rank;float sum=0;
    for(int t=0;t<rows;t++)sum=fma(dy[t*output+o],z[t*rank+r],sum);db[i]=0.0f+sum*scale;
}
__kernel void q35t_lora_da_write(__global const float* dz,__global const float* x,__global float* da,
    int rows,int width,int rank) {
    int i=get_global_id(0);if(i>=rank*width)return;int r=i/width,j=i%width;float sum=0;
    for(int t=0;t<rows;t++)sum=fma(dz[t*rank+r],x[t*width+j],sum);da[i]=0.0f+sum;
}
__kernel void q35t_lora_dx(__global const float* dz,__global const float* a,__global float* dx,
    int rows,int width,int rank) {
    int i=get_global_id(0);if(i>=rows*width)return;int t=i/width,j=i%width;float sum=0;
    for(int r=0;r<rank;r++)sum=fma(dz[t*rank+r],a[r*width+j],sum);dx[i]+=sum;
}
// Masked response-only cross entropy. stats = [loss, max, sum] per row.
__kernel void q35t_ce_stats(__global const float* logits,__global const int* targets,
    __global float* stats,int vocab,int valid) {
    int t=get_group_id(0),tid=get_local_id(0);__local float reduction[128];
    float maxv=-INFINITY;int bad=0;
    for(int j=tid;j<vocab;j+=128){float v=logits[t*vocab+j];bad|=!isfinite(v);maxv=fmax(maxv,v);}
    reduction[tid]=bad?INFINITY:maxv;barrier(CLK_LOCAL_MEM_FENCE);
    for(int s=64;s>0;s>>=1){if(tid<s)reduction[tid]=fmax(reduction[tid],reduction[tid+s]);barrier(CLK_LOCAL_MEM_FENCE);}
    maxv=reduction[0];barrier(CLK_LOCAL_MEM_FENCE);float sum=0;
    for(int j=tid;j<vocab;j+=128)sum+=exp(logits[t*vocab+j]-maxv);
    reduction[tid]=sum;barrier(CLK_LOCAL_MEM_FENCE);
    for(int s=64;s>0;s>>=1){if(tid<s)reduction[tid]+=reduction[tid+s];barrier(CLK_LOCAL_MEM_FENCE);}
    if(tid==0){int target=targets[t];stats[t*3]=!isfinite(maxv)?NAN:target<0?0:(log(reduction[0])+maxv-logits[t*vocab+target])/valid;stats[t*3+1]=maxv;stats[t*3+2]=reduction[0];}
}
__kernel void q35t_ce_grad(__global const float* logits,__global const int* targets,
    __global const float* stats,__global float* grad,int rows,int vocab,int valid) {
    int i=get_global_id(0);if(i>=rows*vocab)return;int t=i/vocab,j=i%vocab,target=targets[t];
    grad[i]=target<0?0:(exp(logits[i]-stats[t*3+1])/stats[t*3+2]-(float)(j==target))/valid;
}
__kernel void q35t_norm2(__global const float* x,__global float* result,int n) {
    int tid=get_local_id(0);__local float v[128];float sum=0;
    for(int i=tid;i<n;i+=128){float g=x[i];sum+=isfinite(g)?g*g:INFINITY;}
    v[tid]=sum;barrier(CLK_LOCAL_MEM_FENCE);
    for(int s=64;s>0;s>>=1){if(tid<s)v[tid]+=v[tid+s];barrier(CLK_LOCAL_MEM_FENCE);}if(tid==0)result[0]=v[0];
}
__kernel void q35t_adam(__global float* w,__global const float* g,__global float* m,__global float* v,
    int n,float lr,float decay,float clip,float correction1,float correction2) {
    int i=get_global_id(0);if(i>=n)return;float grad=g[i]*clip;
    float mm=.9f*m[i]+.1f*grad,vv=.999f*v[i]+.001f*grad*grad;m[i]=mm;v[i]=vv;
    w[i]=w[i]*(1-lr*decay)-lr*(mm/correction1)/(sqrt(vv/correction2)+1e-8f);
}
__kernel void q35t_xpose_reduce(__global const float* partial,__global float* dx,
    int rows,int width,int splits) {
    int i=get_global_id(0);if(i>=rows*width)return;int t=i/width,j=i%width;float sum=0;
    for(int s=0;s<splits;s++)sum+=partial[(t*splits+s)*width+j];dx[i]+=sum;
}
