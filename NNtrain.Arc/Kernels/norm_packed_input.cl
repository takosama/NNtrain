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
