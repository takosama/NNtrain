// Low-rank branches are separate FP32 buffers; the GGUF is never rewritten.
__kernel void q35l_lora_a(__global const float* x, __global const float* a,
    __global float* z, int rows, int width, int rank) {
    int i=get_global_id(0); if(i>=rows*rank) return;
    int t=i/rank, r=i%rank; float sum=0;
    for(int j=0;j<width;j++) sum=fma(x[t*width+j],a[r*width+j],sum);
    z[i]=sum;
}
__kernel void q35l_lora_b(__global const float* z, __global const float* b,
    __global float* y, int rows, int output, int rank, float scale) {
    int i=get_global_id(0); if(i>=rows*output) return;
    int t=i/output, o=i%output; float sum=0;
    for(int r=0;r<rank;r++) sum=fma(z[t*rank+r],b[o*rank+r],sum);
    y[i]+=scale*sum;
}
