// Memory-bounded attention: no global [batch,heads,sequence,sequence] arrays.
// All three kernels launch global=batch*heads*sequence*64, local=64.
// QKV storage is [batch,sequence,3*width]; y/dy is [batch,sequence,width].
// stats is [batch,heads,sequence,2]: pre-quantization maximum and inverse sum.
// delta is [batch,heads,sequence], computed by dQ before the dK/dV dispatch.
//
// No backward calculation uses a potentially quantized forward output.
// delta(q)=sum_k p(q,k)*(dY(q) dot V(k)), exactly the softmax derivative's
// reduction. Avoiding dY dot quantized(Y) is important for mixed BFP8 training.
// Kernels use FP32 FMA/reductions, ordinary exp, and no unsafe fast math.

__attribute__((reqd_work_group_size(64,1,1)))
__kernel void attention_streaming(
    __global const float* qkv, __global float* y, __global float* stats,
    int batch, int seq, int width, int heads, int causal,
    __local float* memory) {
    const int row=get_group_id(0), lane=get_local_id(0);
    const int q=row%seq, h=(row/seq)%heads, b=row/(seq*heads);
    const int d=width/heads, count=causal?q+1:seq;
    const int query=(b*seq+q)*3*width+h*d;
    const float scale=rsqrt((float)d);
    __local float* scores=memory;       // seq floats
    __local float* scratch=memory+seq;  // 64 floats
    float maximum=-INFINITY;
    for(int key=lane;key<count;key+=64) {
        const int keyOffset=(b*seq+key)*3*width+width+h*d;
        float value=0;
        for(int channel=0;channel<d;channel++)
            value=fma(qkv[query+channel],qkv[keyOffset+channel],value);
        value*=scale;scores[key]=value;maximum=fmax(maximum,value);
    }
    scratch[lane]=maximum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int stride=32;stride>0;stride/=2) {
        if(lane<stride)scratch[lane]=fmax(scratch[lane],scratch[lane+stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maximum=scratch[0];
    // All work-items must finish reading maximum before scratch is reused.
    barrier(CLK_LOCAL_MEM_FENCE);
    float total=0;
    for(int key=lane;key<count;key+=64) {
        const float p=exp(scores[key]-maximum);
        scores[key]=p;total+=p;
    }
    scratch[lane]=total;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int stride=32;stride>0;stride/=2) {
        if(lane<stride)scratch[lane]+=scratch[lane+stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inverse=1.0f/scratch[0];
    if(lane==0){stats[2*row]=maximum;stats[2*row+1]=inverse;}
    for(int key=lane;key<count;key+=64)scores[key]*=inverse;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int channel=lane;channel<d;channel+=64) {
        float result=0;
        for(int key=0;key<count;key++)
            result=fma(scores[key],qkv[(b*seq+key)*3*width+2*width+h*d+channel],result);
        y[(b*seq+q)*width+h*d+channel]=result;
    }
}

__attribute__((reqd_work_group_size(64,1,1)))
__kernel void attention_streaming_dq(
    __global const float* qkv, __global const float* dy,
    __global const float* stats, __global float* dx, __global float* delta,
    int batch, int seq, int width, int heads, int causal,
    __local float* memory) {
    const int row=get_group_id(0), lane=get_local_id(0);
    const int q=row%seq, h=(row/seq)%heads, b=row/(seq*heads);
    const int d=width/heads, count=causal?q+1:seq;
    const int query=(b*seq+q)*3*width+h*d;
    const int gradient=(b*seq+q)*width+h*d;
    const float scale=rsqrt((float)d), maximum=stats[2*row], inverse=stats[2*row+1];
    __local float* probabilities=memory;     // seq floats
    __local float* derivatives=memory+seq;   // seq floats
    __local float* scratch=memory+2*seq;     // 64 floats
    float reduction=0;
    for(int key=lane;key<count;key+=64) {
        const int keyOffset=(b*seq+key)*3*width+width+h*d;
        const int valueOffset=keyOffset+width;
        float score=0, dp=0;
        for(int channel=0;channel<d;channel++) {
            score=fma(qkv[query+channel],qkv[keyOffset+channel],score);
            dp=fma(dy[gradient+channel],qkv[valueOffset+channel],dp);
        }
        const float p=exp(score*scale-maximum)*inverse;
        probabilities[key]=p;derivatives[key]=dp;reduction=fma(p,dp,reduction);
    }
    scratch[lane]=reduction;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int stride=32;stride>0;stride/=2) {
        if(lane<stride)scratch[lane]+=scratch[lane+stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float rowDelta=scratch[0];
    if(lane==0)delta[row]=rowDelta;
    for(int key=lane;key<count;key+=64)
        derivatives[key]=probabilities[key]*(derivatives[key]-rowDelta)*scale;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int channel=lane;channel<d;channel+=64) {
        float result=0;
        for(int key=0;key<count;key++)
            result=fma(derivatives[key],qkv[(b*seq+key)*3*width+width+h*d+channel],result);
        dx[query+channel]+=result;
    }
}

__attribute__((reqd_work_group_size(64,1,1)))
__kernel void attention_streaming_dkv(
    __global const float* qkv, __global const float* dy,
    __global const float* stats, __global const float* delta, __global float* dx,
    int batch, int seq, int width, int heads, int causal,
    __local float* memory) {
    const int row=get_group_id(0), lane=get_local_id(0);
    const int key=row%seq, h=(row/seq)%heads, b=row/(seq*heads);
    const int d=width/heads, start=causal?key:0;
    const int keyOffset=(b*seq+key)*3*width+width+h*d;
    const int valueOffset=keyOffset+width;
    const float scale=rsqrt((float)d);
    __local float* probabilities=memory;     // seq floats
    __local float* derivatives=memory+seq;   // seq floats
    for(int q=start+lane;q<seq;q+=64) {
        const int query=(b*seq+q)*3*width+h*d;
        const int gradient=(b*seq+q)*width+h*d;
        const int stat=(b*heads+h)*seq+q;
        float score=0, dp=0;
        for(int channel=0;channel<d;channel++) {
            score=fma(qkv[query+channel],qkv[keyOffset+channel],score);
            dp=fma(dy[gradient+channel],qkv[valueOffset+channel],dp);
        }
        const float p=exp(score*scale-stats[2*stat])*stats[2*stat+1];
        probabilities[q]=p;derivatives[q]=p*(dp-delta[stat])*scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int channel=lane;channel<d;channel+=64) {
        float dk=0,dv=0;
        for(int q=start;q<seq;q++) {
            dk=fma(derivatives[q],qkv[(b*seq+q)*3*width+h*d+channel],dk);
            dv=fma(probabilities[q],dy[(b*seq+q)*width+h*d+channel],dv);
        }
        dx[keyOffset+channel]+=dk;dx[valueOffset+channel]+=dv;
    }
}
