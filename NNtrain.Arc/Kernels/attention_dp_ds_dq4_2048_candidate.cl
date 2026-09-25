// T=2048, D=32 backward candidate. Four query rows share one V/K tile.
// The 4xT local derivative cache uses 32 KiB at T=2048. Together with the
// fixed tiles this remains below the device's 64 KiB local-memory limit.
// dP uses the original channel FMA order; softmax uses the original modulo-64
// reduction; dQ uses the original ascending-key FMA order. dS is also written
// to global memory for the following dK/dV kernel.
// global=(32,round_up(T,4),tileHeads), local=(32,4,1).
__attribute__((reqd_work_group_size(32,4,1)))
__kernel void attention_dp_ds_dq4_t2048_candidate(
    __global const float* qkv, __global const float* dy,
    __global const float* p, __global float* ds, __global float* dx,
    int seq, int width, int heads, int first, int causal,
    __local float* derivatives) {
    const int lane=get_local_id(0),r=get_local_id(1),tid=r*32+lane;
    const int g=get_group_id(2),h=first+g,b=h/heads,d=width/heads,hd=(h%heads)*d;
    const int qb=get_group_id(1)*4,query=qb+r;
    // Rounded causal bound matches the 64-row dQ tile. In particular all
    // invalid keys consumed by dQ are explicitly published as zero.
    const int limit=causal?min(seq,(qb/64+1)*64):seq;
    const int count=causal?min(seq,query+1):seq;
    const int published=causal==2?min(seq,(query/64+1)*64):seq;
    const int qbase=b*seq*3*width+hd,ybase=b*seq*width+hd,pbase=g*seq*seq;
    __local float values[32][33],gradient[4][33],sums[4][64];
    gradient[r][lane]=(query<seq&&lane<d)?dy[ybase+query*width+lane]:0;
    barrier(CLK_LOCAL_MEM_FENCE);

    float sumLow=0,sumHigh=0;
    for(int base=0;base<limit;base+=32) {
        for(int i=tid;i<1024;i+=128) {
            const int key=base+i/32,channel=i%32;
            values[i/32][channel]=(key<seq&&channel<d)?qkv[qbase+key*3*width+2*width+channel]:0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const int key=base+lane;
        float dp=0;
        if(query<seq&&key<count) {
            for(int c=0;c<d;c++)dp=fma(gradient[r][c],values[lane][c],dp);
            const float probability=p[pbase+query*seq+key];
            if((base&32)==0)sumLow=fma(probability,dp,sumLow);
            else sumHigh=fma(probability,dp,sumHigh);
        }
        if(key<seq)derivatives[r*seq+key]=dp;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sums[r][lane]=sumLow;sums[r][lane+32]=sumHigh;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int stride=32;stride;stride/=2) {
        if(lane<stride)sums[r][lane]+=sums[r][lane+stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float delta=sums[r][0],scale=rsqrt((float)d);
    if(query<seq)for(int key=lane;key<published;key+=32) {
        const float value=key<count?p[pbase+query*seq+key]*(derivatives[r*seq+key]-delta)*scale:0;
        ds[pbase+query*seq+key]=value;
        if(key<limit)derivatives[r*seq+key]=value;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float dq=(query<seq&&lane<d)?dx[qbase+query*3*width+lane]:0;
    for(int base=0;base<limit;base+=32) {
        for(int i=tid;i<1024;i+=128) {
            const int key=base+i/32,channel=i%32;
            values[i/32][channel]=(key<seq&&channel<d)?qkv[qbase+key*3*width+width+channel]:0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if(query<seq&&lane<d)for(int key=0;key<32;key++)
            dq=fma(derivatives[r*seq+base+key],values[key][lane],dq);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(query<seq&&lane<d)dx[qbase+query*3*width+lane]=dq;
}
