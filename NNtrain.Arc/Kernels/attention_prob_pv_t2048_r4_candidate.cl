// T2048/D32 probability + PV experiment. Four query rows keep the complete
// probability tile in 32 KiB of local memory, so the FP32 P matrix never
// needs a global write or read in forward. The reduction tree and ordered
// per-key FP32 FMA match attention_probabilities + attention_fp32_pv_d32_aligned.
// Local memory: 4*2048*4 + 32*33*4 + 4*64*4 = 38,016 bytes.
__attribute__((reqd_work_group_size(32,4,1)))
__kernel void attention_prob_pv_fused_t2048_r4_candidate(
    __global const float* qkv, __global const float* scores,
    __global float* output, __global float* stats,
    int seq, int width, int heads, int first, int causal) {
    const int lane=get_local_id(0),r=get_local_id(1),tid=r*32+lane;
    const int g=get_group_id(2),h=first+g,b=h/heads,d=width/heads,hd=(h%heads)*d;
    const int qb=get_group_id(1)*4,query=qb+r;
    // The ordinary probability kernel publishes zeros through a 64-key
    // causal tile; retain that boundary even though each workgroup owns four
    // query rows.
    const int limit=causal?min(seq,((qb/64)+1)*64):seq;
    const int count=causal?min(seq,query+1):seq;
    const int pbase=g*seq*seq,qbase=b*seq*3*width+hd;
    const float scale=rsqrt((float)d);
    __local float probabilities[4][2048],values[32][33],reduction[4][64];

    float maxLow=-INFINITY,maxHigh=-INFINITY;
    for(int key=lane;key<count;key+=32) {
        const float score=scores[pbase+query*seq+key];
        probabilities[r][key]=score;
        if((key&32)==0)maxLow=fmax(maxLow,score*scale);
        else maxHigh=fmax(maxHigh,score*scale);
    }
    reduction[r][lane]=maxLow;
    reduction[r][lane+32]=maxHigh;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int stride=32;stride;stride/=2) {
        if(lane<stride)reduction[r][lane]=fmax(reduction[r][lane],reduction[r][lane+stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float maximum=reduction[r][0];
    barrier(CLK_LOCAL_MEM_FENCE);

    float sumLow=0,sumHigh=0;
    for(int key=lane;key<limit;key+=32) {
        const float p=key<count?exp(probabilities[r][key]*scale-maximum):0;
        probabilities[r][key]=p;
        if((key&32)==0)sumLow+=p;
        else sumHigh+=p;
    }
    reduction[r][lane]=sumLow;
    reduction[r][lane+32]=sumHigh;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int stride=32;stride;stride/=2) {
        if(lane<stride)reduction[r][lane]+=reduction[r][lane+stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inverse=1.0f/reduction[r][0];
    if(lane==0) {
        stats[2*(h*seq+query)]=maximum;
        stats[2*(h*seq+query)+1]=inverse;
    }
    for(int key=lane;key<limit;key+=32)probabilities[r][key]*=inverse;
    barrier(CLK_LOCAL_MEM_FENCE);

    float value=0;
    for(int base=0;base<limit;base+=32) {
        for(int i=tid;i<1024;i+=128) {
            const int key=base+i/32,channel=i%32;
            values[i/32][channel]=qkv[qbase+key*3*width+2*width+channel];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int key=0;key<32;key++)
            value=fma(probabilities[r][base+key],values[key][lane],value);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    output[(b*seq+query)*width+hd+lane]=value;
}
