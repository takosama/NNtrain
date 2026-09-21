// Experimental FP32 backward fusion. This file deliberately does not implement
// QK: existing BF16/XMX QK and its saved FP32 probabilities remain authoritative.
// P and DS are [tileHeads,sequence,sequence], QKV is [batch,sequence,3*width],
// DY is [batch,sequence,width]. Every output has a unique workgroup owner.

// Replaces the separate DS^T*Q and P^T*DY products. Each workgroup owns a
// 32-key x 32-channel tile and keeps both independent FP32 accumulators live.
// Query traversal and FMA ordering match attention_gemm_compact exactly.
// global=(ceil(d/32)*16,ceil(sequence/32)*16,tileHeads), local=(16,16,1).
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void attention_dkv_fused_candidate(
    __global const float* qkv, __global const float* dy,
    __global const float* p, __global const float* ds, __global float* dx,
    int seq, int width, int heads, int first, int causal) {
    const int lx=get_local_id(0),ly=get_local_id(1),tid=ly*16+lx;
    const int g=get_group_id(2),h=first+g,b=h/heads,d=width/heads,hd=(h%heads)*d;
    const int kb=get_group_id(1)*32,cb=get_group_id(0)*32;
    const int pbase=g*seq*seq,qbase=b*seq*3*width+hd,ybase=b*seq*width+hd;
    __local float ps[32][33],ss[32][33],qs[32][33],ys[32][33];
    float2 dk[2],dv[2];
    #pragma unroll
    for(int r=0;r<2;r++) {
        const int key=kb+ly+16*r;
        dk[r]=(float2)(0);dv[r]=(float2)(0);
        if(key<seq) {
            const int off=qbase+key*3*width+width+cb+lx;
            if(cb+lx<d){dk[r].s0=dx[off];dv[r].s0=dx[off+width];}
            if(cb+lx+16<d){dk[r].s1=dx[off+16];dv[r].s1=dx[off+width+16];}
        }
    }
    for(int base=causal?kb:0;base<seq;base+=32) {
        for(int i=tid;i<1024;i+=256) {
            const int query=i/32,key=i%32,channel=i%32;
            const int qi=base+query,ki=kb+key,ci=cb+channel;
            const int score=pbase+qi*seq+ki;
            ps[key][query]=(qi<seq&&ki<seq)?p[score]:0;
            ss[key][query]=(qi<seq&&ki<seq)?ds[score]:0;
            qs[query][channel]=(qi<seq&&ci<d)?qkv[qbase+qi*3*width+ci]:0;
            ys[query][channel]=(qi<seq&&ci<d)?dy[ybase+qi*width+ci]:0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for(int inner=0;inner<32;inner++) {
            if(base+inner<seq) {
                const float2 qv=(float2)(qs[inner][lx],qs[inner][lx+16]);
                const float2 yv=(float2)(ys[inner][lx],ys[inner][lx+16]);
                #pragma unroll
                for(int r=0;r<2;r++) {
                    dk[r]=fma((float2)(ss[ly+16*r][inner]),qv,dk[r]);
                    dv[r]=fma((float2)(ps[ly+16*r][inner]),yv,dv[r]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for(int r=0;r<2;r++) {
        const int key=kb+ly+16*r;
        if(key<seq) {
            const int off=qbase+key*3*width+width+cb+lx;
            if(cb+lx<d){dx[off]=dk[r].s0;dx[off+width]=dv[r].s0;}
            if(cb+lx+16<d){dx[off+16]=dk[r].s1;dx[off+width+16]=dv[r].s1;}
        }
    }
}

// Replaces dP=DY*V^T, the softmax derivative, and dQ=DS*K for d<=32.
// DS stays on chip between these stages, but is also published for dK.
// The two modulo-64 accumulators retain the existing derivative reduction
// order even though there are 32, rather than 64, work-items per query.
// localMemory requires (8*sequence)*sizeof(float); fixed local use is 7,328 B.
// global=(32,ceil(sequence/8)*8,tileHeads), local=(32,8,1).
__attribute__((reqd_work_group_size(32,8,1)))
__kernel void attention_dp_ds_dq_fused_candidate(
    __global const float* qkv, __global const float* dy,
    __global const float* p, __global float* ds, __global float* dx,
    int seq, int width, int heads, int first, int causal,
    __local float* derivatives) {
    const int lane=get_local_id(0),r=get_local_id(1),tid=r*32+lane;
    const int g=get_group_id(2),h=first+g,b=h/heads,d=width/heads,hd=(h%heads)*d;
    const int qb=get_group_id(1)*8,query=qb+r,limit=causal?min(seq,qb+8):seq;
    const int qbase=b*seq*3*width+hd,ybase=b*seq*width+hd,pbase=g*seq*seq;
    const int count=causal?min(seq,query+1):seq;
    __local float values[32][33],gradient[8][33],sums[8][64];
    gradient[r][lane]=(query<seq&&lane<d)?dy[ybase+query*width+lane]:0;
    barrier(CLK_LOCAL_MEM_FENCE);
    float sumLow=0,sumHigh=0;
    for(int base=0;base<limit;base+=32) {
        for(int i=tid;i<1024;i+=256) {
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
    // dK may read to the end of the enclosing 64-row causal tile, not merely
    // this kernel's 8-row tile. Explicitly zero that entire valid read region.
    const int published=causal?min(seq,((query/64)+1)*64):seq;
    if(query<seq)for(int key=lane;key<published;key+=32) {
        const float value=key<count?p[pbase+query*seq+key]*(derivatives[r*seq+key]-delta)*scale:0;
        ds[pbase+query*seq+key]=value;
        if(key<limit)derivatives[r*seq+key]=value;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float dq=(query<seq&&lane<d)?dx[qbase+query*3*width+lane]:0;
    for(int base=0;base<limit;base+=32) {
        for(int i=tid;i<1024;i+=256) {
            const int key=base+i/32,channel=i%32;
            values[i/32][channel]=(key<seq&&channel<d)?qkv[qbase+key*3*width+width+channel]:0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if(query<seq&&lane<d)for(int k=0;k<32&&base+k<limit;k++)
            dq=fma(derivatives[r*seq+base+k],values[k][lane],dq);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(query<seq&&lane<d)dx[qbase+query*3*width+lane]=dq;
}

// Forward companion: retain existing QK/XMX scores, fuse softmax with FP32 PV.
// Scores are not modified. The maximum and sum follow the original 64-lane
// reduction order, and PV retains ordered FP32 FMA over all valid keys.
// Only d<=32 is supported. Fixed local use is 6,272 B plus 8*seq*4 dynamic.
// global=(32,ceil(sequence/8)*8,tileHeads), local=(32,8,1).
__attribute__((reqd_work_group_size(32,8,1)))
__kernel void attention_prob_pv_fused_candidate(
    __global const float* qkv, __global const float* scores,
    __global float* output, __global float* stats,
    int seq, int width, int heads, int first, int causal,
    __local float* probabilities) {
    const int lane=get_local_id(0),r=get_local_id(1),tid=r*32+lane;
    const int g=get_group_id(2),h=first+g,b=h/heads,d=width/heads,hd=(h%heads)*d;
    const int qb=get_group_id(1)*8,query=qb+r,limit=causal?min(seq,qb+8):seq;
    const int count=causal?min(seq,query+1):seq;
    const int pbase=g*seq*seq,qbase=b*seq*3*width+hd;
    const float scale=rsqrt((float)d);
    __local float values[32][33],reduction[8][64];
    float maxLow=-INFINITY,maxHigh=-INFINITY;
    if(query<seq)for(int key=lane;key<count;key+=32) {
        // Retain the unscaled score: the reference exp(score*scale-maximum)
        // may contract its multiply/subtract, unlike a stored scaled score.
        const float score=scores[pbase+query*seq+key];
        probabilities[r*seq+key]=score;
        if((key&32)==0)maxLow=fmax(maxLow,score*scale);else maxHigh=fmax(maxHigh,score*scale);
    }
    reduction[r][lane]=maxLow;reduction[r][lane+32]=maxHigh;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int stride=32;stride;stride/=2) {
        if(lane<stride)reduction[r][lane]=fmax(reduction[r][lane],reduction[r][lane+stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float maximum=reduction[r][0];
    barrier(CLK_LOCAL_MEM_FENCE);
    float sumLow=0,sumHigh=0;
    for(int key=lane;key<limit;key+=32) {
        const float p=(query<seq&&key<count)?exp(probabilities[r*seq+key]*scale-maximum):0;
        probabilities[r*seq+key]=p;
        if((key&32)==0)sumLow+=p;else sumHigh+=p;
    }
    reduction[r][lane]=sumLow;reduction[r][lane+32]=sumHigh;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int stride=32;stride;stride/=2) {
        if(lane<stride)reduction[r][lane]+=reduction[r][lane+stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inverse=query<seq?1.0f/reduction[r][0]:0;
    if(query<seq&&lane==0){stats[2*(h*seq+query)]=maximum;stats[2*(h*seq+query)+1]=inverse;}
    for(int key=lane;key<limit;key+=32)probabilities[r*seq+key]*=inverse;
    barrier(CLK_LOCAL_MEM_FENCE);
    float value=0;
    for(int base=0;base<limit;base+=32) {
        for(int i=tid;i<1024;i+=256) {
            const int key=base+i/32,channel=i%32;
            values[i/32][channel]=(key<seq&&channel<d)?qkv[qbase+key*3*width+2*width+channel]:0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if(query<seq&&lane<d)for(int k=0;k<32&&base+k<limit;k++)
            value=fma(probabilities[r*seq+base+k],values[k][lane],value);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(query<seq&&lane<d)output[(b*seq+query)*width+hd+lane]=value;
}
