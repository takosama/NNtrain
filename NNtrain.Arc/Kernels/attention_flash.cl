// SG16 FlashAttention. Global storage is O(B*T*D), never O(B*H*T*T).
// Q/K/V are BF16 operands; softmax, accumulators and gradients remain FP32.
// FP32 products use a two-component BF16 expansion (including low*low).
// Async mode expresses a work-group collective ping-pong copy, not a claim
// that a particular OpenCL driver overlaps transfer and DPAS in hardware.
#if defined(ARC_FLASH) && defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#define FA_BODY __attribute__((always_inline)) inline
ushort xmx_bf16(float x) {
    uint u=as_uint(x);
    if((u&0x7f800000u)!=0x7f800000u)u+=0x7fffu+((u>>16)&1u);
    else if((u&0x007fffffu)!=0)u|=0x00400000u;
    return (ushort)(u>>16);
}
__attribute__((always_inline)) inline float fa_float(ushort x) { return as_float((uint)x << 16); }
__attribute__((always_inline)) inline ushort fa_low(float x) { return xmx_bf16(x-fa_float(xmx_bf16(x))); }
// XOR pair swizzle keeps contiguous async copies while avoiding the 16-way
// SLM bank conflict of reading one channel from 16 row-major keys at D32.
__attribute__((always_inline)) inline int fa_index(int r,int c,int depth) { return r*depth+(c^((r&(depth/2-1))*2)); }

__kernel void attention_flash_pack(__global const float* x, __global ushort* p,
    int seq, int width, int heads, int padded, int depth, int count) {
    int i=get_global_id(0); if(i>=count)return;
    int c=i%depth, t=(i/depth)%padded, component=(i/(depth*padded))%3;
    int h=i/(depth*padded*3), d=width/heads; c^=(t&(depth/2-1))*2;
    p[i]=c<d && t<seq ? xmx_bf16(x[((h/heads)*seq+t)*3*width+component*width+(h%heads)*d+c]) : (ushort)0;
}
__kernel void attention_flash_pack_dy(__global const float* dy, __global ushort* p,
    int seq, int width, int heads, int padded, int depth, int count) {
    int i=get_global_id(0); if(i>=count)return;
    int c=i%depth, t=(i/depth)%padded, component=(i/(depth*padded))%2;
    int h=i/(depth*padded*2), d=width/heads; c^=(t&(depth/2-1))*2;
    float v=c<d && t<seq ? dy[((h/heads)*seq+t)*width+(h%heads)*d+c] : 0;
    p[i]=component ? fa_low(v) : xmx_bf16(v);
}
// Raw (unquantized) replay output is used here, never the BFP8 result tensor.
__kernel void attention_flash_delta(__global const float* dy, __global const float* raw,
    __global float* delta, int seq, int width, int heads, int count) {
    int i=get_global_id(0); if(i>=count)return;
    int h=i/seq, t=i%seq, d=width/heads;
    int offset=((h/heads)*seq+t)*width+(h%heads)*d;
    float sum=0; for(int c=0;c<d;c++)sum=fma(dy[offset+c],raw[offset+c],sum);
    delta[i]=sum;
}
__attribute__((always_inline)) inline void fa_copy(__local ushort* dest, __global const ushort* source, int count) {
    for(int i=get_local_id(0);i<count;i+=64)dest[i]=source[i];
}
__attribute__((always_inline)) inline short8 fa_a(__local const ushort* a,int row,int k,int depth) {
    short8 v;
    #pragma unroll
    for(int r=0;r<8;r++)v[r]=as_short(a[fa_index(row+r,k+get_sub_group_local_id(),depth)]); return v;
}
__attribute__((always_inline)) inline short8 fa_af(__local const float* a,int row,int k,int low) {
    short8 v;
    #pragma unroll
    for(int r=0;r<8;r++) {
        float f=a[(row+r)*32+k+get_sub_group_local_id()];
        v[r]=as_short((ushort)(low ? fa_low(f) : xmx_bf16(f)));
    } return v;
}
__attribute__((always_inline)) inline int8 fa_bt(__local const ushort* b,int col,int k,int depth) {
    int8 v; int lane=get_sub_group_local_id();
    #pragma unroll
    for(int r=0;r<8;r++)v[r]=as_int((uint)b[fa_index(col+lane,k+2*r,depth)] | ((uint)b[fa_index(col+lane,k+2*r+1,depth)]<<16)); return v;
}
__attribute__((always_inline)) inline int8 fa_b(__local const ushort* b,int col,int k,int depth) {
    int8 v; int lane=get_sub_group_local_id();
    #pragma unroll
    for(int r=0;r<8;r++)v[r]=as_int((uint)b[fa_index(k+2*r,col+lane,depth)] | ((uint)b[fa_index(k+2*r+1,col+lane,depth)]<<16)); return v;
}
__attribute__((always_inline)) inline float8 fa_dot(__local const ushort* a,__local const ushort* b,int row,int col,int depth) {
    float8 sum=0;
    #pragma unroll
    for(int k=0;k<depth;k+=16)
        sum=intel_sub_group_bf16_bf16_matrix_mad_k16(fa_a(a,row,k,depth),fa_bt(b,col,k,depth),sum);
    return sum;
}
__attribute__((always_inline)) inline float8 fa_product(__local const float* a,__local const ushort* b,
    __local const ushort* blow,int row,int col,int depth,int twoB,int xmx) {
    float8 sum=0;
    if(xmx)for(int k=0;k<32;k+=16) {
        short8 hi=fa_af(a,row,k,0),lo=fa_af(a,row,k,1);
        int8 bh=fa_b(b,col,k,depth);
        // Small products first; FP32 accumulation of both components.
        if(twoB) {
            int8 bl=fa_b(blow,col,k,depth);
            sum=intel_sub_group_bf16_bf16_matrix_mad_k16(lo,bl,sum);
            sum=intel_sub_group_bf16_bf16_matrix_mad_k16(hi,bl,sum);
        }
        sum=intel_sub_group_bf16_bf16_matrix_mad_k16(lo,bh,sum);
        sum=intel_sub_group_bf16_bf16_matrix_mad_k16(hi,bh,sum);
    } else for(int k=0;k<32;k++) {
        float bv=fa_float(b[fa_index(k,col+get_sub_group_local_id(),depth)]);
        if(twoB)bv+=fa_float(blow[fa_index(k,col+get_sub_group_local_id(),depth)]);
        for(int r=0;r<8;r++)sum[r]=fma(a[(row+r)*32+k],bv,sum[r]);
    }
    return sum;
}
__attribute__((always_inline)) inline short8 fa_rbits(float8 p,int low) {
    short8 a;
    #pragma unroll
    for(int r=0;r<8;r++)a[r]=as_short((ushort)(low?fa_low(p[r]):xmx_bf16(p[r])));
    return a;
}
__attribute__((always_inline)) inline float8 fa_rproduct(float8 p0,float8 p1,__local const ushort* b,
    __local const ushort* blow,int col,int depth,int twoB) {
    float8 sum=0;
    #pragma unroll
    for(int part=0;part<2;part++) {
        float8 p=part?p1:p0;
        short8 hi=fa_rbits(p,0),lo=fa_rbits(p,1);
        int8 bh=fa_b(b,col,part*16,depth);
        if(twoB) {
            int8 bl=fa_b(blow,col,part*16,depth);
            sum=intel_sub_group_bf16_bf16_matrix_mad_k16(lo,bl,sum);
            sum=intel_sub_group_bf16_bf16_matrix_mad_k16(hi,bl,sum);
        }
        sum=intel_sub_group_bf16_bf16_matrix_mad_k16(lo,bh,sum);
        sum=intel_sub_group_bf16_bf16_matrix_mad_k16(hi,bh,sum);
    }
    return sum;
}

FA_BODY void fa_forward(__global const ushort* packed,__global float* y,__global float* stats,
    int seq,int width,int heads,int padded,int d,int causal,int depth,int xmx,int asyncCopy,
    __local ushort* memory,__local float* probs) {
    int lane=get_sub_group_local_id(),row=get_sub_group_id()*8;
    int first=get_group_id(0)*32,head=get_group_id(1),size=32*depth;
    __global const ushort* base=packed+head*3*padded*depth;
    __local ushort* q=memory; __local ushort* keys=memory+size; __local ushort* vals=memory+(asyncCopy?3:2)*size;
    fa_copy(q,base+first*depth,size);
    int end=causal ? min(padded,first+32) : padded;
    event_t pending=0;
    if(asyncCopy) {
        pending=async_work_group_copy(keys,base+padded*depth,(size_t)size,0);
        pending=async_work_group_copy(vals,base+2*padded*depth,(size_t)size,pending);
    }
    float8 maximum=(float8)(-INFINITY),denom=0;
    float8 result[4];
    #pragma unroll
    for(int c=0;c<depth/16;c++)result[c]=0;
    for(int start=0;start<end;start+=32) {
        int slot=asyncCopy ? (start/32)%2 : 0;
        __local ushort* k=keys+slot*size; __local ushort* v=vals+slot*size;
        if(asyncCopy) {
            wait_group_events(1,&pending);
            if(start+32<end) {
                int other=1-slot;
                pending=async_work_group_copy(keys+other*size,base+padded*depth+(start+32)*depth,(size_t)size,0);
                pending=async_work_group_copy(vals+other*size,base+2*padded*depth+(start+32)*depth,(size_t)size,pending);
            }
        } else { fa_copy(k,base+padded*depth+start*depth,size);fa_copy(v,base+2*padded*depth+start*depth,size); }
        barrier(CLK_LOCAL_MEM_FENCE);
        float8 score[2]; score[0]=fa_dot(q,k,row,0,depth);score[1]=fa_dot(q,k,row,16,depth);
        float8 oldScale;
        #pragma unroll
        for(int r=0;r<8;r++) {
            int qi=first+row+r;
            #pragma unroll
            for(int t=0;t<2;t++) {
                int ki=start+t*16+lane;
                score[t][r]=(qi<seq && ki<seq && (!causal||ki<=qi)) ? score[t][r]*rsqrt((float)d) : -INFINITY;
            }
            float m=fmax(maximum[r],sub_group_reduce_max(fmax(score[0][r],score[1][r])));
            // Entire padded query rows are harmless zeros, not inf-inf NaNs.
            if(qi>=seq)m=0;
            oldScale[r]=exp(maximum[r]-m);
            float p0=exp(score[0][r]-m),p1=exp(score[1][r]-m);
            denom[r]=denom[r]*oldScale[r]+sub_group_reduce_add(p0+p1);
            maximum[r]=m;
            if(xmx){score[0][r]=p0;score[1][r]=p1;}
            else {probs[(row+r)*32+lane]=p0;probs[(row+r)*32+16+lane]=p1;}
        }
        if(!xmx)barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for(int c=0;c<depth/16;c++)result[c]=result[c]*oldScale+(xmx?
            fa_rproduct(score[0],score[1],v,v,c*16,depth,0):fa_product(probs,v,v,row,c*16,depth,0,0));
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for(int r=0;r<8;r++)if(first+row+r<seq) {
        int t=first+row+r;
        if(lane==0) { stats[2*(head*seq+t)]=maximum[r];stats[2*(head*seq+t)+1]=1.0f/denom[r]; }
        #pragma unroll
        for(int c=0;c<depth/16;c++)if(c*16+lane<d)
            y[((head/heads)*seq+t)*width+(head%heads)*d+c*16+lane]=result[c][r]/denom[r];
    }
}

// Separate query owner and key/value owner. Each destination has exactly one
// work-group, so no floating-point atomic or global score/derivative scratch.
FA_BODY void fa_backward(__global const ushort* packed,__global const ushort* g,
    __global const float* stats,__global const float* delta,__global float* dx,
    int seq,int width,int heads,int padded,int d,int causal,int depth,int xmx,int asyncCopy,int keyOwner,
    __local ushort* memory,__local float* probs,__local float* ds) {
    int lane=get_sub_group_local_id(),row=get_sub_group_id()*8;
    int first=get_group_id(0)*32,head=get_group_id(1),size=32*depth;
    __global const ushort* base=packed+head*3*padded*depth;
    __global const ushort* grad=g+head*2*padded*depth;
    __local ushort* a=memory;__local ushort* ah=memory+size;__local ushort* al=memory+2*size;
    __local ushort* blocks=memory+3*size;
    fa_copy(a,base+(keyOwner?padded*depth:0)+first*depth,size);
    fa_copy(ah,(keyOwner?base+2*padded*depth:grad)+first*depth,size);
    if(!keyOwner)fa_copy(al,grad+padded*depth+first*depth,size);
    int begin=causal&&keyOwner?first:0;
    int end=causal&&!keyOwner?min(padded,first+32):padded;
    __global const ushort* src0=base+(keyOwner?0:padded*depth);
    __global const ushort* src1=keyOwner?grad:base+2*padded*depth;
    __global const ushort* src2=keyOwner?grad+padded*depth:base+2*padded*depth;
    event_t pending=0;
    if(asyncCopy) {
        pending=async_work_group_copy(blocks,src0+begin*depth,(size_t)size,0);
        pending=async_work_group_copy(blocks+size,src1+begin*depth,(size_t)size,pending);
        if(keyOwner)pending=async_work_group_copy(blocks+2*size,src2+begin*depth,(size_t)size,pending);
    }
    float8 result[4],value[4];
    #pragma unroll
    for(int c=0;c<depth/16;c++){result[c]=0;value[c]=0;}
    for(int start=begin;start<end;start+=32) {
        int slot=asyncCopy ? ((start-begin)/32)%2 : 0;
        __local ushort* b=blocks+slot*3*size;__local ushort* bh=b+size;__local ushort* bl=b+2*size;
        if(asyncCopy) {
            wait_group_events(1,&pending);
            if(start+32<end) {
                __local ushort* next=blocks+(1-slot)*3*size;
                pending=async_work_group_copy(next,src0+(start+32)*depth,(size_t)size,0);
                pending=async_work_group_copy(next+size,src1+(start+32)*depth,(size_t)size,pending);
                if(keyOwner)pending=async_work_group_copy(next+2*size,src2+(start+32)*depth,(size_t)size,pending);
            }
        } else {fa_copy(b,src0+start*depth,size);fa_copy(bh,src1+start*depth,size);if(keyOwner)fa_copy(bl,src2+start*depth,size);}
        barrier(CLK_LOCAL_MEM_FENCE);
        float8 pr[2],der[2];
        #pragma unroll
        for(int t=0;t<2;t++) {
            float8 score=fa_dot(a,b,row,t*16,depth);
            float8 dp=0;
            if(xmx) {
                dp=fa_dot(ah,bh,row,t*16,depth);
                dp+=keyOwner?fa_dot(ah,bl,row,t*16,depth):fa_dot(al,bh,row,t*16,depth);
            } else for(int c=0;c<depth;c++)for(int r=0;r<8;r++) {
                float av=fa_float(ah[fa_index(row+r,c,depth)]),bv=fa_float(bh[fa_index(t*16+lane,c,depth)]);
                if(keyOwner)bv+=fa_float(bl[fa_index(t*16+lane,c,depth)]);else av+=fa_float(al[fa_index(row+r,c,depth)]);
                dp[r]=fma(av,bv,dp[r]);
            }
            #pragma unroll
            for(int r=0;r<8;r++) {
                int own=first+row+r,other=start+t*16+lane;
                int qi=keyOwner?other:own,ki=keyOwner?own:other;
                float p=0,s=0;
                if(qi<seq && ki<seq && (!causal||ki<=qi)) {
                    p=exp(score[r]*rsqrt((float)d)-stats[2*(head*seq+qi)])*stats[2*(head*seq+qi)+1];
                    s=p*(dp[r]-delta[head*seq+qi])*rsqrt((float)d);
                }
                if(xmx){pr[t][r]=p;der[t][r]=s;}
                else {probs[(row+r)*32+t*16+lane]=p;ds[(row+r)*32+t*16+lane]=s;}
            }
        }
        if(!xmx)barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for(int c=0;c<depth/16;c++) {
            result[c]+=xmx?fa_rproduct(der[0],der[1],b,b,c*16,depth,0):fa_product(ds,b,b,row,c*16,depth,0,0);
            if(keyOwner)value[c]+=xmx?fa_rproduct(pr[0],pr[1],bh,bl,c*16,depth,1):fa_product(probs,bh,bl,row,c*16,depth,1,0);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for(int r=0;r<8;r++)if(first+row+r<seq) {
      #pragma unroll
      for(int c=0;c<depth/16;c++)if(c*16+lane<d) {
        int offset=((head/heads)*seq+first+row+r)*3*width+(head%heads)*d+c*16+lane;
        dx[offset+(keyOwner?width:0)]+=result[c][r];
        if(keyOwner)dx[offset+2*width]+=value[c][r];
      }
    }
}
#define FA_KERNELS(D,X,A) \
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(64,1,1))) \
__kernel void attention_flash_f_d##D##_x##X##_a##A(__global const ushort* p,__global float* y,__global float* stats,int seq,int width,int heads,int padded,int d,int causal) { \
    __local ushort m[(A?5:3)*32*D];__local float pr[X?1:32*32];fa_forward(p,y,stats,seq,width,heads,padded,d,causal,D,X,A,m,pr); } \
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(64,1,1))) \
__kernel void attention_flash_q_d##D##_x##X##_a##A(__global const ushort* p,__global const ushort* g,__global const float* stats,__global const float* delta,__global float* dx,int seq,int width,int heads,int padded,int d,int causal) { \
    __local ushort m[(A?9:6)*32*D];__local float pr[X?1:32*32],ds[X?1:32*32];fa_backward(p,g,stats,delta,dx,seq,width,heads,padded,d,causal,D,X,A,0,m,pr,ds); } \
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(64,1,1))) \
__kernel void attention_flash_kv_d##D##_x##X##_a##A(__global const ushort* p,__global const ushort* g,__global const float* stats,__global const float* delta,__global float* dx,int seq,int width,int heads,int padded,int d,int causal) { \
    __local ushort m[(A?9:6)*32*D];__local float pr[X?1:32*32],ds[X?1:32*32];fa_backward(p,g,stats,delta,dx,seq,width,heads,padded,d,causal,D,X,A,1,m,pr,ds); }
#define FA_SELECT(D,X,A) FA_KERNELS(D,X,A)
#define FA_DEPTH(D) FA_SELECT(D,ARC_FLASH_XMX,ARC_FLASH_ASYNC)
#if ARC_FLASH_DEPTH == 16
FA_DEPTH(16)
#elif ARC_FLASH_DEPTH == 32
FA_DEPTH(32)
#elif ARC_FLASH_DEPTH == 64
FA_DEPTH(64)
#endif
#endif
