// QK attention keeps the same strided addressing and BF16 arithmetic.
#ifdef ARC_XMX
ushort xmx_bf16(float x);
int8 xmx_panel_b(const __local uint* p,int lane);
__attribute__((intel_reqd_sub_group_size(ARC_SG)))
__attribute__((reqd_work_group_size(ARC_SG,16,1)))
__kernel void attention_xmx_panel(__global const float* a,__global const float* b,__global float* c,
 int m,int n,int k,int ar,int ac,int ag,int ab,int ah,int ao,int br,int bc,int bg,int bb,int bh,int bo,
 int cr,int cc,int cg,int cb,int ch,int co,int heads,int first,int add,int causalMode){
    if(causalMode==1&&get_group_id(0)*64>=get_group_id(1)*128+128)return;
    int kStart=0,kCount=k,ta=ac!=1,tb=bc!=1,accumulate=add,relu=0;
    int group=get_group_id(2);
    int ap=att_address(group,first,heads,ag,ab,ah,ao),bp=att_address(group,first,heads,bg,bb,bh,bo),cp=att_address(group,first,heads,cg,cb,ch,co);
    int lane=get_local_id(0), sg=get_local_id(1), tid=sg*ARC_SG+lane;
    int rowBase=get_group_id(1)*128, colBase=get_group_id(0)*64;
    int row=rowBase+sg*8, end=min(k,kStart+kCount);
    __local ushort at[128][33];__local uint bt[16*64];
    float8 sums[64/ARC_SG];
    #pragma unroll
    for(int tile=0;tile<64/ARC_SG;tile++){
        sums[tile]=(float8)(0);int col=colBase+tile*ARC_SG+lane;
        #pragma unroll
        for(int r=0;r<8;r++)if(row+r<m&&col<n){
            if(accumulate)sums[tile][r]=c[cp+(row+r)*cr+col*cc];

        }
    }
    for(int base=kStart;base<end;base+=32){
        for(int i=tid;i<4096;i+=ARC_SG*16){
            int rr=ta?i%128:i/32,kk=ta?i/128:i%32,r=rowBase+rr,q=base+kk;
            float x=0;
            if(r<m&&q<end){int idx=ap+r*ar+q*ac;x=a[idx];}
            at[rr][kk]=xmx_bf16(x);
        }
        for(int i=tid;i<1024;i+=ARC_SG*16){
            int kp=tb?i%16:i/64,nn=tb?i/16:i%64,q=base+kp*2,col=colBase+nn;
            float x=0,y=0;
            if(col<n&&q<end)x=b[bp+q*br+col*bc];
            if(col<n&&q+1<end)y=b[bp+(q+1)*br+col*bc];
            bt[((nn/ARC_SG)*2+kp/8)*8*ARC_SG+(kp%8)*ARC_SG+nn%ARC_SG]=(uint)xmx_bf16(x)|((uint)xmx_bf16(y)<<16);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for(int part=0;part<2;part++){
#if ARC_SG == 16
            short8 av;
            #pragma unroll
            for(int r=0;r<8;r++)av[r]=as_short(at[sg*8+r][part*16+lane]);
#else
            int8 av;
            #pragma unroll
            for(int r=0;r<8;r++)av[r]=as_int((uint)at[sg*8+r][part*16+lane*2]|((uint)at[sg*8+r][part*16+lane*2+1]<<16));
#endif
            #pragma unroll
            for(int tile=0;tile<64/ARC_SG;tile++){
                int8 bv=xmx_panel_b(bt+(tile*2+part)*8*ARC_SG,lane);
                sums[tile]=intel_sub_group_bf16_bf16_matrix_mad_k16(av,bv,sums[tile]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for(int tile=0;tile<64/ARC_SG;tile++){
        int col=colBase+tile*ARC_SG+lane;
        #pragma unroll
        for(int r=0;r<8;r++)if(row+r<m&&col<n)c[cp+(row+r)*cr+col*cc]=relu?fmax(0.0f,sums[tile][r]):sums[tile][r];
    }
}
#endif
