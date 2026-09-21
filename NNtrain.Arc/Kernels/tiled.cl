// Portable OpenCL 1.2 GEMM. No fast-math, subgroup-size assumptions or
// vendor-specific matrix intrinsics. Every reduction uses ordered FP32 FMA.
//
// Launch with local=(16,16), global=(ceil(n/32)*16,ceil(m/32)*16).
// A logical [m,k], B logical [k,n]; ta/tb describe transposed physical storage.
// bf16Operands: bit 0 rounds A, bit 1 rounds B to BF16 round-to-nearest-even.
// gateOperand: 0=none, 1=mask A, 2=mask B. gate has that operand's physical
// storage layout; elements with gate<=0 contribute zero (ReLU backward).
// bias/gate may alias any valid buffer when disabled, and are then never read.
//
// For long reductions, dispatch successive [kStart,kStart+kCount) chunks on
// an in-order queue with the same C buffer. Set accumulate=1 after chunk 0,
// addBias only on chunk 0, and relu only on the final chunk. This bounds each
// dispatch's work without allocating a split-K partial matrix or using atomics.
float arc_tiled_bf16(float value) {
    uint bits=as_uint(value);
    if ((bits & 0x7f800000u)==0x7f800000u) return value;
    return as_float((bits+0x7fffu+((bits>>16)&1u))&0xffff0000u);
}

__attribute__((reqd_work_group_size(16,16,1)))
__kernel void gemm_tiled(
    __global const float* a, __global const float* b, __global float* c,
    __global const float* bias, __global const float* gate,
    int m, int n, int k, int ta, int tb, int bf16Operands,
    int accumulate, int addBias, int relu, int gateOperand,
    int kStart, int kCount) {
    const int lx=get_local_id(0), ly=get_local_id(1);
    const int rowBase=get_group_id(1)*32, colBase=get_group_id(0)*32;
    const int row0=rowBase+ly, row1=row0+16;
    const int col0=colBase+lx, col1=col0+16;
    const int end=min(k,kStart+kCount);
    // Padding prevents transposed cooperative loads from hammering a single
    // local-memory bank. 4,288 bytes of local storage per 256-thread group.
    __local float tileA[32][17];
    __local float tileB[16][33];

    float s00=0, s01=0, s10=0, s11=0;
    if (accumulate) {
        if(row0<m && col0<n)s00=c[row0*n+col0];
        if(row0<m && col1<n)s01=c[row0*n+col1];
        if(row1<m && col0<n)s10=c[row1*n+col0];
        if(row1<m && col1<n)s11=c[row1*n+col1];
    }
    if (addBias) {
        const float bias0=col0<n?bias[col0]:0;
        const float bias1=col1<n?bias[col1]:0;
        s00+=bias0;s10+=bias0;s01+=bias1;s11+=bias1;
    }

    for(int base=kStart;base<end;base+=16) {
        // Swap the cooperative loading coordinates for transposed operands.
        // In both cases the x-lanes read contiguous global addresses.
        const int ar=ta?lx:ly, ak=ta?ly:lx;
        const int bk=tb?lx:ly, bc=tb?ly:lx;
        for(int part=0;part<2;part++) {
            const int r=rowBase+ar+part*16, reductionA=base+ak;
            float av=0;
            if(r<m && reductionA<end) {
                const int index=ta?reductionA*m+r:r*k+reductionA;
                av=a[index];
                if(gateOperand==1 && gate[index]<=0)av=0;
                if(bf16Operands&1)av=arc_tiled_bf16(av);
            }
            tileA[ar+part*16][ak]=av;
            const int column=colBase+bc+part*16, reductionB=base+bk;
            float bv=0;
            if(column<n && reductionB<end) {
                const int index=tb?column*k+reductionB:reductionB*n+column;
                bv=b[index];
                if(gateOperand==2 && gate[index]<=0)bv=0;
                if(bf16Operands&2)bv=arc_tiled_bf16(bv);
            }
            tileB[bk][bc+part*16]=bv;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // Limit the final tile rather than multiplying padding by zero: this
        // preserves IEEE behavior for nonfinite operands and nonmultiple K.
        const int tileCount=min(16,end-base);
        for(int inner=0;inner<tileCount;inner++) {
            const float a0=tileA[ly][inner], a1=tileA[ly+16][inner];
            const float b0=tileB[inner][lx], b1=tileB[inner][lx+16];
            s00=fma(a0,b0,s00);s01=fma(a0,b1,s01);
            s10=fma(a1,b0,s10);s11=fma(a1,b1,s11);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(row0<m && col0<n)c[row0*n+col0]=relu?fmax(0.0f,s00):s00;
    if(row0<m && col1<n)c[row0*n+col1]=relu?fmax(0.0f,s01):s01;
    if(row1<m && col0<n)c[row1*n+col0]=relu?fmax(0.0f,s10):s10;
    if(row1<m && col1<n)c[row1*n+col1]=relu?fmax(0.0f,s11):s11;
}

// Bias gradient uses its own owner per output column. Unlike the original dW
// kernel, this does not redundantly sum db once for every input feature.
// Chunking matches GEMM's long-K policy: db remains device-resident between
// chunks. No floating-point atomic extension is required.
__kernel void linear_db_chunk(
    __global const float* dy, __global const float* y, __global float* db,
    int rows, int width, int relu, int rowStart, int rowCount) {
    const int col=get_global_id(0);
    if(col>=width)return;
    float total=0;
    const int end=min(rows,rowStart+rowCount);
    for(int row=rowStart;row<end;row++) {
        const int index=row*width+col;
        total+=relu && y[index]<=0?0:dy[index];
    }
    db[col]+=total;
}
