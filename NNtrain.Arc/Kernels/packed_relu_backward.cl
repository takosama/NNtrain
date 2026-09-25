float arc_gradient_bf16(float v);
// Fused gate + BF16 operand encoding. Gate storage is read in its native type;
// it never needs an activation-sized FP32 materialization.
__kernel void linear_relu_grad_packed(__global const float* dy,__global const uchar* gate,__global const float* scales,
 __global ushort* encoded,int n,int block,int bfp8){
 int i=get_global_id(0);if(i>=n)return;
 float value=bfp8?(float)((__global const char*)gate)[i]*scales[i/block]
     :as_float((uint)((__global const ushort*)gate)[i]<<16);
 encoded[i]=(ushort)(as_uint(arc_gradient_bf16(value<=0?0:dy[i]))>>16);
}
// Identical modulo8 partial streams and 8->4->2->1 tree to gradient_rows.
__attribute__((reqd_work_group_size(32,8,1)))
__kernel void gradient_rows_packed_bf16(__global const ushort* dy,__global float* parts,int rows,int width){
 int col=get_group_id(0)*32+get_local_id(0),lane=get_local_id(1),group=get_group_id(1);
 float sum=0;int end=min(rows,(group+1)*256);
 for(int row=group*256+lane;row<end;row+=8)if(col<width)sum+=as_float((uint)dy[row*width+col]<<16);
 __local float partial[8][32];int c=get_local_id(0);partial[lane][c]=sum;barrier(CLK_LOCAL_MEM_FENCE);
 for(int stride=4;stride;stride/=2){if(lane<stride)partial[lane][c]+=partial[lane+stride][c];barrier(CLK_LOCAL_MEM_FENCE);}
 if(lane==0&&col<width)parts[group*width+col]=partial[0][c];
}
__kernel void linear_db_packed_bf16_chunk(__global const ushort* dy,__global float* db,int rows,int width,int rowStart,int rowCount){
 int col=get_global_id(0);if(col>=width)return;float sum=0;
 for(int row=rowStart;row<min(rows,rowStart+rowCount);row++)sum+=as_float((uint)dy[row*width+col]<<16);
 db[col]+=sum;
}

// The source is the FP32 gradient, while the gate remains in the published
// activation's native storage. This is exactly the old encode kernel's gate
// comparison and BF16 rounding, including NaN/negative-zero behavior.
ushort linear_relu_gradient_bits(__global const float* dy,__global const uchar* gate,
 __global const float* scales,int index,int block,int bfp8){
 float value=bfp8?(float)((__global const char*)gate)[index]*scales[index/block]
     :as_float((uint)((__global const ushort*)gate)[index]<<16);
 return (ushort)(as_uint(arc_gradient_bf16(value<=0?0:dy[index]))>>16);
}

// Identical [K16][M8][M8][K16] panel addresses to
// xmx_storage_pack_a_linear_vec4_bf16, but without the row-major BF16 image.
__kernel void linear_relu_grad_pack_a_vec4(__global const float* dy,
 __global const uchar* gate,__global const float* scales,__global ushort* panels,
 int m,int k,int offset,int block,int bfp8){
 int i=get_global_id(0)*4,rowGroups=(m+7)/8,kPadded=((k+15)/16)*16;
 int total=rowGroups*8*kPadded;if(i>=total)return;
 int row=i/kPadded,q=i%kPadded;ushort4 value=(ushort4)(0);
 if(row<m){
  for(int z=0;z<4;z++)if(q+z<k)
   value[z]=linear_relu_gradient_bits(dy,gate,scales,offset+row*k+q+z,block,bfp8);
 }
 vstore4(value,0,panels+((q/16)*rowGroups+row/8)*128+(row%8)*16+q%16);
}

// Each 32-column x 256-row group publishes its dX BF16 panels and the same
// eight independent bias partial streams as gradient_rows_packed_bf16. The
// row tile offset is 256-aligned, so a streamed pack preserves global group
// numbers and reduction order instead of starting a new reduction tree.
__attribute__((reqd_work_group_size(32,8,1)))
__kernel void linear_relu_grad_pack_a_bias(__global const float* dy,
 __global const uchar* gate,__global const float* scales,__global ushort* panels,
 __global float* parts,int m,int k,int offset,int block,int bfp8,int groupOffset){
 int col=get_group_id(0)*32+get_local_id(0),lane=get_local_id(1),group=get_group_id(1);
 int rowGroups=(m+7)/8,paddedRows=rowGroups*8,paddedCols=((k+15)/16)*16;
 int end=min(paddedRows,(group+1)*256);float sum=0;
 for(int row=group*256+lane;row<end;row+=8){
  if(col<paddedCols){
   ushort bits=row<m&&col<k
       ?linear_relu_gradient_bits(dy,gate,scales,offset+row*k+col,block,bfp8):0;
   panels[((col/16)*rowGroups+row/8)*128+(row%8)*16+col%16]=bits;
   if(row<m&&col<k)sum+=as_float((uint)bits<<16);
  }
 }
 __local float partial[8][32];int c=get_local_id(0);partial[lane][c]=sum;barrier(CLK_LOCAL_MEM_FENCE);
 for(int stride=4;stride;stride/=2){if(lane<stride)partial[lane][c]+=partial[lane+stride][c];barrier(CLK_LOCAL_MEM_FENCE);}
 if(lane==0&&col<k)parts[(groupOffset+group)*k+col]=partial[0][c];
}

// The full-panel mix8_16 path publishes both gradient A operands from one
// gated BF16 value. A 32x32 local tile keeps transposed panel writes in the
// same order/layout as linear_relu_grad_pack_a_transpose. The outer 256-row
// group and each lane's +8 row sequence preserve the bias reduction tree.
__attribute__((reqd_work_group_size(32,8,1)))
__kernel void linear_relu_grad_pack_a_bias_dual(__global const float* dy,
 __global const uchar* gate,__global const float* scales,
 __global ushort* normalPanels,__global ushort* transposedPanels,
 __global float* parts,int rows,int cols,int block,int bfp8){
 int colLocal=get_local_id(0),lane=get_local_id(1);
 int colBase=get_group_id(0)*32,col=colBase+colLocal,group=get_group_id(1);
 int normalRowGroups=(rows+7)/8,normalPaddedRows=normalRowGroups*8;
 int normalPaddedCols=((cols+15)/16)*16;
 int transposedRowGroups=(cols+7)/8,transposedPaddedRows=((rows+15)/16)*16;
 __local ushort values[32][33];
 __local float partial[8][32];
 float sum=0;
 for(int chunk=0;chunk<8;chunk++){
  int rowBase=group*256+chunk*32;
  for(int pass=0;pass<4;pass++){
   int rowLocal=lane+pass*8,row=rowBase+rowLocal;
   ushort bits=row<rows&&col<cols
       ?linear_relu_gradient_bits(dy,gate,scales,row*cols+col,block,bfp8):0;
   values[rowLocal][colLocal]=bits;
   if(row<normalPaddedRows&&col<normalPaddedCols)
    normalPanels[((col/16)*normalRowGroups+row/8)*128+(row%8)*16+col%16]=bits;
   if(row<rows&&col<cols)sum+=as_float((uint)bits<<16);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  int tid=lane*32+colLocal;
  for(int i=tid;i<1024;i+=256){
   int colOffset=((i/128)%4)*8+(i/16)%8;
   int rowOffset=(i/512)*16+i%16;
   int outputRow=colBase+colOffset,outputReduction=rowBase+rowOffset;
   if(outputRow<transposedRowGroups*8&&outputReduction<transposedPaddedRows)
    transposedPanels[((outputReduction/16)*transposedRowGroups+outputRow/8)*128
       +(outputRow%8)*16+outputReduction%16]=values[rowOffset][colOffset];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
 }
 partial[lane][colLocal]=sum;
 barrier(CLK_LOCAL_MEM_FENCE);
 for(int stride=4;stride;stride/=2){
  if(lane<stride)partial[lane][colLocal]+=partial[lane+stride][colLocal];
  barrier(CLK_LOCAL_MEM_FENCE);
 }
 if(lane==0&&col<cols)parts[group*cols+col]=partial[0][colLocal];
}

// The transposed dW operand uses the same 32x33 SLM permutation as
// xmx_storage_pack_a_transpose_bf16. A source offset selects a contiguous
// reduction slice without changing BFP8 scale addressing.
__attribute__((reqd_work_group_size(256,1,1)))
__kernel void linear_relu_grad_pack_a_transpose(__global const float* dy,
 __global const uchar* gate,__global const float* scales,__global ushort* panels,
 int m,int k,int offset,int block,int bfp8){
 int tid=get_local_id(0),rowBase=get_group_id(0)*32,kBase=get_group_id(1)*32;
 int rowGroups=(m+7)/8,kGroups=(k+15)/16;
 __local ushort values[32][33];
 for(int i=tid;i<1024;i+=256){
  int rr=i%32,kk=i/32,row=rowBase+rr,q=kBase+kk;ushort value=0;
  if(row<m&&q<k)
   value=linear_relu_gradient_bits(dy,gate,scales,offset+q*m+row,block,bfp8);
  values[rr][kk]=value;
 }
 barrier(CLK_LOCAL_MEM_FENCE);
 for(int i=tid;i<1024;i+=256){
  int rr=((i/128)%4)*8+(i/16)%8,kk=(i/512)*16+i%16;
  int rb=(rowBase+rr)/8,kb=(kBase+kk)/16;
  if(rb<rowGroups&&kb<kGroups)
   panels[(kb*rowGroups+rb)*128+(rr%8)*16+kk%16]=values[rr][kk];
 }
}

// Preserve the existing bias reduction tree and its per-chunk update order;
// only replace each load from the intermediate BF16 image with its producer.
__attribute__((reqd_work_group_size(32,8,1)))
__kernel void linear_relu_bias_rows_gate(__global const float* dy,
 __global const uchar* gate,__global const float* scales,__global float* parts,
 int rows,int width,int block,int bfp8){
 int col=get_group_id(0)*32+get_local_id(0),lane=get_local_id(1),group=get_group_id(1);
 float sum=0;int end=min(rows,(group+1)*256);
 for(int row=group*256+lane;row<end;row+=8)if(col<width){
  ushort bits=linear_relu_gradient_bits(dy,gate,scales,row*width+col,block,bfp8);
  sum+=as_float((uint)bits<<16);
 }
 __local float partial[8][32];int c=get_local_id(0);partial[lane][c]=sum;barrier(CLK_LOCAL_MEM_FENCE);
 for(int stride=4;stride;stride/=2){if(lane<stride)partial[lane][c]+=partial[lane+stride][c];barrier(CLK_LOCAL_MEM_FENCE);}
 if(lane==0&&col<width)parts[group*width+col]=partial[0][c];
}
__kernel void linear_relu_bias_chunk_gate(__global const float* dy,
 __global const uchar* gate,__global const float* scales,__global float* db,
 int rows,int width,int block,int bfp8,int rowStart,int rowCount){
 int col=get_global_id(0);if(col>=width)return;float sum=0;
 for(int row=rowStart;row<min(rows,rowStart+rowCount);row++){
  ushort bits=linear_relu_gradient_bits(dy,gate,scales,row*width+col,block,bfp8);
  sum+=as_float((uint)bits<<16);
 }
 db[col]+=sum;
}
