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
