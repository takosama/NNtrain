// A 32-column by 256-row tile. X lanes read consecutive channels;
// Y lanes split the row reduction. Each partial and final value has one owner.
float arc_gradient_bf16(float v);
__attribute__((reqd_work_group_size(32,8,1)))
__kernel void gradient_rows(__global const float* dy,__global const float* x,
 __global const float* stats,__global float* parts,int rows,int width,int relu,int norm){
 int col=get_group_id(0)*32+get_local_id(0),lane=get_local_id(1),group=get_group_id(1);
 int end=min(rows,(group+1)*256);float g=0,b=0;
 for(int row=group*256+lane;row<end;row+=8)if(col<width){
  int idx=row*width+col;float d=relu&&x[idx]<=0?0:dy[idx];
  if(norm==2)d=arc_gradient_bf16(d);
  b+=d;if(norm==1)g+=d*(x[idx]-stats[row*2])*stats[row*2+1];
 }
 __local float gs[8][32],bs[8][32];int c=get_local_id(0);
 gs[lane][c]=g;bs[lane][c]=b;barrier(CLK_LOCAL_MEM_FENCE);
 for(int stride=4;stride;stride/=2){if(lane<stride){gs[lane][c]+=gs[lane+stride][c];bs[lane][c]+=bs[lane+stride][c];}barrier(CLK_LOCAL_MEM_FENCE);}
 if(lane==0&&col<width){parts[group*width+col]=bs[0][c];if(norm==1)parts[((rows+255)/256+group)*width+col]=gs[0][c];}
}
__kernel void linear_db_bf16_chunk(__global const float* dy,__global const float* gate,__global float* db,
 int rows,int width,int relu,int rowStart,int rowCount){
 int col=get_global_id(0);if(col>=width)return;float total=0;
 for(int row=rowStart;row<min(rows,rowStart+rowCount);row++){int i=row*width+col;total+=arc_gradient_bf16(relu&&gate[i]<=0?0:dy[i]);}
 db[col]+=total;
}
__kernel void gradient_rows_finish(__global const float* parts,__global float* db,__global float* dg,int groups,int width,int norm){
 int col=get_global_id(0);if(col>=width)return;float b=0,g=0;
 for(int r=0;r<groups;r++){b+=parts[r*width+col];if(norm)g+=parts[(groups+r)*width+col];}
 db[col]+=b;if(norm)dg[col]+=g;
}
