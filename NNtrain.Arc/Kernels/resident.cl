// Same nearest-even, symmetric block codec as CPU. No fast-relaxed-math.
float arc_gradient_bf16(float v){uint b=as_uint(v);
 if((b&0x7f800000u)!=0x7f800000u)b+=0x7fffu+((b>>16)&1u);
 else if((b&0x007fffffu)!=0)b|=0x00400000u;
 return as_float(b&0xffff0000u);
}
__kernel void matrix_gradient_bf16(__global const float* x,__global const float* gate,__global float* y,int n,int relu){
 int i=get_global_id(0);if(i>=n)return;y[i]=arc_gradient_bf16(relu&&gate[i]<=0?0:x[i]);
}
__kernel void resident_zero(__global float* x,int n){int i=get_global_id(0);if(i<n)x[i]=0;}
__kernel void resident_bf16(__global const float* x,__global ushort* y,int n){int i=get_global_id(0);if(i<n){uint b=as_uint(x[i]);if((b&0x7f800000u)!=0x7f800000u)b+=0x7fffu+((b>>16)&1u);else if((b&0x007fffffu)!=0)b|=0x00400000u;y[i]=(ushort)(b>>16);}}
__kernel void resident_bfp8(__global const float* x,__global char* y,__global float* scales,__global int* status,int n,int block){
    int b=get_global_id(0),start=b*block;if(start>=n)return;int end=min(n,start+block);float m=0;
    int valid=1;for(int i=start;i<end;i++){valid&=isfinite(x[i]);m=fmax(m,fabs(x[i]));}
    if(!valid){atomic_or(status,1);scales[b]=NAN;for(int i=start;i<end;i++)y[i]=0;return;}
    float s=m==0?1:m/127.0f;
    if(!(s>0)||!isfinite(s)){atomic_or(status,1);scales[b]=NAN;for(int i=start;i<end;i++)y[i]=0;return;}
    scales[b]=s;
    for(int i=start;i<end;i++)y[i]=convert_char_sat_rte(clamp(x[i]/s,-127.0f,127.0f));
}
__kernel void resident_clip_total(__global const float* parts,__global float* out,__global const int* status,int n,float maxnorm){float sum=0;for(int i=0;i<n;i++)sum+=parts[i];float norm=sqrt(sum);out[0]=norm;out[1]=norm>maxnorm?maxnorm/(norm+1e-6f):1;out[2]=(float)status[0];}
__kernel void resident_validate_loss(__global float* loss,__global const int* status){if(status[0])loss[0]=NAN;}
__kernel void resident_clip_apply(__global float* g,__global const float* stats,int n){int i=get_global_id(0);if(i<n)g[i]*=stats[1];}
__kernel void resident_sum_slot(__global const float* parts,__global float* out,int n,int slot){float sum=0;for(int i=0;i<n;i++)sum+=parts[i];out[slot]=sum;}
__kernel void resident_finite(__global const float* g,__global int* status,int n){int i=get_global_id(0);if(i<n&&!isfinite(g[i]))atomic_or(status,1);}
