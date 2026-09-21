// Correctness-first kernels. OpenCL 1.2 baseline, no vendor matrix intrinsics.
__kernel void range_back(__global const float* x,__global float* y,int offset,int n){int i=get_global_id(0);if(i<n)y[offset+i]+=x[i];}
__kernel void copy_scale(__global const float* x, __global float* y, int n, float scale, int add) {
    int i=get_global_id(0); if(i<n) y[i]=(add?y[i]:0.0f)+scale*x[i];
}
__kernel void linear(__global const float* x, __global const float* w, __global const float* b,
    __global float* y, int rows,int ni,int no,int relu) {
    int i=get_global_id(0);if(i>=rows*no)return;int r=i/no,c=i%no;float s=b[c];
    for(int k=0;k<ni;k++)s+=x[r*ni+k]*w[c*ni+k];y[i]=relu?fmax(0.0f,s):s;
}
__kernel void linear_dx(__global const float* w,__global const float* dy,__global const float* y,
    __global float* dx,int rows,int ni,int no,int relu) {
    int i=get_global_id(0);if(i>=rows*ni)return;int r=i/ni,k=i%ni;float s=0;
    for(int c=0;c<no;c++){int j=r*no+c;s+=(relu&&y[j]<=0?0:dy[j])*w[c*ni+k];}dx[i]+=s;
}
__kernel void linear_dw(__global const float* x,__global const float* dy,__global const float* y,
    __global float* dw,__global float* db,int rows,int ni,int no,int relu) {
    int i=get_global_id(0);if(i>=ni*no)return;int c=i/ni,k=i%ni;float s=0,b=0;
    for(int r=0;r<rows;r++){int j=r*no+c;float g=relu&&y[j]<=0?0:dy[j];s+=g*x[r*ni+k];b+=g;}dw[i]+=s;if(k==0)db[c]+=b;
}
float mask(uint seed,int i,uint threshold,float scale){uint b=seed+0x9E3779B9u*(uint)(i+1);b^=b>>16;b*=0x7FEB352Du;b^=b>>15;b*=0x846CA68Bu;b^=b>>16;return b<threshold?0:scale;}
__kernel void dropout(__global const float* x,__global const float* residual,__global float* y,
    int n,uint seed,uint threshold,float scale,int add){int i=get_global_id(0);if(i<n)y[i]=x[i]*mask(seed,i,threshold,scale)+(add?residual[i]:0);}
__kernel void dropout_back(__global const float* dy,__global float* dx,int n,uint seed,uint threshold,float scale){int i=get_global_id(0);if(i<n)dx[i]+=dy[i]*mask(seed,i,threshold,scale);}
__kernel void add(__global const float* a,__global const float* b,__global float* y,int n){int i=get_global_id(0);if(i<n)y[i]=a[i]+b[i];}
__kernel void norm(__global const float* x,__global const float* gamma,__global const float* beta,
    __global float* y,__global float* stats,int rows,int width,float eps){int r=get_global_id(0);if(r>=rows)return;
    float mean=0;for(int c=0;c<width;c++)mean+=x[r*width+c];mean/=width;
    float var=0;for(int c=0;c<width;c++){float d=x[r*width+c]-mean;var+=d*d;}float inv=rsqrt(var/width+eps);
    stats[2*r]=mean;stats[2*r+1]=inv;for(int c=0;c<width;c++)y[r*width+c]=(x[r*width+c]-mean)*inv*gamma[c]+beta[c];}
__kernel void norm_dx(__global const float* x,__global const float* gamma,__global const float* dy,
    __global const float* stats,__global float* dx,int rows,int width){int r=get_global_id(0);if(r>=rows)return;float m=stats[2*r],iv=stats[2*r+1],s=0,t=0;
    for(int c=0;c<width;c++){float g=dy[r*width+c]*gamma[c];s+=g;t+=g*(x[r*width+c]-m)*iv;}
    for(int c=0;c<width;c++){int i=r*width+c;dx[i]+=iv*(dy[i]*gamma[c]-(s+(x[i]-m)*iv*t)/width);}}
__kernel void norm_dw(__global const float* x,__global const float* dy,__global const float* stats,
    __global float* dg,__global float* db,int rows,int width){int c=get_global_id(0);if(c>=width)return;float g=0,b=0;
    for(int r=0;r<rows;r++){float d=dy[r*width+c];g+=d*(x[r*width+c]-stats[2*r])*stats[2*r+1];b+=d;}dg[c]+=g;db[c]+=b;}
__kernel void norm_dx_set(__global const float* x,__global const float* gamma,__global const float* dy,
    __global const float* stats,__global float* dx,int rows,int width){int r=get_global_id(0);if(r>=rows)return;float m=stats[2*r],iv=stats[2*r+1],s=0,t=0;
    for(int c=0;c<width;c++){float g=dy[r*width+c]*gamma[c];s+=g;t+=g*(x[r*width+c]-m)*iv;}
    for(int c=0;c<width;c++){int i=r*width+c;dx[i]=iv*(dy[i]*gamma[c]-(s+(x[i]-m)*iv*t)/width);}}
void float_add(volatile __global float* p,float x){volatile __global uint* v=(volatile __global uint*)p;uint old=atomic_add(v,0u),next;do{next=old;old=atomic_cmpxchg(v,next,as_uint(as_float(next)+x));}while(old!=next);}
__kernel void embedding(__global const float* table,__global const float* pos,__global const int* ids,
    __global float* y,int tokens,int width,int sequence,int positions){int i=get_global_id(0);if(i<tokens*width)y[i]=table[ids[i/width]*width+i%width]+(positions?pos[(i/width%sequence)*width+i%width]:0);}
__kernel void embedding_back(__global const float* dy,__global const int* ids,__global float* dt,
    __global float* dp,int tokens,int width,int sequence,int positions){int i=get_global_id(0);if(i>=tokens*width)return;float_add(dt+ids[i/width]*width+i%width,dy[i]);if(positions)float_add(dp+(i/width%sequence)*width+i%width,dy[i]);}
// Group per (batch,head,query), local probabilities are never retained on the host.
__kernel void attention(__global const float* qkv,__global float* y,__global float* prob,
    int batch,int seq,int width,int heads,int causal,__local float* s){int row=get_group_id(0),lane=get_local_id(0),ls=get_local_size(0);
    int q=row%seq,h=(row/seq)%heads,b=row/(seq*heads),d=width/heads,n=causal?q+1:seq;float scale=rsqrt((float)d);
    for(int k=lane;k<seq;k+=ls){float dot=0;if(k<n)for(int c=0;c<d;c++)dot+=qkv[(b*seq+q)*3*width+h*d+c]*qkv[(b*seq+k)*3*width+width+h*d+c];s[k]=k<n?dot*scale:-INFINITY;}
    barrier(CLK_LOCAL_MEM_FENCE);if(lane==0){float m=-INFINITY,z=0;for(int k=0;k<n;k++)m=fmax(m,s[k]);for(int k=0;k<n;k++){s[k]=exp(s[k]-m);z+=s[k];}for(int k=0;k<n;k++)s[k]/=z;for(int k=n;k<seq;k++)s[k]=0;}
    barrier(CLK_LOCAL_MEM_FENCE);for(int k=lane;k<seq;k+=ls)prob[row*seq+k]=s[k];
    for(int c=lane;c<d;c+=ls){float v=0;for(int k=0;k<n;k++)v+=s[k]*qkv[(b*seq+k)*3*width+2*width+h*d+c];y[(b*seq+q)*width+h*d+c]=v;}}
__kernel void attention_ds(__global const float* qkv,__global const float* dy,__global const float* p,
    __global float* ds,int seq,int width,int heads,__local float* s){int row=get_group_id(0),lane=get_local_id(0),ls=get_local_size(0),q=row%seq,h=(row/seq)%heads,b=row/(seq*heads),d=width/heads;
    for(int k=lane;k<seq;k+=ls){float dot=0;for(int c=0;c<d;c++)dot+=dy[(b*seq+q)*width+h*d+c]*qkv[(b*seq+k)*3*width+2*width+h*d+c];s[k]=dot;}
    barrier(CLK_LOCAL_MEM_FENCE);if(lane==0){float z=0;for(int k=0;k<seq;k++)z+=p[row*seq+k]*s[k];s[seq]=z;}
    barrier(CLK_LOCAL_MEM_FENCE);for(int k=lane;k<seq;k+=ls)ds[row*seq+k]=p[row*seq+k]*(s[k]-s[seq])*rsqrt((float)d);}
__kernel void attention_dq(__global const float* qkv,__global const float* ds,__global float* dx,int batch,int seq,int width,int heads){int i=get_global_id(0);if(i>=batch*seq*width)return;int c=i%width,q=i/width%seq,b=i/(width*seq),d=width/heads,h=c/d;float sum=0;for(int k=0;k<seq;k++)sum+=ds[((b*heads+h)*seq+q)*seq+k]*qkv[(b*seq+k)*3*width+width+c];dx[(b*seq+q)*3*width+c]+=sum;}
__kernel void attention_dkv(__global const float* qkv,__global const float* dy,__global const float* p,__global const float* ds,__global float* dx,int batch,int seq,int width,int heads){int i=get_global_id(0);if(i>=batch*seq*width)return;int c=i%width,k=i/width%seq,b=i/(width*seq),d=width/heads,h=c/d;float dk=0,dv=0;for(int q=0;q<seq;q++){int j=((b*heads+h)*seq+q)*seq+k;dk+=ds[j]*qkv[(b*seq+q)*3*width+c];dv+=p[j]*dy[(b*seq+q)*width+c];}dx[(b*seq+k)*3*width+width+c]+=dk;dx[(b*seq+k)*3*width+2*width+c]+=dv;}
__kernel void cross_entropy(__global const float* x,__global const int* labels,__global float* stats,__global float* loss,int rows,int width,int ignore,int valid,float smooth){int r=get_global_id(0);if(r>=rows)return;if(labels[r]==ignore){loss[r]=0;stats[2*r]=0;stats[2*r+1]=0;return;}float m=-INFINITY,s=0,total=0;for(int c=0;c<width;c++)m=fmax(m,x[r*width+c]);for(int c=0;c<width;c++){s+=exp(x[r*width+c]-m);total+=x[r*width+c];}float lse=m+log(s);stats[2*r]=m;stats[2*r+1]=1/s;loss[r]=(lse-(1-smooth)*x[r*width+labels[r]]-smooth*total/width)/valid;}
__kernel void cross_entropy_back(__global const float* x,__global const int* labels,__global const float* stats,__global float* dx,int rows,int width,int ignore,int valid,float smooth,float upstream){int i=get_global_id(0);if(i>=rows*width)return;int r=i/width,c=i%width;if(labels[r]==ignore)return;float p=exp(x[i]-stats[2*r])*stats[2*r+1];dx[i]+=(p-smooth/width-(labels[r]==c?1-smooth:0))*upstream/valid;}
__kernel void sum(__global const float* x,__global float* y,int n,int squared){float s=0;for(int i=0;i<n;i++)s+=squared?x[i]*x[i]:x[i];y[0]=s;}
__kernel void gemm(__global const float* a,__global const float* b,__global float* c,int m,int n,int k,int ta,int tb){int i=get_global_id(0);if(i>=m*n)return;int r=i/n,col=i%n;float s=0;for(int j=0;j<k;j++)s+=a[ta?j*m+r:r*k+j]*b[tb?col*k+j:j*n+col];c[i]=s;}
__kernel void axpby(__global const float* a,__global const float* b,__global float* c,int n,float sa,float sb){int i=get_global_id(0);if(i<n)c[i]=sa*a[i]+sb*b[i];}
__kernel void transpose_scale(__global const float* x,__global float* y,int rows,int cols,int transpose,float scale){int i=get_global_id(0);if(i<rows*cols)y[transpose?(i%cols)*rows+i/cols:i]=x[i]*scale;}
__kernel void moments(__global const float* g,__global float* fast,__global float* slow,__global float* fh,__global float* sh,int n,float bf,float bs,float fc,float sc,int nesterov){int i=get_global_id(0);if(i>=n)return;float f=bf*fast[i]+(1-bf)*g[i],s=bs*slow[i]+(1-bs)*g[i];fast[i]=f;slow[i]=s;fh[i]=f/fc;sh[i]=s/sc;if(nesterov){float d=bf*f+(1-bf)*g[i];fh[i]=d;sh[i]=d;slow[i]=d;}}
__kernel void confidence(__global const float* f,__global const float* s,__global float* stats,int n){float dot=0,ff=0,ss=0,rr=0;for(int i=0;i<n;i++){dot+=f[i]*s[i];ff+=f[i]*f[i];ss+=s[i]*s[i];float d=f[i]-s[i];rr+=d*d;}stats[0]=dot;stats[1]=ff;stats[2]=ss;stats[3]=rr;}
__kernel void adam(__global const float* g,__global float* m,__global float* v,__global float* w,int n,float b1,float b2,float scale,float eps,float decay){int i=get_global_id(0);if(i>=n)return;float f=b1*m[i]+(1-b1)*g[i],s=b2*v[i]+(1-b2)*g[i]*g[i];m[i]=f;v[i]=s;w[i]=w[i]*decay-scale*f/(sqrt(s)+eps);}

// Pairwise reduction avoids the error growth of a million-element serial sum.
__kernel void reduce_sum(__global const float* x,__global float* out,int n,int squared,__local float* scratch) {
    int i=get_global_id(0),l=get_local_id(0);float v=i<n?x[i]:0;scratch[l]=squared?v*v:v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int stride=get_local_size(0)/2;stride>0;stride/=2){if(l<stride)scratch[l]+=scratch[l+stride];barrier(CLK_LOCAL_MEM_FENCE);}
    if(l==0)out[get_group_id(0)]=scratch[0];
}
float round_bf16(float x){uint bits=as_uint(x);if((bits&0x7f800000u)==0x7f800000u)return x;return as_float((bits+0x7fffu+((bits>>16)&1u))&0xffff0000u);}
__kernel void gemm_precision(__global const float* a,__global const float* b,__global float* c,int m,int n,int k,int ta,int tb,int bf16){
    int i=get_global_id(0);if(i>=m*n)return;int r=i/n,col=i%n;float s=0;
    for(int j=0;j<k;j++){float av=a[ta?j*m+r:r*k+j],bv=b[tb?col*k+j:j*n+col];s=fma(bf16?round_bf16(av):av,bf16?round_bf16(bv):bv,s);}c[i]=s;
}
__kernel void ns_coefficient(__global const float* g,__global const float* square,__global float* out,int rows,float a,float b,float c){int i=get_global_id(0);if(i<rows*rows)out[i]=fma(c,square[i],b*g[i])+(i/rows==i%rows?a:0);}
__kernel void confidence_blocks(__global const float* f,__global const float* s,__global float* out,int n,__local float* tmp){
    int i=get_global_id(0),l=get_local_id(0);float a=i<n?f[i]:0,b=i<n?s[i]:0,d=a-b;
    tmp[l]=a*b;tmp[256+l]=a*a;tmp[512+l]=b*b;tmp[768+l]=d*d;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int stride=128;stride>0;stride/=2){if(l<stride)for(int j=0;j<4;j++)tmp[j*256+l]+=tmp[j*256+l+stride];barrier(CLK_LOCAL_MEM_FENCE);}
    if(l<4)out[get_group_id(0)*4+l]=tmp[l*256];
}
__kernel void confidence_finish(__global const float* parts,__global float* out,int groups){int c=get_global_id(0);float s=0;for(int i=0;i<groups;i++)s+=parts[i*4+c];out[c]=s;}
__kernel void copy_range(__global const float* source,__global float* target,int sourceOffset,int targetOffset,int n,int add){int i=get_global_id(0);if(i<n)target[targetOffset+i]=(add?target[targetOffset+i]:0)+source[sourceOffset+i];}
__kernel void round_bf16_values(__global float* values,int n){int i=get_global_id(0);if(i<n)values[i]=round_bf16(values[i]);}
__kernel void loss_total(__global const float* rows,__global float* total,int n){float sum=0;for(int i=0;i<n;i++)sum+=rows[i];total[0]+=sum;}
__kernel void cross_entropy_gradient(__global const float* x,__global const int* labels,__global const float* stats,__global float* dx,int rows,int width,int ignore,int valid,float upstream){int i=get_global_id(0);if(i>=rows*width)return;int r=i/width,c=i%width;dx[i]=labels[r]==ignore?0:(exp(x[i]-stats[2*r])*stats[2*r+1]-(labels[r]==c?1:0))*upstream/valid;}
__kernel void decode_bfp8(__global const char* payload,__global const float* scales,__global float* output,int n,int block,int bf16){int i=get_global_id(0);if(i<n){float x=(float)payload[i]*scales[i/block];output[i]=bf16?round_bf16(x):x;}}
__kernel void decode_bf16(__global const ushort* payload,__global float* output,int n){int i=get_global_id(0);if(i<n)output[i]=as_float(((uint)payload[i])<<16);}
__kernel void scale_in_place(__global float* values,int n,float scale){int i=get_global_id(0);if(i<n)values[i]*=scale;}
__kernel void cross_entropy_rows(__global const float* x,__global const int* labels,__global float* stats,__global float* loss,int rows,int width,int ignore,int valid,__local float* tmp){
    int r=get_group_id(0),lane=get_local_id(0),size=get_local_size(0);
    if(labels[r]==ignore){if(lane==0){loss[r]=0;stats[2*r]=0;stats[2*r+1]=0;}return;}
    float m=-INFINITY;for(int c=lane;c<width;c+=size)m=fmax(m,x[r*width+c]);tmp[lane]=m;
    barrier(CLK_LOCAL_MEM_FENCE);for(int k=size/2;k>0;k/=2){if(lane<k)tmp[lane]=fmax(tmp[lane],tmp[lane+k]);barrier(CLK_LOCAL_MEM_FENCE);}
    m=tmp[0];barrier(CLK_LOCAL_MEM_FENCE);
    float sum=0;for(int c=lane;c<width;c+=size)sum+=exp(x[r*width+c]-m);tmp[lane]=sum;
    barrier(CLK_LOCAL_MEM_FENCE);for(int k=size/2;k>0;k/=2){if(lane<k)tmp[lane]+=tmp[lane+k];barrier(CLK_LOCAL_MEM_FENCE);}
    if(lane==0){stats[2*r]=m;stats[2*r+1]=1/tmp[0];loss[r]=(m+log(tmp[0])-x[r*width+labels[r]])/valid;}
}
