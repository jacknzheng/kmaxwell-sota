"""Projection tiles with softcap/CE statistics, gradient, and HVP epilogues.

No full logits or softmax activation is retained. One temporary derivative
matrix per token chunk feeds a library GEMM back to hidden coordinates. Inputs
and accumulators are FP32. Highest uses IEEE dots; explicitly enabling TF32
permits TF32 dots, as in the benchmark's other GEMMs.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _stats(X,W,B,Y,S,M:tl.constexpr,D:tl.constexpr,V:tl.constexpr,
           NB:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,DOT:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for start in range(tl.cdiv(D,BK)):
        k=start*BK+kk
        x=tl.load(X+rows[:,None]*D+k[None,:],(rows[:,None]<M)&(k[None,:]<D),other=0)
        w=tl.load(W+cols[None,:]*D+k[:,None],(cols[None,:]<V)&(k[:,None]<D),other=0)
        acc=tl.dot(x,w,acc,input_precision=DOT)
    z=acc+tl.load(B+cols,cols<V,other=0)[None,:]
    u=15*z*tl.rsqrt(z*z+225)
    masked=tl.where(cols[None,:]<V,u,float('-inf'))
    maximum=tl.max(masked,1)
    denominator=tl.sum(tl.exp(masked-maximum[:,None]),1)
    target=tl.load(Y+rows,rows<M,other=-1)
    truth=tl.sum(tl.where(cols[None,:]==target[:,None],u,0),1)
    index=(rows*NB+tl.program_id(1))*3
    tl.store(S+index,maximum,rows<M)
    tl.store(S+index+1,denominator,rows<M)
    tl.store(S+index+2,truth,rows<M)


@triton.jit
def _reduce(S,LSE,LOSS,NB:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);cols=tl.arange(0,BLOCK)
    m=tl.load(S+(row*NB+cols)*3,cols<NB,other=float('-inf'))
    total=tl.load(S+(row*NB+cols)*3+1,cols<NB,other=0)
    truth=tl.load(S+(row*NB+cols)*3+2,cols<NB,other=0)
    maximum=tl.max(m,0)
    lse=maximum+tl.log(tl.sum(total*tl.exp(m-maximum),0))
    tl.store(LSE+row,lse);tl.store(LOSS+row,lse-tl.sum(truth,0))


@triton.jit
def _project_derivative(X,R,W,B,Y,LSE,MEAN,OUT,S,M:tl.constexpr,D:tl.constexpr,V:tl.constexpr,
                       NB:tl.constexpr,MODE:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,DOT:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    z=tl.zeros((BM,BN),tl.float32);dz=tl.zeros((BM,BN),tl.float32)
    for start in range(tl.cdiv(D,BK)):
        k=start*BK+kk
        x=tl.load(X+rows[:,None]*D+k[None,:],(rows[:,None]<M)&(k[None,:]<D),other=0)
        w=tl.load(W+cols[None,:]*D+k[:,None],(cols[None,:]<V)&(k[:,None]<D),other=0)
        z=tl.dot(x,w,z,input_precision=DOT)
        if MODE!=0:
            r=tl.load(R+rows[:,None]*D+k[None,:],(rows[:,None]<M)&(k[None,:]<D),other=0)
            dz=tl.dot(r,w,dz,input_precision=DOT)
    z=z+tl.load(B+cols,cols<V,other=0)[None,:]
    inv=tl.rsqrt(z*z+225);first=3375*inv*inv*inv
    lse=tl.load(LSE+rows,rows<M,other=0)
    p=tl.where(cols[None,:]<V,tl.exp(15*z*inv-lse[:,None]),0)
    if MODE==1:
        partial=tl.sum(p*first*dz,1)
        tl.store(S+rows*NB+tl.program_id(1),partial,rows<M)
    else:
        target=tl.load(Y+rows,rows<M,other=-1)
        error=p-tl.where(cols[None,:]==target[:,None],1.,0.)
        if MODE==0:
            result=error*first
        else:
            mean=tl.load(MEAN+rows,rows<M,other=0)
            second=-10125*z*inv*inv*inv*inv*inv
            result=p*(first*dz-mean[:,None])*first+error*second*dz
        tl.store(OUT+rows[:,None]*V+cols[None,:],result,
                 (rows[:,None]<M)&(cols[None,:]<V))


@triton.jit
def _reduce_direction(S,MEAN,NB:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);cols=tl.arange(0,BLOCK)
    values=tl.load(S+row*NB+cols,cols<NB,other=0)
    tl.store(MEAN+row,tl.sum(values,0))


@torch.library.custom_op('critroll_head::statistics',mutates_args=())
def statistics(x:torch.Tensor,weight:torch.Tensor,bias:torch.Tensor,
               targets:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    m,d=x.shape;v=weight.shape[0];nb=triton.cdiv(v,128)
    stats=torch.empty((m,nb,3),device=x.device,dtype=x.dtype)
    lse=torch.empty(m,device=x.device,dtype=x.dtype);loss=torch.empty_like(lse)
    precision='tf32' if torch.backends.cuda.matmul.allow_tf32 else 'ieee'
    _stats[(triton.cdiv(m,16),nb)](x,weight,bias,targets,stats,m,d,v,nb,16,128,64,precision)
    _reduce[(m,)](stats,lse,loss,nb,triton.next_power_of_2(nb))
    return loss,lse


@statistics.register_fake
def _statistics_fake(x,weight,bias,targets):
    return x.new_empty(x.shape[0]),x.new_empty(x.shape[0])


@torch.library.custom_op('critroll_head::derivative',mutates_args=())
def derivative(x:torch.Tensor,weight:torch.Tensor,bias:torch.Tensor,targets:torch.Tensor,
               lse:torch.Tensor,direction:torch.Tensor|None=None)->torch.Tensor:
    m,d=x.shape;v=weight.shape[0];nb=triton.cdiv(v,128)
    out=torch.empty((m,v),device=x.device,dtype=x.dtype)
    mean=torch.empty(m,device=x.device,dtype=x.dtype)
    partial=torch.empty((m,nb),device=x.device,dtype=x.dtype)
    grid=(triton.cdiv(m,16),nb)
    precision='tf32' if torch.backends.cuda.matmul.allow_tf32 else 'ieee'
    if direction is None:
        direction=x
        mode=0
    else:
        direction=direction.contiguous()
        _project_derivative[grid](x,direction,weight,bias,targets,lse,mean,out,partial,
                                 m,d,v,nb,1,16,128,64,precision)
        _reduce_direction[(m,)](partial,mean,nb,triton.next_power_of_2(nb))
        mode=2
    _project_derivative[grid](x,direction,weight,bias,targets,lse,mean,out,partial,
                             m,d,v,nb,mode,16,128,64,precision)
    return out @ weight


@derivative.register_fake
def _derivative_fake(x,weight,bias,targets,lse,direction=None):
    return torch.empty_like(x)
