#!/usr/bin/env python3
import math,torch,torch.nn as nn,torch.nn.functional as F
VOCAB=159
def rope_freqs(dim,max_len,theta=10000.0,device='cpu'):
    inv=1.0/(theta**(torch.arange(0,dim,2,device=device).float()/dim))
    t=torch.arange(max_len,device=device).float()
    f=torch.outer(t,inv)
    return f.cos(),f.sin()
class RoPEAttention(nn.Module):
    def __init__(self,d_model,n_head,dropout):
        super().__init__()
        self.d_model=d_model;self.n_head=n_head;self.head_dim=d_model//n_head
        self.qkv=nn.Linear(d_model,3*d_model,bias=False)
        self.proj=nn.Linear(d_model,d_model,bias=False)
        self.drop=nn.Dropout(dropout)
    def forward(self,x,cos,sin):
        B,T,C=x.shape
        qkv=self.qkv(x).reshape(B,T,3,self.n_head,self.head_dim).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        q=q.reshape(B,self.n_head,T,self.head_dim//2,2)
        k=k.reshape(B,self.n_head,T,self.head_dim//2,2)
        qr,qi=q[...,0],q[...,1]
        kr,ki=k[...,0],k[...,1]
        c=cos[:T].unsqueeze(0).unsqueeze(0);s=sin[:T].unsqueeze(0).unsqueeze(0)
        qr2=qr*c-qi*s;qi2=qr*s+qi*c
        kr2=kr*c-ki*s;ki2=kr*s+ki*c
        q=torch.stack([qr2,qi2],dim=-1).reshape(B,self.n_head,T,self.head_dim)
        k=torch.stack([kr2,ki2],dim=-1).reshape(B,self.n_head,T,self.head_dim)
        att=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        att=att.transpose(1,2).reshape(B,T,C)
        return self.drop(self.proj(att))
class Block(nn.Module):
    def __init__(self,d_model,n_head,d_ff,dropout):
        super().__init__()
        self.ln1=nn.LayerNorm(d_model)
        self.attn=RoPEAttention(d_model,n_head,dropout)
        self.ln2=nn.LayerNorm(d_model)
        self.mlp=nn.Sequential(nn.Linear(d_model,d_ff),nn.GELU(),nn.Linear(d_ff,d_model),nn.Dropout(dropout))
    def forward(self,x,cos,sin):
        x=x+self.attn(self.ln1(x),cos,sin)
        x=x+self.mlp(self.ln2(x))
        return x
class MusicTransformer(nn.Module):
    def __init__(self,vocab=VOCAB,d_model=384,n_head=12,n_layer=8,d_ff=1536,max_len=1024,dropout=0.1):
        super().__init__()
        self.d_model=d_model;self.max_len=max_len
        self.tok_emb=nn.Embedding(vocab,d_model)
        self.drop=nn.Dropout(dropout)
        self.blocks=nn.ModuleList([Block(d_model,n_head,d_ff,dropout) for _ in range(n_layer)])
        self.ln_f=nn.LayerNorm(d_model)
        self.head=nn.Linear(d_model,vocab,bias=False)
        self.head.weight=self.tok_emb.weight
        cos,sin=rope_freqs(d_model//n_head,max_len)
        self.register_buffer('cos',cos,persistent=False)
        self.register_buffer('sin',sin,persistent=False)
        self.apply(self._init)
    def _init(self,m):
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight,0,0.02)
            if m.bias is not None:nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Embedding):
            nn.init.normal_(m.weight,0,0.02)
    def forward(self,idx,targets=None):
        B,T=idx.shape;T=min(T,self.max_len);idx=idx[:,:T]
        x=self.drop(self.tok_emb(idx))
        cos,sin=self.cos.to(idx.device),self.sin.to(idx.device)
        for blk in self.blocks:x=blk(x,cos,sin)
        x=self.ln_f(x);logits=self.head(x)
        if targets is None:return logits,None
        tg=targets[:,:T]
        loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),tg.reshape(-1),ignore_index=-100,label_smoothing=0.02)
        return logits,loss
    @torch.no_grad()
    def generate(self,idx,max_new_tokens=256,temperature=1.0,top_k=40,top_p=0.95,repetition_penalty=1.15):
        self.eval()
        for _ in range(max_new_tokens):
            ctx=idx[:,-self.max_len:]
            logits,_=self(ctx)
            logits=logits[:,-1,:]/max(1e-6,temperature)
            if repetition_penalty!=1.0:
                for tok in set(idx[0].tolist()):
                    logits[0,tok]/=repetition_penalty
            if top_k>0:
                v,_=torch.topk(logits,min(top_k,logits.size(-1)))
                logits[logits<v[:,[-1]]]=-float('inf')
            if top_p<1.0:
                sl,_=torch.sort(logits,descending=True)
                cum=torch.cumsum(F.softmax(sl,dim=-1),dim=-1)
                mask=cum>top_p
                mask[...,1:]=mask[...,:-1].clone()
                mask[...,0]=0
                sl[mask]=-float('inf')
                logits=torch.full_like(logits,-float('inf')).scatter(1,torch.argsort(logits,descending=True),sl)
            probs=F.softmax(logits,dim=-1)
            nxt=torch.multinomial(probs,1)
            idx=torch.cat([idx,nxt],dim=1)
            if nxt.item()==2:break
        return idx
def count_params(m):return sum(p.numel() for p in m.parameters())