#!/usr/bin/env python3
import math,torch,torch.nn as nn,torch.nn.functional as F
VOCAB=149
class MusicTransformer(nn.Module):
    def __init__(self,vocab=VOCAB,d_model=256,n_head=8,n_layer=6,d_ff=1024,max_len=1024,dropout=0.1):
        super().__init__()
        self.d_model=d_model;self.max_len=max_len
        self.tok_emb=nn.Embedding(vocab,d_model)
        self.pos_emb=nn.Embedding(max_len,d_model)
        self.drop=nn.Dropout(dropout)
        layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=n_head,dim_feedforward=d_ff,dropout=dropout,activation='gelu',batch_first=True,norm_first=True)
        self.blocks=nn.TransformerEncoder(layer,num_layers=n_layer)
        self.ln_f=nn.LayerNorm(d_model)
        self.head=nn.Linear(d_model,vocab,bias=False)
        self.head.weight=self.tok_emb.weight
        self.apply(self._init)
    def _init(self,m):
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight,0,0.02)
            if m.bias is not None:nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Embedding):
            nn.init.normal_(m.weight,0,0.02)
    def forward(self,idx,targets=None):
        B,T=idx.shape;T=min(T,self.max_len);idx=idx[:,:T]
        pos=torch.arange(T,device=idx.device).unsqueeze(0)
        x=self.drop(self.tok_emb(idx)+self.pos_emb(pos))
        mask=torch.triu(torch.ones(T,T,device=idx.device,dtype=torch.bool),diagonal=1)
        x=self.blocks(x,mask=mask);x=self.ln_f(x)
        logits=self.head(x)
        if targets is None:return logits,None
        tg=targets[:,:T]
        loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),tg.reshape(-1),ignore_index=-100)
        return logits,loss
    @torch.no_grad()
    def generate(self,idx,max_new_tokens=256,temperature=1.0,top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            ctx=idx[:,-self.max_len:]
            logits,_=self(ctx)
            logits=logits[:,-1,:]/max(1e-6,temperature)
            if top_k>0:
                v,_=torch.topk(logits,min(top_k,logits.size(-1)))
                logits[logits<v[:,[-1]]]=-float('inf')
            probs=F.softmax(logits,dim=-1)
            nxt=torch.multinomial(probs,1)
            idx=torch.cat([idx,nxt],dim=1)
            if nxt.item()==2:break
        return idx
def count_params(m):return sum(p.numel() for p in m.parameters())