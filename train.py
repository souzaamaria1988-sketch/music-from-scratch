#!/usr/bin/env python3
import json,argparse,time,math
from pathlib import Path
import torch
from torch.utils.data import Dataset,DataLoader
from model import MusicTransformer,VOCAB,count_params
class TokDS(Dataset):
    def __init__(self,path,block):
        self.seqs=[];self.block=block
        for line in open(path):
            line=line.strip()
            if not line:continue
            t=json.loads(line)
            if len(t)>=block+2:self.seqs.append(t)
        print('Sequencias: '+str(len(self.seqs)))
    def __len__(self):return len(self.seqs)
    def __getitem__(self,i):
        s=self.seqs[i]
        mx=max(0,len(s)-self.block-1)
        st=0 if mx==0 else torch.randint(0,mx+1,(1,)).item()
        x=s[st:st+self.block];y=s[st+1:st+self.block+1]
        x=x+[0]*(self.block-len(x));y=y+[-100]*(self.block-len(y))
        return torch.tensor(x,dtype=torch.long),torch.tensor(y,dtype=torch.long)
def lr_at(step,total,warmup,lr,min_lr):
    if step<warmup:return lr*step/max(1,warmup)
    p=(step-warmup)/max(1,total-warmup)
    return min_lr+(lr-min_lr)*0.5*(1+math.cos(math.pi*p))
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tokens',default='tokens.jsonl')
    ap.add_argument('--block',type=int,default=512)
    ap.add_argument('--epochs',type=int,default=40)
    ap.add_argument('--batch',type=int,default=24)
    ap.add_argument('--lr',type=float,default=3e-4)
    ap.add_argument('--min-lr',type=float,default=3e-5)
    ap.add_argument('--warmup',type=int,default=200)
    ap.add_argument('--d-model',type=int,default=384)
    ap.add_argument('--n-layer',type=int,default=8)
    ap.add_argument('--n-head',type=int,default=12)
    ap.add_argument('--d-ff',type=int,default=1536)
    ap.add_argument('--out',default='checkpoints')
    a=ap.parse_args()
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: '+dev)
    ds=TokDS(a.tokens,a.block)
    dl=DataLoader(ds,batch_size=a.batch,shuffle=True,num_workers=2,drop_last=True)
    m=MusicTransformer(vocab=VOCAB,d_model=a.d_model,n_head=a.n_head,n_layer=a.n_layer,d_ff=a.d_ff,max_len=a.block).to(dev)
    print('Params: '+str(count_params(m)))
    opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=0.05,betas=(0.9,0.95))
    total=a.epochs*len(dl)
    step=0
    best=float('inf')
    for ep in range(a.epochs):
        m.train();s=0.0;n=0;t0=time.time()
        for x,y in dl:
            x=x.to(dev);y=y.to(dev)
            for g in opt.param_groups:g['lr']=lr_at(step,total,a.warmup,a.lr,a.min_lr)
            _,loss=m(x,y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
            opt.step();opt.zero_grad();step+=1
            s+=loss.item();n+=1
            if n%20==0:print('ep '+str(ep+1)+' step '+str(n)+'/'+str(len(dl))+' loss='+str(round(loss.item(),4))+' lr='+str(round(opt.param_groups[0]['lr'],6)))
        avg=s/max(1,n)
        print('== ep '+str(ep+1)+'/'+str(a.epochs)+' loss='+str(round(avg,4))+' t='+str(round(time.time()-t0,1))+'s')
        if avg<best:
            best=avg;Path(a.out).mkdir(parents=True,exist_ok=True)
            torch.save({'model':m.state_dict(),'vocab':VOCAB,'d_model':a.d_model,'n_head':a.n_head,'n_layer':a.n_layer,'d_ff':a.d_ff,'max_len':a.block},Path(a.out)/'best.pt')
            print('  saved '+str(round(best,4)))
    print('done')
if __name__=='__main__':main()