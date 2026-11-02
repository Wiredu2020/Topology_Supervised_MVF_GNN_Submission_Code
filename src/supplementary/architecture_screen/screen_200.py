"""Exploratory 200-way differentiable multivector-inspired incidence routing screen.
No labels are used in topology features. Test split remains sealed in this script.
Every candidate differs on at least one of four pairing rules, five incidence
normalizers, five local motif priors, two
message-update rules; all 200 train parameters end-to-end.
Not a trained discrete combinatorial multivector field. Soft face-incidence
selection is the trainable proxy; post-hoc blocks can be extracted separately.
"""
import os, sys, json, itertools, time, csv, copy, argparse
from pathlib import Path
os.environ.setdefault('MVF_THREADS','1')
ROOT=Path(__file__).resolve().parent; PR=ROOT.parents[1]/'study'/'router_core'/'project'
sys.path.insert(0,str(PR));sys.path.insert(0,str(PR/'source'))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from run_structural_suite import (load_tu_zip, feature_graph, make_batch, chunks,
    setup_seed, StructuralBlock, StructuralRouter, PlainGINE, scatter_sum)
import pandas as pd
from collections import defaultdict
import networkx as nx

torch.set_num_threads(1)
policies=('mutual','union','source','mean')
normalizers=('sigmoid','softmax','power','degree_sigmoid','centered_sigmoid')
priors=('none','triangle','bridge','degree_gap','cycle_bridge')
updates=('residual','centered')
configs=[(p,n,m,u) for p,n,m,u in itertools.product(policies,normalizers,priors,updates)]
assert len(configs)==200

class MVFBlock(StructuralBlock):
    def __init__(self,h, config, drop=.15):
        super().__init__(h,drop);self.cfg=config
        # Shared learnable motif correction for fair parameter counts.
    def getw(self,h,b,e,mode=None):
        policy,norm,motif,update=self.cfg
        pairs=b['pairs']; a,v=pairs[:,0],pairs[:,1]; n=len(h);m=len(a)
        deg=torch.bincount(pairs.flatten(),minlength=n).to(h.dtype).clamp(min=1)
        tri=b['edge'][:,-4];bridge=b['edge'][:,-3]
        if motif=='none': feat=torch.zeros_like(tri)
        elif motif=='triangle':feat=tri
        elif motif=='bridge':feat=bridge
        elif motif=='degree_gap':feat=(deg[a]-deg[v]).abs()/(deg[a]+deg[v])
        else:feat=tri-bridge
        prior=0.35*feat + self.struct_bias(torch.stack([tri,bridge],-1)).flatten()
        meta1=torch.stack([torch.log1p(deg[a]),torch.log1p(deg[v])],-1)
        meta2=meta1.flip(-1)
        s1=self.scorer(torch.cat([h[a],h[v],e,meta1],-1)).flatten()+prior
        s2=self.scorer(torch.cat([h[v],h[a],e,meta2],-1)).flatten()+prior
        logits=torch.cat([s1,s2]);ids=torch.cat([a,v]);
        def probs(temp):
            z=logits/temp
            if norm=='sigmoid':return torch.sigmoid(z)
            if norm=='degree_sigmoid':return torch.sigmoid(z-.35*torch.log1p(deg[ids]))
            if norm=='centered_sigmoid':
                s=z.new_zeros(n).index_add_(0,ids,z)
                return torch.sigmoid(z-s[ids]/deg[ids])
            mx=z.new_full((n,),-1e9);mx.scatter_reduce_(0,ids,z,reduce='amax',include_self=True)
            exp=torch.exp(z-mx[ids]);
            if norm=='power':exp=exp.square()
            den=z.new_zeros(n).index_add_(0,ids,exp)
            return 1-torch.exp(-deg[ids]*exp/den[ids].clamp(min=1e-8))
        def fused(t):
            q=probs(t);x,y=q[:m],q[m:]
            if policy=='mutual':u=2.5*x*y;d=u
            elif policy=='union':u=x+y-x*y;d=u
            elif policy=='source':u=x;d=y
            else:u=(x+y)/2;d=u
            arc=b['arc_edge_id']; forward=(b['src']==a[arc])
            return torch.where(forward,u[arc],d[arc])
        t1,t2=(0.65,1.50)
        return fused(t1),fused(t2),deg
    def forward(self,h,b,e,mode=None):
        n=len(h);msg=F.relu(h[b['src']]+e[b['arc_edge_id']]);agg=scatter_sum(msg,b['dst'],n)
        w1,w2,deg=self.getw(h,b,e)
        if self.cfg[-1]=='residual':
            c1=w1-1;c2=w2-1
        else:
            c1=w1-scatter_sum(w1[:,None],b['dst'],n)[b['dst'],0]/deg[b['dst']]
            c2=w2-scatter_sum(w2[:,None],b['dst'],n)[b['dst'],0]/deg[b['dst']]
        d1=scatter_sum(c1[:,None]*msg,b['dst'],n)
        d2=scatter_sum(c2[:,None]*msg,b['dst'],n)
        z=torch.cat([(1+self.eps)*h+agg,self.amp*d1/deg.sqrt()[:,None],self.amp*d2/deg.sqrt()[:,None]],-1)
        return F.relu(h+self.drop(self.bn(self.nn(z))))

class MVFRouter(StructuralRouter):
    def __init__(self,dx,de,nclass,h,config):
        super().__init__(dx,de,nclass,h=h)
        self.layers=nn.ModuleList([MVFBlock(h,config) for _ in range(2)])

def data(dataset):
    names={'PTC_MR':'PTC_MR(1)','COX2':'COX2(1)','NCI1':'NCI1(1)'}
    raw,y=load_tu_zip(str(ROOT.parents[1]/'study'/'router_core'/'data'/(names[dataset]+'.zip')),dataset)
    G=[]
    for i,g in enumerate(raw):
        p=np.asarray(g['edges'],np.int64).reshape(-1,2)
        x=np.asarray(g['node'],np.float32)
        ea=np.asarray(g['edge_attr'],np.float32).reshape(len(p),-1)
        G.append(feature_graph(g['n'],p,x,edge=ea,gid=g['original_id']))
    return G,np.asarray(y,dtype=np.int64)

def fit_model(G,y,fit,val,conf=None,epochs=6,seed=2027,h=24):
    setup_seed(seed)
    sc=StandardScaler().fit(np.concatenate([G[i]['x'] for i in fit]))
    model=(MVFRouter(G[0]['x'].shape[1],G[0]['edge'].shape[1],int(y.max())+1,h,conf)
           if conf is not None else PlainGINE(G[0]['x'].shape[1],G[0]['edge'].shape[1],int(y.max())+1,h=h))
    opt=torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=1e-4)
    cw=np.bincount(y[fit],minlength=int(y.max())+1)
    w=torch.tensor(np.sqrt(len(fit)/np.maximum(1,cw)),dtype=torch.float32);w/=w.mean()
    for ep in range(epochs):
        model.train()
        for ids in chunks(G,fit,1600,seed=seed*137+ep):
            b=make_batch(G,ids,sc)
            logits=model(b)  # shared GINE path or MVF routing
            loss=F.cross_entropy(logits,torch.tensor(y[ids]),weight=w)
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
    model.eval();ys=[];pr=[]
    with torch.no_grad():
        for ids in chunks(G,val,1600):
            b=make_batch(G,ids,sc);pred=model(b).argmax(-1).cpu().numpy()
            ys.extend(y[ids]);pr.extend(pred)
    return float(balanced_accuracy_score(ys,pr)),sum(p.numel() for p in model.parameters())

def main():
    ag=argparse.ArgumentParser();ag.add_argument('--dataset',default='PTC_MR');ag.add_argument('--epochs',type=int,default=6);ag.add_argument('--start',type=int,default=0);ag.add_argument('--end',type=int,default=200);ag.add_argument('--out',default=str(ROOT/'screen_200_results.csv'));a=ag.parse_args()
    G,y=data(a.dataset)
    dev,test=next(StratifiedShuffleSplit(n_splits=1,test_size=.20,random_state=20271027).split(np.arange(len(y)),y))
    f,v=next(StratifiedShuffleSplit(n_splits=1,test_size=.23,random_state=20271028).split(dev,y[dev]));fit,val=dev[f],dev[v]
    assert not set(test)&set(fit) and not set(val)&set(fit) and not set(val)&set(test)
    splits=dict(dataset=a.dataset,n_graphs=len(y),n_fit=len(fit),n_val=len(val),n_untouched_test=len(test),fit_ids=fit.tolist(),val_ids=val.tolist(),sealed_test_ids=test.tolist(),split_seed=20271027,inner_seed=20271028,epochs=a.epochs)
    (ROOT/'screen_splits.json').write_text(json.dumps(splits,indent=2))
    outfile=Path(a.out);exists=outfile.exists()
    with outfile.open('a',newline='') as fd:
        fields=['id','policy','normalizer','motif_prior','update','val_BA','train_params','seconds']
        w=csv.DictWriter(fd,fieldnames=fields)
        if not exists:w.writeheader()
        if a.start==0:
            t=time.time();ba,k=fit_model(G,y,fit,val,None,a.epochs)
            w.writerow(dict(id=-1,policy='GINE_h24',normalizer='',motif_prior='',update='',val_BA=ba,train_params=k,seconds=time.time()-t));fd.flush();print('BASELINE GINE_h24',round(100*ba,2),flush=True)
        for k in range(a.start,a.end):
            conf=configs[k];t=time.time()
            ba,npar=fit_model(G,y,fit,val,conf,a.epochs)
            w.writerow(dict(zip(['id','policy','normalizer','motif_prior','update','val_BA','train_params','seconds'],[k,*conf,ba,npar,time.time()-t])))
            fd.flush()
            if k%10==9:print('COMPLETED',k+1,'/',200,'val_BA',round(ba*100,2),'elapsed',round(time.time()-t,2),flush=True)

if __name__=='__main__':main()
