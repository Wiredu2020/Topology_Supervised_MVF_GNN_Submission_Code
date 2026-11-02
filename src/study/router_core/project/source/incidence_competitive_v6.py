"""Graph incidence routing and multivector-learning experiments."""
from __future__ import annotations
import os, random, time, json, argparse, copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from budgeted_mvf_gine import load_tu_zip,preprocess_graph,make_batch,scatter_sum,scatter_mean,chunks,setup_seed

torch.set_num_threads(int(os.getenv('MVF_THREADS','2')))
MODES=('base','uniform','sym','asym','shuffle_sym','shuffle_asym')

class RoutingBlock(nn.Module):
    def __init__(self,h,drop,kind):
        super().__init__();self.kind=kind
        self.eps=nn.Parameter(torch.zeros(1))
        self.scorer=nn.Sequential(nn.Linear(3*h+2,h),nn.ReLU(),nn.Linear(h,1))
        nn.init.constant_(self.scorer[-1].bias,-.05)
        if kind=='direct':in_h=h
        elif kind=='dual':in_h=2*h
        else:raise ValueError(kind)
        self.nn=nn.Sequential(nn.Linear(in_h,2*h),nn.ReLU(),nn.Linear(2*h,h))
        self.bn=nn.BatchNorm1d(h);self.drop=nn.Dropout(drop)
        # learn an appropriate amplitude; start nonzero so score path is active
        self.amp=nn.Parameter(torch.tensor(1.0))
    def weights(self,h,b,e,mode):
        p=b['pairs']; m=len(p);n=len(h)
        if not m:return h.new_zeros((0,)),None
        a=p[:,0];c=p[:,1];degree=torch.bincount(p.flatten(),minlength=n).to(h.dtype)
        def calc(u,v):
            meta=torch.stack([torch.log1p(degree[u]),torch.log1p(degree[v])],-1)
            return self.scorer(torch.cat([h[u],h[v],e,meta],-1)).flatten()
        l1=calc(a,c);l2=calc(c,a)
        # Each vertex competes among its actual incident edges. Normalization is
        # permutation equivariant and uses no label information.
        ix=torch.cat((a,c),0);logit=torch.cat((l1,l2),0)/.7
        mx=logit.new_full((n,),-1e9)
        mx.scatter_reduce_(0,ix,logit,reduce='amax',include_self=True)
        es=torch.exp(logit-mx[ix])
        denom=es.new_zeros(n).index_add_(0,ix,es)
        mass=es/denom[ix].clamp(min=1e-8)
        q=1-torch.exp(-degree[ix]*mass)
        q1,q2=q[:m],q[m:]
        if mode=='base':w=torch.ones_like(q1);other=w
        elif mode=='uniform':w=torch.ones_like(q1)*.85;other=w
        elif mode in ('sym','shuffle_sym'):
            w=2.5*q1*q2;other=w
        elif mode in ('asym','shuffle_asym'):
            w=1.58*q1;other=1.58*q2
        else:raise ValueError(mode)
        if mode.startswith('shuffle'):
            perm=torch.argsort(b['eg'].to(h.dtype)+torch.rand_like(w));w=w[perm];other=other[perm]
        eid=b['arc_edge_id'];res=torch.where(b['src']==p[eid,0],w[eid],other[eid])
        return res,(q1,q2)
    def forward(self,h,b,e,mode):
        n=len(h);msg=F.relu(h[b['src']]+e[b['arc_edge_id']]);agg=scatter_sum(msg,b['dst'],n)
        if mode=='base': arc_w=None
        else: arc_w,_=self.weights(h,b,e,mode)
        if arc_w is not None:
            diff=scatter_sum((arc_w-1.)[:,None]*msg,b['dst'],n)
        else:diff=torch.zeros_like(agg)
        z=(1+self.eps)*h+agg
        if self.kind=='direct':inp=z+self.amp*diff
        else:
            deg=torch.bincount(b['dst'],minlength=n).to(h.dtype).clamp(min=1)
            inp=torch.cat((z, self.amp*diff/deg.sqrt()[:,None]),-1)
        return F.relu(h+self.drop(self.bn(self.nn(inp))))

class IncidenceRouter(nn.Module):
    def __init__(self,dx,de,nclass,hidden=32,depth=2,drop=.15,kind='direct',raw=True):
        super().__init__();self.depth=depth;self.hidden=hidden;self.raw_on=raw
        self.node=nn.Sequential(nn.Linear(dx,hidden),nn.ReLU(),nn.LayerNorm(hidden))
        self.edge=nn.Sequential(nn.Linear(de,hidden),nn.ReLU(),nn.Linear(hidden,hidden))
        self.layers=nn.ModuleList([RoutingBlock(hidden,drop,kind) for _ in range(depth)])
        self.raw=nn.Sequential(nn.Linear(2*dx+2*de+2,hidden),nn.ReLU(),nn.Linear(hidden,hidden))
        self.head=nn.Sequential(nn.Linear(depth*2*hidden+hidden,2*hidden),nn.ReLU(),nn.Dropout(drop),nn.Linear(2*hidden,nclass))
    def forward(self,b,mode='base'):
        h=self.node(b['x']);e=self.edge(b['edge']);outs=[]
        for layer in self.layers:
            h=layer(h,b,e,mode)
            s=scatter_sum(h,b['ng'],b['ngraphs']);ct=torch.bincount(b['ng'],minlength=b['ngraphs']).to(h.dtype).clamp(min=1)
            outs.extend([s/ct[:,None],s/ct.sqrt()[:,None]])
        xmean=scatter_mean(b['x'],b['ng'],b['ngraphs']);xstd=(scatter_mean(b['x']**2,b['ng'],b['ngraphs'])-xmean.square()).clamp(min=1e-8).sqrt()
        if len(b['eg']):
            emean=scatter_mean(b['edge'],b['eg'],b['ngraphs']);estd=(scatter_mean(b['edge']**2,b['eg'],b['ngraphs'])-emean.square()).clamp(min=1e-8).sqrt()
        else:
            emean=e.new_zeros((b['ngraphs'],b['edge'].shape[-1]));estd=emean
        nv=torch.bincount(b['ng'],minlength=b['ngraphs']).to(h.dtype);ne=torch.bincount(b['eg'],minlength=b['ngraphs']).to(h.dtype)
        raw=self.raw(torch.cat((xmean,xstd,emean,estd,torch.log1p(nv)[:,None],torch.log1p(ne)[:,None]),-1))
        if not self.raw_on:raw=torch.zeros_like(raw)
        return self.head(torch.cat(outs+[raw],-1))

def preflight(G,kind):
    gs=G[:3]; sc=StandardScaler().fit(np.concatenate([g['x'] for g in gs]));b=make_batch(gs,list(range(len(gs))),sc)
    model=IncidenceRouter(gs[0]['x'].shape[1],gs[0]['edge'].shape[1],2,hidden=24,kind=kind)
    for mode in MODES:
        model.zero_grad(set_to_none=True);z=model(b,mode);assert torch.isfinite(z).all()
        z.square().sum().backward()
        if mode in ('sym','asym'):
            gr=model.layers[0].scorer[-1].weight.grad
            assert gr is not None and gr.abs().max()>1e-9,(mode,'zero gate gradient')
    for mode in ('base','uniform','sym','asym'):
        model.eval()
        with torch.no_grad():
            a=model(b,mode);f=copy.deepcopy(gs)
            for g in f:
                rng=np.random.default_rng(456);ix=rng.permutation(len(g['pairs']))
                g['pairs']=g['pairs'][ix][:,::-1].copy();g['edge']=g['edge'][ix].copy()
            b2=make_batch(f,list(range(len(f))),sc)
            err=float((a-model(b2,mode)).abs().max());assert err<1e-4,(mode,err)
    return True

def train(G,y,mode,kind,seed=101,split_seed=20261020,folds=3,epochs=25,hidden=32,depth=2,patience=6,node_budget=1600,lr=.0008,raw=True,shuffle_eval=3):
    y=np.asarray(y,dtype=np.int64);nclass=int(y.max())+1;pred=np.full(len(y),-1,dtype=int);probs=np.zeros((len(y),nclass),np.float32);ofold=np.full(len(y),-1,dtype=int)
    rows=[];t0=time.time()
    outer=StratifiedKFold(n_splits=folds,random_state=split_seed,shuffle=True)
    for fold,(itr,ite) in enumerate(outer.split(np.arange(len(y)),y)):
        setup_seed(seed+fold*101)
        inn=StratifiedShuffleSplit(n_splits=1,test_size=.18,random_state=seed+fold*31)
        i1,i2=next(inn.split(itr,y[itr]));fit=itr[i1];val=itr[i2]
        scaler=StandardScaler().fit(np.concatenate([G[i]['x'] for i in fit]))
        model=IncidenceRouter(G[0]['x'].shape[1],G[0]['edge'].shape[1],nclass,hidden,depth,kind=kind,raw=raw)
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
        counts=np.bincount(y[fit],minlength=nclass)
        cw=torch.tensor(np.sqrt(counts.sum()/np.maximum(1,counts)),dtype=torch.float);cw/=cw.mean()
        best=np.inf;state=None;stale=0;bestep=0
        for ep in range(epochs):
            model.train()
            for ids in chunks(G,fit,node_budget,seed=seed*1000+fold*97+ep):
                inp=make_batch(G,ids,scaler)
                z=model(inp,mode)
                loss=F.cross_entropy(z,torch.tensor(y[ids]),weight=cw)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),2.0);optimizer.step()
            model.eval();vl=0.;total=0
            with torch.no_grad():
                for ids in chunks(G,val,node_budget):
                    inp=make_batch(G,ids,scaler)
                    if mode.startswith('shuffle'):
                        z=torch.stack([model(inp,mode) for _ in range(shuffle_eval)]).mean(0)
                    else:z=model(inp,mode)
                    vl+=float(F.cross_entropy(z,torch.tensor(y[ids]),weight=cw))*len(ids);total+=len(ids)
            vl/=max(1,total)
            if vl<best-1e-4:best=vl;bestep=ep+1;stale=0;state=copy.deepcopy(model.state_dict())
            else:stale+=1
            if stale>=patience and ep>=10:break
        model.load_state_dict(state);model.eval()
        with torch.no_grad():
            for ids in chunks(G,ite,node_budget):
                inp=make_batch(G,ids,scaler)
                if mode.startswith('shuffle'):prob=torch.stack([model(inp,mode).softmax(-1) for _ in range(shuffle_eval)]).mean(0).numpy()
                else:prob=model(inp,mode).softmax(-1).numpy()
                probs[ids]=prob;pred[ids]=prob.argmax(-1);ofold[ids]=fold
        ba=float(balanced_accuracy_score(y[ite],pred[ite]));rows.append(dict(fold=fold,balanced_accuracy=ba,best_epoch=bestep,n_test=len(ite)))
        print('FOLD',kind,mode,seed,fold,round(100*ba,2),'epoch',bestep,flush=True)
    assert np.all(pred>=0)
    return dict(accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),pred=pred,probs=probs,ofold=ofold,folds=rows,seconds=time.time()-t0,nparams=sum(p.numel() for p in model.parameters()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['COX2','PTC_MR','MUTAG']);p.add_argument('--modes',nargs='+',default=['base','uniform','sym','asym']);p.add_argument('--kind',choices=['direct','dual'],default='direct');p.add_argument('--seeds',nargs='+',type=int,default=[101]);p.add_argument('--folds',type=int,default=3);p.add_argument('--epochs',type=int,default=25);p.add_argument('--hidden',type=int,default=32);p.add_argument('--depth',type=int,default=2);p.add_argument('--patience',type=int,default=6);p.add_argument('--node_budget',type=int,default=1600);p.add_argument('--split_seed',type=int,default=20261020);p.add_argument('--lr',type=float,default=.0008);p.add_argument('--raw_off',action='store_true');p.add_argument('--out',default='/mnt/data/mvf_work/new_routing');p.add_argument('--shuffle_eval',type=int,default=3);a=p.parse_args()
    out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    (out/'config.json').write_text(json.dumps(vars(a),indent=2))
    for name in a.datasets:
        graphs,y=load_tu_zip(str(Path(os.environ.get('research study_DATA_ROOT','/mnt/data/mvf_work/datasets'))/(name+'.zip')),name);G=[preprocess_graph(g) for g in graphs]
        print('DATA',name,len(y),np.bincount(y).tolist(),'dim',G[0]['x'].shape[1],G[0]['edge'].shape[1],flush=True)
        print('PREFLIGHT',preflight(G,a.kind),flush=True)
        for seed in a.seeds:
            for mode in a.modes:
                tag=f'{name}_{a.kind}_{mode}_s{seed}'
                if (out/(tag+'.json')).exists():continue
                r=train(G,y,mode,a.kind,seed,a.split_seed,a.folds,a.epochs,a.hidden,a.depth,a.patience,a.node_budget,a.lr,not a.raw_off,a.shuffle_eval)
                np.savez_compressed(out/(tag+'.npz'),y=y,pred=r['pred'],probs=r['probs'],fold=r['ofold'],id=np.array([g['id'] for g in G]))
                d={k:v for k,v in r.items() if k not in ('pred','probs','ofold')};d.update(dataset=name,kind=a.kind,mode=mode,seed=seed)
                (out/(tag+'.json')).write_text(json.dumps(d,indent=2))
                print('RESULT',name,a.kind,mode,seed,round(r['balanced_accuracy']*100,2),'time',round(r['seconds'],1),flush=True)

if __name__=='__main__':main()
