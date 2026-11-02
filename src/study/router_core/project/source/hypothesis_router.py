"""Graph incidence routing and multivector-learning experiments."""
from __future__ import annotations
import argparse, copy, json, os, time, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from budgeted_mvf_gine import load_tu_zip,preprocess_graph,make_batch,scatter_sum,scatter_mean,chunks,setup_seed
from incidence_competitive_v6 import IncidenceRouter, RoutingBlock

torch.set_num_threads(int(os.getenv('MVF_THREADS','1')))

class ExploreRoutingBlock(RoutingBlock):
    def __init__(self,hidden,drop,variant):
        super().__init__(hidden,drop,'dual')
        self.variant=variant
        if variant=='multitau':
            self.nn=nn.Sequential(nn.Linear(3*hidden,2*hidden),nn.ReLU(),nn.Linear(2*hidden,hidden))
        if variant in ('context','context_align'):
            self.meta_scale=nn.Sequential(nn.Linear(4,12),nn.Tanh(),nn.Linear(12,1))
            nn.init.zeros_(self.meta_scale[-1].weight);nn.init.constant_(self.meta_scale[-1].bias,0)
        if variant in ('context_align', 'spectral'):
            self.struct_bias=nn.Parameter(torch.tensor(0.0))
        if variant=='confidence':
            self.conf_scale=nn.Parameter(torch.tensor(1.0))

    def getw(self,h,b,e,mode):
        p=b['pairs'];m=len(p);n=len(h);a=p[:,0];v=p[:,1];
        deg=torch.bincount(p.flatten(),minlength=n).to(h.dtype)
        def logits(u,w):
            meta=torch.stack([torch.log1p(deg[u]),torch.log1p(deg[w])],-1)
            score=self.scorer(torch.cat([h[u],h[w],e,meta],-1)).squeeze(1)
            if self.variant in ('spectral','context_align'):
                sim=F.cosine_similarity(h[u],h[w],dim=-1).detach()
                score=score + self.struct_bias*sim
            return score
        log=torch.cat([logits(a,v),logits(v,a)],0);ix=torch.cat([a,v],0)
        def competition(tau):
            if self.variant=='temp_sharp': tau_node=tau*(deg[ix].clamp(min=1)/2.0).pow(-0.5)
            elif self.variant=='temp_smooth': tau_node=tau*(deg[ix].clamp(min=1)/2.0).pow(0.5)
            else: tau_node=tau
            z=log/tau_node;mx=z.new_full((n,),-1e9);mx.scatter_reduce_(0,ix,z,reduce='amax',include_self=True)
            val=torch.exp(z-mx[ix]);denom=z.new_zeros(n).index_add_(0,ix,val)
            q=1-torch.exp(-deg[ix]*val/denom[ix].clamp(min=1e-8))
            if self.variant=='confidence':
                # A centered selector suppresses small variations but keeps gradient to logits
                q=torch.sigmoid(self.conf_scale*(q-.5)*4)
            u,w=q[:m],q[m:]
            return (2.5*u*w),q
        w1,q1=competition(.7)
        if self.variant=='multitau':w2,q2=competition(1.5)
        else:w2=w1
        if self.variant=='ste_top2':
            # Straight-through top-2 threshold on locally ranked endpoint logits.
            # Hard forward decisions are not claimed differentiable; gradients flow
            # solely through the smooth competitive probabilities.
            z=log/.7
            mx=z.new_full((n,),-1e9)
            mx.scatter_reduce_(0,ix,z,reduce='amax',include_self=True)
            top_count=z.new_zeros(n).index_add_(0,ix,(z>=mx[ix]-1e-6).to(z.dtype))
            second_logits=torch.where(z<mx[ix]-1e-6,z,torch.full_like(z,-1e9))
            second=z.new_full((n,),-1e9)
            second.scatter_reduce_(0,ix,second_logits,reduce='amax',include_self=True)
            cut=torch.where((deg<=1)|(top_count>=2),mx,second)
            hard=(z>=cut[ix]-1e-6).to(z.dtype).detach()
            hard_edge=hard[:m]*hard[m:]
            w1=w1+0.4*(1.3*hard_edge-w1).detach()
            w2=w1
        if mode=='uniform':w1=torch.ones_like(w1)*.85;w2=torch.ones_like(w2)*.85
        if mode=='shuffle':
            # permute incidences only within each graph, preserving marginal gate distribution
            ids=b['eg'];r=torch.empty(m,dtype=torch.long,device=h.device)
            for k in range(b['ngraphs']):
                loc=(ids==k).nonzero().flatten()
                r[loc]=loc[torch.randperm(len(loc),device=h.device)]
            w1=w1[r];w2=w2[r]
        eids=b['arc_edge_id']
        return w1[eids],w2[eids],deg

    def forward(self,h,b,e,mode):
        n=len(h);msg=F.relu(h[b['src']]+e[b['arc_edge_id']])
        agg=scatter_sum(msg,b['dst'],n)
        w1,w2,deg=self.getw(h,b,e,mode)
        d1=scatter_sum((w1-1.)[:,None]*msg,b['dst'],n)
        d2=scatter_sum((w2-1.)[:,None]*msg,b['dst'],n)
        count=deg.clamp(min=1)
        if self.variant in ('context','context_align'):
            g=b['ng'];ng=b['ngraphs'];
            avg=scatter_mean(deg[:,None],g,ng).squeeze(1)
            dev=(scatter_mean((deg[:,None]-avg[g,None])**2,g,ng).squeeze(1)+1e-5).sqrt()
            size=torch.bincount(g,minlength=ng).to(h.dtype)
            density=avg # mean degree
            feats=torch.stack([torch.log1p(size),torch.log1p(avg),torch.log1p(dev),torch.log1p(density)],-1)
            scale=2*torch.sigmoid(self.meta_scale(feats)).squeeze(1)
            d1=d1*scale[g,None]
        z=(1+self.eps)*h+agg
        if self.variant=='multitau': inp=torch.cat([z,self.amp*d1/count.sqrt()[:,None],self.amp*d2/count.sqrt()[:,None]],-1)
        else: inp=torch.cat([z,self.amp*d1/count.sqrt()[:,None]],-1)
        return F.relu(h+self.drop(self.bn(self.nn(inp))))

class Explorer(IncidenceRouter):
    def __init__(self,dx,de,nclass,hidden=32,depth=2,drop=.15,variant='context'):
        super().__init__(dx,de,nclass,hidden,depth,drop,'dual',True)
        self.layers=nn.ModuleList([ExploreRoutingBlock(hidden,drop,variant) for _ in range(depth)])
        self.variant=variant
    def forward(self,b,mode='learned'):
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
        return self.head(torch.cat(outs+[raw],-1))

def preflight(G,nclass,variant):
    gs=G[:3]; sc=StandardScaler().fit(np.concatenate([g['x'] for g in gs]));b=make_batch(gs,list(range(len(gs))),sc)
    net=Explorer(b['x'].shape[1],b['edge'].shape[1],nclass,hidden=16,variant=variant)
    net.eval()
    for mode in ('learned','uniform','shuffle'):
        net.zero_grad(set_to_none=True);logits=net(b,mode)
        assert torch.isfinite(logits).all()
        logits.square().sum().backward()
        if mode=='learned':
            gr=net.layers[0].scorer[-1].weight.grad
            assert gr is not None and gr.abs().max()>1e-10,(variant,mode,'missing scorer gradient')
    with torch.no_grad():
        a=net(b,'learned'); gs2=copy.deepcopy(gs)
        for g in gs2:
            if len(g['pairs']):
                rng=np.random.default_rng(10);p=rng.permutation(len(g['pairs']))
                g['pairs']=g['pairs'][p][:,::-1].copy();g['edge']=g['edge'][p].copy()
        b2=make_batch(gs2,list(range(len(gs2))),sc)
        err=(a-net(b2,'learned')).abs().max().item()
        assert err<1e-4,('endpoint/edge order invariance failure',err)
    return dict(grad=True,invariance_error=err,params=sum(p.numel() for p in net.parameters()))

def train(G,y,variant,mode,seed=101,split_seed=20261020,epochs=25,hidden=32,depth=2,patience=6,node_budget=1600,lr=.0008):
    y=np.asarray(y,dtype=np.int64);nclass=int(y.max())+1;pred=np.full(len(y),-1,dtype=int);probs=np.zeros((len(y),nclass),np.float32);ofold=np.full(len(y),-1,dtype=int)
    rows=[];t0=time.time()
    outer=StratifiedKFold(n_splits=3,random_state=split_seed,shuffle=True)
    for fold,(itr,ite) in enumerate(outer.split(np.arange(len(y)),y)):
        setup_seed(seed+fold*101)
        inn=StratifiedShuffleSplit(n_splits=1,test_size=.18,random_state=seed+fold*31)
        i1,i2=next(inn.split(itr,y[itr]));fit=itr[i1];val=itr[i2]
        scaler=StandardScaler().fit(np.concatenate([G[i]['x'] for i in fit]))
        net=Explorer(G[0]['x'].shape[1],G[0]['edge'].shape[1],nclass,hidden=hidden,depth=depth,variant=variant)
        opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-4)
        counts=np.bincount(y[fit],minlength=nclass);cw=torch.tensor(np.sqrt(counts.sum()/np.maximum(1,counts)),dtype=torch.float);cw/=cw.mean()
        best=np.inf;state=None;stale=0;bestep=0
        for ep in range(epochs):
            net.train()
            for ids in chunks(G,fit,node_budget,seed=seed*1000+fold*97+ep):
                inp=make_batch(G,ids,scaler);z=net(inp,mode)
                loss=F.cross_entropy(z,torch.tensor(y[ids]),weight=cw)
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
            net.eval();vl=0.;total=0
            with torch.no_grad():
                for ids in chunks(G,val,node_budget):
                    z=net(make_batch(G,ids,scaler),mode)
                    vl+=float(F.cross_entropy(z,torch.tensor(y[ids]),weight=cw))*len(ids);total+=len(ids)
            vl/=max(1,total)
            if vl<best-1e-4:best=vl;bestep=ep+1;stale=0;state=copy.deepcopy(net.state_dict())
            else:stale+=1
            if stale>=patience and ep>=10:break
        net.load_state_dict(state);net.eval()
        with torch.no_grad():
            for ids in chunks(G,ite,node_budget):
                b=make_batch(G,ids,scaler)
                if mode=='shuffle': prob=torch.stack([net(b,mode).softmax(-1) for _ in range(5)]).mean(0).numpy()
                else:prob=net(b,mode).softmax(-1).numpy()
                probs[ids]=prob;pred[ids]=prob.argmax(-1);ofold[ids]=fold
        ba=float(balanced_accuracy_score(y[ite],pred[ite]));rows.append(dict(fold=fold,balanced_accuracy=ba,best_epoch=bestep,n_test=len(ite)))
        print('FOLD',variant,mode,seed,fold,round(100*ba,2),'epoch',bestep,flush=True)
    assert np.all(pred>=0)
    return dict(accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),pred=pred,probs=probs,ofold=ofold,folds=rows,seconds=time.time()-t0,nparams=sum(p.numel() for p in net.parameters()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['NCI1','COX2','PTC_MR']);p.add_argument('--variants',nargs='+',default=['context','confidence','spectral','multitau']);p.add_argument('--modes',nargs='+',default=['learned']);p.add_argument('--seeds',nargs='+',type=int,default=[101]);p.add_argument('--data_root',default='/mnt/data/mvf_work/datasets');p.add_argument('--out',default='/mnt/data/mvf_hypothesis_search/results');p.add_argument('--split_seed',type=int,default=20261020);p.add_argument('--epochs',type=int,default=25);p.add_argument('--node_budget',type=int,default=1600);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/'protocol.json').write_text(json.dumps(vars(a),indent=2))
    for dataset in a.datasets:
        graphs,y=load_tu_zip(str(Path(a.data_root)/(dataset+'.zip')),dataset);G=[preprocess_graph(g) for g in graphs]
        for variant in a.variants:
            print('PREFLIGHT',dataset,variant,preflight(G,int(np.max(y))+1,variant),flush=True)
            for seed in a.seeds:
                for mode in a.modes:
                    tag=f'{dataset}_{variant}_{mode}_s{seed}'
                    if (out/(tag+'.json')).exists():continue
                    r=train(G,y,variant,mode,seed=seed,split_seed=a.split_seed,epochs=a.epochs,node_budget=a.node_budget)
                    np.savez_compressed(out/(tag+'.npz'),y=y,pred=r['pred'],probs=r['probs'],fold=r['ofold'],id=np.array([g['id'] for g in G]))
                    d={k:v for k,v in r.items() if k not in ('pred','probs','ofold')};d.update(dataset=dataset,variant=variant,mode=mode,seed=seed,split_seed=a.split_seed,epochs=a.epochs)
                    (out/(tag+'.json')).write_text(json.dumps(d,indent=2))
                    print('RESULT',dataset,variant,mode,seed,round(r['balanced_accuracy']*100,2),'time',round(r['seconds'],1),flush=True)

if __name__=='__main__': main()
