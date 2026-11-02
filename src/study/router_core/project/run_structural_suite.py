"""Graph incidence routing and multivector-learning experiments."""
import sys,os,copy,json,time,argparse,random,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'source'))
import numpy as np
import networkx as nx
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit,StratifiedGroupKFold
from budgeted_mvf_gine import (preprocess_graph,load_tu_zip,make_batch,scatter_sum,scatter_mean,chunks,setup_seed,ExactEquationGINE)
from incidence_competitive_v6 import IncidenceRouter
from hypothesis_router import Explorer

torch.set_num_threads(1)
ROOT=Path(os.environ.get('MVF_ROUTER_ROOT','/tmp/mvf_router_experiments'));ROOT.mkdir(parents=True,exist_ok=True)


def feature_graph(n,pairs,x,edge=None,gid=0,truth=None):
    pairs=np.array(sorted(tuple(sorted(e)) for e in pairs),np.int64).reshape(-1,2)
    g=nx.Graph();g.add_nodes_from(range(n));g.add_edges_from(pairs.tolist())
    deg=np.asarray([d for _,d in g.degree()],np.float32)
    tri=np.asarray([len(set(g[u]).intersection(g[v])) for u,v in pairs],np.float32)
    br=set(tuple(sorted(p)) for p in nx.bridges(g));bridge=np.asarray([int(tuple(p) in br) for p in pairs],np.float32)
    tri=np.log1p(tri);tri=np.clip(tri/2,0,2)
    local=np.array([nx.clustering(g,u) for u in range(n)],np.float32)
    bridge_inc=np.zeros(n,np.float32)
    for (u,v),b in zip(pairs,bridge):bridge_inc[u]+=b;bridge_inc[v]+=b
    bridge_inc/=np.maximum(deg,1)
    x=np.asarray(x,np.float32)
    # Shared structural inputs for every baseline and routing model.
    xx=np.column_stack([x, np.log1p(deg),deg/max(1,float(deg.max())),local,bridge_inc,np.ones(n)]).astype(np.float32)
    edge=np.zeros((len(pairs),1),np.float32) if edge is None else np.asarray(edge,np.float32).reshape(len(pairs),-1)
    ee=np.column_stack([edge,tri,bridge,abs(deg[pairs[:,0]]-deg[pairs[:,1]])/max(1,float(deg.max())),np.ones(len(pairs))]).astype(np.float32)
    ans=dict(n=n,pairs=pairs,x=xx,edge=ee,id=int(gid))
    if truth is not None:ans['truth']=np.asarray(truth,np.int64)
    ans['graph_stats']={'n':int(n),'m':int(len(pairs)),'mean_deg':float(deg.mean()),'bridge_rate':float(bridge.mean()),'triangle_edge_rate':float((tri>0).mean())}
    return ans


def synthetic(n_each=80,n=24,degree=4,seed=30077,signal=.10):
    """Assortative vs cross-linked graphs by degree-preserving edge swaps.
    Same node count, degree, and vertex-feature marginal for both classes.
    Ground-truth relevant edges connect vertex colors.
    """
    rng=np.random.default_rng(seed);graphs=[];y=[]
    assert (n%2==0 and (n//2)*degree%2==0 and n//2>degree)
    for label in [0,1]:
      for rep in range(n_each):
        a=nx.random_regular_graph(degree,n//2,seed=int(rng.integers(1,2**30)))
        b=nx.random_regular_graph(degree,n//2,seed=int(rng.integers(1,2**30)))
        gg=nx.disjoint_union(a,b)
        half=n//2
        # Swap disjoint intra-community links into cross-community links.
        # Two-edge switch preserves every vertex degree and graph size.
        target=2 if label==0 else int(.42*n*degree/2)
        for _ in range(target//2):
          edges0=[(u,v) for u,v in gg.edges if u<half and v<half]
          edges1=[(u,v) for u,v in gg.edges if u>=half and v>=half]
          if not edges0 or not edges1:break
          u,v=edges0[int(rng.integers(len(edges0)))]; w,z=edges1[int(rng.integers(len(edges1)))]
          gg.remove_edges_from([(u,v),(w,z)])
          gg.add_edges_from([(u,w),(v,z)])
        # Random vertex permutation eliminates absolute vertex index and group-order shortcuts.
        perm=rng.permutation(n);gg=nx.relabel_nodes(gg,{i:int(perm[i]) for i in range(n)})
        latent=np.zeros(n,np.int64);latent[perm[half:]]=1
        # Identical marginal distribution of features for both labels, noisy community color.
        col=(2*latent-1).astype(np.float32)+rng.normal(0,signal,n)
        xx=np.column_stack([col,rng.normal(0,1,n),rng.normal(0,1,n)])
        pairs=np.array(sorted(tuple(sorted(e)) for e in gg.edges),np.int64)
        truth=(latent[pairs[:,0]]!=latent[pairs[:,1]]).astype(np.int64)
        graphs.append(feature_graph(n,pairs,xx,gid=len(graphs),truth=truth));y.append(label)
    return graphs,np.asarray(y,np.int64)


def synthetic_paired(n_each=90,n=28,degree=4,seed=30443,signal=1.2):
    """Paired-label experiment: the graph topology is exactly the same for each
    class pair, but the balanced vertex-color partition is aligned vs random.
    Node color counts, edge counts, degree sequence and all topology-derived
    features are matched *within every pair*; label is an edge-alignment property.
    """
    rng=np.random.default_rng(seed);Gs=[];ys=[]
    half=n//2
    for rep in range(n_each):
        aa=nx.random_regular_graph(degree,half,seed=int(rng.integers(1,2**30)))
        bb=nx.random_regular_graph(degree,half,seed=int(rng.integers(1,2**30)))
        g=nx.disjoint_union(aa,bb)
        e0=[(u,v) for u,v in g.edges if u<half and v<half]
        e1=[(u,v) for u,v in g.edges if u>=half and v>=half]
        u,v=e0[int(rng.integers(len(e0)))];w,z=e1[int(rng.integers(len(e1)))]
        g.remove_edges_from([(u,v),(w,z)]);g.add_edges_from([(u,w),(v,z)])
        perm=rng.permutation(n);g=nx.relabel_nodes(g,{i:int(perm[i]) for i in range(n)})
        true=np.zeros(n,np.int64);true[perm[half:]]=1
        # label 1 has high color-assortativity and label 0 has random colors
        random_colors=np.zeros(n,np.int64);random_colors[rng.choice(n,size=half,replace=False)]=1
        background=rng.normal(0,1,(n,2))
        ps=np.array(sorted(tuple(sorted(e)) for e in g.edges),np.int64)
        for label,colors in [(1,true),(0,random_colors)]:
            color=(2*colors-1).astype(np.float32)+rng.normal(0,signal,n)
            features=np.column_stack([color,background])
            truth=(colors[ps[:,0]]!=colors[ps[:,1]]).astype(np.int64)
            Gs.append(feature_graph(n,ps,features,gid=len(Gs),truth=truth));ys.append(label)
    return Gs,np.asarray(ys,np.int64)

def regular_blind_pair(n=24):
    # 1-WL-indistinguishable graphs: single n-cycle versus two n/2 cycles.
    graphs=[]
    for i,gg in enumerate([nx.cycle_graph(n),nx.disjoint_union(nx.cycle_graph(n//2),nx.cycle_graph(n//2))]):
      gg=nx.convert_node_labels_to_integers(gg)
      graphs.append(feature_graph(n,list(gg.edges),np.ones((n,3),np.float32),gid=i))
    return graphs

class PlainGINE(nn.Module):
    def __init__(self,dx,de,nclass,h=32,depth=2,drop=.15):
        super().__init__();self.h=h
        self.node=nn.Sequential(nn.Linear(dx,h),nn.ReLU(),nn.LayerNorm(h))
        self.edge=nn.Sequential(nn.Linear(de,h),nn.ReLU(),nn.Linear(h,h))
        self.layers=nn.ModuleList([ExactEquationGINE(h,drop) for _ in range(depth)])
        self.raw=nn.Sequential(nn.Linear(2*dx+2*de+2,h),nn.ReLU(),nn.Linear(h,h))
        self.head=nn.Sequential(nn.Linear(2*depth*h+h,2*h),nn.ReLU(),nn.Dropout(drop),nn.Linear(2*h,nclass))
    def forward(self,b,mode=None):
        h=self.node(b['x']);e=self.edge(b['edge']);outs=[]
        for layer in self.layers:
            h=layer(h,b,e)
            s=scatter_sum(h,b['ng'],b['ngraphs']);ct=torch.bincount(b['ng'],minlength=b['ngraphs']).to(h.dtype).clamp(min=1)
            outs.extend([s/ct[:,None],s/ct.sqrt()[:,None]])
        xmean=scatter_mean(b['x'],b['ng'],b['ngraphs']);xstd=(scatter_mean(b['x']**2,b['ng'],b['ngraphs'])-xmean.square()).clamp(min=1e-8).sqrt()
        if len(b['eg']):
            emean=scatter_mean(b['edge'],b['eg'],b['ngraphs']);estd=(scatter_mean(b['edge']**2,b['eg'],b['ngraphs'])-emean.square()).clamp(min=1e-8).sqrt()
        else:
            emean=h.new_zeros((b['ngraphs'],b['edge'].shape[-1]));estd=emean
        nv=torch.bincount(b['ng'],minlength=b['ngraphs']).to(h.dtype);ne=torch.bincount(b['eg'],minlength=b['ngraphs']).to(h.dtype)
        raw=self.raw(torch.cat([xmean,xstd,emean,estd,torch.log1p(nv)[:,None],torch.log1p(ne)[:,None]],-1))
        return self.head(torch.cat(outs+[raw],-1))

class StructuralBlock(nn.Module):
    """Two competiting scales with unsupervised structural priors on incidences.
    Every competitor gets IDENTICAL edge/node motif inputs; no oracle edge truth.
    """
    def __init__(self,h,drop=.15):
        super().__init__();self.eps=nn.Parameter(torch.zeros(1));self.amp=nn.Parameter(torch.tensor(1.0))
        self.scorer=nn.Sequential(nn.Linear(3*h+2,h),nn.ReLU(),nn.Linear(h,1))
        self.struct_bias=nn.Linear(2,1,bias=False) # triangle and bridge local signals
        nn.init.zeros_(self.struct_bias.weight)
        self.temp_cal=nn.Linear(3,1) # log degree, triangle rate and bridge rate
        nn.init.zeros_(self.temp_cal.weight);nn.init.zeros_(self.temp_cal.bias)
        self.nn=nn.Sequential(nn.Linear(3*h,2*h),nn.ReLU(),nn.Linear(2*h,h))
        self.bn=nn.BatchNorm1d(h);self.drop=nn.Dropout(drop)
    def getw(self,h,b,e,mode):
        p=b['pairs'];n=len(h);m=len(p);a=p[:,0];v=p[:,1]
        deg=torch.bincount(p.flatten(),minlength=n).to(h.dtype)
        tri=b['edge'][:,-4];bridge=b['edge'][:,-3] # as in feature_graph
        prior=self.struct_bias(torch.stack([tri,bridge],-1)).flatten()
        def logit(u,v):
            meta=torch.stack([torch.log1p(deg[u]),torch.log1p(deg[v])],-1)
            return self.scorer(torch.cat([h[u],h[v],e,meta],-1)).flatten()+prior
        logits=torch.cat([logit(a,v),logit(v,a)]);ids=torch.cat([a,v])
        triangle_rate=torch.zeros(n,device=h.device,dtype=h.dtype);bridge_rate=torch.zeros_like(triangle_rate)
        triangle_rate.index_add_(0,a,tri);triangle_rate.index_add_(0,v,tri)
        bridge_rate.index_add_(0,a,bridge);bridge_rate.index_add_(0,v,bridge)
        triangle_rate/=deg.clamp(min=1);bridge_rate/=deg.clamp(min=1)
        local=torch.stack([torch.log1p(deg),triangle_rate,bridge_rate],-1)
        tau_scale=.6+1.2*torch.sigmoid(self.temp_cal(local)).flatten()
        def weights(tau):
            z=logits/(tau*tau_scale[ids]);mx=z.new_full((n,),-1e9);mx.scatter_reduce_(0,ids,z,reduce='amax',include_self=True)
            e0=torch.exp(z-mx[ids]);denom=z.new_zeros(n).index_add_(0,ids,e0)
            q=1-torch.exp(-deg[ids]*e0/denom[ids].clamp(min=1e-8))
            return 2.5*q[:m]*q[m:]
        w1=weights(.65);w2=weights(1.5)
        if mode=='uniform':w1=torch.ones_like(w1)*.85;w2=w1
        elif mode=='shuffle':
            # Graph-local shuffling; correct mapping preserves marginal weights.
            perm=torch.empty(m,dtype=torch.long,device=h.device)
            for i in range(b['ngraphs']):
                loc=(b['eg']==i).nonzero().flatten();perm[loc]=loc[torch.randperm(len(loc),device=h.device)]
            w1=w1[perm];w2=w2[perm]
        elif mode!='learned':raise ValueError(mode)
        return w1[b['arc_edge_id']],w2[b['arc_edge_id']],deg
    def forward(self,h,b,e,mode):
        n=len(h);msg=F.relu(h[b['src']]+e[b['arc_edge_id']]);agg=scatter_sum(msg,b['dst'],n)
        w1,w2,deg=self.getw(h,b,e,mode)
        d1=scatter_sum((w1-1.)[:,None]*msg,b['dst'],n);d2=scatter_sum((w2-1.)[:,None]*msg,b['dst'],n)
        z=torch.cat([(1+self.eps)*h+agg,self.amp*d1/deg.clamp(min=1).sqrt()[:,None],self.amp*d2/deg.clamp(min=1).sqrt()[:,None]],-1)
        return F.relu(h+self.drop(self.bn(self.nn(z))))

class StructuralRouter(PlainGINE):
    def __init__(self,dx,de,nclass,h=32,depth=2,drop=.15):
        super().__init__(dx,de,nclass,h,depth,drop)
        self.layers=nn.ModuleList([StructuralBlock(h,drop) for _ in range(depth)])
    def forward(self,b,mode='learned'):
        h=self.node(b['x']);e=self.edge(b['edge']);outs=[]
        for layer in self.layers:
            h=layer(h,b,e,mode)
            s=scatter_sum(h,b['ng'],b['ngraphs']);ct=torch.bincount(b['ng'],minlength=b['ngraphs']).to(h.dtype).clamp(min=1)
            outs.extend([s/ct[:,None],s/ct.sqrt()[:,None]])
        xmean=scatter_mean(b['x'],b['ng'],b['ngraphs']);xstd=(scatter_mean(b['x']**2,b['ng'],b['ngraphs'])-xmean.square()).clamp(min=1e-8).sqrt()
        emean=scatter_mean(b['edge'],b['eg'],b['ngraphs']);estd=(scatter_mean(b['edge']**2,b['eg'],b['ngraphs'])-emean.square()).clamp(min=1e-8).sqrt()
        nv=torch.bincount(b['ng'],minlength=b['ngraphs']).to(h.dtype);ne=torch.bincount(b['eg'],minlength=b['ngraphs']).to(h.dtype)
        raw=self.raw(torch.cat([xmean,xstd,emean,estd,torch.log1p(nv)[:,None],torch.log1p(ne)[:,None]],-1))
        return self.head(torch.cat(outs+[raw],-1))


def make_model(typ,dx,de,nclass,h):
    if typ in ('gine32','gine39','gine41'):
        return PlainGINE(dx,de,nclass,{'gine32':32,'gine39':39,'gine41':41}[typ])
    if typ in ('struct_learned','struct_uniform','struct_shuffle'):
        return StructuralRouter(dx,de,nclass,h=h)
    if typ in ('old_learned',):
        return IncidenceRouter(dx,de,nclass,hidden=h,kind='dual')
    if typ in ('multitau','multitau_uniform','multitau_shuffle'):
        return Explorer(dx,de,nclass,hidden=h,variant='multitau')
    raise ValueError(typ)


def gradient_and_invariance_check(G,y):
    scaler=StandardScaler().fit(np.concatenate([x['x'] for x in G[:4]]));b=make_batch(G,np.arange(4),scaler)
    nclass=int(max(y))+1;model=make_model('struct_learned',G[0]['x'].shape[1],G[0]['edge'].shape[1],nclass,20)
    model.eval();mode='learned';logits=model(b,mode);logits.square().sum().backward()
    scoregrad=float(model.layers[0].scorer[-1].weight.grad.abs().max())
    structgrad=float(model.layers[0].struct_bias.weight.grad.abs().max())
    assert scoregrad>1e-9 and structgrad>1e-9,(scoregrad,structgrad)
    with torch.no_grad():
      orig=model(b,mode);alter=copy.deepcopy(G[:4]);
      for g in alter:
        pp=np.random.default_rng(44).permutation(len(g['pairs']))
        g['pairs']=g['pairs'][pp][:,::-1].copy();g['edge']=g['edge'][pp].copy()
      b2=make_batch(alter,np.arange(4),scaler)
      err=float((orig-model(b2,mode)).abs().max())
    assert err<1e-4,err
    return {'scorer_gradient_max':scoregrad,'structural_prior_gradient_max':structgrad,'edge_order_invariance_max_abs_logit':err}


def test_regular_indistinguishability():
    G=regular_blind_pair();scaler=StandardScaler().fit(np.concatenate([x['x'] for x in G]));b=make_batch(G,[0,1],scaler)
    results={}
    with torch.no_grad():
      for typ in ('gine32','struct_learned','struct_uniform'):
        setup_seed(4242);model=make_model(typ,G[0]['x'].shape[1],G[0]['edge'].shape[1],2,32);model.eval()
        mode='learned' if typ=='struct_learned' else 'uniform' if typ=='struct_uniform' else None
        o=model(b,mode);results[typ]=float((o[0]-o[1]).abs().max())
    return results


def run_cv(G,y,dataset,typ,seed=101,split_seed=20261051,epochs=18,folds=3,budget=1350,patience=5,h=32,out=None):
    y=np.asarray(y,np.int64);nclass=int(y.max())+1;pred=np.full(len(y),-1,np.int64);probs=np.zeros((len(y),nclass),np.float32)
    ofold=np.full(len(y),-1,np.int64);logs=[];t0=time.time()
    if typ=='struct_learned':mode='learned'
    elif typ=='struct_uniform':mode='uniform'
    elif typ=='struct_shuffle':mode='shuffle'
    elif typ=='old_learned':mode='sym'
    elif typ=='multitau':mode='learned'
    elif typ=='multitau_uniform':mode='uniform'
    elif typ=='multitau_shuffle':mode='shuffle'
    else:mode=None
    groups=(np.arange(len(y))//2 if dataset=='SYNPAIR' else None)
    if groups is None: split_iter=StratifiedKFold(n_splits=folds,shuffle=True,random_state=split_seed).split(np.arange(len(y)),y)
    else:split_iter=StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=split_seed).split(np.arange(len(y)),y,groups)
    for fold,(it,te) in enumerate(split_iter):
        setup_seed(seed+fold*101)
        if groups is None:
            inner=StratifiedShuffleSplit(n_splits=1,test_size=.18,random_state=seed+fold*31)
            ia,iv=next(inner.split(it,y[it]))
        else:
            inner=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed+fold*31)
            ia,iv=next(inner.split(it,y[it],groups[it]))
            assert not set(groups[it[ia]]).intersection(groups[it[iv]])
            assert not set(groups[it]).intersection(groups[te])
        fit=it[ia];val=it[iv]
        sc=StandardScaler().fit(np.concatenate([G[i]['x'] for i in fit]))
        model=make_model(typ,G[0]['x'].shape[1],G[0]['edge'].shape[1],nclass,h)
        opt=torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=1e-4)
        counts=np.bincount(y[fit],minlength=nclass);cw=torch.tensor(np.sqrt(counts.sum()/np.maximum(1,counts)),dtype=torch.float);cw/=cw.mean()
        best=np.inf;state=None;stale=0;bestep=0
        for ep in range(epochs):
            model.train()
            for ids in chunks(G,fit,budget,seed=seed*1000+fold*97+ep):
                x=make_batch(G,ids,sc);zz=model(x,mode)
                loss=F.cross_entropy(zz,torch.tensor(y[ids]),weight=cw)
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
            model.eval();vl=0.;total=0
            with torch.no_grad():
                for ids in chunks(G,val,budget):
                    xx=make_batch(G,ids,sc)
                    # averaged randomized inference is used for shuffle controls
                    prob=torch.stack([model(xx,mode).softmax(-1) for _ in range(3)]).mean(0) if mode=='shuffle' else model(xx,mode).softmax(-1)
                    vl+=float(F.nll_loss(prob.clamp(min=1e-9).log(),torch.tensor(y[ids]),weight=cw))*len(ids);total+=len(ids)
            vl/=max(1,total)
            if vl<best-1e-4:best=vl;stale=0;bestep=ep+1;state=copy.deepcopy(model.state_dict())
            else:stale+=1
            if stale>=patience and ep>=9:break
        model.load_state_dict(state);model.eval()
        with torch.no_grad():
            for ids in chunks(G,te,budget):
                xx=make_batch(G,ids,sc)
                pr=(torch.stack([model(xx,mode).softmax(-1) for _ in range(5)]).mean(0) if mode=='shuffle' else model(xx,mode).softmax(-1)).numpy()
                probs[ids]=pr;pred[ids]=pr.argmax(-1);ofold[ids]=fold
        a=float(balanced_accuracy_score(y[te],pred[te]));logs.append({'fold':fold,'ba':a,'epoch':bestep,'test_n':len(te)});
        print('FOLD',dataset,typ,seed,fold,round(100*a,2),'bestepoch',bestep,flush=True)
    elapsed=time.time()-t0
    ba=float(balanced_accuracy_score(y,pred));params=sum(p.numel() for p in model.parameters());active=params
    if typ=='struct_uniform':active-=sum(p.numel() for layer in model.layers for mod in (layer.scorer,layer.struct_bias,layer.temp_cal) for p in mod.parameters())
    res={'dataset':dataset,'model':typ,'seed':int(seed),'split_seed':split_seed,'ba':ba,'nparams_total':params,'nparams_active_approx':active,'epochs':epochs,'folds':logs,'seconds':elapsed,'n_graphs':len(y)}
    if out:
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        (out/f'{dataset}_{typ}_s{seed}.json').write_text(json.dumps(res,indent=2))
        np.savez_compressed(out/f'{dataset}_{typ}_s{seed}.npz',label=y,pred=pred,prob=probs,fold=ofold)
    print('COMPLETE',dataset,typ,seed,round(100*ba,2),'secs',round(elapsed,1),flush=True)
    return res


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['SYN24','SYN48','SYNPAIR','COX2','MUTAG','NCI1','PTC_MR','ENZYMES','PROTEINS','DD'],default='SYN24')
    p.add_argument('--models',nargs='+',default=['gine32','gine39','struct_uniform','struct_shuffle','struct_learned'])
    p.add_argument('--seeds',nargs='+',type=int,default=[101,202,303])
    p.add_argument('--epochs',type=int,default=18);p.add_argument('--n_each',type=int,default=80);p.add_argument('--noise',type=float,default=.10);p.add_argument('--pair_n',type=int,default=28);p.add_argument('--pair_degree',type=int,default=4);p.add_argument('--pair_seed',type=int,default=30443);p.add_argument('--split_seed',type=int,default=20261051);p.add_argument('--budget',type=int,default=1350)
    a=p.parse_args();print('CONFIG',vars(a),flush=True)
    if a.dataset=='SYNPAIR':
        G,y=synthetic_paired(n_each=a.n_each,n=a.pair_n,degree=a.pair_degree,seed=a.pair_seed,signal=a.noise)
    elif a.dataset.startswith('SYN'):
        n,degree=(24,4) if a.dataset=='SYN24' else (48,6)
        G,y=synthetic(n_each=a.n_each,n=n,degree=degree,seed=31001 if n==24 else 31071,signal=a.noise)
    else:
        filename={'COX2':'COX2(1)','MUTAG':'MUTAG (1)(1)','NCI1':'NCI1(1)','PTC_MR':'PTC_MR(1)','ENZYMES':'ENZYMES(2)','PROTEINS':'PROTEINS(2)','DD':'DD (1)'}[a.dataset]
        raw,y=load_tu_zip(str(Path(os.environ.get('research study_DATA_ROOT','/mnt/data'))/(filename+'.zip')),a.dataset)
        G=[]
        for i,g in enumerate(raw):
            p=np.asarray(g['edges'],np.int64).reshape(-1,2)
            x=np.asarray(g['node'],np.float32)
            ed=np.asarray(g['edge_attr'],np.float32).reshape(len(p),-1)
            G.append(feature_graph(g['n'],p,x,edge=ed,gid=g['original_id']))
    checks=gradient_and_invariance_check(G,y);blind=test_regular_indistinguishability()
    (ROOT/'mechanism_checks.json').write_text(json.dumps(dict(checks=checks,blind=blind),indent=2))
    print('CHECKS',checks,'REGULAR_BLIND',blind,flush=True)
    output=(ROOT/f'results_{a.dataset}_GROUPED_n{a.pair_n}_d{a.pair_degree}_generator{a.pair_seed}_noise{a.noise:.2f}' if a.dataset=='SYNPAIR' and (a.pair_n!=28 or a.pair_seed!=30443) else ROOT/f'results_{a.dataset}{"_GROUPED" if a.dataset=="SYNPAIR" else ""}_noise{a.noise:.2f}' if a.dataset.startswith('SYN') else ROOT/'results');(ROOT/'last_config.json').write_text(json.dumps(vars(a),indent=2))
    for typ in a.models:
        for seed in a.seeds:
            fn=output/f'{a.dataset}_{typ}_s{seed}.json'
            if fn.exists(): print('SKIP_EXISTING',fn,flush=True);continue
            run_cv(G,y,a.dataset,typ,seed=seed,split_seed=a.split_seed,epochs=a.epochs,budget=a.budget,out=output)

if __name__=='__main__': main()
