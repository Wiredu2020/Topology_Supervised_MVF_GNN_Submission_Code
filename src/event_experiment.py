"""Edge-event-supervised incidence and generalized multivector feedback experiments.

This module evaluates whether supervising global bridge-versus-cycle-support events
improves the learned relation between vertex and edge cells. Ground-truth bridge
indicators are used only for training loss and held-out diagnostics, never as
inference features. Hard multivector membership is stop-gradient; continuous cell
weights and the supervised incidence probabilities remain differentiable.
"""
from __future__ import annotations
import os, sys, json, time, copy, zipfile, hashlib, argparse, random, math
from pathlib import Path
import numpy as np
import networkx as nx
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'study'/'cycle_poset'),str(ROOT/'study'/'block_feedback')]
from run import CyclePosetFeedback, CyclePosetRouter, cycle_graph_features, make_model as old_make_model
from block_feedback_experiment import (feature_graph, make_batch, chunks, setup_seed, load_tu_zip,
                                       StructuralBlock, PlainGINE, scatter_mean, scatter_sum)
torch.set_num_threads(max(1,int(os.environ.get('MVF_THREADS','1'))))

DATASETS=('BZR','DHFR','COX2','ENZYMES','PROTEINS','NCI109','PTC_FM','PTC_MM','PTC_FR')
MODES=('gine','gine40','gine_aux','old_poset','event','event_noaux','event_shuffle','event_uniform','event_softonly')


def locate_zip(name, allow_download=False):
    p=ROOT/'data'/f'{name}.zip'
    if p.is_file():return p
    if allow_download:
        import urllib.request
        p.parent.mkdir(parents=True,exist_ok=True)
        url=f'https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip'
        urllib.request.urlretrieve(url,str(p))
        if zipfile.is_zipfile(p):return p
        p.unlink(missing_ok=True)
    raise FileNotFoundError(f'{name}.zip not packaged. Provide the corresponding TU dataset archive or enable automatic retrieval.')


def load_graphs(name,allow_download=False):
    """One undirected simple edge per pair; preserve categorical/continuous edge attributes."""
    raw,y=load_tu_zip(str(locate_zip(name,allow_download)),name)
    out=[];realigned=0
    for g in raw:
        p=np.asarray(g['edges'],dtype=np.int64).reshape(-1,2)
        edge=np.asarray(g['edge_attr'],dtype=np.float32).reshape(len(p),-1)
        can=np.sort(p,axis=1)
        order=np.lexsort((can[:,1],can[:,0])) if len(p) else np.empty(0,int)
        realigned+=int(not np.array_equal(order,np.arange(len(p))))
        p=can[order];edge=edge[order]
        if len(np.unique(p,axis=0))!=len(p):raise ValueError('Duplicate undirected edges in data')
        n=int(g['n']); graph=nx.Graph();graph.add_nodes_from(range(n));graph.add_edges_from(p.tolist())
        br=set(tuple(sorted(uv)) for uv in nx.bridges(graph))
        target=np.asarray([int(tuple(uv) in br) for uv in p],dtype=np.int64)
        # Graph-derived closed-walk inputs are shared by every model.
        nf,ef=cycle_graph_features(n,p)
        x=np.column_stack((np.asarray(g['node'],np.float32),nf)).astype(np.float32)
        e=np.column_stack((edge,ef)).astype(np.float32)
        z=feature_graph(n,p,x,edge=e,gid=int(g['original_id']))
        # No oracle bridge labels or bridge fractions in model inputs.
        # feature_graph appends [..., bridge_incidence_rate, constant] to nodes,
        # and [..., triangle, edge_bridge_truth, degree_delta, constant] to edges.
        z['x'][:,-2]=0.
        z['edge'][:,-3]=0.
        z['bridge_target']=target
        assert np.array_equal(z['pairs'],p)
        out.append(z)
    print('DATASET',name,'graphs',len(out),'reordered_graphs',realigned,
          'bridge_fraction',round(float(np.mean(np.concatenate([g['bridge_target'] for g in out if len(g['bridge_target'])]))),4),flush=True)
    return out,np.asarray(y,int)


def batch_of(G,ids,scaler):
    b=make_batch(G,ids,scaler)
    b['bridge_target']=torch.cat([torch.tensor(G[int(i)]['bridge_target'],dtype=torch.long) for i in ids])
    return b


def shuffled_probs(p,b):
    """Fixed per-graph score misalignment; independent of class and graph training split."""
    out=p.clone()
    for gi,gid in enumerate(b['ids']):
        loc=torch.nonzero(b['eg']==gi).flatten()
        if len(loc)>1:
            # Graph IDs are canonical TU graph IDs; the edge list is already canonicalized.
            seed=(int(gid)*1103515245+9131)&0xFFFFFFFF
            perm=torch.as_tensor(np.random.default_rng(seed).permutation(len(loc)),dtype=torch.long,device=p.device)
            out[loc]=p[loc[perm]]
    return out


class EdgeEventBlock(CyclePosetFeedback):
    """Two event-conditioned soft transports and two hard generalized MVFs.

    Event type 1: bridge edge (deletion increases component count).
    Event type 0: nonbridge edge (participates in an undirected cycle).
    Each induced incidence-connected cell block is order convex in the graph face poset.
    """
    def __init__(self,h,kind='learned',hard=True,soft=True):
        super().__init__(h,kind='learned',topk=1,hard=hard,soft=soft)
        self.requested_kind=kind
        self.event_head=nn.Sequential(nn.Linear(3*h,h),nn.ReLU(),nn.Linear(h,2))
        self.bridge_cycle_mix=nn.Sequential(nn.Linear(2*h,h),nn.ReLU(),nn.Linear(h,h))
        self.hard_pair_mix=nn.Sequential(nn.Linear(2*h,h),nn.ReLU(),nn.Linear(h,h))
        self.last_event_logits=None
        self.last_probs=None
        self.override=None

    def forward(self,h,b,e,mode='learned'):
        base=StructuralBlock.forward(self,h,b,e,'learned')
        p=b['pairs'];u=p[:,0];v=p[:,1];n=len(base);m=len(p)
        ctx=torch.cat((base[u]+base[v],torch.abs(base[u]-base[v]),e),dim=-1)
        logits=self.event_head(ctx)
        self.last_event_logits=logits
        learned=torch.softmax(logits,dim=-1)
        self.last_probs=learned
        routing=self.override or self.requested_kind
        if routing=='shuffled':probs=shuffled_probs(learned,b)
        elif routing=='uniform':probs=torch.full_like(learned,0.5)
        else:probs=learned
        # Directed node-to-edge membership affinities, differentiated end-to-end.
        a=torch.sigmoid(self.inc_gate(torch.cat([base[u],e],dim=-1)).flatten())
        c=torch.sigmoid(self.inc_gate(torch.cat([base[v],e],dim=-1)).flatten())
        def propagate(s):
            aa=a*s;cc=c*s
            ev=(aa[:,None]*base[u]+cc[:,None]*base[v])/(aa+cc).clamp(min=1e-7)[:,None]
            ev=self.edge_update(torch.cat([e,ev],dim=-1))
            out=base.new_zeros(n,base.shape[1]);tot=base.new_zeros(n)
            out.index_add_(0,u,aa[:,None]*ev);out.index_add_(0,v,cc[:,None]*ev)
            tot.index_add_(0,u,aa);tot.index_add_(0,v,cc)
            return out/tot.clamp(min=1e-7)[:,None]
        delta=base.new_zeros(base.shape)
        if self.soft_enabled:
            cycle=propagate(probs[:,0]);bridge=propagate(probs[:,1])
            delta=delta+torch.sigmoid(self.soft_alpha)*self.soft_ln(self.bridge_cycle_mix(torch.cat([cycle,bridge],dim=-1)))
        if self.hard_enabled:
            # The selection is discrete and not differentiated through; the
            # weighted cell embeddings inside field_pool have score gradients.
            cycle_blocks,_=self.field_pool(base,b,e,probs[:,0][b['arc_edge_id']])
            bridge_blocks,_=self.field_pool(base,b,e,probs[:,1][b['arc_edge_id']])
            pooled=self.hard_pair_mix(torch.cat([cycle_blocks,bridge_blocks],dim=-1))
            delta=delta+torch.sigmoid(self.hard_alpha)*self.feedback_norm(pooled)
        return F.relu(base+delta)


class EventMVFRouter(CyclePosetRouter):
    def __init__(self,dx,de,nclasses,h=24,kind='learned',hard=True,soft=True):
        super().__init__(dx,de,nclasses,h=h,kind='learned',hard=hard,soft=soft)
        self.layers=nn.ModuleList([EdgeEventBlock(h,kind,hard=hard,soft=soft) for _ in range(2)])

    def event_logits(self):
        return torch.stack([layer.last_event_logits for layer in self.layers]).mean(0)

    def set_override(self,mode=None):
        for layer in self.layers:layer.override=mode


class GINEWithEvent(PlainGINE):
    """Auxiliary-edge supervised GINE control with the same bridge targets."""
    def __init__(self,dx,de,nclasses,h=36):
        super().__init__(dx,de,nclasses,h=h,depth=2)
        self.event_head=nn.Sequential(nn.Linear(3*h,h),nn.ReLU(),nn.Linear(h,2))
        self._event_logits=None

    def forward(self,b,mode=None):
        h=self.node(b['x']);e=self.edge(b['edge']);outs=[]
        for layer in self.layers:
            h=layer(h,b,e)
            sums=scatter_sum(h,b['ng'],b['ngraphs'])
            ct=torch.bincount(b['ng'],minlength=b['ngraphs']).to(h.dtype).clamp(min=1)
            outs.extend([sums/ct[:,None],sums/ct.sqrt()[:,None]])
        p=b['pairs'];u=p[:,0];v=p[:,1]
        self._event_logits=self.event_head(torch.cat([h[u]+h[v],abs(h[u]-h[v]),e],dim=-1))
        xm=scatter_mean(b['x'],b['ng'],b['ngraphs'])
        xs=(scatter_mean(b['x']**2,b['ng'],b['ngraphs'])-xm.square()).clamp(min=1e-8).sqrt()
        em=scatter_mean(b['edge'],b['eg'],b['ngraphs'])
        es=(scatter_mean(b['edge']**2,b['eg'],b['ngraphs'])-em.square()).clamp(min=1e-8).sqrt()
        nv=torch.bincount(b['ng'],minlength=b['ngraphs']).to(h.dtype)
        ne=torch.bincount(b['eg'],minlength=b['ngraphs']).to(h.dtype)
        raw=self.raw(torch.cat([xm,xs,em,es,torch.log1p(nv)[:,None],torch.log1p(ne)[:,None]],dim=-1))
        return self.head(torch.cat(outs+[raw],dim=-1))

    def event_logits(self):return self._event_logits


def make_model(mode,dx,de,nc):
    if mode=='gine':return old_make_model('gine_cycle',dx,de,nc,44)
    if mode=='gine40':return old_make_model('gine_cycle',dx,de,nc,40)
    if mode=='gine_aux':return GINEWithEvent(dx,de,nc,h=42)
    if mode=='old_poset':return old_make_model('poset_cycle',dx,de,nc,24)
    if mode=='event':return EventMVFRouter(dx,de,nc,h=24)
    if mode=='event_noaux':return EventMVFRouter(dx,de,nc,h=24)
    if mode=='event_shuffle':return EventMVFRouter(dx,de,nc,h=24,kind='shuffled')
    if mode=='event_uniform':return EventMVFRouter(dx,de,nc,h=24,kind='uniform')
    if mode=='event_softonly':return EventMVFRouter(dx,de,nc,h=24,hard=False,soft=True)
    raise ValueError(mode)


def aux_loss(model,b,mode,alpha):
    if not hasattr(model,'event_logits') or mode=='event_noaux' or alpha<=0:
        return b['edge'].new_zeros(()),float('nan')
    logits=model.event_logits()
    y=b['bridge_target']
    if not len(y):return logits.sum()*0.,float('nan')
    # Fixed soft class-balance avoids excluding bridge-free mini-batches.
    pos=y.float().mean().clamp(min=.05,max=.95)
    cw=torch.stack([(0.5/(1-pos)),(0.5/pos)]).to(logits.device)
    loss=F.cross_entropy(logits,y,weight=cw)
    return alpha*loss,float((logits.argmax(-1)==y).float().mean().detach())


def dry_check(G,y):
    from sklearn.preprocessing import StandardScaler
    subset=G[:min(3,len(G))];ids=list(range(len(subset)))
    sc=StandardScaler().fit(np.concatenate([g['x'] for g in subset]))
    b=batch_of(subset,ids,sc)
    m=make_model('event',b['x'].shape[-1],b['edge'].shape[-1],int(max(y))+1)
    m.eval();out=m(b);loss,_=aux_loss(m,b,'event',.2);(out.square().mean()+loss).backward()
    grads={'event_head':float(m.layers[0].event_head[-1].weight.grad.abs().sum()),
           'cell':float(m.layers[0].cell[0].weight.grad.abs().sum()),
           'incidence_gate':float(m.layers[0].inc_gate[0].weight.grad.abs().sum())}
    assert all(np.isfinite(list(grads.values()))) and min(grads.values())>0,grads
    # Oracle target must not enter model forward, nor be exposed as input.
    assert all(not np.any(g['x'][:,-2]) and not np.any(g['edge'][:,-3]) for g in subset)
    import copy
    altered=copy.deepcopy(subset)
    for g in altered:
        rng=np.random.default_rng(91)
        ix=rng.permutation(len(g['pairs']))
        g['pairs']=g['pairs'][ix][:,::-1].copy();g['edge']=g['edge'][ix].copy()
        g['bridge_target']=g['bridge_target'][ix].copy()
    with torch.no_grad():
        old=m(b);new=m(batch_of(altered,ids,sc))
    err=float((old-new).abs().max())
    assert err<3e-4,(err,grads)
    return {'edge_endpoint_order_max_abs':err,'gradient_abs_sum':grads,
            'n':len(b['x']),'m':len(b['pairs']),'parameters':sum(z.numel() for z in m.parameters())}


def export_results_zip(output):
    """Atomic, resumable export of completed and partial folds."""
    import zipfile
    folder=Path(output)
    path=Path(os.environ.get('MVF_RESULTS_ZIP',str(folder.parent/'MVF_Event_Results.zip')))
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for f in folder.rglob('*'):
            if f.is_file():z.write(f,arcname=str(f.relative_to(folder)))
    tmp.replace(path)
    return path


def train_one(G,y,dataset,mode,seed=101,folds=3,epochs=30,patience=8,
              split_seed=20261121,alpha=.2,budget=1100,output='output',fold_limit=None):
    outdir=Path(output)/'runs'/dataset;outdir.mkdir(parents=True,exist_ok=True)
    stem=f'{dataset}_{mode}_s{seed}'
    final=outdir/(stem+'.json')
    protocol={'dataset':dataset,'mode':mode,'seed':seed,'folds':folds,'epochs':epochs,'patience':patience,
              'split_seed':split_seed,'aux_weight':alpha,'node_budget':budget,'preprocess':'event_target_hidden_v1'}
    hashstr=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()[:16]
    if final.is_file():
        previous=json.loads(final.read_text())
        if previous.get('protocol_hash')==hashstr and previous.get('complete'):
            print('SKIP COMPLETED',stem,flush=True);return previous
        raise RuntimeError(f'Conflicting run parameters for {final.name}; move old outputs to a different folder')
    scores=np.full(len(y),-1,dtype=np.int64);probs=np.zeros((len(y),int(y.max())+1),np.float32)
    foldarray=np.full(len(y),-1,dtype=np.int64);logs=[];all_aux_pred=[];all_aux_y=[];interventions=[]
    cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=split_seed)
    partial_json=outdir/(stem+'_partial.json')
    partial_npz=outdir/(stem+'_partial.npz')
    if partial_json.exists() and partial_npz.exists():
        prior=json.loads(partial_json.read_text())
        if prior.get('protocol_hash')!=hashstr:
            raise RuntimeError('Incompatible prior partial run; use a new output folder.')
        loaded=np.load(partial_npz)
        scores=loaded['pred'].copy();probs=loaded['probs'].copy();foldarray=loaded['fold'].copy()
        logs=prior['completed_folds']
        all_aux_y=prior.get('aux_y',[]);all_aux_pred=prior.get('aux_pred',[])
        interventions=prior.get('interventions',[])
        print('RESUME',stem,'completed_folds',len(logs),flush=True)
    started=time.monotonic()
    model=make_model(mode,G[0]['x'].shape[-1],G[0]['edge'].shape[-1],int(y.max())+1)
    for fold,(tr,te) in enumerate(cv.split(np.arange(len(y)),y)):
        if any(z['fold']==fold for z in logs):continue
        if fold_limit is not None and fold>=fold_limit:break
        setup_seed(seed+fold*101)
        ai,vi=next(StratifiedShuffleSplit(n_splits=1,test_size=.18,random_state=seed+fold*31).split(tr,y[tr]))
        fit=tr[ai];val=tr[vi]
        scaler=StandardScaler().fit(np.concatenate([G[i]['x'] for i in fit]))
        model=make_model(mode,G[0]['x'].shape[-1],G[0]['edge'].shape[-1],int(y.max())+1)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
        cnt=np.bincount(y[fit],minlength=int(y.max())+1)
        cw=torch.tensor(np.sqrt(len(fit)/np.maximum(cnt,1)),dtype=torch.float32);cw/=cw.mean()
        best=float('inf');best_ep=0;state=None;stale=0
        for ep in range(epochs):
            model.train()
            for indices in chunks(G,fit,budget,seed=seed*1000+fold*97+ep):
                b=batch_of(G,indices,scaler)
                z=model(b)
                a,_=aux_loss(model,b,mode,alpha)
                loss=F.cross_entropy(z,torch.tensor(y[indices]),weight=cw)+a
                opt.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
            model.eval();vl=0.;count=0
            with torch.no_grad():
                for indices in chunks(G,val,budget):
                    b=batch_of(G,indices,scaler);z=model(b)
                    a,_=aux_loss(model,b,mode,alpha)
                    lo=F.cross_entropy(z,torch.tensor(y[indices]),weight=cw)+a
                    vl+=float(lo)*len(indices);count+=len(indices)
            vl/=max(count,1)
            if vl<best-1e-4:
                best=vl;best_ep=ep+1;state=copy.deepcopy(model.state_dict());stale=0
            else:stale+=1
            if stale>=patience and ep>=10:break
        model.load_state_dict(state);model.eval()
        fold_pred=[];fold_y=[]
        with torch.no_grad():
            for indices in chunks(G,te,budget):
                b=batch_of(G,indices,scaler)
                z=model(b);p=z.softmax(-1).cpu().numpy()
                scores[indices]=p.argmax(1);probs[indices]=p;foldarray[indices]=fold
                if hasattr(model,'event_logits'):
                    zz=model.event_logits().softmax(-1)[:,1].cpu().numpy()
                    fold_pred.extend(zz.tolist());fold_y.extend(b['bridge_target'].cpu().numpy().tolist())
                if mode=='event':
                    original=z.argmax(1)
                    for override in ('shuffled','uniform'):
                        model.set_override(override)
                        new=model(b).argmax(1)
                        interventions.append({'fold':fold,'override':override,'changed':int((original!=new).sum()),'graphs':len(indices)})
                    model.set_override(None)
        if fold_y:
            all_aux_y.extend(fold_y);all_aux_pred.extend(fold_pred)
        fold_ba=float(balanced_accuracy_score(y[te],scores[te]))
        logs.append({'fold':fold,'balanced_accuracy':fold_ba,'best_epoch':best_ep,'validation_objective':best})
        print('FOLD',dataset,mode,'seed',seed,'fold',fold,'BA',round(fold_ba*100,2),'best_ep',best_ep,flush=True)
        # Write fold-complete checkpoint, preserving interrupted Kaggle sessions.
        np.savez_compressed(outdir/(stem+'_partial.npz'),y=y,pred=scores,probs=probs,fold=foldarray)
        (outdir/(stem+'_partial.json')).write_text(json.dumps({'protocol':protocol,'protocol_hash':hashstr,'completed_folds':logs,'aux_y':all_aux_y,'aux_pred':all_aux_pred,'interventions':interventions},indent=2))
        export_results_zip(output)
    complete=(fold_limit is None or fold_limit>=folds) and np.all(foldarray>=0)
    ba=float(balanced_accuracy_score(y,scores)) if complete else None
    auc=float(roc_auc_score(all_aux_y,all_aux_pred)) if all_aux_y and len(set(all_aux_y))==2 else None
    result={**protocol,'protocol_hash':hashstr,'complete':bool(complete),'balanced_accuracy':ba,'aux_edge_auc':auc,
            'seconds':time.monotonic()-started,'parameters':sum(p.numel() for p in model.parameters()),'fold_records':logs}
    if interventions:
        result['same_checkpoint_interventions']={k:{'changed_fraction':sum(q['changed'] for q in interventions if q['override']==k)/max(1,sum(q['graphs'] for q in interventions if q['override']==k))} for k in ('shuffled','uniform')}
    if complete:
        np.savez_compressed(outdir/(stem+'.npz'),y=y,pred=scores,probs=probs,fold=foldarray,
                            edge_y=np.asarray(all_aux_y,dtype=np.int8),edge_score=np.asarray(all_aux_pred,np.float32))
        final.write_text(json.dumps(result,indent=2))
        (outdir/(stem+'_partial.npz')).unlink(missing_ok=True)
        (outdir/(stem+'_partial.json')).unlink(missing_ok=True)
        export_results_zip(output)
    return result


def summarize(output):
    import pandas as pd
    output=Path(output);r=[]
    for f in (output/'runs').glob('*/*.json'):
        try:
            x=json.loads(f.read_text())
            if x.get('complete'):r.append(x)
        except Exception:pass
    if not r:return None
    df=pd.DataFrame([{k:r0.get(k) for k in ('dataset','mode','seed','balanced_accuracy','aux_edge_auc','seconds','parameters','epochs','folds')} for r0 in r])
    df['balanced_accuracy_pct']=df.balanced_accuracy*100
    df.to_csv(output/'completed_runs.csv',index=False)
    sm=df.groupby(['dataset','mode']).agg(seeds=('seed','count'),mean_ba=('balanced_accuracy_pct','mean'),sd_ba=('balanced_accuracy_pct','std'),mean_aux_auc=('aux_edge_auc','mean')).reset_index()
    sm.to_csv(output/'summary.csv',index=False)
    return sm


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--datasets',nargs='+',default=['BZR','DHFR'])
    p.add_argument('--modes',nargs='+',default=['gine','gine_aux','old_poset','event','event_noaux','event_shuffle','event_uniform'])
    p.add_argument('--seeds',nargs='+',type=int,default=[101,202,303])
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--folds',type=int,default=3)
    p.add_argument('--patience',type=int,default=8);p.add_argument('--aux_weight',type=float,default=.2)
    p.add_argument('--split_seed',type=int,default=20261121);p.add_argument('--node_budget',type=int,default=1100)
    p.add_argument('--output',default=str(ROOT/'output'));p.add_argument('--fold_limit',type=int,default=None)
    p.add_argument('--allow_download',action='store_true');p.add_argument('--preflight',action='store_true')
    a=p.parse_args()
    Path(a.output).mkdir(parents=True,exist_ok=True)
    for ds in a.datasets:
        G,y=load_graphs(ds,allow_download=a.allow_download)
        if a.preflight:print('CHECK',ds,json.dumps(dry_check(G,y)),flush=True)
        for seed in a.seeds:
            for mode in a.modes:
                z=train_one(G,y,ds,mode,seed,a.folds,a.epochs,a.patience,a.split_seed,a.aux_weight,a.node_budget,a.output,a.fold_limit)
                print('RESULT',ds,mode,seed,z.get('balanced_accuracy'),flush=True)
                summarize(a.output)
    print('SUMMARY',summarize(a.output),flush=True)

if __name__=='__main__':main()
