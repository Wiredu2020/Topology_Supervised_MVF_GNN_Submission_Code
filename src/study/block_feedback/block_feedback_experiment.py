"""Experiments on FINITE FACE-POSET multivector block feedback during GNN learning.
Not a differentiable topology claim: hard incidence top-k is stop-gradient;
learnable incidence score weights and vertex/edge cell embeddings ARE differentiable.
Requires the exact recovered original StructuralRouter project alongside this file.
"""
import os,sys, json,time, copy,argparse,csv, random
from pathlib import Path
os.environ.setdefault('MVF_THREADS','1')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parent
RECOVERED=ROOT.parent/'router_core'/'project'
sys.path[:0]=[str(RECOVERED),str(RECOVERED/'source')]
from run_structural_suite import (feature_graph,load_tu_zip,make_batch,chunks,setup_seed,StructuralRouter,StructuralBlock,PlainGINE,scatter_sum,scatter_mean)
torch.set_num_threads(1)


def load_data(dataset):
    path=ROOT.parent/'router_core'/'data'/(dataset+'(1).zip')
    raw,y=load_tu_zip(str(path),dataset)
    graphs=[]
    for g in raw:
        p=np.asarray(g['edges'],dtype=np.int64).reshape(-1,2)
        edge=np.asarray(g['edge_attr'],dtype=np.float32).reshape(len(p),-1)
        # CRITICAL: reorder edge attributes together with canonicalized edge pairs.
        # Some PTC_MR TU archives store edges in nonlexicographic order.
        canonical=np.sort(p,axis=1)
        order=np.lexsort((canonical[:,1],canonical[:,0]))
        p=p[order];edge=edge[order]
        graphs.append(feature_graph(g['n'],p,np.asarray(g['node'],dtype=np.float32),edge,gid=g['original_id']))
    return graphs,np.asarray(y,dtype=np.int64)


def connected_labels(n,pairs,selection):
    """Cell-graph CC: n vertex cells + m edge cells; each selected incidence joins v to edge e.
    All such partitions are order convex on 1D face posets, even for disconnected blocks.
    """
    m=len(pairs);parent=list(range(n+m));sz=[1]*(n+m)
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        i=find(i);j=find(j)
        if i!=j:
            if sz[i]<sz[j]:i,j=j,i
            parent[j]=i;sz[i]+=sz[j]
    for eid,(u,v) in enumerate(pairs):
        if selection[eid,0]:union(int(u),n+eid)
        if selection[eid,1]:union(int(v),n+eid)
    roots=[find(i) for i in range(n+m)]
    dic={}; lab=[]
    for r in roots:
        if r not in dic:dic[r]=len(dic)
        lab.append(dic[r])
    return np.asarray(lab,np.int64),len(dic)


def topk_incidence(n,pairs,w,k):
    """Tie inclusive vertex-local selection of incidence links; full graph partition.
    Score `w` is one undirected edge priority from trained symmetric endpoint gates.
    """
    acc=np.zeros((len(pairs),2),bool)
    inc=[[] for _ in range(n)]
    for j,(u,v) in enumerate(pairs):
        inc[int(u)].append((float(w[j]),j,0));inc[int(v)].append((float(w[j]),j,1))
    for entries in inc:
        if entries:
            threshold=sorted((x[0] for x in entries),reverse=True)[min(k,len(entries))-1]
            for score,j,endpoint in entries:
                if score>=threshold-1e-6:acc[j,endpoint]=True
    return acc


def permute_priorities(w,pairs,gid):
    """Same graph-local learned score multiset, assigned to other edges reproducibly.
    Sorting canonical unordered endpoints makes permutation independent of edge storage order.
    """
    if len(w)<2:return w
    order=sorted(range(len(pairs)),key=lambda i:tuple(sorted(map(int,pairs[i]))))
    rnd=np.random.default_rng(912751+int(gid)*31)
    p=rnd.permutation(len(w))
    new=w.copy()
    for ix,j in enumerate(order):new[j]=w[order[p[ix]]]
    return new


class MVFBlockFeedback(StructuralBlock):
    def __init__(self,h,kind='learned',topk=1,feedback=True):
        super().__init__(h,drop=.15)
        self.kind=kind;self.topk=int(topk);self.feedback=feedback
        self.cell=nn.Sequential(nn.Linear(h+1,h),nn.ReLU(),nn.Linear(h,h))
        self.block_proj=nn.Sequential(nn.Linear(2*h+2,h),nn.ReLU(),nn.Linear(h,h))
        self.feedback_norm=nn.LayerNorm(h)
        self.feedback_scale=nn.Parameter(torch.tensor(-1.1))
        self.final_signal=None
    def field_pool(self,h,b,e,weights):
        # The two stored directed copies of an edge must be mapped back by edge id.
        # This is independent of the order of arcs inside a batched graph.
        m=len(b['pairs']);n=len(h)
        sums=weights.new_zeros(m).index_add_(0,b['arc_edge_id'],weights)
        w=sums/2.
        pairs=b['pairs'].detach().cpu().numpy();eg=b['eg'].detach().cpu().numpy()
        ng=b['ng'].detach().cpu().numpy()
        ids=b['ids']
        score_cpu=w.detach().cpu().numpy()
        vl=[];el=[];total=0;nblocks=[];naccepted=0
        for gi,gid in enumerate(ids):
            vs=np.where(ng==gi)[0];es=np.where(eg==gi)[0]
            localpairs=pairs[es]-int(vs[0]);score=score_cpu[es].copy()
            if self.kind=='shuffled':score=permute_priorities(score,localpairs,gid)
            if self.kind=='uniform':score[:]=1
            accept=topk_incidence(len(vs),localpairs,score,self.topk)
            labels,k=connected_labels(len(vs),localpairs,accept)
            vl.extend((labels[:len(vs)]+total).tolist());el.extend((labels[len(vs):]+total).tolist());total+=k
            naccepted+=int(accept.sum());nblocks.append(k)
        vlabel=torch.tensor(vl,device=h.device,dtype=torch.long)
        elabel=torch.tensor(el,device=h.device,dtype=torch.long)
        nv=torch.bincount(vlabel,minlength=total).to(h.dtype)
        ve=scatter_sum(h,vlabel,total)/nv.clamp(min=1)[:,None]
        # A continuous edge-cell mass carries gradients to the trained gate scores.
        # For shuffled/uniform controls, the block construction is de-aligned but
        # the model capacity, score network and weighted edge-cell channel remain equal.
        weight=torch.sigmoid(w)[:,None]
        emb=self.cell(torch.cat([e,weight],-1))
        ee=scatter_sum(emb*(.2+weight),elabel,total)
        ne=scatter_sum(.2+weight,elabel,total)[:,0]
        emean=ee/ne.clamp(min=1e-6)[:,None]
        block=self.block_proj(torch.cat([ve,emean,torch.log1p(nv)[:,None],torch.log1p(ne)[:,None]],-1))
        diag={'blocks_per_graph':float(np.mean(nblocks)),'blocks_per_cell':float(total/(n+m)),'incidence_fraction':float(naccepted/(2*max(1,m)))}
        return block[vlabel],diag
    def forward(self,h,b,e,mode='learned'):
        base=super().forward(h,b,e,'learned')
        # The field is extracted from scores computed from current hidden state,
        # so it evolves across trainable layers.  Hard assignments are stop-gradient.
        weights=self.getw(h,b,e,'learned')[0]
        signal,diag=self.field_pool(base,b,e,weights)
        self.final_signal=diag
        if self.feedback:
            return F.relu(base+torch.sigmoid(self.feedback_scale)*self.feedback_norm(signal))
        # For the readout control, the exact same block computation is run, but
        # its channel is injected after the last GNN layer, not between layers.
        self.readout_signal=signal
        return base

class MVFRouter(StructuralRouter):
    def __init__(self,dx,de,nclass,h,kind='learned',topk=1,feedback=True):
        super().__init__(dx,de,nclass,h=h,depth=2)
        self.feedback=feedback
        self.layers=nn.ModuleList([MVFBlockFeedback(h,kind,topk,feedback) for _ in range(2)])
        if not feedback:
            self.readout_gate=nn.Parameter(torch.tensor(-1.1))
            # Fixed size h output appended without increasing classifier input.
    def forward(self,b,mode='learned'):
        if self.feedback:return super().forward(b,'learned')
        h=self.node(b['x']);e=self.edge(b['edge']);outs=[]
        for layer in self.layers:
            h=layer(h,b,e)
            # Block readout appears after the GNN layer and *never* feeds its next layer.
            s=scatter_sum(h,b['ng'],b['ngraphs']);ct=torch.bincount(b['ng'],minlength=b['ngraphs']).to(h.dtype).clamp(min=1)
            extra=torch.sigmoid(self.readout_gate)*layer.readout_signal
            pooled=scatter_mean(extra,b['ng'],b['ngraphs'])
            outs.extend([s/ct[:,None]+pooled,s/ct.sqrt()[:,None]])
        xmean=scatter_mean(b['x'],b['ng'],b['ngraphs']);xstd=(scatter_mean(b['x']**2,b['ng'],b['ngraphs'])-xmean.square()).clamp(min=1e-8).sqrt()
        emean=scatter_mean(b['edge'],b['eg'],b['ngraphs']);estd=(scatter_mean(b['edge']**2,b['eg'],b['ngraphs'])-emean.square()).clamp(min=1e-8).sqrt()
        nv=torch.bincount(b['ng'],minlength=b['ngraphs']).to(h.dtype);ne=torch.bincount(b['eg'],minlength=b['ngraphs']).to(h.dtype)
        raw=self.raw(torch.cat([xmean,xstd,emean,estd,torch.log1p(nv)[:,None],torch.log1p(ne)[:,None]],-1))
        return self.head(torch.cat(outs+[raw],-1))


def make_model(name,dx,de,nclasses,h):
    if name=='gine':return PlainGINE(dx,de,nclasses,h=h)
    if name=='gine35':return PlainGINE(dx,de,nclasses,h=35)
    if name=='struct':return StructuralRouter(dx,de,nclasses,h=h)
    if name=='mvf_feedback':return MVFRouter(dx,de,nclasses,h,kind='learned',topk=1,feedback=True)
    if name=='mvf_shuffle':return MVFRouter(dx,de,nclasses,h,kind='shuffled',topk=1,feedback=True)
    if name=='mvf_uniform':return MVFRouter(dx,de,nclasses,h,kind='uniform',topk=1,feedback=True)
    if name=='mvf_readout':return MVFRouter(dx,de,nclasses,h,kind='learned',topk=1,feedback=False)
    if name=='mvf_feedback_k2':return MVFRouter(dx,de,nclasses,h,kind='learned',topk=2,feedback=True)
    raise ValueError(name)


def invariance_and_gradient_check(G,y):
    setup_seed(8071)
    sc=StandardScaler().fit(np.concatenate([x['x'] for x in G[:4]]));b=make_batch(G,np.arange(4),sc)
    model=make_model('mvf_feedback',b['x'].shape[1],b['edge'].shape[1],int(max(y))+1,20)
    model.eval();out=model(b);out.square().sum().backward()
    scorer=float(model.layers[0].scorer[-1].weight.grad.abs().sum())
    cell=float(model.layers[0].cell[0].weight.grad.abs().sum())
    block=float(model.layers[0].block_proj[0].weight.grad.abs().sum())
    import copy
    altered=copy.deepcopy(G[:4])
    for g in altered:
        pp=np.random.default_rng(901).permutation(len(g['pairs']))
        g['pairs']=g['pairs'][pp][:,::-1].copy();g['edge']=g['edge'][pp].copy()
    with torch.no_grad():
        original=model(b);changed=model(make_batch(altered,np.arange(4),sc))
    err=float((original-changed).abs().max())
    return {'score_gradient_abs_sum':scorer,'cell_gradient_abs_sum':cell,'block_gradient_abs_sum':block,'edge_order_invariance_max_abs':err,'sample_last_layer':model.layers[-1].final_signal}


def cv(G,y,dataset,name,seed=101,h=24,epochs=18,split_seed=20261101,folds=3,budget=1500,patience=5,out=None):
    y=np.asarray(y,dtype=int);ncl=int(y.max())+1
    pred=np.full(len(y),-1,dtype=int);probs=np.zeros((len(y),ncl),np.float32)
    foldids=np.full(len(y),-1,dtype=int);losslogs=[];started=time.time()
    cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=split_seed)
    for fold,(tr,te) in enumerate(cv.split(np.arange(len(y)),y)):
        setup_seed(seed+fold*101)
        ia,iv=next(StratifiedShuffleSplit(n_splits=1,test_size=.18,random_state=seed+fold*31).split(tr,y[tr]));fit=tr[ia];val=tr[iv]
        sc=StandardScaler().fit(np.concatenate([G[i]['x'] for i in fit]));m=make_model(name,G[0]['x'].shape[1],G[0]['edge'].shape[1],ncl,h)
        opt=torch.optim.AdamW(m.parameters(),lr=.0008,weight_decay=1e-4)
        cnt=np.bincount(y[fit],minlength=ncl);cw=torch.tensor(np.sqrt(len(fit)/np.maximum(1,cnt)),dtype=torch.float32);cw/=cw.mean()
        best=float('inf');state=None;be=0;stale=0
        for ep in range(epochs):
            m.train()
            for ids in chunks(G,fit,budget,seed=seed*1000+fold*97+ep):
                batch=make_batch(G,ids,sc);z=m(batch)
                loss=F.cross_entropy(z,torch.tensor(y[ids]),weight=cw)
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),2.0);opt.step()
            m.eval();val_loss=0.;count=0
            with torch.no_grad():
                for ids in chunks(G,val,budget):
                    z=m(make_batch(G,ids,sc));loss=F.cross_entropy(z,torch.tensor(y[ids]),weight=cw)
                    val_loss+=float(loss)*len(ids);count+=len(ids)
            val_loss/=max(1,count)
            if val_loss<best-1e-4:best=val_loss;state=copy.deepcopy(m.state_dict());be=ep+1;stale=0
            else:stale+=1
            if stale>=patience and ep>=9:break
        m.load_state_dict(state);m.eval()
        with torch.no_grad():
            for ids in chunks(G,te,budget):
                zz=m(make_batch(G,ids,sc)).softmax(-1).cpu().numpy()
                pred[ids]=zz.argmax(-1);probs[ids]=zz;foldids[ids]=fold
        ba=float(balanced_accuracy_score(y[te],pred[te]));losslogs.append(dict(fold=fold,BA=ba,best_epoch=be,validation_loss=best))
        print('FOLD',dataset,name,'s',seed,fold,'BA',round(ba*100,2),'epoch',be,'elapsed',round(time.time()-started,1),flush=True)
    ba=float(balanced_accuracy_score(y,pred));pcount=sum(p.numel() for p in m.parameters())
    result=dict(dataset=dataset,mode=name,seed=seed,BA=ba,params=pcount,epochs=epochs,split_seed=split_seed,folds=folds,h=h,seconds=time.time()-started,fold_records=losslogs)
    if out:
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        stem=f'{dataset}_{name}_s{seed}'
        np.savez_compressed(out/(stem+'.npz'),y=y,pred=pred,probs=probs,fold=foldids,ids=np.asarray([g['id'] for g in G]))
        (out/(stem+'.json')).write_text(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='PTC_MR');p.add_argument('--modes',nargs='+',default=['gine','struct','mvf_feedback','mvf_shuffle','mvf_uniform','mvf_readout']);p.add_argument('--seeds',nargs='+',type=int,default=[101,202,303]);p.add_argument('--h',type=int,default=24);p.add_argument('--epochs',type=int,default=18);p.add_argument('--split_seed',type=int,default=20261101);p.add_argument('--out',default=str(ROOT/'results'));args=p.parse_args()
    G,y=load_data(args.dataset)
    c=invariance_and_gradient_check(G,y)
    print('PREFLIGHT',json.dumps(c),flush=True)
    if args.out:(ROOT/'checks.json').write_text(json.dumps(c,indent=2))
    for seed in args.seeds:
        for name in args.modes:
            if (Path(args.out)/f'{args.dataset}_{name}_s{seed}.json').exists():continue
            r=cv(G,y,args.dataset,name,seed=seed,h=args.h,epochs=args.epochs,split_seed=args.split_seed,out=args.out)
            print('DONE',name,'seed',seed,'BA',round(100*r['BA'],2),'params',r['params'],flush=True)
if __name__=='__main__':main()
