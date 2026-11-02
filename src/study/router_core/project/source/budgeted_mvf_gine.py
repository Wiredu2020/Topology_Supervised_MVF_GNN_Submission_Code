"""Experimental learned symmetric/asymmetric multivector pooling on a GINE-equation backbone.

No PyTorch Geometric is installed in this runtime. GINE is implemented with the exact
published sum+ReLU+MLP update, not labelled as the PyG canonical source implementation.
Endpoint score -> hard incidence CC on detached scores; surrogate score gradients through
continuous block-mass and cell-feature weights. This is NOT differentiable topology.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, random, time, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, accuracy_score
sys.path.insert(0,str(Path(__file__).parent / "source"))
from mvf_real_validation import load_tu_zip

NTHREADS=int(os.environ.get('MVF_THREADS','2'));torch.set_num_threads(NTHREADS)

def setup_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def preprocess_graph(g):
    n=g['n'];pairs=np.asarray(g['edges'],dtype=np.int64).reshape(-1,2);m=len(pairs)
    degree=np.bincount(pairs.ravel(),minlength=n).astype(np.float32)
    x=np.column_stack([g['node'],np.log1p(degree),degree/max(1.,float(degree.max())),np.ones(n,dtype=np.float32)]).astype(np.float32)
    edge=np.asarray(g['edge_attr'],dtype=np.float32).reshape(m,-1)
    # degree differences included for all methods; pure TU edge features may be empty.
    a=np.column_stack([edge,np.ones(m,dtype=np.float32),np.abs(np.log1p(degree[pairs[:,0]])-np.log1p(degree[pairs[:,1]]))]).astype(np.float32) if m else np.zeros((0,edge.shape[1]+2),np.float32)
    return {'n':n,'pairs':pairs,'x':x,'edge':a,'id':g['original_id']}

def make_batch(G,ids,scaler,device='cpu'):
    xs=[];ee=[];ps=[];src=[];dst=[];ng=[];eg=[];rawids=[];arc_eid=[];noff=0;mooff=0
    for j,idx in enumerate(ids):
        g=G[int(idx)];n=g['n'];p=g['pairs'];m=len(p)
        x=scaler.transform(g['x']).clip(-10,10).astype(np.float32)
        xs.append(torch.from_numpy(x));ee.append(torch.from_numpy(g['edge']))
        ps.append(torch.from_numpy(p+noff));ng.extend([j]*n);eg.extend([j]*m)
        if m:
            src.append(torch.from_numpy(np.r_[p[:,0],p[:,1]]+noff))
            dst.append(torch.from_numpy(np.r_[p[:,1],p[:,0]]+noff))
            arc_eid.append(torch.from_numpy(np.r_[np.arange(m),np.arange(m)]+mooff))
        noff+=n;mooff+=m;rawids.append(g['id'])
    edge=torch.cat(ee).to(device);pairs=torch.cat(ps).to(device)
    src=torch.cat(src).long().to(device) if src else torch.empty(0,dtype=torch.long,device=device)
    dst=torch.cat(dst).long().to(device) if dst else torch.empty(0,dtype=torch.long,device=device)
    return dict(x=torch.cat(xs).to(device),edge=edge,pairs=pairs,src=src,dst=dst,
        ng=torch.tensor(ng,dtype=torch.long,device=device),eg=torch.tensor(eg,dtype=torch.long,device=device),
        arc_edge_id=torch.cat(arc_eid).long().to(device) if arc_eid else torch.empty(0,dtype=torch.long,device=device),
        ngraphs=len(ids),ids=rawids)

def scatter_sum(x,idx,k):
    out=x.new_zeros((k,x.shape[-1]));out.index_add_(0,idx,x);return out

def scatter_mean(x,idx,k):
    if len(idx)==0:return x.new_zeros((k,x.shape[-1]))
    counts=x.new_zeros((k,));counts.index_add_(0,idx,x.new_ones(len(idx)))
    return scatter_sum(x,idx,k)/counts.clamp(min=1)[:,None]

# Mathematical GINEConv: x'_i = h_Theta((1+eps)*x_i + sum_{j in N(i)} ReLU(x_j + e_{ji})).
class ExactEquationGINE(nn.Module):
    def __init__(self,h,drop):
        super().__init__()
        self.eps=nn.Parameter(torch.zeros(1))
        self.nn=nn.Sequential(nn.Linear(h,2*h),nn.ReLU(),nn.Linear(2*h,h))
        self.bn=nn.BatchNorm1d(h)
        self.drop=nn.Dropout(drop)
    def forward(self,h,b,edge_enc):
        agg=h.new_zeros(h.shape)
        if len(b['src']):
            e2=edge_enc[b['arc_edge_id']]
            msg=F.relu(h[b['src']]+e2)
            agg.index_add_(0,b['dst'],msg)
        z=self.bn(self.nn((1+self.eps)*h+agg))
        return F.relu(h+self.drop(z))


# Optional *actual PyTorch Geometric* implementation of the canonical GINE operator.
# This backend requires torch_geometric.
class OfficialPyGGINEResidual(nn.Module):
    def __init__(self,h,drop):
        super().__init__()
        from torch_geometric.nn import GINEConv
        self.conv=GINEConv(nn=nn.Sequential(nn.Linear(h,2*h),nn.ReLU(),nn.Linear(2*h,h)),train_eps=True)
        self.bn=nn.BatchNorm1d(h)
        self.drop=nn.Dropout(drop)
    def forward(self,h,b,edge_enc):
        edge_index=torch.stack([b['src'],b['dst']],dim=0)
        z=self.bn(self.conv(h,edge_index,edge_enc[b['arc_edge_id']]))
        return F.relu(h+self.drop(z))

class LearnedMvfGINE(nn.Module):
    def __init__(self,dx,de,nclass,h=48,depth=3,drop=.15,tau=.55,backend='reference'):
        super().__init__();self.h=h;self.tau=tau
        self.node=nn.Sequential(nn.Linear(dx,h),nn.ReLU(),nn.LayerNorm(h))
        self.edge=nn.Sequential(nn.Linear(de,h),nn.ReLU(),nn.Linear(h,h))
        self.layers=nn.ModuleList([(OfficialPyGGINEResidual(h,drop) if backend=='pyg' else ExactEquationGINE(h,drop)) for _ in range(depth)])
        self.score_sym=nn.Sequential(nn.Linear(3*h,h),nn.ReLU(),nn.Linear(h,1))
        self.score_query=nn.Linear(h,h,bias=False)
        self.score_key=nn.Linear(h,h,bias=False)
        nn.init.constant_(self.score_sym[-1].bias,-.65)
        self.score_strength=nn.Parameter(torch.tensor(.0))
        self.cell=nn.Sequential(nn.Linear(h+2,h),nn.ReLU(),nn.Linear(h,h))
        self.block=nn.Sequential(nn.Linear(2*h+2,h),nn.ReLU(),nn.Linear(h,h))
        self.block_gate=nn.Sequential(nn.Linear(3*h,h),nn.ReLU(),nn.Linear(h,1))
        nn.init.constant_(self.block_gate[-1].bias,-2.0)
        self.raw=nn.Sequential(nn.Linear(2*dx+2*de+2,2*h),nn.ReLU(),nn.Linear(2*h,h))
        self.classifier=nn.Sequential(nn.Linear(depth*2*h+3*h,2*h),nn.ReLU(),nn.Dropout(drop),nn.Linear(2*h,nclass))

    def score(self,enc,b):
        p=b['pairs'];m=len(p)
        if not m:
            z=enc.new_zeros((0,));return z,z
        u,v=p[:,0],p[:,1];e=self.edge(b['edge'])
        cov=torch.cat([enc[u]+enc[v],torch.abs(enc[u]-enc[v]),e],dim=1)
        s=self.score_sym(cov).squeeze(1)
        qu=self.score_query(enc);kv=self.score_key(enc)
        a=((qu[u]*kv[v]).sum(1)-(qu[v]*kv[u]).sum(1))/(self.h**.5)
        a=torch.tanh(a)*torch.exp(self.score_strength.clamp(-2,2))
        return s,a

    @staticmethod
    def field_labels(n,pairs,accept):
        # Two-level graph cell order, so all connected incidence blocks are valid multivectors.
        m=len(pairs);parent=list(range(n+m));size=[1]*(n+m)
        def find(x):
            while x!=parent[x]:parent[x]=parent[parent[x]];x=parent[x]
            return x
        def union(u,v):
            u=find(u);v=find(v)
            if u!=v:
                if size[u]<size[v]:u,v=v,u
                parent[v]=u;size[u]+=size[v]
        for j,(u,v) in enumerate(pairs):
            if accept[j,0]:union(int(u),n+j)
            if accept[j,1]:union(int(v),n+j)
        roots=[find(i) for i in range(n+m)];_,labels=np.unique(roots,return_inverse=True)
        return labels.astype(np.int64),len(set(roots))

    @staticmethod
    def budget_accept(n,pairs,q,k=2,symmetric=True):
        # Endpoint-local top-k with tie inclusion, preserving permutation equivariance.
        # Symmetric field uses mutually chosen edges;
        # asymmetric field accepts each endpoint independently. No labels enter ranking.
        chosen=np.zeros((len(pairs),2),dtype=np.bool_)
        inc=[[] for _ in range(n)]
        for j,(u,v) in enumerate(pairs):
            inc[int(u)].append((float(q[j,0]),j,0))
            inc[int(v)].append((float(q[j,1]),j,1))
        for vals in inc:
            if not vals:continue
            # Keep every candidate tied at the kth score (within a numerical tolerance).
            # Picking arbitrary first-k edges is NOT permutation equivariant.
            cutoff=sorted([v[0] for v in vals],reverse=True)[min(k,len(vals))-1]
            for score,j,end in vals:
                if score>=cutoff-1e-5:chosen[j,end]=True
        if symmetric:
            mutual=np.logical_and(chosen[:,0],chosen[:,1])
            chosen[:]=mutual[:,None]
        return chosen

    def learned_block_pool(self,h0,h,b,mode,s,a):
        p=b['pairs'];m=len(p);nnodes=len(h)
        if mode in ('learned_sym','shuffle_sym','fixed_sym','budget_sym','budget_shuffle_sym','context_budget_sym','context_budget_shuffle_sym'):
            q=torch.sigmoid(s);q2=torch.stack([q,q],dim=1)
        elif mode in ('learned_asym','shuffle_asym','budget_asym','budget_shuffle_asym','context_budget_asym','context_budget_shuffle_asym'):
            q2=torch.sigmoid(torch.stack((s+a,s-a),dim=1))
        else:raise ValueError(mode)
        # Fixed competitor uses meaningful degree similarity instead of trainable q for hard grouping.
        # Its scored edge channels still flow into GINE exactly as in the learned variants.
        if mode=='fixed_sym':
            # x includes log-degree at [-3], normalized degree [-2]
            logd=h0.new_zeros(nnodes)
            logd.index_add_(0,p[:,0],h0.new_ones(m));logd.index_add_(0,p[:,1],h0.new_ones(m))
            v=logd[p[:,0]];w=logd[p[:,1]]
            qfix=torch.exp(-torch.abs(v-w)/(v.std(unbiased=False).clamp(min=.1)))
            q2=qfix[:,None].repeat(1,2)
        if 'shuffle' in mode:
            # Shuffling within a graph preserves the distribution of accepted edge patterns,
            # and detaches the alignment between the learned score and specific edge endpoints.
            # Deterministic index permutation by graph indices, not graph labels or global RNG.
            qn=q2.clone()
            eg=b['eg']; graph_ids=b['ids']
            for gi,gid in enumerate(graph_ids):
                loc=(eg==gi).nonzero().flatten();k=len(loc)
                if k>1:
                    rng=np.random.default_rng(33017+int(gid)*173)
                    perm=torch.as_tensor(rng.permutation(k),device=loc.device)
                    qn[loc]=q2[loc[perm]]
            q2=qn
        p_cpu=p.detach().cpu().numpy()
        q_cpu=q2.detach().cpu().numpy()
        ng=b['ng'];eg=b['eg'];cellnode=[];celledge=[];blockgraph=[];total=0;accepted_inc=0;possible_inc=0
        for gi,gid in enumerate(b['ids']):
            nv=(ng==gi).nonzero().flatten();ei=(eg==gi).nonzero().flatten()
            # global node indices are monotonically increasing with batching
            off=int(nv[0]);localpairs=p_cpu[ei.cpu().numpy()]-off
            if 'budget' in mode:
                ac=self.budget_accept(len(nv),localpairs,q_cpu[ei.cpu().numpy()],k=2,
                    symmetric=('sym' in mode and 'asym' not in mode))
            else:ac=q_cpu[ei.cpu().numpy()]>=self.tau
            accepted_inc+=int(ac.sum());possible_inc+=int(ac.size)
            labels,k=self.field_labels(len(nv),localpairs,ac)
            cellnode.extend((labels[:len(nv)]+total).tolist());celledge.extend((labels[len(nv):]+total).tolist())
            blockgraph.extend([gi]*k);total+=k
        vi=torch.tensor(cellnode,device=h.device,dtype=torch.long)
        ei=torch.tensor(celledge,device=h.device,dtype=torch.long)
        bg=torch.tensor(blockgraph,device=h.device,dtype=torch.long)
        # Differentiable surrogate: cell masses and edge-cell signals retain gradient to scores.
        e=self.edge(b['edge']);# Edge-cell features must be independent of arbitrary endpoint ordering.
        q_invariant=torch.stack([q2.mean(1),(q2[:,0]-q2[:,1]).abs()],dim=1)
        ec=self.cell(torch.cat([e,q_invariant],dim=1))
        qmass=.2+q2.mean(1)
        v_sum=scatter_sum(h,vi,total);e_sum=scatter_sum(ec*qmass[:,None],ei,total)
        nv=h.new_zeros(total).index_add_(0,vi,torch.ones_like(vi,dtype=h.dtype))
        ne=h.new_zeros(total).index_add_(0,ei,qmass)
        vmean=v_sum/nv.clamp(min=1)[:,None]
        emean=e_sum/ne.clamp(min=1e-8)[:,None]
        blocks=self.block(torch.cat([vmean,emean,torch.log1p(nv)[:,None],torch.log1p(ne)[:,None]],dim=1))
        # non-directed block-level pooling; both symmetric and asymmetric methods use it.
        node_signal=scatter_mean(blocks[vi],ng,b['ngraphs'])
        block_signal=scatter_mean(blocks,bg,b['ngraphs'])
        pooled=torch.cat([node_signal,block_signal],dim=1)
        # Returns diagnostics that do not depend on class labels.
        accept_count=accepted_inc/max(1,possible_inc)
        return pooled,q2.mean(),float(total/b['ngraphs']),accept_count

    def forward(self,b,mode,details=False):
        h=self.node(b['x']);h0=h;enc=self.edge(b['edge']);outputs=[]
        # Learnable incidence scores built from the initial (label-independent) node/edge inputs.
        s,a=self.score(h0,b)
        for layer in self.layers:
            h=layer(h,b,enc)
            gs=scatter_sum(h,b['ng'],b['ngraphs']);count=h.new_zeros(b['ngraphs']).index_add_(0,b['ng'],h.new_ones(len(h)))
            outputs.extend([gs/count[:,None].clamp(min=1),gs/count[:,None].clamp(min=1).sqrt()])
        pooled=torch.cat(outputs,dim=1)
        if mode.startswith('context_'):
            s,a=self.score(h,b)
        # Global raw graph moments supply an identical reference information path to every model.
        gx=scatter_mean(b['x'],b['ng'],b['ngraphs']);g2=scatter_mean(b['x']**2,b['ng'],b['ngraphs'])
        if len(b['eg']):
            ee=scatter_mean(b['edge'],b['eg'],b['ngraphs']);e2=scatter_mean(b['edge']**2,b['eg'],b['ngraphs'])
        else:
            ee=b['edge'].new_zeros(b['ngraphs'],b['edge'].shape[-1]);e2=ee
        size=torch.log1p(count)[:,None];es=b['edge'].new_zeros(b['ngraphs']).index_add_(0,b['eg'],b['edge'].new_ones(len(b['eg'])))
        stats=torch.cat([gx,(g2-gx*gx).clamp(min=1e-8).sqrt(),ee,(e2-ee*ee).clamp(min=1e-8).sqrt(),size,torch.log1p(es)[:,None]],dim=1)
        raw=self.raw(stats)
        if mode=='base':extra=raw.new_zeros((b['ngraphs'],2*self.h)); diagnostics=(0.,0.,0.)
        else:
            block,meanq,nb,acc=self.learned_block_pool(h0,h,b,mode,s,a)
            gate=torch.sigmoid(self.block_gate(torch.cat([raw,block],dim=1)))
            extra=block*gate
            diagnostics=(float(meanq.detach()),nb,acc)
        result=self.classifier(torch.cat([pooled,raw,extra],dim=1))
        return (result,diagnostics) if details else result


def chunks(G,ids,nodebudget=1400,seed=None):
    ids=np.asarray(ids).copy()
    if seed is not None:np.random.default_rng(seed).shuffle(ids)
    out=[];buf=[];nodes=0
    for i in ids:
        n=G[int(i)]['n']
        if buf and nodes+n>nodebudget:
            out.append(buf);buf=[];nodes=0
        buf.append(int(i));nodes+=n
    if buf:out.append(buf)
    return out

def test_preflight(G):
    sample=G[:min(len(G),3)];classes=2
    scaler=StandardScaler().fit(np.concatenate([g['x'] for g in sample]));bt=make_batch(sample,list(range(len(sample))),scaler)
    model=LearnedMvfGINE(bt['x'].shape[1],bt['edge'].shape[1],classes,h=24,depth=2)
    for mode in ['base','budget_sym','budget_asym','budget_shuffle_sym','budget_shuffle_asym']:
        logits,d=model(bt,mode,True);assert logits.shape==(len(sample),2) and torch.isfinite(logits).all()
        model.zero_grad();logits.sum().backward();
        if mode in ('learned_sym','learned_asym','budget_sym','budget_asym','context_budget_sym','context_budget_asym'):
            assert model.score_sym[-1].weight.grad is not None and model.score_sym[-1].weight.grad.abs().sum().item()>1e-12, (mode,'scorer gradient')
    return 'shapes, finite logits and scorer gradients PASS'

def evaluate(G,y,mode,seed=101,folds=3,epochs=22,h=48,depth=3,patience=6,budget=1400,backend='reference',split_seed=20261017):
    y=np.asarray(y,dtype=int);nclass=len(set(y));p=np.full(len(y),-1,dtype=int);proba=np.zeros((len(y),nclass),np.float32)
    cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=split_seed)
    rows=[];t0=time.time()
    for fold,(tr,te) in enumerate(cv.split(np.arange(len(y)),y)):
        setup_seed(seed+fold*199)
        s=StratifiedShuffleSplit(n_splits=1,test_size=.15,random_state=seed+fold*53)
        ia,iv=next(s.split(tr,y[tr]));fit=tr[ia];val=tr[iv]
        scaler=StandardScaler().fit(np.concatenate([G[int(i)]['x'] for i in fit]))
        mdl=LearnedMvfGINE(G[0]['x'].shape[1],G[0]['edge'].shape[1],nclass,h=h,depth=depth,backend=backend)
        opt=torch.optim.AdamW(mdl.parameters(),lr=.0008,weight_decay=1e-4)
        counts=np.bincount(y[fit],minlength=nclass);weights=torch.tensor(np.sqrt(counts.sum()/np.maximum(counts,1)),dtype=torch.float32);weights/=weights.mean()
        best=np.inf;state=None;bestepoch=0;stale=0;diagnostics=[]
        for ep in range(epochs):
            mdl.train();mtr=[]
            for ids in chunks(G,fit,budget,seed=seed*10000+fold*100+ep):
                bt=make_batch(G,ids,scaler);logits,stats=mdl(bt,mode,True)
                loss=F.cross_entropy(logits,torch.tensor(y[ids],dtype=torch.long),weight=weights)
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(mdl.parameters(),2.);opt.step();mtr.append(float(loss.detach()))
            mdl.eval();vtotal=0.;nval=0;dd=[]
            with torch.no_grad():
                for ids in chunks(G,val,budget):
                    logits,stat=mdl(make_batch(G,ids,scaler),mode,True)
                    l=F.cross_entropy(logits,torch.tensor(y[ids],dtype=torch.long),weight=weights)
                    vtotal+=float(l)*len(ids);nval+=len(ids);dd.append(stat)
            vl=vtotal/max(1,nval)
            if vl<best-1e-4:
                best=vl;state=copy.deepcopy(mdl.state_dict());bestepoch=ep+1;stale=0;diagnostics=np.mean(dd,axis=0).tolist() if dd else []
            else:stale+=1
            if stale>=patience and ep>=11:break
        mdl.load_state_dict(state);mdl.eval()
        with torch.no_grad():
            for ids in chunks(G,te,budget):
                pp=mdl(make_batch(G,ids,scaler),mode).softmax(-1).cpu().numpy();p[ids]=pp.argmax(1);proba[ids]=pp
        r=dict(fold=fold,BA=float(balanced_accuracy_score(y[te],p[te])),epoch=bestepoch,val_loss=best,diag=diagnostics)
        rows.append(r)
        print(f'FOLD {os.environ.get("DATASET")} {mode} {seed} {fold+1}/{folds} BA={r["BA"]:.4f} ep={bestepoch} elapsed={time.time()-t0:.1f}s',flush=True)
    assert np.all(p>=0)
    return dict(BA=float(balanced_accuracy_score(y,p)),ACC=float(accuracy_score(y,p)),p=p,proba=proba,folds=rows,params=sum(x.numel() for x in mdl.parameters()),seconds=time.time()-t0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--datasets',nargs='+',default=['MUTAG','COX2_MD','PROTEINS','ENZYMES'])
    ap.add_argument('--modes',nargs='+',default=['base','budget_sym','budget_asym','budget_shuffle_sym'])
    ap.add_argument('--seeds',nargs='+',type=int,default=[101]);ap.add_argument('--folds',type=int,default=3)
    ap.add_argument('--epochs',type=int,default=22);ap.add_argument('--h',type=int,default=48);ap.add_argument('--depth',type=int,default=3)
    ap.add_argument('--budget',type=int,default=1400);ap.add_argument('--data_root',default='/mnt/data');ap.add_argument('--out',default='/mnt/data/mvf_learned/results')
    ap.add_argument('--backend',choices=['reference','pyg'],default='reference')
    ap.add_argument('--outer_split_seed',type=int,default=20261017)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    meta={'model': 'Official PyG GINEConv when backend=pyg; exact GINE equation when backend=reference', 'backend': args.backend,'tau':.55,'budget_topk':2,'folds':args.folds,'max_epochs':args.epochs,'node_budget':args.budget,'h':args.h,'depth':args.depth,'fixed_split_seed':args.outer_split_seed,'modes':args.modes,'datasets':args.datasets,'seeds':args.seeds}
    (out/'RUN_CONFIG.json').write_text(json.dumps(meta,indent=2))
    for name in args.datasets:
        os.environ['DATASET']=name
        raw,y=load_tu_zip(str(Path(args.data_root)/f'{name}.zip'),name);G=[preprocess_graph(g) for g in raw]
        print('DATA',name,len(y),np.bincount(y).tolist(),G[0]['x'].shape,G[0]['edge'].shape,flush=True)
        print('CHECK',test_preflight(G),flush=True)
        for seed in args.seeds:
            for mode in args.modes:
                tag=f'{name}_{mode}_s{seed}'
                if (out/f'{tag}.json').exists():print('SKIP',tag,flush=True);continue
                r=evaluate(G,y,mode,seed=seed,folds=args.folds,epochs=args.epochs,h=args.h,depth=args.depth,budget=args.budget,backend=args.backend,split_seed=args.outer_split_seed)
                np.savez_compressed(out/f'{tag}.npz',y=y,pred=r['p'],probs=r['proba'],ids=np.asarray([g['id'] for g in G]))
                d={k:v for k,v in r.items() if k not in ('p','proba')};d.update({'dataset':name,'mode':mode,'seed':seed,'folds_n':args.folds,'outer_split_seed':args.outer_split_seed})
                (out/f'{tag}.json').write_text(json.dumps(d,indent=2))
                print('DONE',tag,'BA',round(r['BA']*100,2),'time',round(r['seconds'],1),flush=True)

if __name__=='__main__':main()
