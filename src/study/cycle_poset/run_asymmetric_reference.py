"""Graph incidence routing and multivector-learning experiments."""
import os,sys,io,zipfile,pickle,copy,random,json,time,argparse
from pathlib import Path
import numpy as np,torch,networkx as nx
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
ROOT=Path(__file__).resolve().parent; STUDY=ROOT.parent
sys.path.insert(0,str(STUDY/'block_feedback'))
from block_feedback_experiment import (MVFBlockFeedback,MVFRouter,connected_labels,topk_incidence,permute_priorities,scatter_sum,scatter_mean,PlainGINE,StructuralRouter,feature_graph,make_batch,chunks,setup_seed)
torch.set_num_threads(1)
K_RANGE=list(range(3,11))
def cycle_graph_features(n,pairs):
    """Random-walk return and edge-walk intensities; no class labels or graph IDs.
    P=A/degree; features use P^k (walks); invariant under relabeling.
    Fixed 16x scaling, same inputs provided to all model variants.
    """
    A=np.zeros((n,n),np.float64)
    for u,v in pairs:A[u,v]=A[v,u]=1.
    deg=A.sum(1);P=A/np.maximum(deg[:,None],1.);power=np.eye(n);nodes=[];edges=[]
    for k in range(1,max(K_RANGE)+1):
        power=power@P
        if k in K_RANGE:
            nodes.append(np.diag(power)*16.)
            edges.append(power[pairs[:,0],pairs[:,1]]*16.)
    return np.stack(nodes,axis=1).astype('float32'),np.stack(edges,axis=1).astype('float32')

def load_csl():
    with zipfile.ZipFile(str(STUDY/'CSL.zip')) as z:
        matrices=pickle.loads(z.read('graphs_Kary_Deterministic_Graphs.pkl'))
        y=torch.load(io.BytesIO(z.read('y_Kary_Deterministic_Graphs.pt')),weights_only=True).numpy()
    graphs=[]
    for i,a in enumerate(matrices):
        a=a.tocoo();p=np.stack([a.row,a.col],axis=1);p=p[p[:,0]<p[:,1]];p=np.unique(p,axis=0)
        nf,ef=cycle_graph_features(a.shape[0],p)
        g=feature_graph(a.shape[0],p,np.column_stack([np.ones((a.shape[0],1),np.float32),nf]),edge=np.column_stack([np.ones((len(p),1),np.float32),ef]),gid=i)
        graphs.append(g)
    return graphs,np.asarray(y,dtype=np.int64)

def load_cox2():
    from block_feedback_experiment import load_data
    G,y=load_data('COX2')
    for g in G:
        nf,ef=cycle_graph_features(g['n'],g['pairs'])
        g['x']=np.column_stack([g['x'],nf]).astype(np.float32)
        g['edge']=np.column_stack([g['edge'][:,:-4],ef,g['edge'][:,-4:]]).astype(np.float32)
    return G,y

def load_brec_pairs(pair_ids):
    with zipfile.ZipFile(str(STUDY/'BREC_data_all.zip')) as z:
        arr=np.load(io.BytesIO(z.read('brec_v3.npy')),allow_pickle=False)
    out={}
    for i in pair_ids:
        gs=[];ys=[]
        for j in range(64):
            gg=nx.from_graph6_bytes(bytes(arr[i*64+j]).rstrip(b'\x00'))
            ps=np.asarray(sorted(tuple(sorted(e)) for e in gg.edges()),dtype=np.int64).reshape(-1,2)
            nf,ef=cycle_graph_features(len(gg),ps)
            gs.append(feature_graph(len(gg),ps,np.column_stack([np.ones((len(gg),1),np.float32),nf]),edge=np.column_stack([np.ones((len(ps),1),np.float32),ef]),gid=100000+i*64+j));ys.append(j%2)
        out[i]=(gs,np.array(ys))
    return out

class CyclePosetFeedback(MVFBlockFeedback):
    """Explicit hard generalized MVF blocks + differentiable vertex/edge cell diffusion.
    Hard connected block partitions are stop-gradient; all soft incidence weights,
    cell maps and feedback channels are trained through CE loss. In 1D, each partition is order convex.
    """
    def __init__(self,h,kind='learned',topk=1,hard=True,soft=True):
        super().__init__(h,kind=kind,topk=topk,feedback=True)
        self.hard_enabled=hard;self.soft_enabled=soft
        self.inc_gate=nn.Sequential(nn.Linear(2*h,h),nn.ReLU(),nn.Linear(h,1))
        self.edge_update=nn.Sequential(nn.Linear(2*h,h),nn.ReLU(),nn.Linear(h,h))
        self.soft_out=nn.Sequential(nn.Linear(2*h,h),nn.ReLU(),nn.Linear(h,h))
        self.soft_ln=nn.LayerNorm(h)
        self.soft_alpha=nn.Parameter(torch.tensor(-1.1))
        self.hard_alpha=nn.Parameter(torch.tensor(-1.1))
    def forward(self,h,b,e,mode='learned'):
        base=super(MVFBlockFeedback,self).forward(h,b,e,'learned')
        p=b['pairs'];u=p[:,0];v=p[:,1];n=len(base);m=len(p)
        a=torch.sigmoid(self.inc_gate(torch.cat([base[u],e],-1)).flatten())
        c=torch.sigmoid(self.inc_gate(torch.cat([base[v],e],-1)).flatten())
        # Valid score-to-incidence alignment control: permute by edge within each graph.
        if self.kind=='shuffled':
            perm=torch.empty(m,dtype=torch.long,device=base.device)
            for gi in range(b['ngraphs']):
                loc=(b['eg']==gi).nonzero().flatten();rr=np.random.default_rng(24691+int(b['ids'][gi]))
                pp=b['pairs'][loc].detach().cpu().numpy(); canonical=np.lexsort((np.max(pp,axis=1),np.min(pp,axis=1)))
                ordered=loc[torch.tensor(canonical,device=base.device)];reordered=ordered[torch.tensor(rr.permutation(len(loc)),device=base.device)]
                perm[ordered]=reordered
            a=a[perm];c=c[perm]
        elif self.kind=='uniform':
            # Unit capacities give the same edge-cell encoder, without learned affinity differences.
            a=torch.ones_like(a)*.5;c=torch.ones_like(c)*.5
        div=(a+c).clamp(min=1e-6)
        def diffusion(h0):
            ev=(a[:,None]*h0[u]+c[:,None]*h0[v])/div[:,None]
            ev=self.edge_update(torch.cat([e,ev],-1))
            nm=base.new_zeros((n,base.shape[-1]));tot=base.new_zeros(n)
            nm.index_add_(0,u,a[:,None]*ev);nm.index_add_(0,v,c[:,None]*ev)
            tot.index_add_(0,u,a);tot.index_add_(0,v,c)
            return nm/tot.clamp(min=1e-6)[:,None]
        s1=diffusion(base);s2=diffusion(s1)
        delta=0.
        if self.soft_enabled:
            out=self.soft_out(torch.cat([s1,s2],-1))
            delta=delta+torch.sigmoid(self.soft_alpha)*self.soft_ln(out)
        if self.hard_enabled:
            w=self.getw(base,b,e,'learned')[0]
            hard,_=self.field_pool(base,b,e,w)
            delta=delta+torch.sigmoid(self.hard_alpha)*self.feedback_norm(hard)
        return F.relu(base+delta)

class CyclePosetRouter(StructuralRouter):
    def __init__(self,dx,de,nclass,h=24,kind='learned',hard=True,soft=True,adaptive=False):
        super().__init__(dx,de,nclass,h=h,depth=2)
        self.layers=nn.ModuleList([CyclePosetFeedback(h,kind=kind,hard=hard,soft=soft) for _ in range(2)])
        if adaptive:
            for layer in self.layers:
                with torch.no_grad():
                    layer.soft_alpha.fill_(-3.0)
                    layer.hard_alpha.fill_(-0.5)

def make_model(mode,dx,de,nc,h):
    if mode=='gine_cycle':return PlainGINE(dx,de,nc,h=h)
    if mode=='struct_cycle':return StructuralRouter(dx,de,nc,h=h)
    if mode=='hard_cycle':return MVFRouter(dx,de,nc,h,kind='learned',topk=1,feedback=True)
    if mode=='poset_cycle':return CyclePosetRouter(dx,de,nc,h,'learned',True,True)
    if mode=='poset_shuffle':return CyclePosetRouter(dx,de,nc,h,'shuffled',True,True)
    if mode=='poset_uniform':return CyclePosetRouter(dx,de,nc,h,'uniform',True,True)
    if mode=='poset_adaptive':return CyclePosetRouter(dx,de,nc,h,'learned',True,True,adaptive=True)
    if mode=='poset_adaptive_shuffle':return CyclePosetRouter(dx,de,nc,h,'shuffled',True,True,adaptive=True)
    if mode=='poset_softonly':return CyclePosetRouter(dx,de,nc,h,'learned',False,True)
    if mode=='poset_hardonly':return CyclePosetRouter(dx,de,nc,h,'learned',True,False)
    raise ValueError(mode)

def check(G,y):
    sc=StandardScaler().fit(np.concatenate([v['x'] for v in G[:4]]));b=make_batch(G,[0,1,2,3],sc)
    setup_seed(890);net=make_model('poset_cycle',G[0]['x'].shape[1],G[0]['edge'].shape[1],int(max(y))+1,20)
    net.eval();out=net(b);out.square().mean().backward()
    grads={k:float(v.grad.abs().sum()) for k,v in net.named_parameters() if v.grad is not None and any(z in k for z in ['inc_gate.0.weight','cell.0.weight','scorer.0.weight','soft_out.0.weight'])}
    changed=copy.deepcopy(G[:4]);rng=np.random.default_rng(94)
    for g in changed:
        ids=rng.permutation(len(g['pairs']));g['pairs']=g['pairs'][ids][:,::-1].copy();g['edge']=g['edge'][ids].copy()
    with torch.no_grad():err=float((net(b)-net(make_batch(changed,[0,1,2,3],sc))).abs().max())
    assert err<2e-4,err
    assert all(v>0 for v in grads.values()),grads
    return {'invariance_max_abs':err,'grads':grads}

def cv(G,y,mode,seed=101,epochs=65,folds=3,split_seed=20261027,h=24,patience=12,budget=1100,out=ROOT/'runs'):
    y=np.asarray(y,int);ncl=int(y.max())+1;pred=np.full(len(G),-1,int);probs=np.zeros((len(G),ncl),np.float32)
    foldid=np.full(len(G),-1,int);times=[];start=time.time()
    for fold,(tr,te) in enumerate(StratifiedKFold(n_splits=folds,shuffle=True,random_state=split_seed).split(range(len(y)),y)):
        setup_seed(seed+fold*101)
        ai,vi=next(StratifiedShuffleSplit(n_splits=1,test_size=.18,random_state=seed+fold*31).split(tr,y[tr]));fit=tr[ai];val=tr[vi]
        sc=StandardScaler().fit(np.concatenate([G[i]['x'] for i in fit]));net=make_model(mode,G[0]['x'].shape[1],G[0]['edge'].shape[1],ncl,h)
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-4)
        cnt=np.bincount(y[fit],minlength=ncl);cw=torch.tensor(np.sqrt(len(fit)/np.maximum(cnt,1)),dtype=torch.float32);cw/=cw.mean()
        best=float('inf');best_ep=0;state=None;stale=0
        for ep in range(epochs):
            net.train()
            for ids in chunks(G,fit,budget,seed=seed*1000+fold*97+ep):
                logits=net(make_batch(G,ids,sc));loss=F.cross_entropy(logits,torch.tensor(y[ids]),weight=cw)
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
            net.eval();vloss=0.;nnn=0
            with torch.no_grad():
                for ids in chunks(G,val,budget):
                    zz=net(make_batch(G,ids,sc));l=F.cross_entropy(zz,torch.tensor(y[ids]),weight=cw);vloss+=float(l)*len(ids);nnn+=len(ids)
            vloss/=max(nnn,1)
            if vloss<best-1e-4:best=vloss;best_ep=ep+1;state=copy.deepcopy(net.state_dict());stale=0
            else:stale+=1
            if stale>=patience and ep>=15:break
        net.load_state_dict(state);net.eval()
        with torch.no_grad():
            for ids in chunks(G,te,budget):
                z=net(make_batch(G,ids,sc)).softmax(-1).numpy();pred[ids]=z.argmax(1);probs[ids]=z;foldid[ids]=fold
        scores=balanced_accuracy_score(y[te],pred[te]);times.append({'fold':fold,'BA':scores,'best_epoch':best_ep})
        print('FOLD',mode,seed,fold,round(100*scores,2),'best_ep',best_ep,'sec',round(time.time()-start,1),flush=True)
    metrics={'model':mode,'seed':seed,'BA':float(balanced_accuracy_score(y,pred)),'folds':times,'params':sum(p.numel() for p in net.parameters()),'seconds':time.time()-start,'epochs_max':epochs,'split_seed':split_seed,'n':len(y)}
    out=Path(out);out.mkdir(parents=True,exist_ok=True);stem=f'{mode}_s{seed}_{len(G)}'
    np.savez_compressed(out/(stem+'.npz'),y=y,pred=pred,probs=probs,fold=foldid);(out/(stem+'.json')).write_text(json.dumps(metrics,indent=2))
    return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['csl','brec','cox2'],default='csl');p.add_argument('--pairs',nargs='+',type=int,default=[160,265,278]);p.add_argument('--modes',nargs='+',default=['gine_cycle','hard_cycle','poset_cycle','poset_shuffle','poset_softonly']);p.add_argument('--seeds',nargs='+',type=int,default=[101]);p.add_argument('--epochs',type=int,default=65);p.add_argument('--folds',type=int,default=3);p.add_argument('--h',type=int,default=24);p.add_argument('--patience',type=int,default=12);args=p.parse_args()
    if args.dataset=='csl':
        G,y=load_csl();print('PREFLIGHT',json.dumps(check(G,y)),flush=True)
        for seed in args.seeds:
            for mode in args.modes:
                print('START',mode,seed,flush=True);print('RESULT',json.dumps(cv(G,y,mode,seed=seed,epochs=args.epochs,folds=args.folds,h=args.h,patience=args.patience)),flush=True)
    elif args.dataset=='cox2':
        G,y=load_cox2();print('COX2',len(G),flush=True)
        for seed in args.seeds:
            for mode in args.modes:
                print('RESULT',json.dumps(cv(G,y,mode,seed=seed,epochs=args.epochs,folds=args.folds,h=args.h,patience=args.patience,split_seed=20261030,out=ROOT/'cox2')),flush=True)
    else:
        for pi,(G,y) in load_brec_pairs(args.pairs).items():
            print('BREC',pi,flush=True)
            for seed in args.seeds:
                for mode in args.modes:
                    print('RESULT',pi,json.dumps(cv(G,y,mode,seed=seed,epochs=args.epochs,folds=2,h=args.h,patience=args.patience,split_seed=20261029,out=ROOT/f'brec{pi}')),flush=True)
