"""Controlled expressivity diagnostic: constant-feature single cycle vs two cycles.
This is a deliberately easy homology/component-count task, NOT evidence of
real-world superiority. Identical local node and edge features, graph size,
degree sequence and zero triangles/bridges make two local GINE layers 1-WL blind.
"""
import sys, argparse, json
from pathlib import Path
import networkx as nx, numpy as np
from block_feedback_experiment import feature_graph,cv,invariance_and_gradient_check
ROOT=Path(__file__).resolve().parent

def data(n_each=90,n=24,seed=18191):
    rng=np.random.default_rng(seed);G=[];y=[]
    for i in range(2*n_each):
        label=i%2
        g=nx.cycle_graph(n) if label==0 else nx.disjoint_union(nx.cycle_graph(n//2),nx.cycle_graph(n//2))
        perm=rng.permutation(n);p=sorted((min(int(perm[u]),int(perm[v])),max(int(perm[u]),int(perm[v]))) for u,v in g.edges())
        G.append(feature_graph(n,p,np.ones((n,1),np.float32),edge=np.zeros((len(p),1),np.float32),gid=i))
        y.append(label)
    return G,np.asarray(y,np.int64)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[101,202,303]);p.add_argument('--epochs',type=int,default=22);a=p.parse_args()
    G,y=data();print('SYN_DATA',len(y),'both have 24 nodes, 24 edges, constant features, all degrees 2; graph components (1,2)',flush=True)
    print('CHECK',invariance_and_gradient_check(G,y),flush=True)
    for seed in a.seeds:
        for name in ['gine','struct','mvf_feedback','mvf_shuffle','mvf_uniform','mvf_readout']:
            if (ROOT/'results'/f'SYNCYCLES_{name}_s{seed}.json').exists():continue
            r=cv(G,y,'SYNCYCLES',name,seed=seed,h=24,epochs=a.epochs,split_seed=20261101,out=ROOT/'results')
            print('DONE',name,'seed',seed,'BA',r['BA'],'params',r['params'],flush=True)
