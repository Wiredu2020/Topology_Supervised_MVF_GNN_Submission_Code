import sys,json,numpy as np,torch,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import load_csl,make_model,make_batch,StandardScaler,setup_seed
G,y=load_csl();old=copy.deepcopy(G[:3]);new=copy.deepcopy(old);rng=np.random.default_rng(1781)
for g in new:
 n=g['n'];pi=rng.permutation(n);xx=np.zeros_like(g['x']);xx[pi]=g['x'];g['x']=xx
 g['pairs']=pi[g['pairs']][:,::-1].copy()
 idx=rng.permutation(len(g['pairs']));g['pairs']=g['pairs'][idx];g['edge']=g['edge'][idx]
sc=StandardScaler().fit(np.concatenate([v['x'] for v in G[:40]]))
b=make_batch(old,list(range(3)),sc);bnew=make_batch(new,list(range(3)),sc)
for m in ['gine_cycle','hard_cycle','poset_cycle','poset_softonly','poset_uniform']:
 setup_seed(555);net=make_model(m,G[0]['x'].shape[1],G[0]['edge'].shape[1],10,24);net.eval()
 with torch.no_grad():d=float((net(b)-net(bnew)).abs().max())
 print(m,d,flush=True)
