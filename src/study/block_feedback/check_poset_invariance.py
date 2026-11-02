import json,copy
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
from block_feedback_experiment import (ROOT,load_data,make_model,make_batch,topk_incidence,connected_labels)

def check(dataset):
 G,y=load_data(dataset);sc=StandardScaler().fit(np.concatenate([g['x'] for g in G[:4]]))
 b=make_batch(G,np.arange(4),sc)
 torch.manual_seed(712);m=make_model('mvf_feedback',b['x'].shape[1],b['edge'].shape[1],2,24);m.eval()
 altered=copy.deepcopy(G[:4])
 rng=np.random.default_rng(33)
 for g in altered:
  n=g['n'];perm=rng.permutation(n);xx=np.empty_like(g['x']);xx[perm]=g['x'];g['x']=xx
  p=perm[g['pairs']];p.sort(axis=1)
  order=np.lexsort((p[:,1],p[:,0]));g['pairs']=p[order];g['edge']=g['edge'][order]
 with torch.no_grad():
  orig=m(b);alter=m(make_batch(altered,np.arange(4),sc))
 perm_error=float((orig-alter).abs().max())
 # Isolate the field_pool pathway; unlike whole-model gradients, this proves
 # soft edge-cell weights alone backpropagate into the incidence scorer.
 m.zero_grad(set_to_none=True)
 h=m.node(b['x']);e=m.edge(b['edge']);layer=m.layers[0]
 ww=layer.getw(h,b,e,'learned')[0]
 signal,_=layer.field_pool(h,b,e,ww)
 signal.square().sum().backward()
 block_only_scorer_grad=float(layer.scorer[-1].weight.grad.abs().sum())
 # exact combinatorial partition coverage and automatic convexity
 n=5;p=np.array([[0,1],[1,2],[2,3],[3,4]],int)
 q=np.array([.7,.2,.8,.4]);ac=topk_incidence(n,p,q,1)
 labels,k=connected_labels(n,p,ac)
 assert len(labels)==n+len(p) and set(labels)==set(range(k))
 return {'dataset':dataset,'permutation_max_abs_logit':perm_error,'partition_example_cells':len(labels),'partition_example_multivectors':k,'poset_height':2,'block_only_scorer_gradient_abs_sum':block_only_scorer_grad}

if __name__=='__main__':
 out={k:check(k) for k in ['PTC_MR','COX2']}
 (ROOT/'relabeling_checks.json').write_text(json.dumps(out,indent=2))
 print(json.dumps(out,indent=2))
