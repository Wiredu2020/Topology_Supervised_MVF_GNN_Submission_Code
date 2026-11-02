from screen_200 import data,configs,MVFRouter
from run_structural_suite import make_batch,setup_seed
import torch,copy,numpy as np,json
from sklearn.preprocessing import StandardScaler
G,y=data('PTC_MR')
sc=StandardScaler().fit(np.concatenate([g['x'] for g in G[:8]]))
b=make_batch(G,list(range(8)),sc)
ans=[]
for idx in (0,38,78,101,138,169,188,199):
 setup_seed(123);mm=MVFRouter(G[0]['x'].shape[1],G[0]['edge'].shape[1],2,24,configs[idx]);mm.eval()
 with torch.enable_grad():
  z=mm(b);z.square().sum().backward()
 g=max(float(l.scorer[-1].weight.grad.abs().max()) for l in mm.layers)
 G2=copy.deepcopy(G[:8])
 for gg in G2:
  p=np.random.default_rng(42).permutation(len(gg['pairs']))
  gg['pairs']=gg['pairs'][p][:,::-1].copy();gg['edge']=gg['edge'][p].copy()
 with torch.no_grad():delta=float((mm(b)-mm(make_batch(G2,list(range(8)),sc))).abs().max())
 ans.append(dict(id=idx,config=configs[idx],max_scorer_gradient=g,max_edge_orientation_difference=delta))
 print(idx,'grad',g,'orientation_diff',delta)
open(str(__import__('pathlib').Path(__file__).parent/'screen_checks.json'),'w').write(json.dumps(ans,indent=2))
