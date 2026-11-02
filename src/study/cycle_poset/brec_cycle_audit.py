import sys,io,zipfile,numpy as np,networkx as nx,pandas as pd
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import cycle_graph_features
with zipfile.ZipFile(str(Path(__file__).resolve().parent.parent/'BREC_data_all.zip')) as z: a=np.load(io.BytesIO(z.read('brec_v3.npy')),allow_pickle=False)
rows=[]
for i in range(400):
 feats=[]
 for j in [0,1]:
  g=nx.from_graph6_bytes(bytes(a[64*i+j]).rstrip(b'\x00'))
  p=np.array(sorted(tuple(sorted(e)) for e in g.edges),int).reshape(-1,2)
  nf,ef=cycle_graph_features(len(g),p)
  feats.append(np.mean(nf,axis=0))
 d=np.abs(feats[0]-feats[1]); rows.append({'pair':i,'cycle_walk_dist':float(np.linalg.norm(d)),'walk_diff':bool((d>1e-5).any())})
f=pd.DataFrame(rows);f.to_csv(str(Path(__file__).resolve().parent/'brec_cycle_audit.csv'),index=False)
print('total distinct at k3..10',f.walk_diff.sum());print('selected',f[f.pair.isin([0,60,160,260,265,278,297,360,380])].to_string(index=False))
