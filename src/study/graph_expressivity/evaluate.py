"""Exploratory CSL training and BREC structural audit using recovered incidence/MVF code.
CSL loader strips stored self-loops; avoids node-ID features. BREC audit is NOT RPC.
"""
import os,io,zipfile,pickle, collections,sys,json,time
from pathlib import Path
os.environ['MVF_THREADS']='1'
import numpy as np, torch, networkx as nx, pandas as pd
from sklearn.model_selection import StratifiedKFold
ROOT=Path(__file__).resolve().parent
BASE=Path(os.environ.get('MVF_STUDY_ROOT',str(ROOT.parent)))
SRC=BASE/'block_feedback'
sys.path.insert(0,str(SRC))
from block_feedback_experiment import feature_graph,make_model,cv
from run_structural_suite import make_batch,setup_seed
from sklearn.preprocessing import StandardScaler

def load_csl():
    with zipfile.ZipFile(str(BASE/'CSL.zip')) as z:
      graph=pickle.loads(z.read('graphs_Kary_Deterministic_Graphs.pkl'))
      y=torch.load(io.BytesIO(z.read('y_Kary_Deterministic_Graphs.pt')),weights_only=True).numpy()
    G=[]
    for i,A in enumerate(graph):
      A=A.tocoo();pairs=np.stack([A.row,A.col],axis=1);pairs=pairs[pairs[:,0]<pairs[:,1]]
      pairs=np.unique(pairs,axis=0)
      g=feature_graph(A.shape[0],pairs,np.ones((A.shape[0],1),np.float32),edge=np.ones((len(pairs),1),np.float32),gid=i)
      G.append(g)
    return G,np.asarray(y,np.int64)

def audit_csl(G,y):
  rows=[]
  for i,g in enumerate(G):
    gg=nx.Graph();gg.add_nodes_from(range(g['n']));gg.add_edges_from(g['pairs'].tolist())
    deg=[d for _,d in gg.degree()]
    rows.append(dict(id=i,label=int(y[i]),n=g['n'],edges=len(g['pairs']),beta0=nx.number_connected_components(gg),beta1=len(g['pairs'])-g['n']+nx.number_connected_components(gg),triangles=sum(nx.triangles(gg).values())//3,degree_min=min(deg),degree_max=max(deg)))
  df=pd.DataFrame(rows);df.to_csv(ROOT/'CSL_graph_metadata.csv',index=False)
  print('CSL y',df.groupby('label').agg({'id':'count','triangles':['min','max'],'beta0':['min','max']}).to_string(),flush=True)
  sc=StandardScaler().fit(np.concatenate([x['x'] for x in G]));b=make_batch(G,[0,15,30,45,60,75,90,105,120,135],sc)
  res={}
  for mode in ['gine','struct','mvf_feedback','mvf_shuffle','mvf_uniform']:
    setup_seed(2026);m=make_model(mode,b['x'].shape[1],b['edge'].shape[1],10,24);m.eval()
    with torch.no_grad():
      pred=m(b).numpy()
      res[mode]=dict(max_abs_logit_spread=float(np.max(np.abs(pred-pred[0]))),max_adjacent_diff=float(np.max(np.abs(np.diff(pred,axis=0)))))
  (ROOT/'CSL_untrained_probe.json').write_text(json.dumps(res,indent=2));print('UNTRAINED PROBE',res,flush=True)
  return df

def audit_brec():
  with zipfile.ZipFile(str(BASE/'BREC_data_all.zip')) as z:
    ar=np.load(io.BytesIO(z.read('brec_v3.npy')),allow_pickle=False)
  assert len(ar)==51200
  def decode(i):return nx.from_graph6_bytes(bytes(ar[i]).rstrip(b'\x00'))
  # Dataset convention: first 400*64 elements are original 400 pairs; within each 64-block even/odd alternate isomorphic relabelings of the pair members.
  rows=[]
  for i in range(400):
    a,b=decode(i*64),decode(i*64+1)
    def detail(g):
      n=g.number_of_nodes();e=g.number_of_edges();c=nx.number_connected_components(g)
      d=tuple(sorted(dict(g.degree()).values()))
      tris=sum(nx.triangles(g).values())//3
      edges_tri=[]
      for u,v in g.edges: edges_tri.append(len(set(g[u]).intersection(g[v])))
      return dict(n=n,m=e,beta0=c,beta1=e-n+c,triangles=tris,degree_hist=str(tuple(sorted(collections.Counter(d).items()))),edge_triangle_hist=str(tuple(sorted(collections.Counter(edges_tri).items()))),bridge_count=len(list(nx.bridges(g))))
    x,y=detail(a),detail(b)
    row={'pair':i,'category':'Basic' if i<60 else 'Regular' if i<160 else 'Extension' if i<260 else 'CFI' if i<360 else '4Vertex' if i<380 else 'DistanceRegular'}
    for k in x:row[k+'_a']=x[k];row[k+'_b']=y[k];row['diff_'+k]=int(x[k]!=y[k])
    rows.append(row)
  df=pd.DataFrame(rows);df.to_csv(ROOT/'BREC_400_pair_topological_audit.csv',index=False)
  cols=[f'diff_{x}' for x in ['n','m','beta0','beta1','triangles','degree_hist','edge_triangle_hist','bridge_count']]
  summ=df.groupby('category')[cols].sum().astype(int)
  summ['pairs']=df.groupby('category').size();summ.loc['ALL']=df[cols].sum().astype(int).tolist()+[400]
  summ.to_csv(ROOT/'BREC_category_summary.csv')
  print('BREC AUDIT\n',summ.to_string(),flush=True)
  # Verify within-pair graph relabeling and the first reliability boundary.
  spot=[]
  for i in [0,60,160,260,360,380,399]:
    a=decode(i*64);aa=decode(i*64+2);b=decode(i*64+1);bb=decode(i*64+3)
    spot.append(dict(pair=i,even_isomorphic=nx.is_isomorphic(a,aa),odd_isomorphic=nx.is_isomorphic(b,bb)))
  (ROOT/'BREC_pair_integrity.json').write_text(json.dumps(spot,indent=2))
  return df

if __name__=='__main__':
  import argparse
  p=argparse.ArgumentParser();p.add_argument('task',choices=['audit','train']);p.add_argument('--seeds',nargs='+',type=int,default=[101]);p.add_argument('--models',nargs='+',default=['gine','struct','mvf_feedback','mvf_shuffle','mvf_uniform']);p.add_argument('--epochs',type=int,default=20);a=p.parse_args()
  G,y=load_csl();audit_csl(G,y)
  if a.task=='audit':audit_brec();sys.exit()
  for seed in a.seeds:
    for mode in a.models:
      print('TRAINING',mode,seed,flush=True)
      r=cv(G,y,'CSL',mode,seed=seed,h=24,epochs=a.epochs,split_seed=20261027,folds=5,budget=1100,patience=6,out=ROOT/'runs')
      print('RESULT',mode,seed,round(100*r['BA'],3),r['seconds'],flush=True)
