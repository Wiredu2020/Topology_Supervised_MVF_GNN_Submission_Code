"""Graph incidence routing and multivector-learning experiments."""
import sys, os, json, csv, argparse, hashlib, time, random
from pathlib import Path
import numpy as np, torch, networkx as nx
from rdkit import Chem
from collections import Counter
BASE=Path(__file__).resolve().parent; DATA=BASE.parents[1]/'data'
sys.path.insert(0,str(BASE.parent/'cycle_poset'))
from run_asymmetric_reference import cycle_graph_features, cv,check
from block_feedback_experiment import feature_graph, load_tu_zip
SEEDS=[101,202,303]
MODES={'gine_cycle':40,'poset_cycle':24,'poset_shuffle':24,'poset_uniform':24,'poset_softonly':24}
PROTO={'datasets':['DHFR','BACE','NCI109'],'seeds':SEEDS,'models':MODES,'folds':3,'split_seed':20261027,'max_epochs':36,'patience':10,'batch_node_budget':1100,'optimizer':'AdamW(lr=0.001,wd=0.0001)','cycle_features':'random walk length 3..10 factor16 for all architectures','metric':'pooled out-of-fold balanced accuracy','status':'protocol saved before fitting any new dataset','BACE_atom_features':'atomic number onehot + atom degree + aromatic/ring/formal charge/H count/hybridization','BACE_bond_features':'type, conjugation, ring, stereo','notes':'Datasets chosen based on topology vs BZR, no adjustment based on labels. New data not untouched after model selection. Include all negative results.'}

def smiles_graphs():
 import pandas as pd
 D=pd.read_csv(DATA/'bace.csv');G=[]; y=[]; bad=[]
 atoms=[1,5,6,7,8,9,15,16,17,35,53]
 hybs=[Chem.rdchem.HybridizationType.SP,Chem.rdchem.HybridizationType.SP2,Chem.rdchem.HybridizationType.SP3]
 bonds=[Chem.rdchem.BondType.SINGLE,Chem.rdchem.BondType.DOUBLE,Chem.rdchem.BondType.TRIPLE,Chem.rdchem.BondType.AROMATIC]
 for i,(s,label) in enumerate(zip(D.SMILES,D.label)):
  mol=Chem.MolFromSmiles(s)
  if mol is None:bad.append(i);continue
  n=mol.GetNumAtoms()
  x=[]
  for a in mol.GetAtoms():
   x.append([float(a.GetAtomicNum()==z) for z in atoms]+[float(a.GetAtomicNum() not in atoms),float(a.GetDegree()),float(a.GetIsAromatic()),float(a.IsInRing()),float(a.GetFormalCharge()),float(a.GetTotalNumHs())]+[float(a.GetHybridization()==hyb) for hyb in hybs])
  emap={}
  for b in mol.GetBonds():
   u=b.GetBeginAtomIdx();v=b.GetEndAtomIdx();e=tuple(sorted((u,v)))
   emap[e]=[float(b.GetBondType()==t) for t in bonds]+[float(b.GetIsConjugated()),float(b.IsInRing()),float(b.GetStereo()!=Chem.rdchem.BondStereo.STEREONONE)]
  pairs=np.asarray(sorted(emap),dtype=np.int64).reshape(-1,2)
  edge=np.asarray([emap[tuple(k)] for k in pairs],np.float32).reshape(len(pairs),7)
  nf,ef=cycle_graph_features(n,pairs)
  X=np.column_stack([np.asarray(x,np.float32),nf]).astype(np.float32)
  E=np.column_stack([edge,ef]).astype(np.float32)
  G.append(feature_graph(n,pairs,X,edge=E,gid=i)); y.append(int(label))
 return G,np.asarray(y,dtype=np.int64),{'invalid_smiles':bad,'source':'bace.csv','atom_feature_dim':len(x[0]) if x else None,'edge_feature_dim':7}

def tu_graphs(name):
 p=DATA/('DHFR (1).zip' if name=='DHFR' else 'NCI109 (2).zip' if name=='NCI109' else 'BZR (1).zip')
 raw,y=load_tu_zip(str(p),name);G=[];unsorted=0;dup=0;selfloops=0
 for item in raw:
  ps=np.asarray(item['edges'],np.int64).reshape(-1,2)
  ea=np.asarray(item['edge_attr'],np.float32).reshape(len(ps),-1)
  can=np.sort(ps,axis=1)
  order=np.lexsort((can[:,1],can[:,0])) if len(ps) else np.empty(0,dtype=int)
  unsorted+=int(not np.array_equal(order,np.arange(len(ps))))
  dup+=int(len(np.unique(can,axis=0))!=len(ps)); selfloops+=int(np.any(ps[:,0]==ps[:,1]))
  ps=can[order];ea=ea[order]
  nf,ef=cycle_graph_features(item['n'],ps)
  xx=np.column_stack([np.asarray(item['node'],np.float32),nf])
  ee=np.column_stack([ea,ef])
  g=feature_graph(item['n'],ps,xx,edge=ee,gid=item['original_id'])
  assert np.array_equal(g['pairs'],ps)
  G.append(g)
 return G,np.asarray(y,int),{'source':str(p.name),'edge_sort_was_nontrivial_graphs':unsorted,'duplicate_graphs':dup,'selfloop_graphs':selfloops}

def get(name):
 if name=='BACE': return smiles_graphs()
 return tu_graphs(name)

def audit_one(name,raw_stats):
 vals=[]
 for g in raw_stats:
  n=g['n'];p=g['pairs'];gg=nx.Graph();gg.add_nodes_from(range(n));gg.add_edges_from(p.tolist())
  m=len(p);c=nx.number_connected_components(gg)
  deg=np.asarray([d for _,d in gg.degree()],dtype=float)
  br=list(nx.bridges(gg));tri=sum(nx.triangles(gg).values())/3
  vals.append({'dataset':name,'n':n,'m':m,'beta0':c,'beta1':m-n+c,'mean_degree':float(deg.mean()),'max_degree':float(deg.max()),'bridge_fraction':len(br)/max(1,m),'triangle_count':tri,'clustering':nx.average_clustering(gg),'largest_cc_frac':len(max(nx.connected_components(gg),key=len))/n,'edge_density':2*m/max(n*(n-1),1)})
 return vals

def audit():
 allrows=[];stat={};target=None
 for name in ['BZR','DHFR','BACE','NCI109']:
  G,y,meta=get(name); rr=audit_one(name,G);allrows+=rr
  st={'graphs':len(G),'classes':dict(Counter(map(int,y))),'features':{'nodes':int(G[0]['x'].shape[1]),'edges':int(G[0]['edge'].shape[1])},'source_info':meta}
  for feat in ['n','m','beta0','beta1','mean_degree','bridge_fraction','triangle_count','clustering','largest_cc_frac','edge_density']:
   vv=np.array([r[feat] for r in rr]);st[feat]={'mean':round(float(vv.mean()),4),'median':round(float(np.median(vv)),4),'q10':round(float(np.quantile(vv,.1)),4),'q90':round(float(np.quantile(vv,.9)),4)}
  stat[name]=st
  print('AUDIT',name,'n',len(G),'avg|V|',st['n']['mean'],'avg|E|',st['m']['mean'],'beta1',st['beta1']['mean'],'components',st['beta0']['mean'],'bridges',st['bridge_fraction']['mean'],flush=True)
 with open(BASE/'topology_per_graph.csv','w') as f:
  wr=csv.DictWriter(f,fieldnames=allrows[0].keys());wr.writeheader();wr.writerows(allrows)
 (BASE/'topology_summary.json').write_text(json.dumps(stat,indent=2))
 return stat

def train(name,mode,seed):
 assert name in PROTO['datasets'] and mode in MODES and seed in SEEDS
 G,y,meta=get(name)
 respath=BASE/'runs'/name/(f'{mode}_s{seed}_{len(G)}.json')
 if respath.exists():
  print('EXISTING',name,mode,seed,respath,flush=True);return json.loads(respath.read_text())
 print('START',name,mode,seed,'graphs',len(G),flush=True)
 if name!='NCI109' or mode=='poset_cycle':
  chk=check(G[:4],y[:4]);print('MODEL_CHECK',name,mode,chk,flush=True)
 res=cv(G,y,mode,seed=seed,epochs=PROTO['max_epochs'],folds=PROTO['folds'],split_seed=PROTO['split_seed'],h=MODES[mode],patience=PROTO['patience'],budget=PROTO['batch_node_budget'],out=BASE/'runs'/name)
 print('DONE',name,mode,seed,'BA',res['BA']*100,'seconds',res['seconds'],flush=True)
 return res

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['audit','train']);ap.add_argument('--name');ap.add_argument('--mode');ap.add_argument('--seed',type=int,default=101);x=ap.parse_args()
 (BASE/'FROZEN_PROTOCOL.json').write_text(json.dumps(PROTO,indent=2))
 if x.action=='audit':audit()
 else:train(x.name,x.mode,x.seed)
