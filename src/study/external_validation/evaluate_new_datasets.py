"""Graph incidence routing and multivector-learning experiments."""
from pathlib import Path
import json,sys, time,hashlib, zipfile, os
import numpy as np
ROOT=Path(__file__).resolve().parent
STUDY=ROOT.parent  # portable: package root contains original source and dataset archives
sys.path[:0]=[str(STUDY/'cycle_poset'),str(STUDY/'block_feedback')]
from run_asymmetric_reference import cycle_graph_features,cv,check
from block_feedback_experiment import feature_graph, load_tu_zip

PROTO={'datasets':['BZR','PROTEINS','ENZYMES'],
 'models':['gine_cycle','poset_cycle','poset_shuffle','poset_uniform','poset_softonly'],
 'seeds':[101,202,303], 'folds':3, 'split_seed':20261027,
 'max_epochs':36, 'patience':10, 'val_fraction':.18,
 'h':{'gine_cycle':40,'poset_cycle':24,'poset_shuffle':24,'poset_uniform':24,'poset_softonly':24},
 'lr':.001,'weight_decay':.0001,'node_scaling':'fit_only',
 'cycle_features':'random-walk returns P^k k3..10, factor16; SAME for all models',
 'metric':'pooled out-of-fold balanced accuracy, matched splits and seeds',
 'notes':'Use no model selection from test predictions. Inspect first all model-family results. Replication across unseen graph families is still development because model defined following CSL/COX2/BREC.'}

def raw_graphs(name):
 path=STUDY.parent/'data'/f'{name}.zip'
 raw,y=load_tu_zip(str(path),name)
 graphs=[]; stats={'mismatched_attr_order':0,'isolates':0,'max_n':0,'max_m':0,'self_loops':0,'duplicates':0,'n':len(y),'counts':{str(int(k)):int(v) for k,v in zip(*np.unique(y,return_counts=True))}, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 for gr in raw:
  p=np.asarray(gr['edges'],dtype=np.int64).reshape(-1,2)
  ea=np.asarray(gr['edge_attr'],dtype=np.float32).reshape(len(p),-1)
  # Sorted endpoint representation and aligned attributes--always move together.
  can=np.sort(p,axis=1)
  order=np.lexsort((can[:,1],can[:,0])) if len(p) else np.empty(0,dtype=int)
  stats['mismatched_attr_order']+=int(not np.array_equal(order,np.arange(len(p))))
  stats['duplicates']+=int(len(np.unique(can,axis=0))!=len(p))
  p=can[order];ea=ea[order]
  n=gr['n'];stats['max_n']=max(stats['max_n'],n);stats['max_m']=max(stats['max_m'],len(p));stats['isolates']+=int(len(np.unique(p.flatten()))<n)
  nf,ef=cycle_graph_features(n,p)
  x=np.column_stack([np.asarray(gr['node'],np.float32),nf])
  ee=np.column_stack([ea,ef])
  graphs.append(feature_graph(n,p,x,edge=ee,gid=gr['original_id']))
  # feature_graph canonicalizes pairs; verify that edge feature alignment is not accidentally changed.
  assert np.array_equal(graphs[-1]['pairs'],p),('pair/order',name,gr['original_id'])
 return graphs,y,stats

def main():
 (ROOT/'FROZEN_PROTOCOL.json').write_text(json.dumps(PROTO,indent=2))
 for name in PROTO['datasets']:
  t=time.time();print('LOADING',name,flush=True)
  G,y,st=raw_graphs(name)
  (ROOT/f'{name}_integrity.json').write_text(json.dumps(st,indent=2))
  print('AUDIT',name,json.dumps(st), 'sec',round(time.time()-t,1),flush=True)
  try:
   print('CHECK',name,json.dumps(check(G,y)),flush=True)
  except Exception as exc:
   print('FAILED_PREFLIGHT',name,repr(exc),flush=True);raise
  for seed in PROTO['seeds']:
   for mode in PROTO['models']:
    f=ROOT/'runs'/name/f'{mode}_s{seed}_{len(G)}.json'
    if f.is_file():
     print('EXISTING',name,mode,seed,flush=True);continue
    print('START',name,mode,seed,flush=True)
    try:
     result=cv(G,y,mode,seed=seed,epochs=PROTO['max_epochs'],folds=PROTO['folds'],split_seed=PROTO['split_seed'],h=PROTO['h'][mode],patience=PROTO['patience'],budget=1100,out=ROOT/'runs'/name)
     print('DONE',name,mode,seed,round(result['BA']*100,3), 'sec',round(result['seconds'],1),flush=True)
    except Exception as exc:
     print('ERROR',name,mode,seed,repr(exc),flush=True);raise
 print('ALL_COMPLETE',flush=True)

if __name__=='__main__':main()
