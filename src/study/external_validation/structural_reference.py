"""Additional fair control: prior (non-MVF-feedback) trainable structural incidence router.
Same dataset loader and exact cross-validation settings as the frozen study.
"""
from evaluate_new_datasets import raw_graphs,PROTO,ROOT,cv
from pathlib import Path
import sys, json
PLAN={'mode':'struct_cycle','datasets':['BZR','PROTEINS','ENZYMES'],'seeds':[101,202,303], 'h':24,'same_other_settings_as':'FROZEN_PROTOCOL.json'}
(ROOT/'STRUCTURAL_REFERENCE_PLAN.json').write_text(json.dumps(PLAN,indent=2))
for ds in PLAN['datasets']:
 print('LOADING',ds,flush=True)
 G,y,_=raw_graphs(ds)
 for seed in PLAN['seeds']:
  target=ROOT/'runs'/ds/f'struct_cycle_s{seed}_{len(G)}.json'
  if target.is_file():continue
  print('START',ds,'struct_cycle',seed,flush=True)
  r=cv(G,y,'struct_cycle',seed=seed,epochs=PROTO['max_epochs'],folds=PROTO['folds'],split_seed=PROTO['split_seed'],h=24,patience=PROTO['patience'],budget=1100,out=ROOT/'runs'/ds)
  print('DONE',ds,'struct_cycle',seed,round(r['BA']*100,3),flush=True)
