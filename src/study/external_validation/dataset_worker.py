from evaluate_new_datasets import raw_graphs,PROTO,ROOT,cv,check
from pathlib import Path
import json,sys,time
name=sys.argv[1]
assert name in PROTO['datasets']
G,y,st=raw_graphs(name)
(ROOT/f'{name}_integrity.json').write_text(json.dumps(st,indent=2));print('AUDIT',name,st,flush=True)
print('PREFLIGHT',name,json.dumps(check(G,y)),flush=True)
for seed in PROTO['seeds']:
 for mode in PROTO['models']:
  out=ROOT/'runs'/name
  path=out/f'{mode}_s{seed}_{len(G)}.json'
  if path.is_file():print('EXISTING',name,mode,seed,flush=True);continue
  print('START',name,mode,seed,flush=True)
  r=cv(G,y,mode,seed=seed,epochs=PROTO['max_epochs'],folds=PROTO['folds'],split_seed=PROTO['split_seed'],h=PROTO['h'][mode],patience=PROTO['patience'],budget=1100,out=out)
  print('DONE',name,mode,seed,round(r['BA']*100,3),round(r['seconds'],1),flush=True)
print('DATASET_COMPLETE',name,flush=True)
