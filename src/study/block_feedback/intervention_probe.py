"""Paired same-checkpoint block feedback interventions on exact held-out graph IDs.
Trains the MVF model with identical schedule to block_feedback_experiment.py;
then disables block feedback or permutes score-defined block links WITHOUT retraining.
These interventions impose distribution shift; use as mechanism-dependence probes only.
"""
import json,copy,time,argparse
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from block_feedback_experiment import (ROOT,load_data,make_model,make_batch,chunks,setup_seed)
from synthetic_topology_test import data as synthetic

def run(dataset,seed=101,epochs=16,h=24,split_seed=20261101,budget=1500,patience=5):
 G,y=(synthetic() if dataset=='SYNCYCLES' else load_data(dataset))
 y=np.asarray(y,int);classes=int(y.max())+1
 pred={k:np.full(len(y),-1,dtype=int) for k in ('aligned','off','shuffled')}
 probs={k:np.zeros((len(y),classes),np.float32) for k in pred}
 fold_id=np.full(len(y),-1,int);log=[]
 for fold,(tr,te) in enumerate(StratifiedKFold(3,shuffle=True,random_state=split_seed).split(np.arange(len(y)),y)):
  setup_seed(seed+fold*101)
  ia,iv=next(StratifiedShuffleSplit(n_splits=1,test_size=.18,random_state=seed+fold*31).split(tr,y[tr]))
  fit=tr[ia];val=tr[iv];sc=StandardScaler().fit(np.concatenate([G[i]['x'] for i in fit]))
  m=make_model('mvf_feedback',G[0]['x'].shape[1],G[0]['edge'].shape[1],classes,h)
  opt=torch.optim.AdamW(m.parameters(),lr=.0008,weight_decay=1e-4)
  cnt=np.bincount(y[fit],minlength=classes);cw=torch.tensor(np.sqrt(len(fit)/np.maximum(1,cnt)),dtype=torch.float32);cw/=cw.mean()
  best=float('inf');state=None;be=0;stale=0
  for ep in range(epochs):
   m.train()
   for ids in chunks(G,fit,budget,seed=seed*1000+fold*97+ep):
    zz=m(make_batch(G,ids,sc));loss=F.cross_entropy(zz,torch.tensor(y[ids]),weight=cw)
    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2.0);opt.step()
   m.eval();vl=0;total=0
   with torch.no_grad():
    for ids in chunks(G,val,budget):
     l=F.cross_entropy(m(make_batch(G,ids,sc)),torch.tensor(y[ids]),weight=cw)
     vl+=float(l)*len(ids);total+=len(ids)
   vl/=max(1,total)
   if vl<best-1e-4:best=vl;state=copy.deepcopy(m.state_dict());be=ep+1;stale=0
   else:stale+=1
   if stale>=patience and ep>=9:break
  m.load_state_dict(state);m.eval()
  for name in pred:
   for layer in m.layers:
    layer.kind='shuffled' if name=='shuffled' else 'learned'
    if name=='off':layer.feedback_scale.data.fill_(-35.0)
   # Restore learned feedback_scale precisely for ordinary/shuffled conditions.
   if name!='off':
    for i,layer in enumerate(m.layers):layer.feedback_scale.data.copy_(state[f'layers.{i}.feedback_scale'])
   with torch.no_grad():
    for ids in chunks(G,te,budget):
     pr=m(make_batch(G,ids,sc)).softmax(-1).cpu().numpy()
     pred[name][ids]=pr.argmax(-1);probs[name][ids]=pr
  fold_id[te]=fold
  log.append({'fold':fold,'selected_epoch':be,'aligned_BA':float(balanced_accuracy_score(y[te],pred['aligned'][te]))})
  print('INTERVENTION',dataset,'seed',seed,'fold',fold,'scores',{k:round(100*balanced_accuracy_score(y[te],v[te]),2) for k,v in pred.items()},flush=True)
 ans={'dataset':dataset,'seed':seed,'epochs':epochs,'split_seed':split_seed,'n':len(y),'folds':log,'BA':{k:float(balanced_accuracy_score(y,v)) for k,v in pred.items()},'prediction_change_rate':{k:float(np.mean(pred[k]!=pred['aligned'])) for k in ('off','shuffled')},'mean_abs_p1_difference':{k:float(np.mean(np.abs(probs[k][:,1]-probs['aligned'][:,1]))) for k in ('off','shuffled')}}
 out=ROOT/'interventions';out.mkdir(exist_ok=True)
 (out/f'{dataset}_s{seed}.json').write_text(json.dumps(ans,indent=2))
 np.savez_compressed(out/f'{dataset}_s{seed}.npz',y=y,fold=fold_id,aligned=pred['aligned'],off=pred['off'],shuffled=pred['shuffled'],probs_aligned=probs['aligned'],probs_off=probs['off'],probs_shuffled=probs['shuffled'])
 print('FINISH',ans,flush=True)
 return ans

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',default='SYNCYCLES');p.add_argument('--seeds',type=int,nargs='+',default=[101,202,303]);p.add_argument('--epochs',type=int,default=16);a=p.parse_args()
 for seed in a.seeds:run(a.dataset,seed,epochs=a.epochs)
