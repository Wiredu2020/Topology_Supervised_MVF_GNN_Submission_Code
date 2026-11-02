"""One predeclared outer fold, exact model/training routines from archived source.
A pilot, not a complete repeated-CV validation. One test fold is accessed only once per architecture.
"""
import os, sys, json, time, argparse, copy
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score,accuracy_score
from new_validation import get, MODES, BASE
sys.path.insert(0,str(BASE.parent/'cycle_poset'))
from run_asymmetric_reference import make_model, chunks, make_batch,setup_seed

def fit(name,mode,seed=101,fold=0,epochs=36):
 out=BASE/'pilot_3fold'/name;out.mkdir(parents=True,exist_ok=True)
 stem=f'{mode}_seed{seed}_fold{fold}_epochs{epochs}'
 if (out/(stem+'.json')).exists():
  print('EXISTING',out/stem);return
 G,y,info=get(name);ncl=int(y.max())+1
 spl=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=20261027).split(np.arange(len(y)),y))
 tr,te=spl[fold]; ia,vi=next(StratifiedShuffleSplit(n_splits=1,test_size=.18,random_state=seed+fold*31).split(tr,y[tr]));fitidx=tr[ia];validx=tr[vi]
 assert len(set(te)&set(fitidx))==0 and len(set(te)&set(validx))==0
 setup_seed(seed+fold*101)
 sc=StandardScaler().fit(np.concatenate([G[i]['x'] for i in fitidx]));net=make_model(mode,G[0]['x'].shape[1],G[0]['edge'].shape[1],ncl,MODES[mode]);optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-4)
 ct=np.bincount(y[fitidx],minlength=ncl);cw=torch.tensor(np.sqrt(len(fitidx)/np.maximum(ct,1)),dtype=torch.float32);cw/=cw.mean()
 best=float('inf');state=None;bestep=0;stale=0;start=time.time();history=[]
 for ep in range(epochs):
  net.train();trainloss=[]
  for ids in chunks(G,fitidx,1100,seed=seed*1000+fold*97+ep):
   logits=net(make_batch(G,ids,sc));loss=F.cross_entropy(logits,torch.tensor(y[ids]),weight=cw)
   optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2.);optimizer.step();trainloss.append(float(loss))
  net.eval();vloss=0;nv=0
  with torch.no_grad():
   for ids in chunks(G,validx,1100):
    logits=net(make_batch(G,ids,sc));lv=F.cross_entropy(logits,torch.tensor(y[ids]),weight=cw);vloss+=float(lv)*len(ids);nv+=len(ids)
  vloss/=max(1,nv);history.append([ep+1,float(np.mean(trainloss)),vloss]);
  if vloss<best-1e-4:best=vloss;bestep=ep+1;state=copy.deepcopy(net.state_dict());stale=0
  else:stale+=1
  if (ep+1)%8==0:print('EPOCH',name,mode,'seed',seed,ep+1,'val_loss',round(vloss,4),'best_ep',bestep, 'secs',round(time.time()-start,1),flush=True)
  if stale>=10 and ep>=15:break
 net.load_state_dict(state);net.eval();probs=np.zeros((len(te),ncl),np.float32);pred=np.zeros(len(te),int);cursor=0
 with torch.no_grad():
  for ids in chunks(G,te,1100):
   zz=net(make_batch(G,ids,sc)).softmax(-1).numpy()
   # chunks may preserve te order in inference (seed None); use per-ID robustly
   mapping={int(v):j for j,v in enumerate(te)}
   for i,j in enumerate(ids):probs[mapping[int(j)]]=zz[i]
 pred=probs.argmax(1)
 metrics={'dataset':name,'model':mode,'seed':seed,'fold':fold,'max_epochs':epochs,'n_train':len(fitidx),'n_val':len(validx),'n_test':len(te),'balanced_accuracy':float(balanced_accuracy_score(y[te],pred)),'accuracy':float(accuracy_score(y[te],pred)),'best_epoch':bestep,'fit_seconds':round(time.time()-start,2),'parameters':sum(p.numel() for p in net.parameters()),'class_counts':np.bincount(y[te],minlength=ncl).tolist(),'label_features':'feature construction deterministic and label independent'}
 np.savez_compressed(out/(stem+'.npz'),ids=te,y=y[te],pred=pred,probs=probs,train=fitidx,val=validx)
 (out/(stem+'.json')).write_text(json.dumps(metrics,indent=2));(out/(stem+'_learning_curve.json')).write_text(json.dumps(history))
 print('DONE',json.dumps(metrics),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('name',choices=['BACE','DHFR','NCI109']);ap.add_argument('mode',choices=list(MODES));ap.add_argument('--seed',type=int,default=101);ap.add_argument('--fold',type=int,default=0);ap.add_argument('--epochs',type=int,default=36)
 x=ap.parse_args();fit(x.name,x.mode,x.seed,x.fold,x.epochs)
