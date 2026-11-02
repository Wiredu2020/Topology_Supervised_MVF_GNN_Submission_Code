"""Post-screen exploratory PTC_MR holdout follow-up; same archived dataset previously inspected.
Selections from 4-epoch validation ranking predeclared: 38,138,188.
All candidates and comparator GINE24, GINE30, and original StructuralRouter24 trained 3 seeds.
Test graphs are disjoint from the screen's fit and validation graphs, but are
NOT independent of prior years of methods development on PTC_MR.
"""
from pathlib import Path
import sys,json,copy,csv,time,os,argparse
sys.path.insert(0,str(Path(__file__).parent))
from screen_200 import data, configs, MVFRouter
from run_structural_suite import PlainGINE, StructuralRouter, make_batch, chunks, setup_seed
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
ROOT=Path(__file__).parent

def train_one(G,y,split,typ,seed,epochs=24):
    fit=np.array(split['fit_ids']);val=np.array(split['val_ids']);test=np.array(split['sealed_test_ids'])
    setup_seed(seed)
    sc=StandardScaler().fit(np.concatenate([G[i]['x'] for i in fit]))
    dx,de=G[0]['x'].shape[1],G[0]['edge'].shape[1];nc=y.max()+1
    if typ=='GINE24':model=PlainGINE(dx,de,nc,h=24);mode=None
    elif typ=='GINE30':model=PlainGINE(dx,de,nc,h=30);mode=None
    elif typ=='ORIG_STRUCT24':model=StructuralRouter(dx,de,nc,h=24);mode='learned'
    else:model=MVFRouter(dx,de,nc,24,configs[int(typ)]);mode=None
    opt=torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=1e-4)
    count=np.bincount(y[fit],minlength=nc);w=torch.tensor(np.sqrt(len(fit)/np.maximum(1,count)),dtype=torch.float32);w/=w.mean()
    best=np.inf;state=None;stop=0;be=0;t0=time.time()
    for ep in range(epochs):
        model.train()
        for ids in chunks(G,fit,1600,seed=seed*137+ep):
            inp=make_batch(G,ids,sc)
            z=model(inp,mode);loss=F.cross_entropy(z,torch.tensor(y[ids]),weight=w)
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
        model.eval();va=0.;nt=0
        with torch.no_grad():
            for ids in chunks(G,val,1600):
                p=model(make_batch(G,ids,sc),mode).softmax(-1)
                va+=F.nll_loss(p.clamp(min=1e-8).log(),torch.tensor(y[ids]),weight=w).item()*len(ids);nt+=len(ids)
        va/=max(1,nt)
        if va<best-1e-5:best=va;be=ep+1;state=copy.deepcopy(model.state_dict());stop=0
        else:stop+=1
        if stop>=6 and ep>=10:break
    model.load_state_dict(state);model.eval();pr=[];ys=[];ids_seen=[]
    with torch.no_grad():
        for ids in chunks(G,test,1600):
            prob=model(make_batch(G,ids,sc),mode).softmax(-1).numpy()
            pr.extend(prob.tolist());ys.extend(y[ids].tolist());ids_seen.extend(list(ids))
    yp=np.argmax(pr,axis=1);ba=balanced_accuracy_score(ys,yp)
    params=sum(p.numel() for p in model.parameters())
    return {'model':str(typ),'seed':seed,'test_BA':float(ba),'best_epoch':be,'parameters':params,'runtime_seconds':time.time()-t0},np.array(ids_seen),np.array(ys),np.asarray(pr)

def main():
    G,y=data('PTC_MR');split=json.loads((ROOT/'screen_splits.json').read_text())
    out=ROOT/'followup_holdout';out.mkdir(exist_ok=True)
    models=('GINE24','GINE30','ORIG_STRUCT24','38','138','188')
    for typ in models:
        for seed in (2027,3030,4040):
            fn=out/f'{typ}_{seed}.json'
            if fn.exists():continue
            row,ids,ys,pr=train_one(G,y,split,typ,seed)
            fn.write_text(json.dumps(row,indent=2))
            np.savez_compressed(out/f'{typ}_{seed}.npz',id=ids,label=ys,prob=pr)
            print('DONE',typ,seed,'BA',round(row['test_BA']*100,2),'params',row['parameters'],'epoch',row['best_epoch'],flush=True)
    dat=[json.loads(z.read_text()) for z in out.glob('*.json')]
    import pandas as pd
    df=pd.DataFrame(dat).sort_values(['model','seed'])
    df.to_csv(out/'all_results.csv',index=False)
    summary=df.groupby('model')['test_BA'].agg(['count','mean','std']).sort_values('mean',ascending=False)
    summary.to_csv(out/'summary.csv');print(summary.to_string())
if __name__=='__main__': main()
