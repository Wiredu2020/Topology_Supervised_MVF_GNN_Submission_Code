"""Graph incidence routing and multivector-learning experiments."""
import os,sys,json,hashlib,zipfile,time,math,traceback,urllib.request
from pathlib import Path
from collections import Counter
import numpy as np,pandas as pd,networkx as nx
from scipy.stats import wasserstein_distance
BASE=Path(__file__).resolve().parent
STUDY=BASE/'study'
DATA=BASE/'data';DATA.mkdir(parents=True,exist_ok=True)
OUT=BASE/'output';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(STUDY/'cycle_poset'))
sys.path.insert(0,str(STUDY/'block_feedback'))
from run import make_model,cycle_graph_features,cv,check
from block_feedback_experiment import feature_graph,load_tu_zip
from sklearn.metrics import balanced_accuracy_score
import torch
TU_URL='https://www.chrsmrrs.com/graphkerneldatasets/{}.zip'
NEW=['PTC_FM','PTC_FR','PTC_MM','AIDS','Mutagenicity']
# All are official small/medium TU graph classification datasets.
CACHED={'BZR','DHFR','NCI109','PROTEINS','ENZYMES','BACE'}


def ensure_dataset(name,can_download=True):
    if name=='BACE':
        p=DATA/'bace.csv'
        if not p.exists():raise FileNotFoundError('bace.csv was not bundled')
        return p
    p=DATA/f'{name}.zip'
    if p.exists() and zipfile.is_zipfile(p):return p
    # Kaggle's supplyed archive search as an offline fallback
    k=Path('/kaggle/input')
    if k.is_dir():
        hits=list(k.rglob(f'{name}.zip'))
        if hits:
            p.write_bytes(hits[0].read_bytes());return p
    if not can_download:
        raise FileNotFoundError(f'{name}.zip missing (enable Kaggle internet or provide it in the data directory)')
    url=TU_URL.format(name)
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=50) as r:
            payload=r.read()
        p.write_bytes(payload)
        if not zipfile.is_zipfile(p):raise IOError('Downloaded content is not a ZIP archive')
        print('DOWNLOADED',name,len(payload),'bytes',flush=True)
    except Exception:
        if p.exists():p.unlink()
        raise
    return p


def _rdkit_graphs():
    from rdkit import Chem
    D=pd.read_csv(DATA/'bace.csv');G=[];y=[]
    atoms=[1,5,6,7,8,9,15,16,17,35,53]
    hybs=[Chem.rdchem.HybridizationType.SP,Chem.rdchem.HybridizationType.SP2,Chem.rdchem.HybridizationType.SP3]
    bonds=[Chem.rdchem.BondType.SINGLE,Chem.rdchem.BondType.DOUBLE,Chem.rdchem.BondType.TRIPLE,Chem.rdchem.BondType.AROMATIC]
    bad=[]
    for i,(s,label) in enumerate(zip(D.SMILES,D.label)):
        mol=Chem.MolFromSmiles(str(s))
        if mol is None:
            bad.append(i);continue
        n=mol.GetNumAtoms();x=[];edges={}
        for a in mol.GetAtoms():
            x.append([float(a.GetAtomicNum()==z) for z in atoms]+[float(a.GetAtomicNum() not in atoms),float(a.GetDegree()),float(a.GetIsAromatic()),float(a.IsInRing()),float(a.GetFormalCharge()),float(a.GetTotalNumHs())]+[float(a.GetHybridization()==hy) for hy in hybs])
        for b in mol.GetBonds():
            u,v=sorted([b.GetBeginAtomIdx(),b.GetEndAtomIdx()])
            edges[(u,v)]=[float(b.GetBondType()==t) for t in bonds]+[float(b.GetIsConjugated()),float(b.IsInRing()),float(b.GetStereo()!=Chem.rdchem.BondStereo.STEREONONE)]
        pairs=np.array(sorted(edges),dtype=np.int64).reshape(-1,2)
        edge=np.array([edges[tuple(p)] for p in pairs],dtype=np.float32).reshape(len(pairs),7)
        G.append({'n':n,'pairs':pairs,'rawx':np.array(x,dtype=np.float32),'rawe':edge,'id':i})
        y.append(int(label))
    if bad:raise ValueError(f'BACE invalid SMILES rows {bad[:15]} -- do not silently discard examples')
    return G,np.array(y,dtype=int)


def _tu_graphs(name):
    p=ensure_dataset(name,can_download=False)
    raw,y=load_tu_zip(str(p),name);G=[];n_reordered=0;bad=0
    for z in raw:
        pairs=np.asarray(z['edges'],dtype=np.int64).reshape(-1,2)
        ea=np.asarray(z['edge_attr'],dtype=np.float32).reshape(len(pairs),-1)
        can=np.sort(pairs,axis=1)
        order=np.lexsort((can[:,1],can[:,0])) if len(can) else np.empty(0,int)
        n_reordered+=int(not np.array_equal(order,np.arange(len(can))))
        ps=can[order];ee=ea[order]
        if len(ps) and (len(np.unique(ps,axis=0))!=len(ps) or np.any(ps[:,0]==ps[:,1])):
            bad+=1
        xx=np.asarray(z['node'],np.float32)
        if xx.ndim!=2:raise ValueError('Malformed node tensor')
        G.append({'n':int(z['n']),'pairs':ps,'rawx':xx,'rawe':ee,'id':int(z['original_id'])})
    if bad:raise ValueError(f'{name}: duplicate or loop edges in {bad} graphs; confirm normalization before training')
    print('LOADED',name,'graphs',len(G),'edge-order realignments',n_reordered,flush=True)
    return G,np.asarray(y,int)


def load_raw(name):
    return _rdkit_graphs() if name=='BACE' else _tu_graphs(name)


def statistics(G,name):
    rows=[]
    for g in G:
        n=g['n'];pairs=g['pairs']; m=len(pairs)
        nxg=nx.Graph();nxg.add_nodes_from(range(n));nxg.add_edges_from(pairs.tolist())
        d=np.fromiter((t for _,t in nxg.degree()),float,count=n)
        cc=nx.number_connected_components(nxg)
        tri=sum(nx.triangles(nxg).values())/3.
        bridges=sum(1 for _ in nx.bridges(nxg))
        rows.append({'dataset':name,'id':g['id'],'n':n,'m':m,'beta0':cc,'beta1':m-n+cc,'bridge_fraction':bridges/max(1,m),'triangles':tri,'mean_degree':float(d.mean()),'leaf_fraction':float((d==1).mean()),'clustering':float(nx.average_clustering(nxg)),'largest_cc_fraction':float(len(max(nx.connected_components(nxg),key=len))/n)})
    return rows

AUDIT_COLS=['n','m','beta0','beta1','bridge_fraction','triangles','mean_degree','leaf_fraction','clustering','largest_cc_fraction']
WEIGHTS=np.array([.12,.08,.10,.18,.15,.08,.05,.08,.06,.10])

def distances(stats):
    data=pd.DataFrame(stats)
    ref=data[data.dataset=='BZR']
    if len(ref)!=405:print('WARNING: expected 405 BZR reference graphs; got',len(ref))
    rows=[]
    for name,group in data.groupby('dataset'):
        parts={}
        for c in AUDIT_COLS:
            x=ref[c].values;y=group[c].values
            scale=max(np.quantile(x,.9)-np.quantile(x,.1),.25 if c.startswith('beta') else .08)
            parts[c]=float(wasserstein_distance(x,y)/scale)
        weighted=float(sum(w*parts[c] for c,w in zip(AUDIT_COLS,WEIGHTS)))
        rows.append({'dataset':name,'graphs':len(group),'BZR_distance':weighted,**{f'd_{c}':v for c,v in parts.items()},**{f'mean_{c}':float(group[c].mean()) for c in AUDIT_COLS}})
    result=pd.DataFrame(rows).sort_values('BZR_distance')
    return result


def build_features(raw,name):
    """Same edge/node cycle-walk inputs for GINE and every MVF baseline."""
    G=[]
    from scipy import sparse
    for i,z in enumerate(raw):
        n=z['n'];p=z['pairs']
        if n<=150:
            nf,ef=cycle_graph_features(n,p)
        else:
            A=sparse.csr_matrix((np.ones(2*len(p)),(np.r_[p[:,0],p[:,1]],np.r_[p[:,1],p[:,0]])),shape=(n,n),dtype=np.float64)
            deg=np.asarray(A.sum(axis=1)).ravel();P=sparse.diags(1/np.maximum(deg,1))@A
            power=sparse.eye(n,format='csr',dtype=np.float64);nf=[];ef=[]
            for k in range(1,11):
                power=power@P
                if k>=3:
                    nf.append(power.diagonal()*16)
                    ef.append(0.5*(np.asarray(power[p[:,0],p[:,1]]).reshape(-1)+np.asarray(power[p[:,1],p[:,0]]).reshape(-1))*16)
            nf=np.stack(nf,axis=-1).astype(np.float32)
            ef=np.stack(ef,axis=-1).astype(np.float32) if len(p) else np.zeros((0,8),np.float32)
        x=np.column_stack([z['rawx'],nf]).astype(np.float32)
        ea=np.column_stack([z['rawe'],ef]).astype(np.float32)
        gg=feature_graph(n,p,x,edge=ea,gid=z['id'])
        if not np.array_equal(gg['pairs'],p):raise ValueError(f'{name}: edge attribute order mismatch for graph {i}')
        G.append(gg)
        if (i+1)%1000==0:print('PREPARED',name,i+1,flush=True)
    return G

MODES={'gine_cycle':40,'struct_cycle':24,'poset_cycle':24,'poset_shuffle':24,'poset_uniform':24,'poset_softonly':24}

def train_one(name,mode,seed,proto,cache):
    G,y=cache[name]
    out=OUT/'runs'/name;out.mkdir(parents=True,exist_ok=True)
    stem=f'{mode}_s{seed}_{len(G)}'
    path=out/(stem+'.json')
    if path.exists():
        prev=json.loads(path.read_text())
        if prev.get('epochs_max')==proto['epochs'] and prev.get('split_seed')==proto['split_seed'] and prev.get('n')==len(y):
            print('RESUME SKIP',name,mode,seed,flush=True)
            return prev
        raise RuntimeError(f'Conflicting previous experiment: {path}')
    info=cv(G,y,mode,seed=seed,epochs=proto['epochs'],folds=proto['folds'],split_seed=proto['split_seed'],h=MODES[mode],patience=proto['patience'],budget=proto['batch_node_budget'],out=out)
    info['dataset']=name
    path.write_text(json.dumps(info,indent=2))
    return info


def summarize():
    rows=[]
    for p in (OUT/'runs').glob('*/*.json'):
        try:
            j=json.loads(p.read_text())
            if 'BA' in j:rows.append({'dataset':p.parent.name,'model':j['model'],'seed':j['seed'],'balanced_accuracy':100*j['BA'],'seconds':j['seconds'],'params':j['params'],'epochs_max':j['epochs_max'],'split_seed':j['split_seed'],'folds':len(j['folds'])})
        except Exception as e:print('SUMMARY SKIP',p,e)
    if not rows:return pd.DataFrame()
    a=pd.DataFrame(rows).sort_values(['dataset','seed','model']);a.to_csv(OUT/'all_completed_runs.csv',index=False)
    summary=a.groupby(['dataset','model']).agg(n_seeds=('balanced_accuracy','count'),mean_BA=('balanced_accuracy','mean'),sd_BA=('balanced_accuracy','std'),parameters=('params','first')).reset_index()
    summary.to_csv(OUT/'model_summary.csv',index=False)
    return summary


def paired():
    r=OUT/'all_completed_runs.csv'
    if not r.exists():return pd.DataFrame()
    d=pd.read_csv(r);rows=[]
    for ds,g in d.groupby('dataset'):
        pv=g.pivot(index='seed',columns='model',values='balanced_accuracy')
        for a,b in [('poset_cycle','gine_cycle'),('poset_cycle','poset_shuffle'),('poset_cycle','poset_uniform'),('poset_cycle','struct_cycle'),('poset_cycle','poset_softonly')]:
            if a not in pv or b not in pv:continue
            dd=(pv[a]-pv[b]).dropna()
            if len(dd):rows.append(dict(dataset=ds,comparison=f'{a} - {b}',n_paired_seeds=len(dd),difference_pp=dd.mean(),sd_pp=dd.std(),per_seed=';'.join(f'{x:.3f}' for x in dd.values)))
    o=pd.DataFrame(rows);o.to_csv(OUT/'paired_effects.csv',index=False);return o


def dataset_health(name):
    G,y=load_raw(name)
    if len(np.unique(y))<2:raise ValueError(f'{name}: fewer than 2 target classes')
    if min(np.bincount(y))<6: raise ValueError(f'{name}: too few class examples for 3-fold evaluation')
    return G,y
