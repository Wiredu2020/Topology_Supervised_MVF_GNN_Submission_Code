"""NON-OFFICIAL BREC paired-permutation classifier pilot (not the RPC benchmark)."""
import io,os,sys,json,zipfile,argparse
from pathlib import Path
import numpy as np,networkx as nx
os.environ['MVF_THREADS']='1'
ROOT=Path(__file__).resolve().parent
BASE=Path(os.environ.get('MVF_STUDY_ROOT',str(ROOT.parent)))
sys.path.insert(0,str(BASE/'block_feedback'))
from block_feedback_experiment import feature_graph,cv
p=argparse.ArgumentParser();p.add_argument('--pair',type=int,required=True);p.add_argument('--model',choices=['gine','struct','mvf_feedback','mvf_uniform','mvf_shuffle'],required=True);p.add_argument('--epochs',type=int,default=16);p.add_argument('--seed',type=int,default=101);a=p.parse_args()
with zipfile.ZipFile(str(BASE/'BREC_data_all.zip')) as z:
 ar=np.load(io.BytesIO(z.read('brec_v3.npy')),allow_pickle=False)
G=[];y=[]
for j in range(64):
 g=nx.from_graph6_bytes(bytes(ar[a.pair*64+j]).rstrip(b'\x00'))
 gg=feature_graph(g.number_of_nodes(),list(g.edges()),np.ones((g.number_of_nodes(),1),np.float32),edge=np.ones((g.number_of_edges(),1),np.float32),gid=a.pair*1000+j)
 G.append(gg);y.append(j%2)
res=cv(G,np.asarray(y),f'BRECpair{a.pair}',a.model,seed=a.seed,h=24,epochs=a.epochs,split_seed=20261027,folds=2,budget=1100,patience=5,out=ROOT/'brec_runs')
print('RESULT',a.pair,a.model,100*res['BA'],flush=True)
