"""TU archive parser extracted from the archived experiment (no fixed-descriptor routines)."""
import os,re,io,zipfile,pathlib,urllib.request
import numpy as np
def parse_table(blob, dtype=float):
    lines=[x.strip() for x in blob.decode('utf-8-sig').splitlines() if x.strip()]
    return np.asarray([[dtype(k.strip()) for k in re.split(r'\s*,\s*|\s+',s) if k.strip()] for s in lines])

def load_tu_zip(path, name):
    """Canonical TUDataset graph loader, retaining raw graph ids; no label-derived features."""
    with zipfile.ZipFile(path) as z:
        src={pathlib.PurePosixPath(f).name:z.read(f) for f in z.namelist() if not f.endswith('/')}
    def load(sfx,dt=float):
        key=f'{name}_{sfx}.txt'
        return parse_table(src[key],dt) if key in src else None
    gi=load('graph_indicator',int).ravel()-1
    yraw=load('graph_labels',int).ravel()
    edge_global=load('A',int)-1
    nodes_lab=load('node_labels',int)
    edge_lab=load('edge_labels',int)
    nodes_attr=load('node_attributes',float)
    edges_attr=load('edge_attributes',float)
    classes=sorted(np.unique(yraw).tolist());y=np.array([classes.index(v) for v in yraw],int)
    n_graphs=len(y); nodes_by_graph=[np.flatnonzero(gi==k) for k in range(n_graphs)]
    graph_edges=[[] for _ in range(n_graphs)]; ea=[{} for _ in range(n_graphs)]
    for ii,(u,v) in enumerate(edge_global):
        gu=int(gi[u]);gv=int(gi[v]);
        if gu!=gv:raise ValueError('Cross-graph edge found')
        if u==v:continue
        if u>v: u,v=v,u
        e=(int(u),int(v));
        if e not in ea[gu]: ea[gu][e]=[];graph_edges[gu].append(e)
        al=[]
        if edge_lab is not None: al.extend(edge_lab[ii].ravel().tolist())
        if edges_attr is not None: al.extend(edges_attr[ii].ravel().tolist())
        ea[gu][e].append(np.array(al,float))
    nlabdim=0 if nodes_lab is None else int(np.max(nodes_lab)-np.min(nodes_lab)+1)
    elabdim=0 if edge_lab is None else int(np.max(edge_lab)-np.min(edge_lab)+1)
    nlbase=0 if nodes_lab is None else int(np.min(nodes_lab))
    elbase=0 if edge_lab is None else int(np.min(edge_lab))
    graphs=[]
    for g in range(n_graphs):
        nodes=nodes_by_graph[g]
        local={v:j for j,v in enumerate(nodes)}
        E=np.array([(local[u],local[v]) for u,v in graph_edges[g]],dtype=int).reshape(-1,2)
        xx=[]
        if nodes_lab is not None:
            ids=(nodes_lab[nodes,0].astype(int)-nlbase)
            xx.append(np.eye(nlabdim,dtype=float)[ids])
        if nodes_attr is not None:xx.append(nodes_attr[nodes])
        X=np.concatenate(xx,axis=1) if xx else np.zeros((len(nodes),0),float)
        # Reverse observations of each simple support edge are averaged, preserving categories by one-hot encoding
        EV=[]
        for e in graph_edges[g]:
            vv=np.asarray(ea[g][e]);
            # Encoded edge categorical ID, if present, is transformed to a one-hot part.
            if len(vv)==0 or vv.size==0:EV.append(np.zeros(elabdim+(0 if edges_attr is None else edges_attr.shape[1])));continue
            means=vv.mean(0)
            if elabdim:
                catidx=int(round(means[0]))-elbase
                row=np.r_[np.eye(elabdim)[catidx],means[edge_lab.shape[1]:]]
            else:row=means
            EV.append(row)
        width=elabdim+(0 if edges_attr is None else edges_attr.shape[1])
        EA=np.asarray(EV,dtype=float).reshape(len(E),width) if E.size else np.empty((0,width))
        graphs.append({'n':len(nodes),'edges':E,'node':X,'edge_attr':EA,'node_label':nodes_lab[nodes,0].astype(int) if nodes_lab is not None else None,
                       'edge_label':EA[:,:elabdim].argmax(1) if elabdim and len(EA) else None,'original_id':g+1})
    assert len(graphs)==len(y)
    return graphs,y

def ensure_tu_archive(name, data_dir, download_allowed=False):
    path=pathlib.Path(data_dir);path.mkdir(parents=True,exist_ok=True)
    for candidate in [path/f'{name}.zip',path/name/f'{name}.zip',path/'raw'/f'{name}.zip']:
        if candidate.exists():return candidate
    found=list(path.glob(f'**/{name}.zip'))
    if found:return found[0]
    if not download_allowed:raise FileNotFoundError(f'{name}.zip not found in {data_dir}. Dataset archive not found.')
    url=f'https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip'
    out=path/f'{name}.zip';urllib.request.urlretrieve(url,out)
    if not zipfile.is_zipfile(out):raise IOError(f'Not a zip: {url}')
    return out

