from pathlib import Path
import json, csv, collections, math, statistics
ROOT=Path(__file__).resolve().parent
D=[]
for p in sorted((ROOT/'runs').glob('*/*.json')):
 z=json.loads(p.read_text());ds=p.parent.name
 D.append(dict(dataset=ds,model=z['model'],seed=z['seed'],BA_percent=round(100*z['BA'],4),params=z['params'],seconds=round(z['seconds'],1),split_seed=z['split_seed'],epochs_max=z['epochs_max'],folds=len(z['folds'])))
fields=list(D[0].keys()) if D else []
with (ROOT/'completed_runs.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(D)
by=collections.defaultdict(list)
for x in D:by[(x['dataset'],x['model'])].append(x['BA_percent'])
rows=[]
for (ds,m),v in sorted(by.items()):
 rows.append(dict(dataset=ds,model=m,n_seeds=len(v),mean_BA=round(statistics.mean(v),3),seed_SD=round(statistics.stdev(v),3) if len(v)>1 else '',seeds=';'.join(str(x['seed']) for x in D if x['dataset']==ds and x['model']==m)))
with (ROOT/'model_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0].keys()) if rows else ['dataset','model','n_seeds','mean_BA','seed_SD','seeds']);w.writeheader();w.writerows(rows)
by2={(x['dataset'],x['model'],x['seed']):x['BA_percent'] for x in D}
pa=[]
for ds in sorted(set(x['dataset'] for x in D)):
 for c in ['gine_cycle','poset_shuffle','poset_uniform','poset_softonly']:
  seeds=sorted(set(x['seed'] for x in D if x['dataset']==ds))
  matches=[s for s in seeds if (ds,'poset_cycle',s) in by2 and (ds,c,s) in by2]
  if matches:
   diffs=[by2[ds,'poset_cycle',s]-by2[ds,c,s] for s in matches]
   pa.append(dict(dataset=ds,comparison='poset_cycle minus '+c,seeds=len(diffs),paired_mean_pp=round(statistics.mean(diffs),3),paired_sd_pp=round(statistics.stdev(diffs),3) if len(diffs)>1 else '',individual_pp=';'.join(f'{x:.3f}' for x in diffs)))
with (ROOT/'paired_comparisons.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(pa[0].keys()) if pa else ['dataset','comparison','seeds','paired_mean_pp','paired_sd_pp','individual_pp']);w.writeheader();w.writerows(pa)
print('Completed:',len(D),'\n')
for row in rows:print(row)
print('\nMatched:')
for row in pa:print(row)
