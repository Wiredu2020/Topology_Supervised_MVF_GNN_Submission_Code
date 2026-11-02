from pathlib import Path
import json,csv,statistics
p=Path(__file__).resolve().parent
rows=[]
for fn in sorted((p/'pilot_3fold').glob('*/*.json')):
 if fn.name.endswith('_learning_curve.json'):continue
 r=json.loads(fn.read_text())
 rows.append({k:r[k] for k in ['dataset','model','seed','fold','max_epochs','n_train','n_val','n_test','balanced_accuracy','accuracy','best_epoch','fit_seconds','parameters']})
with (p/'pilot_results.csv').open('w') as f:
 wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
for ds in ['BACE','DHFR','NCI109']:
 print(ds)
 for m in ['gine_cycle','poset_cycle','poset_shuffle','poset_uniform','poset_softonly']:
  ll=[r for r in rows if r['dataset']==ds and r['model']==m]
  if ll:print(m,len(ll),round(statistics.mean(r['balanced_accuracy'] for r in ll)*100,3))
