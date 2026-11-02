"""Recompute graph-level balanced accuracy from stored prediction arrays."""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
records = defaultdict(list)
failures = []
for path in sorted((ROOT / 'results').rglob('*.npz')):
    try:
        with np.load(path, allow_pickle=False) as arrays:
            if 'y' in arrays.files:
                y = arrays['y']
            elif 'label' in arrays.files:
                y = arrays['label']
            else:
                continue
            if 'pred' in arrays.files:
                pred = arrays['pred']
            elif 'prob' in arrays.files:
                pred = np.argmax(arrays['prob'], axis=1)
            else:
                continue
            if y.shape != pred.shape:
                failures.append((str(path), 'prediction shape differs from target shape'))
                continue
            if 'fold' in arrays.files and np.any(np.asarray(arrays['fold']) < 0):
                continue
            score = float(balanced_accuracy_score(y, pred))
            records[path.parts[path.parts.index('results')+1]].append(score)
            meta = path.with_suffix('.json')
            if meta.is_file():
                saved = json.loads(meta.read_text())
                reported = next((saved[k] for k in ('balanced_accuracy', 'BA', 'ba', 'test_BA') if k in saved), None)
                if reported is not None and abs(float(reported)-score) > 1e-8:
                    failures.append((str(path.relative_to(ROOT)), f'reported {reported}, recomputed {score}'))
    except Exception as exc:
        failures.append((str(path.relative_to(ROOT)), str(exc)))
print('Recorded prediction arrays:')
for cohort, vals in sorted(records.items()):
    print(f'{cohort}: {len(vals)} verified arrays, mean balanced accuracy {100*np.mean(vals):.4f}%')
print('Score mismatches / unreadable arrays:', len(failures))
for p, reason in failures[:30]: print('ERROR', p, reason)
raise SystemExit(1 if failures else 0)
