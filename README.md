# Topology-Supervised Multivector Incidence Routing for Graph Neural Networks

**Authors:** Donald Woukeng and Bright Wiredu Nuakoh  
**Affiliation:** African Institute for Mathematical Sciences (AIMS), Rwanda

This archive contains the experiment implementation, graph inputs available for redistribution in this research package, saved out-of-fold predictions, protocol records, and accuracy-verification utilities associated with trainable vertex–edge incidence routing, multivector block feedback, and edge-event supervision.

## Contents

- `src/event_experiment.py`: topology-supervised vertex–edge routing and multivector-block models.
- `src/study/router_core/`: structural-incidence baseline and graph-loading utilities.
- `src/study/block_feedback/`: intermediate-layer hard-block feedback and synthetic connectivity comparison.
- `src/study/cycle_poset/`: cycle-conditioned hard/soft incidence routing.
- `src/study/external_validation/`: matched comparisons on molecular and biological graphs.
- `src/study/graph_expressivity/`: CSL and selected BREC pair tests.
- `src/study_pipeline.py`: extended structural-similarity and graph-classification evaluation.
- `src/supplementary/architecture_screen/`: corrected incidence-configuration screening code.
- `src/data/`: supplied benchmark graph datasets and molecular structures.
- `results/`: original saved experiment records, out-of-fold predictions, and measured summaries, separated by study and protocol.
- `scripts/verify_predictions.py`: recomputes balanced accuracy from prediction arrays.
- `requirements.txt`: Python dependencies.
- `PROTOCOLS.md`: experiment definitions and known coverage boundaries.
- `FILE_MANIFEST.sha256`: content checksums.

## Setup

```bash
python -m pip install -r requirements.txt
python scripts/verify_predictions.py
```

The score verification reads completed result files and does not train models. The event-supervision cohort also includes a summary recomputed from all 63 completed per-run records.

## Main experimental commands

The commands below are executed from the archive root. Their output is written to a separate directory so the supplied results remain unchanged.

```bash
python src/event_experiment.py --datasets BZR COX2 DHFR --modes gine gine_aux old_poset event event_noaux event_shuffle event_uniform --seeds 101 202 303 --epochs 35 --folds 3 --split_seed 20261121 --aux_weight 0.2 --output fresh_results/event
python src/study/block_feedback/block_feedback_experiment.py --dataset COX2 --modes gine struct mvf_feedback mvf_shuffle mvf_uniform mvf_readout --seeds 101 202 303 --out fresh_results/block_feedback
python src/study/external_validation/evaluate_new_datasets.py
```

The first command reproduces the topology-supervised study. The other source modules provide the block-feedback and cycle-conditioned experiments. Their exact settings and cohort-specific training budgets are recorded in the provided protocol files and saved run metadata.

**Reproducibility boundary:** Prediction files verify the reported numerical outputs, but full training has not been rerun during assembly of this archive. Some broader validation datasets (`PTC_FM`, `PTC_FR`, `PTC_MM`, and `Mutagenicity`) have saved results without locally available original raw archives. Their official download locations appear in `DATASETS.md`. The archive contains all recovered recorded results for these studies, but cannot be described as fully self-contained for rerunning those four datasets until those archives are obtained. The completed COX2 uniform-event third seed is included in the recomputed numerical summary; it was absent from an earlier cohort summary.
