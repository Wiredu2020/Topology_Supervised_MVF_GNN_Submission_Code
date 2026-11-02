# Experiment protocols and recorded outputs

## Primary evidence

1. **Intermediate-layer multivector feedback.** `results/01_component_feedback/` contains the constant-feature component-detection task, BZR-independent COX2/PTC_MR experiments, trained-checkpoint interventions, and three-seed model comparisons. The corresponding code is in `src/study/block_feedback/`.
2. **Cycle-conditioned incidence routing.** `results/02_cycle_conditioned/` contains the real-data BZR, PROTEINS and ENZYMES experiments and additional COX2 cycle-routing runs. The manuscript's BZR cohort used three seeds, three folds and a maximum of 36 epochs. The code is in `src/study/cycle_poset/` and `src/study/external_validation/`. The cycle-conditioned evaluation archive uses the exact edge-walk input convention of the recorded runs; the topology-supervision implementation uses symmetric edge-walk values. These cohorts must not be numerically conflated, and the BZR comparison should be rerun with symmetric edge-walk inputs before making a definitive vertex-relabeling invariance claim for that historical protocol.
3. **Edge-event-supervised multivector routing.** `results/03_event_supervision/` contains all provided complete and partial model records for BZR, COX2 and DHFR. Three seeds and three folds were specified for each comparator, up to 35 epochs. The originally saved cohort summary omitted a completed COX2 uniform-routing run for seed 303. The independent `completed_runs_from_metadata.csv` and `summary_from_metadata.csv` include all 63 verified completed model-seed records. Labels indicating bridge/cycle-support are training targets, not classifier input features. The corresponding source is `src/event_experiment.py`.

## Additional evaluations

4. **CSL/BREC graph-structure expressivity.** `results/04_expressivity/` includes CSL classification and selected BREC pair-pilot predictions. These are not official full BREC pairwise significance scores.
5. **Topological similarity and additional datasets.** `results/05_external_topology_validation/` preserves the completed long-run PTC data and Mutagenicity comparisons, alongside the topology metadata. Results on missing raw datasets can be verified from saved predictions without downloading data.
6. **BZR-analog structural pilots.** `results/06_bzr_analog_pilot/` includes BACE and DHFR partial pilot records, not a completed multi-seed final evaluation.
7. **Structural-router baseline reproduction.** `results/07_original_incidence_controls/` contains recovered baseline and incidence-router runs.
8. **Incidence-routing configuration screen.** `results/08_incidence_configuration_screen/` contains the corrected 200-configuration screening results and associated limited follow-up runs. These exploratory results are not independent external validation or tests of trainable discrete partitions. Some earlier PTC_MR baseline experiments may be affected by edge-attribute sorting and should not be treated as final evidence without alignment checks.

All results are retained by their original study protocol; do not average classification scores across incompatible splits, architectures, or epoch budgets.
