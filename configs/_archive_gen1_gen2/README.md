# Gen1 & Gen2 Archived Configs

This directory contains 70 experiment configurations from Gen1 and Gen2 hyperparameter exploration.
These configs are preserved for reference but are **not intended for active use**.

## Archive Date
December 4, 2025

## Directory Structure

```
_archive_gen1_gen2/
├── frodo_clean_ice/     (12 configs) - Window/overlap/filter experiments
├── frodo_debris_ice/    (12 configs) - Window/overlap/filter experiments  
├── frodo_multiclass/    (12 configs) - Window/overlap/filter experiments
├── bilbo_clean_ice/     (7 configs)  - Physics channel experiments
├── bilbo_debris_ice/    (7 configs)  - Physics channel experiments
├── bilbo_multiclass/    (7 configs)  - Physics channel experiments
└── desktop2_debris_ice/ (13 configs) - Hyperparameter tuning (batch size, LR, dropout)
```

## Why Archived

1. **Epoch settings too high**: Original configs used epochs=500, early_stopping=100 which led to:
   - Training hitting time limits (493-720 min)
   - Severe overfitting (train-val IoU gaps of 0.15-0.47)
   - Best models actually converged at 130-140 epochs

2. **Metric labeling bugs identified**: Older runs had swapped metric labels that made analysis confusing

3. **Key learnings captured**: See FINDINGS.md for what we learned from these experiments

## Key Findings Summary

- **Window size**: w512 and w256 comparable, w512 slightly better
- **Overlap**: o64 optimal (vs o32, o128)
- **Filter**: f1.0 (no filtering) optimal
- **Physics channels**: phys64_s075 (CI), phys64_s1 (DCI) best performers
- **Velocity**: 2x improvement for debris detection (Gen2 breakthrough)

## Gen3 Direction

New configs in `configs/{server}/{task}/` use:
- epochs: 150 (down from 500)
- early_stopping: 75 (down from 100)
- Cleaner naming with `_gen3` suffix
- Focus on: baselines, physics, velocity, combined experiments
