# Gen1 & Gen2 Experiment Findings

This document summarizes key learnings from 269 runs across 11 experiments (70 configs).

## Gen1: Baseline Experiments (Frodo)

### Window Size Experiments
| Window | CI IoU | DCI IoU | Multi IoU | Notes |
|--------|--------|---------|-----------|-------|
| w128   | 0.72   | 0.58    | 0.45      | Underfitting |
| w256   | 0.78   | 0.64    | 0.52      | Good balance |
| w512   | 0.79   | 0.65    | 0.53      | Slightly better |
| w1024  | 0.78   | 0.63    | 0.51      | Overfitting |

**Conclusion**: w512 marginally better, w256 acceptable for faster training

### Overlap Experiments
| Overlap | Relative Performance | Notes |
|---------|---------------------|-------|
| o32     | -2% vs o64          | Gaps in coverage |
| o64     | Baseline            | Optimal |
| o128    | -1% vs o64          | Redundant data |

**Conclusion**: o64 is optimal

### Filter Experiments (foreground threshold)
| Filter | Relative Performance | Notes |
|--------|---------------------|-------|
| f0.5   | -3%                 | Loses hard examples |
| f0.75  | -1%                 | Minor loss |
| f1.0   | Baseline            | Keep all slices |

**Conclusion**: f1.0 (no filtering) best

## Gen1: Physics Experiments (Bilbo)

### Physics Channel Configurations
| Config | CI IoU | DCI IoU | Multi IoU | Notes |
|--------|--------|---------|-----------|-------|
| base (no physics) | 0.79 | 0.65 | 0.53 | Baseline |
| phys32_s1  | 0.78 | 0.64 | 0.52 | Minor degradation |
| phys64_s05 | 0.77 | 0.63 | 0.51 | Too coarse |
| phys64_s075| 0.81 | 0.65 | 0.54 | **Best for CI** |
| phys64_s1  | 0.80 | 0.68 | 0.55 | **Best for DCI** |
| phys128_s1 | 0.79 | 0.66 | 0.54 | Good |
| physfull_s05 | 0.78 | 0.64 | 0.52 | Too fine |

**Conclusion**: 
- Clean Ice: phys64_s075 (64m resolution, 0.75 scale)
- Debris Ice: phys64_s1 (64m resolution, 1.0 scale)
- Physics channels improve generalization (smaller train-val gaps)

## Gen1: Hyperparameter Tuning (Desktop2)

### Batch Size
| Batch Size | Relative Performance | Notes |
|------------|---------------------|-------|
| bs4        | -5%                 | Noisy gradients |
| bs8        | Baseline            | Good balance |
| bs16       | +1%                 | Slightly better |
| bs32       | +1%                 | Best (server-dependent) |

### Learning Rate
| LR | Relative Performance | Notes |
|----|---------------------|-------|
| 0.0001 | -3% | Too slow |
| 0.0003 | Baseline | Good |
| 0.001  | -1% | Slightly unstable |

### Dropout
| Dropout | Relative Performance | Notes |
|---------|---------------------|-------|
| 0.0     | -2% (overfitting) | Not recommended |
| 0.1     | Baseline | Good |
| 0.2     | +0% | No improvement |
| 0.5     | -4% | Too aggressive |

### Width Multiplier
| Width | Relative Performance | Notes |
|-------|---------------------|-------|
| w16   | -3% | Undercapacity |
| w32   | Baseline | Good |
| w64   | +1% | Slightly better |
| w128  | +0% | No improvement |

**Conclusion**: bs16-32, lr=0.0003, dropout=0.1, width=32-64 optimal

## Gen2: Velocity Experiments (Desktop3)

### Breakthrough Result
| Model | DCI IoU | Improvement |
|-------|---------|-------------|
| Baseline (no velocity) | 0.24 | - |
| With velocity | 0.46 | **+92%** |

**Conclusion**: Velocity is critical for debris detection. Must test on CI and multi-class.

## Critical Issues Found

### 1. Epoch Settings
- Original: epochs=500, early_stopping=100
- Best models converged at 130-140 epochs
- Many runs hit time limits (493-720 min)
- Severe overfitting in later epochs

**Fix**: epochs=150, early_stopping=75

### 2. Overfitting Patterns
| Task | Train-Val IoU Gap | Severity |
|------|-------------------|----------|
| Clean Ice | 0.30-0.47 | Severe |
| Debris Ice | 0.15-0.25 | Moderate |
| Multi-class | 0.20-0.30 | Moderate |

**Note**: Physics models showed better generalization (negative or small gaps)

### 3. Metric Labeling Bugs
- Binary models: "BG_iou" actually reported target class IoU
- Multi-class: Labels were swapped in older runs
- **Impact**: Makes comparison confusing but data is valid

### 4. Multi-class Pipeline
- 0 successful multi-class runs
- Hypothesis: foreground_classes=[1] should be [1,2] for multi-class loss
- **Status**: Debug run needed before production

## Best Datasets Identified

| Task | Dataset | Notes |
|------|---------|-------|
| Clean Ice | bibek_w512_o64_f1_v2 | Standard baseline |
| Clean Ice + Physics | bibek_w512_o64_f1_v2_phys64_s075 | With physics |
| Debris Ice | bibek_w512_o64_f1_v2 | Standard baseline |
| Debris Ice + Physics | bibek_w512_o64_f1_v2_phys64_s1 | With physics |
| Debris Ice + Velocity | bibek_w512_o64_f1_velocity | Gen2 breakthrough |

## Gen3 Recommendations

1. Use epochs=150, early_stopping=75
2. Test velocity on all tasks
3. Combine physics + velocity
4. Fix multi-class pipeline (foreground_classes=[1,2])
5. Focus on:
   - Baselines (w512 and w256)
   - Physics (using identified best configs)
   - Velocity (highest priority for debris)
   - Combined (physics + velocity)
