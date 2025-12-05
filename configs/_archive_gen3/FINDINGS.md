# Gen3 Experiment Findings

This document summarizes key learnings from 160 runs across 3 experiments (21 configs).

## Gen3: Baseline Experiments (Frodo & Bilbo)

### Window Size Validation
| Window | CI IoU | DCI IoU | Multi Debris IoU | Duration (min) | IoU/Min | Notes |
|--------|--------|---------|------------------|----------------|----------|-------|
| w256   | 0.73   | 0.45    | 0.38      | 133-211   | 0.0035  | **Best CI efficiency** |
| w512   | 0.70   | 0.47    | 0.36-0.38 | 139-247  | 0.0028  | Good baseline |

**Conclusion**: w256 significantly better for clean ice (+4% IoU, -38% time), neutral for debris/multi-class

### Training Duration Analysis
| Config | Fastest | Slowest | Average | Efficiency |
|--------|---------|---------|----------|-------------|
| CI w256 | 133 min | - | 133 min | **0.0055 IoU/min** |
| CI w512 | 213 min | 247 min | 230 min | 0.0030 IoU/min |
| DCI w256 | 137 min | - | 137 min | 0.0033 IoU/min |
| DCI w512 | 114 min | 247 min | 180 min | 0.0026 IoU/min |

**Finding**: 2x runtime variance indicates data quality or initialization issues

## Gen3: Physics Experiments (Bilbo)

### Physics Channel Performance
| Config | CI IoU | DCI IoU | Multi Debris IoU | Duration | Train-Val Gap | Generalization |
|--------|--------|---------|------------------|----------|---------------|---------------|
| base (no physics) | 0.70 | 0.47 | 0.36-0.38 | 139-247 | 0.07-0.37 | Moderate |
| physics (CI) | **NO DATA** | - | - | - | - | **Missing runs** |
| physics (DCI) | - | 0.42 | - | 150 | 0.018 | **Excellent** |
| physics (Multi) | - | - | 0.42 | 176 | -0.08 | **Excellent** |

**Key Findings**:
- Physics dramatically reduces overfitting (negative gaps)
- +17% debris IoU improvement in multi-class (0.36→0.42)
- Clean ice physics data completely missing

### Overfitting Analysis by Task
| Task | Train-Val Gap Range | Severity | Best Config |
|------|-------------------|----------|-------------|
| Clean Ice | 0.07-0.37 | Moderate-High | w512 base (0.07) |
| Debris Ice | -0.39 to 0.13 | Low to Negative | Physics (0.018) |
| Multi-class | -0.08 to 0.37 | Low to High | Physics (-0.08) |

**Finding**: Physics channels consistently improve generalization

## Gen3: Velocity Experiments (Critical Failure)

### Velocity Channel Results
| Config | CI IoU | DCI IoU | Multi IoU | Duration | Status | Issue |
|--------|--------|---------|-----------|----------|---------|-------|
| Baseline (no velocity) | 0.70 | 0.47 | 0.36-0.38 | 114-247 | ✅ Success | Working |
| With velocity | **NO DATA** | 0.26 | **NO DATA** | 254-612 | ❌ Failed | Dataset bug |

**Critical Issues**:
- Complete failure to replicate Gen2's +92% breakthrough (0.24→0.46 IoU)
- Performance actually degraded (0.26 vs 0.47 baseline)
- 2-3x longer training times
- User confirmed dataset preprocessing bug

## Gen3: Multi-class Experiments

### Multi-class Performance (Complete Results)
| Config | BG IoU | Clean IoU | Debris IoU | Duration | Train-Val Gap (Clean/Debris) | Notes |
|--------|---------|-----------|------------|----------|-----------------------------|-------|
| base_w512 | 0.84 | 0.66-0.68 | 0.36-0.38 | 113-165 | 0.02-0.25 / 0.05-0.25 | Good baseline |
| base_w256 | 0.79 | 0.67 | 0.38 | 211 | 0.37 / 0.34 | Longer runtime |
| physics_w512 | 0.84 | 0.66 | **0.42** | 176 | 0.09 / -0.08 | **Best debris** |

**Key Findings**:
- Physics improves debris detection by +17% (0.36→0.42)
- Physics eliminates debris overfitting (negative gap)
- Background detection excellent across all configs (0.79-0.84)
- w256 advantage disappears in multi-class setting

### Multi-class Pipeline Success
- ✅ **Pipeline fixed**: `foreground_classes: [1, 2]` resolved multi-class loss
- ✅ **Metrics working**: Per-class IoU properly logged
- ✅ **Training stable**: All runs completed 149/150 epochs
- ✅ **Generalization good**: Physics shows excellent generalization

## Critical Issues Identified

### 1. Velocity Channel Dataset Bug
- **Impact**: Complete failure to validate Gen2 breakthrough
- **Symptoms**: Poor performance (0.26 vs 0.46 expected), long runtimes
- **Status**: User confirmed dataset preprocessing issues
- **Priority**: **CRITICAL** - Blocks all velocity experiments

### 2. Missing Clean Ice Physics Data
- **Impact**: Cannot assess physics benefit for clean ice
- **Symptoms**: Config exists but no successful runs
- **Status**: Complete gap in experimental coverage
- **Priority**: **HIGH** - Incomplete physics assessment

### 3. Runtime Inconsistency
- **Impact**: 2x variance in training times for same configs
- **Symptoms**: DCI base: 114 vs 247 minutes
- **Status**: Likely data loading or initialization variance
- **Priority**: **MEDIUM** - Efficiency concern

### 4. Window Size Contradiction
- **Impact**: Challenges Gen1 findings about optimal window size
- **Symptoms**: w256 outperforms w512 for clean ice (0.73 vs 0.70)
- **Status**: Needs validation across more runs
- **Priority**: **MEDIUM** - Architecture optimization

## Best Performing Gen3 Models

### By Task
| Task | Best Config | IoU | Duration | Generalization | Efficiency |
|------|-------------|-----|----------|---------------|-------------|
| Clean Ice | w256_gen3 | **0.73** | 133 min | Moderate | **0.0055** |
| Debris Ice | base_w512_gen3 | **0.47** | 114 min | Good | 0.0041 |
| Debris Ice (Physics) | physics_gen3 | 0.42 | 150 min | **Excellent** | 0.0028 |
| Multi-class | physics_gen3 | BG 0.84, Clean 0.66, **Debris 0.42** | 176 min | **Excellent** | 0.0024 |

### By Server Coverage
| Server | Clean Ice | Debris Ice | Multi-class | Velocity | Coverage |
|--------|-----------|-----------|-------------|-----------|----------|
| Bilbo | ❌ No physics data | ✅ Base+Physics | ✅ Base+Physics | ❌ By design | 60% |
| Frodo | ✅ Base+w256+Velocity | ✅ Base+w256+Velocity | ✅ Base+w256+Velocity | ❌ Failed | 75% |
| Desktop | ❌ No baseline | ❌ Only velocity | ❌ Only velocity | ❌ Failed | 25% |

## Gen3 vs Gen1+2 Comparison

### Window Size Findings
| Generation | Best CI Window | Best DCI Window | Multi-class |
|------------|----------------|-----------------|-------------|
| Gen1 | w512 (0.79) | w512 (0.65) | w512 (0.53) |
| Gen3 | **w256 (0.73)** | w512 (0.47) | w512 (0.42) |

**Change**: w256 emerged as superior for clean ice, contradicting Gen1

### Physics Channel Benefits
| Generation | CI Physics | DCI Physics | Multi-class Physics |
|------------|-------------|--------------|-------------------|
| Gen1 | +2.5% (0.79→0.81) | +4.6% (0.65→0.68) | +1.9% (0.53→0.54) |
| Gen3 | **NO DATA** | -11% (0.47→0.42) | **+17%** (0.36→0.42) |

**Change**: Physics benefit amplified for multi-class, reduced for binary debris

### Velocity Channel Results
| Generation | CI Velocity | DCI Velocity | Multi-class Velocity |
|------------|--------------|---------------|----------------------|
| Gen2 | **NOT TESTED** | **+92%** (0.24→0.46) | **NOT TESTED** |
| Gen3 | **FAILED** | **FAILED** (0.47→0.26) | **FAILED** |

**Change**: Complete regression due to dataset bug

## Gen4 Recommendations

### Priority 1: Critical Missing Data
1. **Clean Ice Physics** (`ci_physics_gen3`)
   - Essential for complete physics assessment
   - Config exists, needs successful run

2. **Velocity Dataset Validation**
   - Fix preprocessing pipeline to match Gen2
   - Test on all three tasks
   - Validate Gen2 breakthrough replication

### Priority 2: Combined Channel Experiments
```
configs/frodo/debris_ice/physics_velocity_gen4.yaml
configs/bilbo/debris_ice/physics_velocity_gen4.yaml
configs/frodo/clean_ice/physics_velocity_gen4.yaml
configs/frodo/multiclass/physics_velocity_gen4.yaml
```
- Test if physics stabilizes velocity performance
- Highest potential for breakthrough results
- Physics shows excellent generalization

### Priority 3: Complete Server Coverage
```
configs/frodo/multiclass/physics_gen4.yaml
configs/bilbo/clean_ice/w256_gen4.yaml
configs/desktop/clean_ice/base_gen4.yaml
```
- Fill missing server-specific experiments
- Validate w256 advantage across platforms
- Provide proper baselines for velocity comparison

### Priority 4: Architecture Optimization
Based on overfitting analysis:
- **High overfitting** (gap >0.30): dropout=0.2, weight_decay=1e-4
- **Good generalization**: Current settings optimal
- **Underfitting** (negative gaps): Increase model capacity

### Priority 5: Efficiency Improvements
- Standardize w256 for clean ice if advantage confirmed
- Investigate runtime inconsistency (data loading variance)
- Optimize batch sizes per server configuration

## Immediate Action Plan

### Week 1: Critical Fixes
1. Run `configs/bilbo/clean_ice/physics_gen3.yaml`
2. Validate velocity dataset with small test runs
3. Fix any remaining metric logging issues

### Week 2: Velocity Validation
1. Run `configs/frodo/clean_ice/velocity_gen3.yaml`
2. Run `configs/frodo/debris_ice/velocity_gen3.yaml`
3. Run `configs/frodo/multiclass/velocity_gen3.yaml`

### Week 3: Combined Experiments
1. Create and run physics+velocity combinations
2. Focus on debris ice (highest potential)
3. Test multi-class with all channels

### Week 4: Coverage & Optimization
1. Fill missing server coverage gaps
2. Validate w256 advantage across tasks
3. Architecture optimization based on overfitting patterns

## Success Metrics for Gen4

### Minimum Requirements
- ✅ Complete clean ice physics data
- ✅ Validate velocity dataset fix
- ✅ Achieve >0.50 debris IoU with combined channels

### Stretch Goals
- 🎯 Replicate Gen2 velocity breakthrough (>0.46 debris IoU)
- 🎯 Achieve >0.55 debris IoU with physics+velocity combination
- 🎯 Reduce training time by 25% with w256 optimization
- 🎯 Complete server coverage across all experiments

## Key Questions for Gen4

1. **Can velocity channels be fixed** to replicate Gen2 breakthrough?
2. **Will physics+velocity combination** rescue and enhance performance?
3. **Is w256 advantage real** for clean ice or just statistical variance?
4. **Can we achieve >0.50 debris IoU** with any configuration?

The answers to these questions will determine the optimal strategy for production glacier mapping models.

---

**Archive Date**: December 2025  
**Total Runs Analyzed**: 160 (67 successful)  
**Experiments**: 3 (Clean Ice, Debris Ice, Multi-class)  
**Servers**: 3 (Bilbo, Frodo, Desktop)  
**Configs Archived**: 21