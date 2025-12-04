# TODO

## Code Improvements

### Multi-class Threshold Handling

**Issue:** The threshold parameter in multi-class configs is currently not fully utilized in the code.

**Current State:**
- Multi-class configs now correctly specify `threshold: [0.5, 0.5, 0.5]` (one per class)
- However, the actual code has hardcoded `> 0.5` threshold checks in several places:
  - `glacier_mapping/lightning/glacier_module.py:242` - Validation metrics use hardcoded `y_prob[:, class_idx] > 0.5`
  - `glacier_mapping/lightning/glacier_module.py:440-445` - Multi-class prediction uses `argmax()` which doesn't use threshold at all

**Recommendation:**
- Consider refactoring multi-class prediction and validation to use the configurable threshold array
- This would allow different thresholds per class (e.g., `[0.5, 0.6, 0.4]` for BG, CleanIce, Debris)
- Current hardcoded `0.5` is reasonable for now, but configurability would improve flexibility

**Priority:** Low (current implementation works fine, this is for future flexibility)
