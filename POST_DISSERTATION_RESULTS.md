# Post-Dissertation Replication and Robustness Study

## Scope

This report summarizes experiments completed after the dissertation defense.
It is a replication and robustness study, not a replacement for the defended
results. The later work used a changed software environment, reconstructed
datasets, additional evaluation masks, and many exploratory runs. Comparisons
are therefore separated by protocol and interpreted conservatively.

The study addressed four questions:

1. Can the Aryal et al. boundary-aware U-Net result be reproduced in the current
   harness?
2. Which implementation changes account for later apparent improvements?
3. Can the HKH imagery be rebuilt from currently available public products?
4. Do velocity-v2 improvements persist across paired random seeds?

## Methods

The Aryal reproduction preserved the published-code split and augmentation
semantics, eight Landsat bands, custom U-Net, Adam optimizer, plateau scheduler,
and original-style paired CI/DCI evaluation. Physical batch size was reduced
from eight to four with gradient accumulation because of GPU memory; effective
batch size remained eight.

The updated dataset study used Landsat Collection 2 Level 1 raw DN products. It
compared the original acquisition selection, recoverable exact legacy dates,
several SLC-gap donor policies, strict and relaxed valid masks, and date-aware
ITS_LIVE features. Fair dataset comparisons were rescored on common-valid
pixels. Confirmatory comparisons used paired seeds rather than selecting the
best observed run.

The repository's canonical configs preserve the final protocols. Historical
intermediate configs and audit programs are recoverable from commit `9793e17`.

## Aryal baseline reproduction

The frozen U0 parity condition uses a single seed-42 generator shared across
training, validation, and test DataLoaders, matching the public Aryal code. The
current harness closely reproduced both published IoUs.

| Target | Aryal reported | U0 reproduction | Difference |
|---|---:|---:|---:|
| CI IoU | 0.681700 | 0.683959 | +0.002259 |
| DCI IoU | 0.359400 | 0.363852 | +0.004452 |

The small residual difference is compatible with the non-identical runtime,
library versions, and physical batch adaptation. This constitutes a practical
replication of the published baseline from the locally retained source cells.

Canonical configs:

- `configs/local/clean_ice/aryal_u0.yaml`
- `configs/local/debris_ice/aryal_u0.yaml`

## Modernized model stability

U5 combined an SMP U-Net with a ResNet-18 encoder, modern loss behavior, AdamW,
and the plateau scheduler. It performed substantially better than U0 for three
seeds but collapsed for seed 2026.

| Seed | U0 CI | U5 CI | Delta | U0 DCI | U5 DCI | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.685775 | 0.732231 | +0.046456 | 0.393777 | 0.496844 | +0.103067 |
| 41 | 0.683959 | 0.733378 | +0.049419 | 0.363852 | 0.500915 | +0.137063 |
| 123 | 0.680239 | 0.727991 | +0.047752 | 0.390077 | 0.519916 | +0.129839 |
| 2026 | 0.690458 | 0.000001 | -0.690458 | 0.385744 | 0.029942 | -0.355802 |

Across four seeds, the paired mean changes were -0.136708 for CI and +0.003542
for DCI. Neither aggregate supported a stable improvement. The failed DCI model
predicted nearly all foreground; its learned uncertainty value was not clamped,
so boundary-clamp saturation alone does not explain the failure.

The defensible conclusion is conditional: U5 reached approximately 0.73 CI and
0.50 DCI IoU in three observed seeds, but it was not robust under the tested
multi-seed protocol.

Canonical configs:

- `configs/local/clean_ice/aryal_u5.yaml`
- `configs/local/debris_ice/aryal_u5.yaml`

## Dataset reconstruction

An availability audit found all 41 required scenes in Collection 2 Level 1 and
none in the former Collection 1 locations. Exact Collection 1 regeneration is
therefore impossible from the current catalog. Collection 2 raw DN was much
closer to the legacy byte-domain imagery than Collection 2 TOA, making it the
supported replacement.

The first rebuilt dataset used strict QA validity and discarded large fractions
of labeled glacier pixels. On the training split, clean-ice pixels fell from
23.44 million in the legacy data to 12.16 million. The documented relaxed policy
retains pixels when target data are present or an SLC fill succeeds; it restored
22.89 million clean-ice pixels and 2.75 million DCI pixels. Where both datasets
were valid, the class labels did not flip.

The exact-date Collection 2 agreement dataset was evaluated over three seeds on
the same common-valid pixels as the legacy data:

| Target | Legacy IoU | C02 exact-date IoU | Delta |
|---|---:|---:|---:|
| CI | 0.728758 | 0.731553 | +0.002796 |
| DCI | 0.533659 | 0.540917 | +0.007258 |

These are small improvements, not evidence for a new benchmark. Native-mask
scores differed because validity policies changed. The multiclass exact-date
run underperformed for DCI and was not pursued.

Canonical configs:

- `configs/local/clean_ice/c02_agreement.yaml`
- `configs/local/debris_ice/c02_agreement.yaml`
- `configs/local/clean_ice/c02_legacy_dates.yaml`
- `configs/local/debris_ice/c02_legacy_dates.yaml`
- `configs/local/multiclass/c02_agreement.yaml`

## Velocity-v2 paired results

Velocity-v2 used date-aware ITS_LIVE aggregation with provenance-derived target
dates. Four paired seeds compared identical model and training settings with and
without velocity features.

| Comparison | Baseline mean | Variant mean | Mean delta | Paired t p |
|---|---:|---:|---:|---:|
| DCI speed | 0.520808 | 0.536484 | +0.015676 | 0.119 |
| DCI quality | 0.520808 | 0.536081 | +0.015273 | 0.141 |
| Multiclass DCI speed | 0.543134 | 0.547269 | +0.004134 | 0.665 |

The binary DCI changes were positive in all four observed pairs, but the sample
was small and the conventional paired tests were not conclusive. The multiclass
effect was smaller and inconsistent. Velocity-v2 should be treated as promising
replication evidence rather than a publication-ready gain.

Canonical configs:

- `configs/local/debris_ice/velocity_baseline.yaml`
- `configs/local/debris_ice/velocity_speed.yaml`
- `configs/local/debris_ice/velocity_quality.yaml`
- `configs/local/multiclass/velocity_baseline.yaml`
- `configs/local/multiclass/velocity_speed.yaml`

## Threats to validity

- The original Collection 1 data can no longer be regenerated from the current
  public Earth Engine catalog.
- The locally retained Aryal source cells are not redistributed.
- The modern runtime and GPU differ from the original Aryal environment.
- Dataset validity masks materially affect native metrics; fair comparisons
  require a common mask.
- Hundreds of exploratory runs reused the same test split. Their maxima are not
  unbiased estimates and are intentionally omitted here.
- Confirmatory seed counts are small, so uncertainty remains large even where
  every observed paired delta has the same sign.

## Conclusion

The strongest post-dissertation result is the close reproduction of the Aryal
baseline. The updated Collection 2 pipeline also provides a documented,
rebuildable replacement for unavailable Collection 1 inputs, with small
common-mask improvements. Later architecture and velocity results were not
stable or conclusive enough to support a new performance claim. They are best
retained as a replication and robustness record.
