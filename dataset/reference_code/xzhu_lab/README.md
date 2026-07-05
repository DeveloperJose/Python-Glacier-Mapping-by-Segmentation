# XZhu-lab reference code archive

Acquired: 2026-06-30

Source page:

- <https://xzhu-lab.github.io/post/open-source-code/>

Purpose in this repository:

- Preserve original upstream gap-filling and supporting evaluation code used as reference material for the HKH Landsat SLC-off gap-fill tournament.
- Support local reimplementation/comparison only. These files are not imported by production scripts.
- Keep original archives and extracted contents together so algorithm behavior can be audited against upstream code.

## Archived downloads

| Archive | Upstream item | Why kept |
|---|---|---|
| `archives/nspi_update_20100824.zip` | NSPI | Original single- and multiple-donor ETM+ SLC-off gap-fill reference (`FILLGAP_SINGLE_V2.pro`, `FILLGAP_MULTIPLE_V2.pro`). |
| `archives/gnspi_update_20130317.zip` | GNSPI | Geostatistical SLC-off gap-fill reference. Not currently implemented in Python tournament because practical runtime was too high, but kept for audit. |
| `archives/mnspi_cloud_remove_update_20210507.zip` | Modified NSPI / cloud removal | Related NSPI variant; useful for fallback/confidence/edge-handling ideas. |
| `archives/nspi_time_series_python_version-updated_20200627.zip` | NSPI time series Python | Most relevant candidate for future multi-date/per-pixel donor selection ideas. |
| `archives/python_code_and_sample_data_for_computing_accuracy_metrics.zip` | APA / optimal accuracy metrics | Supporting metrics beyond RMSE for tournament reporting. |

## Integrity

SHA256 hashes are in:

```text
archives/SHA256SUMS.txt
```

## Extracted locations

```text
extracted/nspi_update_20100824/
extracted/gnspi_update_20130317/
extracted/mnspi_cloud_remove_update_20210507/
extracted/nspi_time_series_python_version-updated_20200627/
extracted/python_code_and_sample_data_for_computing_accuracy_metrics/
```

## Notes

- Original code languages include IDL and Python.
- Original directory/file names are preserved where possible.
- Do not edit extracted upstream files directly. If adapting logic, implement it in local scripts and cite this folder in comments/docs.
- Current tournament already contains a Python NSPI-style implementation and USGS-style local regression baseline; these archived files are reference/audit sources, not runtime dependencies.
