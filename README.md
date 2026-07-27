# Physics-Guided Glacier Mapping

This repository is the research software companion to the 2025 dissertation
*Physics-Guided Strategies for Enhancing Neural Networks Trained With Limited
Data*. It contains the HKH glacier segmentation pipeline for clean ice (CI),
debris-covered ice (DCI), and multiclass mapping from Landsat imagery.

Results reported in the dissertation are preserved as defended. A subsequent
2026 replication and robustness study closely reproduced the Aryal et al.
baseline, while finding that later model improvements were sensitive to dataset
construction, training semantics, and random seed. Because the follow-up
protocols do not recreate every condition of the defended experiments, the
dissertation and post-dissertation findings are presented separately. This
repository makes no new state-of-the-art claim.

The detailed follow-up is in [POST_DISSERTATION_RESULTS.md](POST_DISSERTATION_RESULTS.md).

## Defended dissertation results

These are the results reported in the dissertation, not re-estimates from the
later protocol.

| Model | CI IoU | DCI IoU |
|---|---:|---:|
| Standard U-Net | 65.60% | 28.50% |
| Boundary-aware U-Net (Aryal et al.) | 68.17% | 35.94% |
| Flow only | 63.50% | 38.50% |
| Full static physics | **71.22%** | 45.92% |
| Velocity channels | 70.78% | 32.40% |
| Velocity channels and loss | 61.83% | 41.91% |
| Complete physics-informed model | 65.85% | **46.07%** |

## Reproducibility boundary

The original dataset was built from Landsat Collection 1 assets and private
Earth Engine assets used by Aryal's scripts. Collection 1 is no longer available
in the current Earth Engine catalog, so that exact input cannot be regenerated
from public services. The four original JavaScript files are preserved unchanged
under `google_earth_engine/boundary_aware_unet_paper/`.

The supported rebuild uses public Landsat Collection 2 Level 1 products,
NASADEM, and ITS_LIVE. The canonical scene selection, donor selection, exact-date
overrides, spatial grids, and source links are recorded in
`dataset/hkh_rebuild_manifest.json`.

Glacier labels are not redistributed. Obtain them from ICIMOD's
[Clean Ice and Debris Covered Glaciers of the HKH Region](https://rds.icimod.org/metadata/c6a59a04-e6f7-4bf6-a6a8-f1cd534a6b62)
record. The accompanying source report is
[The Status of Glaciers in the Hindu Kush Himalayan Region](https://lib.icimod.org/records/wt6cp-2bt35).

## Dataset families

| Dataset | Purpose | Recipe |
|---|---|---|
| Dissertation-era `comprehensive_v3` | Defended experiments and later legacy comparisons | `configs/datasets/dissertation.yaml` |
| Aryal 2023 eight-band reproduction | Public-code parity study | `configs/datasets/aryal_2023.yaml` |
| C02 current-date agreement | Updated DN-domain rebuild | `configs/datasets/c02_agreement.yaml` |
| C02 exact-legacy-date agreement | Fair date-controlled comparison | `configs/datasets/c02_legacy_dates.yaml` |
| C02 agreement with velocity v2 | Date-aware ITS_LIVE comparison | `configs/datasets/c02_velocity_v2.yaml` |

## Installation

Use `uv` from the repository root:

```bash
uv pip install -e .
uv pip install -e ".[dev]"
uv run python scripts/test.py --unit
```

## Local paths

`configs/servers.yaml` contains portable repository-relative defaults. Put
machine-specific paths in `configs/servers.local.yaml`; this file is ignored by
Git and merged over the public defaults.

```yaml
local:
  output_path: /path/to/run/output
  raw_data_path: /path/to/HKH_raw
  image_dir: /path/to/HKH_raw/Landsat7_2005
  dem_dir: /path/to/HKH_raw/DEM
  velocity_dir: /path/to/HKH_raw/Velocity
  labels_dir: /path/to/HKH_raw/labels
  processed_data_path: /path/to/processed/HKH
  num_workers: 4
```

The local `output/` directory is ignored by Git and is the durable experiment
record. Repository cleanup commands do not manage it.

## Rebuild the updated dataset

Set an Earth Engine project explicitly. The exporter has no personal project or
remote endpoint embedded in it.

```bash
export EE_PROJECT=your-earth-engine-project

# Validate the complete Landsat export plan without contacting Earth Engine.
uv run python google_earth_engine/export_hkh_dataset.py \
  --variant c02_current --all --dry-run

# Queue the Landsat target and donor exports.
uv run python google_earth_engine/export_hkh_dataset.py \
  --variant c02_current --all

# Queue fishnet-aligned NASADEM elevation, slope, aspect, and curvature.
uv run python google_earth_engine/export_hkh_dataset.py \
  --variant c02_current --asset dem --all
```

Download the Landsat exports into separate `targets/` and `donors/`
directories. Then construct the fishnet rasters. `HKH_RAW_ROOT` is optional;
explicit paths take precedence.

```bash
uv run python dataset/build_hkh_fishnet.py \
  --dataset-variant c02_current \
  --target-dir /path/to/HKH_raw/HKH_c02_current_raw_scenes/targets \
  --donor-dir /path/to/HKH_raw/HKH_c02_current_raw_scenes/donors \
  --template-dir /path/to/HKH_raw/Landsat7_2005 \
  --output-root /path/to/HKH_raw \
  --variant-folder-prefix HKH_full8_c02t1_dn \
  --variants raw_target,agreement_quality_step3

uv run python dataset/apply_relaxed_valid_mask.py \
  --raw-root /path/to/HKH_raw \
  --dataset-variant c02_current \
  --target-dir /path/to/HKH_raw/HKH_c02_current_raw_scenes/targets \
  --source-prefix HKH_full8_c02t1_dn \
  --variants raw_target_relaxed_valid,agreement_quality_step3_relaxed_valid
```

Use `--variant c02_legacy_dates` in the exporter and
`--dataset-variant c02_legacy_dates` in the builders for the exact-date dataset.
The reselected-donor overlay is named `c02_legacy_dates_reselected`.

Generate ITS_LIVE velocity rasters after the Landsat fishnet exists:

```bash
# Dissertation-era seven-year velocity mosaic.
uv run python scripts/create_velocity_from_itslive_mosaic.py \
  --server local

# Date-aware v2 product using per-pixel provenance.
uv run python scripts/create_velocity_from_itslive_mosaic_v2.py \
  --server local --variant agreement_quality_step3
```

## Preprocess and train

Preprocessing slices the raw rasters and labels, derives terrain and spectral
channels, and packs normalized `X.npy`/`y.npy` arrays.

```bash
uv run python scripts/preprocess.py \
  --server local --config configs/datasets/c02_agreement.yaml \
  --regenerate-full
```

For the velocity-v2 experiments, first create the full source dataset and then
pack the three documented channel recipes:

```bash
uv run python scripts/preprocess.py \
  --server local --config configs/datasets/c02_velocity_v2.yaml \
  --regenerate-full

uv run python scripts/preprocess.py \
  --server local --config configs/datasets/c02_velocity_v2.yaml \
  --recipes c02t1_dn_agreement_ldem_v2_baseline,c02t1_dn_agreement_ldem_v2_speed,c02t1_dn_agreement_ldem_v2_quality
```

The Aryal reproduction intentionally has a separate preprocessing entry point:

```bash
uv run python dataset/create_aryal_2023_dataset.py \
  --server local --labels /path/to/HKH_CIDC_5basins_all.shp
```

Training uses the four-level merge
`train.yaml -> servers.yaml -> tasks/<task>.yaml -> experiment.yaml`. Canonical
experiments are under `configs/local/`; change `training_opts.seed` for the
documented seed sets.

```bash
uv run python scripts/train.py \
  --config configs/local/debris_ice/c02_agreement.yaml \
  --server local --gpu 0
```

No checkpoints are stored in Git. Each run writes its resolved configuration,
TensorBoard events, checkpoints, and test metrics beneath the configured output
directory.

Evaluate one model or a paired CI/DCI model from retained local checkpoints:

```bash
uv run python scripts/predict.py \
  --ci-run-name <ci_run> --deb-run-name <dci_run> \
  --server local --gpu 0 --split test
```

## Optional MLflow and ntfy

Install optional MLflow support with `uv pip install -e ".[tracking]"`.

Both integrations are disabled by default. Training never attempts a network
connection unless they are explicitly configured.

```bash
export MLFLOW_TRACKING_URI=https://your-mlflow-server

uv run python scripts/train.py \
  --config configs/local/debris_ice/c02_agreement.yaml \
  --server local --gpu 0 --mlflow-enabled true

uv run python scripts/upload_to_mlflow.py output/<run_name> \
  --server local

uv run python scripts/plot_mlflow_run.py --run-name <run_name>
```

MLflow artifact uploads remain independently disabled unless
`training_opts.mlflow_artifacts_enabled: true` or
`--mlflow-artifacts-enabled true` is supplied.

The sequential runner enables ntfy only when both a topic and server are given,
or when the topic is a complete publish URL:

```bash
export NTFY_URL=https://ntfy.sh
export NTFY_TOPIC=my-private-topic
uv run bash run_sequential_training.sh local --tasks debris_ice
```

Add `MLFLOW_TRACKING_URI` to the environment or pass `--mlflow-uri` to enable
MLflow for the same batch.

## Repository map

```text
configs/                         portable defaults and canonical experiments
dataset/                         manifest and local dataset builders
glacier_mapping/                 data, Lightning, model, and utility modules
google_earth_engine/             one supported exporter and original Aryal scripts
scripts/                         preprocess, train, predict, velocity, tests, MLflow
POST_DISSERTATION_RESULTS.md      replication and robustness report
```

The complete pre-cleanup research snapshot is commit `9793e17`. Removed
exploratory configs and audit helpers remain available through Git history. The
Gradio demo and its sample arrays were archived outside this repository.
