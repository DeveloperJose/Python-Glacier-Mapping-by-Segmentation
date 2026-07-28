# Physics-Guided Glacier Mapping

Code and workflows for the 2025 dissertation
*Physics-Guided Strategies for Enhancing Neural Networks Trained With Limited
Data*. It implements HKH glacier segmentation for clean ice (CI),
debris-covered ice (DCI), and multiclass mapping from Landsat imagery.

The results below are those reported in the defended dissertation.

## Defended dissertation results

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

The original dataset used Landsat Collection 1 and the private Earth Engine
assets referenced by Aryal's scripts. Collection 1 is no longer available in
the Earth Engine catalog, so the original inputs cannot be regenerated from
public services. The original JavaScript files are in
`google_earth_engine/boundary_aware_unet_paper/`.

A public-data rebuild uses Landsat Collection 2 Level 1, NASADEM, and ITS_LIVE.
Scene selection, donor selection, spatial grids, and source links are in
`dataset/hkh_rebuild_manifest.json`.

Glacier labels are not redistributed. Obtain them from ICIMOD's
[Clean Ice and Debris Covered Glaciers of the HKH Region](https://rds.icimod.org/metadata/c6a59a04-e6f7-4bf6-a6a8-f1cd534a6b62)
record and [The Status of Glaciers in the Hindu Kush Himalayan Region](https://lib.icimod.org/records/wt6cp-2bt35).

## Dataset families

| Dataset | Purpose | Recipe |
|---|---|---|
| Dissertation-era `comprehensive_v3` | Defended experiments | `configs/datasets/dissertation.yaml` |
| Aryal eight-band workflow | Reference implementation workflow | `configs/datasets/aryal_2023.yaml` |
| Public-data rebuild | Build from currently available public inputs | `configs/datasets/` |

## Installation

Use `uv` from the repository root:

```bash
uv pip install -e .
uv pip install -e ".[dev]"
uv run python scripts/test.py --unit
```

## Local paths

`configs/servers.yaml` provides repository-relative defaults. Add
machine-specific paths to `configs/servers.local.yaml`; Git ignores this file
and merges it over the defaults.

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

Git ignores `output/`, which stores local experiment runs.

## Rebuild public inputs

Set an Earth Engine project before exporting.

```bash
export EE_PROJECT=your-earth-engine-project

# Inspect the Landsat export plan without contacting Earth Engine.
uv run python google_earth_engine/export_hkh_dataset.py \
  --variant c02_current --all --dry-run

# Queue Landsat target and donor exports.
uv run python google_earth_engine/export_hkh_dataset.py \
  --variant c02_current --all

# Queue fishnet-aligned NASADEM elevation, slope, aspect, and curvature.
uv run python google_earth_engine/export_hkh_dataset.py \
  --variant c02_current --asset dem --all
```

Download Landsat exports into separate `targets/` and `donors/` directories,
then build the fishnet rasters. `HKH_RAW_ROOT` is optional; explicit paths take
precedence.

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

Generate the ITS_LIVE velocity mosaic after building the Landsat fishnet:

```bash
# Dissertation-era seven-year velocity mosaic.
uv run python scripts/create_velocity_from_itslive_mosaic.py \
  --server local
```

## Preprocess and train

Preprocessing slices the raw rasters and labels, derives terrain and spectral
channels, and packs normalized `X.npy`/`y.npy` arrays.

```bash
uv run python scripts/preprocess.py \
  --server local --config configs/datasets/dissertation.yaml \
  --regenerate-full
```

The Aryal workflow has a separate preprocessing command:

```bash
uv run python dataset/create_aryal_2023_dataset.py \
  --server local --labels /path/to/HKH_CIDC_5basins_all.shp
```

Training combines
`train.yaml -> servers.yaml -> tasks/<task>.yaml -> experiment.yaml`.
Experiment configs are in `configs/local/`.

```bash
uv run python scripts/train.py \
  --config configs/local/debris_ice/dissertation_dataset.yaml \
  --server local --gpu 0
```

Git does not store checkpoints. Each run writes the resolved configuration,
TensorBoard events, checkpoints, and test metrics to the configured output
directory.

Evaluate one model or a paired CI/DCI model from retained local checkpoints:

```bash
uv run python scripts/predict.py \
  --ci-run-name <ci_run> --deb-run-name <dci_run> \
  --server local --gpu 0 --split test
```

## Optional MLflow and ntfy

Install optional MLflow support with `uv pip install -e ".[tracking]"`.

MLflow and ntfy are disabled by default. Configure them explicitly to enable
network access.

```bash
export MLFLOW_TRACKING_URI=https://your-mlflow-server

uv run python scripts/train.py \
  --config configs/local/debris_ice/dissertation_dataset.yaml \
  --server local --gpu 0 --mlflow-enabled true

uv run python scripts/upload_to_mlflow.py output/<run_name> \
  --server local

uv run python scripts/plot_mlflow_run.py --run-name <run_name>
```

MLflow artifact uploads are disabled unless
`training_opts.mlflow_artifacts_enabled: true` or
`--mlflow-artifacts-enabled true` is supplied.

The sequential runner sends ntfy notifications when both a topic and server are
set, or when the topic is a complete publish URL:

```bash
export NTFY_URL=https://ntfy.sh
export NTFY_TOPIC=my-private-topic
uv run bash run_sequential_training.sh local --tasks debris_ice
```

Set `MLFLOW_TRACKING_URI` or pass `--mlflow-uri` to enable MLflow for the same
batch.

## Repository map

```text
configs/                         defaults and experiment configs
dataset/                         manifest and local dataset builders
glacier_mapping/                 data, Lightning, model, and utility modules
google_earth_engine/             Earth Engine exporter and original Aryal scripts
scripts/                         preprocess, train, predict, velocity, tests, MLflow
```
