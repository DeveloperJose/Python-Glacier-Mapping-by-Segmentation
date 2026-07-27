#!/usr/bin/env bash
# Regenerate full preprocessing datasets for configs under configs/datasets.
# Usage:
#   uv run bash run_all_preprocess.sh local
#   uv run bash run_all_preprocess.sh local 'c02_*.yaml'
# Extra args after the optional glob are passed to scripts/preprocess.py.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <server> [config_glob] [extra preprocess args...]"
  exit 1
fi

SERVER="$1"
shift

CONFIG_DIR="configs/datasets"
CONFIG_GLOB="*.yaml"
if [[ $# -gt 0 && "$1" == *.yaml ]]; then
  CONFIG_GLOB="$1"
  shift
fi

if [[ ! -d "${CONFIG_DIR}" ]]; then
  echo "Config directory not found: ${CONFIG_DIR}"
  exit 1
fi

shopt -s nullglob
configs=("${CONFIG_DIR}"/${CONFIG_GLOB})
shopt -u nullglob

if [[ ${#configs[@]} -eq 0 ]]; then
  echo "No configs matched: ${CONFIG_DIR}/${CONFIG_GLOB}"
  exit 1
fi

for config in "${configs[@]}"; do
  if [[ "$(basename "${config}")" == "aryal_2023.yaml" ]]; then
    echo ">>> Skipping ${config}; use dataset/create_aryal_2023_dataset.py"
    continue
  fi
  echo ">>> Regenerating ${config} for server ${SERVER}"
  uv run python scripts/preprocess.py --server "${SERVER}" --config "${config}" --regenerate-full "$@"
done
