#!/usr/bin/env bash
set -o pipefail

usage() {
    printf '%s\n' \
        "Usage: $0 SERVER [options]" \
        "" \
        "Options:" \
        "  --gpu N              GPU index (default: 0)" \
        "  --tasks LIST         clean_ice,debris_ice,multiclass (default: all)" \
        "  --pause SECONDS      Delay between runs (default: 0)" \
        "  --dry-run            Print commands without training" \
        "  --mlflow-uri URI     Enable MLflow and use this tracking URI" \
        "  --ntfy-topic TOPIC   Enable ntfy notifications" \
        "  --ntfy-url URL       ntfy server base URL" \
        "  -h, --help           Show this help" \
        "" \
        "MLFLOW_TRACKING_URI, NTFY_TOPIC, and NTFY_URL may also be set in the environment."
}

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

SERVER=$1
shift
GPU=0
TASKS="clean_ice,debris_ice,multiclass"
PAUSE_SECONDS=0
DRY_RUN=false
MLFLOW_URI=${MLFLOW_TRACKING_URI:-}
NTFY_TOPIC=${NTFY_TOPIC:-}
NTFY_URL=${NTFY_URL:-}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU=$2; shift 2 ;;
        --tasks) TASKS=$2; shift 2 ;;
        --pause) PAUSE_SECONDS=$2; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --mlflow-uri) MLFLOW_URI=$2; shift 2 ;;
        --ntfy-topic) NTFY_TOPIC=$2; shift 2 ;;
        --ntfy-url) NTFY_URL=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage; exit 2 ;;
    esac
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR" || exit 1

IFS=',' read -r -a TASK_NAMES <<< "$TASKS"
CONFIGS=()
for task in "${TASK_NAMES[@]}"; do
    task=${task//[[:space:]]/}
    case "$task" in
        ci) task=clean_ice ;;
        dci) task=debris_ice ;;
        multi) task=multiclass ;;
    esac
    config_dir="configs/$SERVER/$task"
    if [[ ! -d "$config_dir" ]]; then
        printf 'Skipping missing config directory: %s\n' "$config_dir" >&2
        continue
    fi
    while IFS= read -r config; do
        CONFIGS+=("$config")
    done < <(find "$config_dir" -maxdepth 1 -type f -name '*.yaml' | sort)
done

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    printf 'No experiment configs found for server %s\n' "$SERVER" >&2
    exit 1
fi

if [[ -n "$NTFY_TOPIC" && "$NTFY_TOPIC" != http://* && "$NTFY_TOPIC" != https://* && -z "$NTFY_URL" ]]; then
    printf 'ntfy topic names require --ntfy-url or NTFY_URL\n' >&2
    exit 2
fi

ntfy_publish_url() {
    if [[ "$NTFY_TOPIC" == http://* || "$NTFY_TOPIC" == https://* ]]; then
        printf '%s' "$NTFY_TOPIC"
    else
        printf '%s/%s' "${NTFY_URL%/}" "$NTFY_TOPIC"
    fi
}

notify() {
    local title=$1
    local message=$2
    [[ -z "$NTFY_TOPIC" ]] && return 0
    if ! command -v curl >/dev/null 2>&1; then
        printf 'ntfy skipped: curl is unavailable\n' >&2
        return 0
    fi
    curl -fsS -H "Title: $title" -d "$message" "$(ntfy_publish_url)" >/dev/null || \
        printf 'ntfy notification failed: %s\n' "$title" >&2
}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="sequential_training_${TIMESTAMP}.log"
notify "Glacier training started" "Server: $SERVER; runs: ${#CONFIGS[@]}; GPU: $GPU"

SUCCESS=0
FAILED=0
for config in "${CONFIGS[@]}"; do
    command=(
        uv run python scripts/train.py
        --config "$config"
        --server "$SERVER"
        --gpu "$GPU"
    )
    if [[ -n "$MLFLOW_URI" ]]; then
        command+=(--mlflow-enabled true --tracking-uri "$MLFLOW_URI")
    else
        command+=(--mlflow-enabled false)
    fi

    printf 'Running: ' | tee -a "$LOG_FILE"
    printf '%q ' "${command[@]}" | tee -a "$LOG_FILE"
    printf '\n' | tee -a "$LOG_FILE"
    if [[ "$DRY_RUN" == true ]]; then
        continue
    fi

    if "${command[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        ((SUCCESS += 1))
    else
        ((FAILED += 1))
        notify "Glacier training failed" "Server: $SERVER; config: $config"
    fi
    if [[ "$PAUSE_SECONDS" -gt 0 ]]; then
        sleep "$PAUSE_SECONDS"
    fi
done

if [[ "$DRY_RUN" == true ]]; then
    printf 'Dry run complete: %s commands\n' "${#CONFIGS[@]}"
    exit 0
fi

notify "Glacier training finished" "Server: $SERVER; successful: $SUCCESS; failed: $FAILED"
printf 'Finished: %s successful, %s failed. Log: %s\n' "$SUCCESS" "$FAILED" "$LOG_FILE"
[[ "$FAILED" -eq 0 ]]
