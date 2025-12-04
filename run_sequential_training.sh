#!/bin/bash
################################################################################
# Sequential Training Script for Glacier Mapping
#
# Usage: ./run_sequential_training.sh [desktop|bilbo|frodo] [OPTIONS]
#
# This script runs all config files for a specified server sequentially,
# continuing on failure and logging results to both console and summary file.
################################################################################

set -o pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Default values
GPU=0
MLFLOW_URI="https://mlflow.developerjose.duckdns.org/"
DRY_RUN=false
PAUSE_SECONDS=60

# New priority and filtering options
TASK_FILTER=""                # Empty = all tasks
PRIORITY_ORDER="dci,ci,multi" # Default priority: DCI → CI → Multi
EXCLUDE_BASE=false
ONLY_BASE=false

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${BOLD}$1${NC}"
    echo -e "${CYAN}============================================================${NC}"
}

print_subheader() {
    echo -e "${BLUE}------------------------------------------------------------${NC}"
    echo -e "${BOLD}$1${NC}"
    echo -e "${BLUE}------------------------------------------------------------${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

format_duration() {
    local duration=$1
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))

    if [ $hours -gt 0 ]; then
        printf "%dh %02dm %02ds" $hours $minutes $seconds
    elif [ $minutes -gt 0 ]; then
        printf "%dm %02ds" $minutes $seconds
    else
        printf "%ds" $seconds
    fi
}

show_usage() {
    cat <<EOF
${BOLD}Sequential Training Script for Glacier Mapping${NC}

${BOLD}USAGE:${NC}
    ./run_sequential_training.sh SERVER [OPTIONS]

${BOLD}ARGUMENTS:${NC}
    SERVER              Server name: desktop, bilbo, or frodo

${BOLD}BASIC OPTIONS:${NC}
    --gpu N             GPU device number (default: 0)
    --dry-run           Show what would run without executing
    --pause N           Pause N seconds between runs (default: 60)
    --mlflow-uri URI    MLflow tracking URI (default: https://mlflow.developerjose.duckdns.org/)

${BOLD}FILTERING & PRIORITY OPTIONS:${NC}
    --tasks TASKS       Comma-separated tasks to run (default: dci,ci,multi)
                        Example: --tasks dci,ci
    --priority ORDER    Task execution priority (default: dci,ci,multi)
                        Example: --priority ci,dci,multi
    --exclude-base      Skip all base dataset configs
    --only-base         Run only base dataset configs

${BOLD}EXAMPLES:${NC}
    # Run all configs in priority order (DCI → CI → Multi, base last)
    ./run_sequential_training.sh bilbo

    # Run only DCI and CI tasks
    ./run_sequential_training.sh bilbo --tasks dci,ci

    # Run all configs except base datasets
    ./run_sequential_training.sh bilbo --exclude-base

    # Run only base datasets (for baseline comparison)
    ./run_sequential_training.sh bilbo --only-base

    # Preview execution order
    ./run_sequential_training.sh bilbo --dry-run

    # Custom priority order (CI first)
    ./run_sequential_training.sh bilbo --priority ci,dci,multi

    # Run DCI only, exclude base
    ./run_sequential_training.sh bilbo --tasks dci --exclude-base

${BOLD}DEFAULT EXECUTION ORDER:${NC}
    1. DCI (Debris-Covered Ice) - phys32 → phys64 variants → phys128 → physfull
    2. CI (Clean Ice) - phys32 → phys64 variants → phys128 → physfull
    3. Multi (Multi-class) - phys32 → phys64 variants → phys128 → physfull
    4. Base datasets - DCI base → CI base → Multi base (at the very end)

${BOLD}NOTES:${NC}
    - Configs are auto-detected from configs/{SERVER}_*.yaml
    - Script continues on failure (failed runs are logged)
    - Logs are saved to sequential_training_summary_{TIMESTAMP}.log
    - Base datasets run last by default (can be changed with --only-base)
    - Within each task, configs are sorted by sample size (ascending)

EOF
}

################################################################################
# Config Sorting and Filtering Functions
################################################################################

sort_configs_by_priority() {
    # Sort configs using Python for complex multi-key sorting
    # Pass configs as arguments

    python3 -c '
import sys
import os

# Get configs from arguments
configs = sys.argv[1:]

def parse_config(filename):
    """Extract task, dataset, sample_size, scale from filename."""
    # New structure: configs/{server}/{task}/{experiment}.yaml
    parts = filename.split("/")
    base = os.path.basename(filename).replace(".yaml", "")
    
    # Extract task from path (e.g., "configs/frodo/clean_ice/base.yaml")
    if len(parts) >= 3:
        task_folder = parts[2]  # clean_ice, debris_ice, multiclass
        # Map to old naming convention
        if task_folder == "clean_ice":
            task = "ci"
        elif task_folder == "debris_ice":
            task = "dci"
        elif task_folder == "multiclass":
            task = "multi"
        else:
            task = "unknown"
    else:
        task = "unknown"
    
    # Determine if base
    is_base = (base == "base")
    
    # Extract sample size from experiment name
    if "phys32" in base:
        sample_size = 32
    elif "phys64" in base:
        sample_size = 64
    elif "phys128" in base:
        sample_size = 128
    elif "physfull" in base:
        sample_size = 999
    else:
        sample_size = 0
    
    # Extract scale (for secondary sort within same sample size)
    if "s05" in base:
        scale = 0.5
    elif "s075" in base:
        scale = 0.75
    elif "s1" in base:
        scale = 1.0
    else:
        scale = 1.0
    
    return {
        "filename": filename,
        "task": task,
        "is_base": is_base,
        "sample_size": sample_size,
        "scale": scale
    }

# Parse all configs
parsed = [parse_config(c) for c in configs]

# Define task priority from PRIORITY_ORDER environment variable
priority_order = os.environ.get("PRIORITY_ORDER", "dci,ci,multi").split(",")
task_priority = {task.strip(): i for i, task in enumerate(priority_order)}

# Sort
sorted_configs = sorted(parsed, key=lambda x: (
    task_priority.get(x["task"], 999),
    1 if x["is_base"] else 0,
    x["sample_size"],
    x["scale"]
))

# Output sorted filenames
for item in sorted_configs:
    print(item["filename"])
' "$@"
}

filter_configs_by_tasks() {
    # Filter configs by task type
    # Arguments: task_filter (comma-separated), config files (space-separated)

    local task_filter="$1"
    shift
    local configs=("$@")

    if [ -z "$task_filter" ]; then
        # No filter, return all
        printf '%s\n' "${configs[@]}"
        return
    fi

    # Convert comma-separated tasks to array
    IFS=',' read -ra tasks <<<"$task_filter"

    # Filter configs (new structure: configs/{server}/{task}/{experiment}.yaml)
    for config in "${configs[@]}"; do
        for task in "${tasks[@]}"; do
            task=$(echo "$task" | xargs) # Trim whitespace
            # Map task abbreviation to folder name
            if [[ "$task" == "ci" && "$config" =~ /clean_ice/ ]]; then
                echo "$config"
                break
            elif [[ "$task" == "dci" && "$config" =~ /debris_ice/ ]]; then
                echo "$config"
                break
            elif [[ "$task" == "multi" && "$config" =~ /multiclass/ ]]; then
                echo "$config"
                break
            fi
        done
    done
}

filter_base_configs() {
    # Filter configs based on base/non-base
    # Arguments: mode (exclude|only), config files (space-separated)

    local mode="$1"
    shift
    local configs=("$@")

    for config in "${configs[@]}"; do
        local basename=$(basename "$config" .yaml)
        local is_base=false

        if [[ "$basename" =~ _base$ ]]; then
            is_base=true
        fi

        if [ "$mode" = "exclude" ] && [ "$is_base" = false ]; then
            echo "$config"
        elif [ "$mode" = "only" ] && [ "$is_base" = true ]; then
            echo "$config"
        fi
    done
}

################################################################################
# Parse Arguments
################################################################################

if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

# Check for help flag first
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_usage
    exit 0
fi

SERVER=$1
shift

while [[ $# -gt 0 ]]; do
    case $1 in
    --gpu)
        GPU="$2"
        shift 2
        ;;
    --dry-run)
        DRY_RUN=true
        shift
        ;;
    --pause)
        PAUSE_SECONDS="$2"
        shift 2
        ;;
    --mlflow-uri)
        MLFLOW_URI="$2"
        shift 2
        ;;
    --tasks)
        TASK_FILTER="$2"
        shift 2
        ;;
    --priority)
        PRIORITY_ORDER="$2"
        shift 2
        ;;
    --exclude-base)
        EXCLUDE_BASE=true
        shift
        ;;
    --only-base)
        ONLY_BASE=true
        shift
        ;;
    --help)
        show_usage
        exit 0
        ;;
    *)
        echo -e "${RED}Error: Unknown option: $1${NC}"
        show_usage
        exit 1
        ;;
    esac
done

################################################################################
# Validate Arguments
################################################################################

# Validate server
if [[ ! "$SERVER" =~ ^(desktop|bilbo|frodo)$ ]]; then
    print_error "Invalid server: $SERVER"
    echo "Valid servers: desktop, bilbo, frodo"
    exit 1
fi

# Check if server exists in servers.yaml
if ! grep -q "^${SERVER}:" configs/servers.yaml 2>/dev/null; then
    print_error "Server '$SERVER' not found in configs/servers.yaml"
    exit 1
fi

# Validate conflicting options
if [ "$EXCLUDE_BASE" = true ] && [ "$ONLY_BASE" = true ]; then
    print_error "Cannot use both --exclude-base and --only-base together"
    exit 1
fi

# Validate task filter
if [ -n "$TASK_FILTER" ]; then
    IFS=',' read -ra tasks <<<"$TASK_FILTER"
    for task in "${tasks[@]}"; do
        task=$(echo "$task" | xargs)
        if [[ ! "$task" =~ ^(ci|dci|multi)$ ]]; then
            print_error "Invalid task in --tasks: $task"
            echo "Valid tasks: ci, dci, multi"
            exit 1
        fi
    done
fi

# Validate priority order
IFS=',' read -ra priority_tasks <<<"$PRIORITY_ORDER"
for task in "${priority_tasks[@]}"; do
    task=$(echo "$task" | xargs)
    if [[ ! "$task" =~ ^(ci|dci|multi)$ ]]; then
        print_error "Invalid task in --priority: $task"
        echo "Valid tasks: ci, dci, multi"
        exit 1
    fi
done

# Export variables for use in Python sorting function
export SERVER
export PRIORITY_ORDER

################################################################################
# Find and Sort Config Files
################################################################################

# Step 1: Discover all configs (new hierarchical structure)
mapfile -t ALL_CONFIGS < <(find configs/${SERVER} -name "*.yaml" -type f 2>/dev/null | sort)

if [ ${#ALL_CONFIGS[@]} -eq 0 ]; then
    print_error "No config files found in: configs/${SERVER}/"
    exit 1
fi

# Step 2: Apply task filter if specified
if [ -n "$TASK_FILTER" ]; then
    mapfile -t FILTERED_CONFIGS < <(filter_configs_by_tasks "$TASK_FILTER" "${ALL_CONFIGS[@]}")
else
    FILTERED_CONFIGS=("${ALL_CONFIGS[@]}")
fi

# Step 3: Apply base filtering
if [ "$EXCLUDE_BASE" = true ]; then
    mapfile -t FILTERED_CONFIGS < <(filter_base_configs "exclude" "${FILTERED_CONFIGS[@]}")
elif [ "$ONLY_BASE" = true ]; then
    mapfile -t FILTERED_CONFIGS < <(filter_base_configs "only" "${FILTERED_CONFIGS[@]}")
fi

# Step 4: Sort by priority
mapfile -t CONFIG_FILES < <(sort_configs_by_priority "${FILTERED_CONFIGS[@]}")

# Check if we have any configs after filtering
if [ ${#CONFIG_FILES[@]} -eq 0 ]; then
    print_error "No config files match the specified criteria"
    echo ""
    echo "Applied filters:"
    [ -n "$TASK_FILTER" ] && echo "  Tasks: $TASK_FILTER"
    [ "$EXCLUDE_BASE" = true ] && echo "  Exclude base: yes"
    [ "$ONLY_BASE" = true ] && echo "  Only base: yes"
    exit 1
fi

################################################################################
# Setup Logging
################################################################################

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="sequential_training_summary_${SERVER}_${TIMESTAMP}.log"

# Function to log to both console and file
log() {
    echo -e "$@" | tee -a "$LOG_FILE"
}

log_only() {
    echo -e "$@" >>"$LOG_FILE"
}

################################################################################
# Display Plan
################################################################################

print_header "Sequential Training Script"
log ""
log "${BOLD}Configuration:${NC}"
log "  Server:        $SERVER"
log "  GPU:           $GPU"
log "  MLflow URI:    $MLFLOW_URI"
log "  Priority:      $PRIORITY_ORDER"
log "  Tasks:         ${TASK_FILTER:-all (dci,ci,multi)}"
log "  Exclude base:  $EXCLUDE_BASE"
log "  Only base:     $ONLY_BASE"
log "  Dry run:       $DRY_RUN"
log "  Pause:         ${PAUSE_SECONDS}s between runs"
log "  Log file:      $LOG_FILE"
log ""
log "${BOLD}Execution order (${#CONFIG_FILES[@]} configs):${NC}"
log ""

# Display in actual execution order with task labels
for i in "${!CONFIG_FILES[@]}"; do
    config="${CONFIG_FILES[$i]}"
    basename=$(basename "$config" .yaml)

    # Color code by task
    if [[ "$basename" =~ _dci_ ]] || [[ "$basename" =~ _dci$ ]]; then
        task_label="${MAGENTA}[DCI]${NC}"
    elif [[ "$basename" =~ _ci_ ]] || [[ "$basename" =~ _ci$ ]]; then
        task_label="${GREEN}[CI]${NC}"
    elif [[ "$basename" =~ _multi_ ]] || [[ "$basename" =~ _multi$ ]]; then
        task_label="${BLUE}[Multi]${NC}"
    else
        task_label="[???]"
    fi

    log "  $((i + 1)). $task_label $config"
done
log ""

print_header ""

if [ "$DRY_RUN" = true ]; then
    echo ""
    print_info "DRY RUN MODE - Commands that would be executed:"
    echo ""
    for config in "${CONFIG_FILES[@]}"; do
        echo "uv run python scripts/train.py \\"
        echo "  --config $config \\"
        echo "  --server $SERVER \\"
        echo "  --gpu $GPU \\"
        echo "  --mlflow-enabled true \\"
        echo "  --tracking-uri $MLFLOW_URI"
        echo ""
    done
    exit 0
fi

################################################################################
# Signal Handling
################################################################################

INTERRUPTED=false

cleanup() {
    INTERRUPTED=true
    echo ""
    print_error "Interrupted by user (Ctrl+C)"
    echo ""
    # Summary will be printed by the main loop
}

trap cleanup SIGINT SIGTERM

################################################################################
# Run Training Sequentially
################################################################################

TOTAL_CONFIGS=${#CONFIG_FILES[@]}
SUCCESSFUL_RUNS=0
FAILED_RUNS=0
declare -a FAILED_CONFIGS
declare -a RUN_DURATIONS

OVERALL_START=$(date +%s)

for i in "${!CONFIG_FILES[@]}"; do
    if [ "$INTERRUPTED" = true ]; then
        break
    fi

    CONFIG=${CONFIG_FILES[$i]}
    CONFIG_NUM=$((i + 1))

    echo ""
    print_header "[$CONFIG_NUM/$TOTAL_CONFIGS] Running: $(basename $CONFIG)"
    log_only ""
    log_only "============================================================"
    log_only "[$CONFIG_NUM/$TOTAL_CONFIGS] Running: $(basename $CONFIG)"
    log_only "============================================================"

    RUN_START=$(date +%s)
    RUN_START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

    log "Started: $RUN_START_TIME"
    log ""

    # Build command
    CMD="uv run python scripts/train.py \
--config $CONFIG \
--server $SERVER \
--gpu $GPU \
--mlflow-enabled true \
--tracking-uri $MLFLOW_URI"

    log_only "Command: $CMD"
    log_only ""

    print_subheader "Training Output"

    # Run training and capture exit code
    if $CMD 2>&1 | tee -a "$LOG_FILE"; then
        EXIT_CODE=${PIPESTATUS[0]}
    else
        EXIT_CODE=$?
    fi

    RUN_END=$(date +%s)
    RUN_DURATION=$((RUN_END - RUN_START))
    RUN_DURATION_FMT=$(format_duration $RUN_DURATION)
    RUN_DURATIONS+=("$RUN_DURATION_FMT")

    echo ""

    if [ $EXIT_CODE -eq 0 ]; then
        print_success "SUCCESS (Duration: $RUN_DURATION_FMT)"
        log_only "✓ SUCCESS (Duration: $RUN_DURATION_FMT)"
        ((SUCCESSFUL_RUNS++))
    else
        print_error "FAILED (Exit code: $EXIT_CODE, Duration: $RUN_DURATION_FMT)"
        log_only "✗ FAILED (Exit code: $EXIT_CODE, Duration: $RUN_DURATION_FMT)"
        FAILED_CONFIGS+=("$CONFIG (exit code: $EXIT_CODE)")
        ((FAILED_RUNS++))
    fi

    print_header ""

    # Pause between runs if specified
    if [ $CONFIG_NUM -lt $TOTAL_CONFIGS ] && [ $PAUSE_SECONDS -gt 0 ]; then
        print_info "Pausing for ${PAUSE_SECONDS}s before next run..."
        sleep $PAUSE_SECONDS
    fi
done

OVERALL_END=$(date +%s)
OVERALL_DURATION=$((OVERALL_END - OVERALL_START))
OVERALL_DURATION_FMT=$(format_duration $OVERALL_DURATION)

################################################################################
# Print Summary
################################################################################

echo ""
print_header "SUMMARY"
log ""
log "${BOLD}Execution Summary:${NC}"
log "  Server:           $SERVER"
log "  Total configs:    $TOTAL_CONFIGS"
log "  Successful runs:  ${GREEN}$SUCCESSFUL_RUNS${NC}"
log "  Failed runs:      ${RED}$FAILED_RUNS${NC}"
log "  Total duration:   $OVERALL_DURATION_FMT"
log ""

if [ $FAILED_RUNS -gt 0 ]; then
    log "${RED}${BOLD}Failed Configs:${NC}"
    for failed in "${FAILED_CONFIGS[@]}"; do
        log "  ${RED}✗${NC} $failed"
    done
    log ""
fi

if [ "$INTERRUPTED" = true ]; then
    log "${YELLOW}${BOLD}Note: Execution was interrupted by user${NC}"
    log ""
fi

log "${BOLD}Individual Run Durations:${NC}"
for i in "${!CONFIG_FILES[@]}"; do
    if [ $i -lt ${#RUN_DURATIONS[@]} ]; then
        CONFIG_NAME=$(basename "${CONFIG_FILES[$i]}")
        DURATION="${RUN_DURATIONS[$i]}"
        log "  $((i + 1)). $CONFIG_NAME - $DURATION"
    fi
done
log ""

log "${BOLD}Full log saved to:${NC} $LOG_FILE"
print_header ""

################################################################################
# Exit with appropriate code
################################################################################

if [ "$INTERRUPTED" = true ]; then
    exit 130 # Standard exit code for SIGINT
elif [ $FAILED_RUNS -gt 0 ]; then
    exit 1
else
    exit 0
fi
