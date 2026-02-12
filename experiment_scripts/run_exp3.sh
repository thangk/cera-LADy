#!/bin/bash

#===============================================================================
# EXPERIMENT 3: Baseline Comparison and Results Collection
# Purpose: Collect baseline results and exp2 results for final comparison
#          Runs baselines only if not already cached
#
# Prerequisites:
#   1. cd /path/to/LADy-kap  
#   2. ./experiment_scripts/run_exp3.sh -i <exp2_output>
#
# Note: This script primarily copies results, only runs baselines if needed
#===============================================================================

# Default values
INPUT_PATH=""
TOY_MODE=false
BACKGROUND_MODE=false

# Help menu
show_help() {
    cat << 'EOF'
EXPERIMENT 3: Baseline Comparison and Results Collection

DESCRIPTION:
    Collects baseline results from cache (or runs them if needed) and copies
    exp2 results to create a complete comparison dataset for data collection.

USAGE:
    ./experiment_scripts/run_exp3.sh -i <exp2_output_dir> [OPTIONS]

REQUIRED:
    -i, --input PATH       Path to Exp2 output directory 
                          (e.g., experiment_output/exp2_top_models)

OPTIONS:
    -h, --help             Show this help message and exit
    --toy                  Run in toy mode (currently not used, for future)
    --background           Run in background mode (internal use)

PREREQUISITES:
    1. cd /path/to/LADy-kap
    2. Complete run_exp2.sh first

WHAT THIS SCRIPT DOES:
    • Looks for baseline datasets in experiment_datasets/semeval_baselines/
    • Checks if baseline results exist in experiment_datasets/semeval_baselines/output/
    • If baseline results exist, copies them to exp3 output
    • If not, runs baselines for all architecture models and caches results
    • Copies all exp2 results to exp3 output for data collection

OUTPUT:
    • Normal mode: experiment_output/exp3_baseline_comparison[_runN]/
    • Baseline cache: experiment_datasets/semeval_baselines/output/
    • Summary: {output_dir}/baseline_comparison_summary.txt

MONITORING (when running baselines):
    • Process status: ps aux | grep run_exp3.sh
    • Progress logs: tail -f experiment_output/*exp3_*/experiments.log

EXAMPLES:
    # Collect results from exp2
    ./experiment_scripts/run_exp3.sh -i experiment_output/exp2_top_models
    
    # With specific exp2 run
    ./experiment_scripts/run_exp3.sh -i experiment_output/exp2_top_models_run2

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -i|--input)
                INPUT_PATH="$2"
                shift 2
                ;;
            --toy)
                TOY_MODE=true
                shift
                ;;
            --background)
                BACKGROUND_MODE=true
                shift
                ;;
            *)
                echo "❌ ERROR: Unknown option: $1"
                echo "Use -h or --help for usage information."
                exit 1
                ;;
        esac
    done
}

# Validate arguments
validate_args() {
    # Check required input
    if [[ -z "$INPUT_PATH" ]]; then
        echo "❌ ERROR: Input path is required. Use -i or --input to specify."
        echo "Use -h or --help for usage information."
        exit 1
    fi
    
    # Check if input exists
    if [[ ! -d "$INPUT_PATH" ]]; then
        echo "❌ ERROR: Input directory does not exist: $INPUT_PATH"
        exit 1
    fi
}

# Function to find next available directory name
get_unique_output_dir() {
    local base_name="exp3_baseline_comparison"
    if [[ "$TOY_MODE" == true ]]; then
        base_name="toy_exp3_baseline_comparison"
    fi
    
    local base_dir="experiment_output/${base_name}"
    local output_dir="${base_dir}"
    local counter=2
    
    # Check if base directory exists
    while [[ -d "${output_dir}" ]]; do
        output_dir="${base_dir}_run${counter}"
        counter=$((counter + 1))
    done
    
    echo "${output_dir}"
}

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${OUTPUT_DIR}/experiments.log"
}

# Function to wait for baselines to complete
wait_for_baselines() {
    local baseline_pid=$1
    local check_interval=10
    
    log_message "Waiting for baseline experiments to complete (PID: $baseline_pid)..."
    
    while kill -0 "$baseline_pid" 2>/dev/null; do
        # Check baseline status file if it exists
        local baseline_status_file="${BASELINE_CACHE_DIR}/status.log"
        if [[ -f "$baseline_status_file" ]]; then
            local status=$(grep "STATUS:" "$baseline_status_file" | tail -1 | awk '{print $3}')
            if [[ "$status" == "COMPLETED" ]]; then
                log_message "Baselines completed successfully"
                break
            elif [[ "$status" == "FAILED" ]]; then
                log_message "WARNING: Some baselines failed, continuing anyway"
                break
            fi
        fi
        sleep $check_interval
    done
    
    # Final check
    if ! kill -0 "$baseline_pid" 2>/dev/null; then
        log_message "Baseline process finished"
    fi
}

# Main execution starts here
parse_args "$@"

# Skip checks if running in background mode
if [[ "$BACKGROUND_MODE" != true ]]; then
    validate_args
    
    # Check if we're in the right directory
    if [[ ! -f "src/main.py" ]] || [[ ! -d "experiment_scripts" ]]; then
        echo "❌ ERROR: Please run this script from the LADy-kap project root directory"
        exit 1
    fi

    # Check if conda environment is activated
    if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
        echo "❌ ERROR: Please activate your conda environment for this project first"
        exit 1
    fi

    # Check if baselines need to be run
    BASELINE_DIR="experiment_datasets/semeval_baselines"
    BASELINE_CACHE_DIR="${BASELINE_DIR}/output"
    
    # Find all baseline XML files
    BASELINE_FILES=()
    if [[ -d "$BASELINE_DIR" ]]; then
        while IFS= read -r -d '' file; do
            BASELINE_FILES+=("$file")
        done < <(find "$BASELINE_DIR" -maxdepth 1 -name "*.xml" -type f -print0)
    fi
    
    if [[ ${#BASELINE_FILES[@]} -eq 0 ]]; then
        echo "❌ ERROR: No baseline datasets found in ${BASELINE_DIR}"
        exit 1
    fi
    
    # Check if any baselines need to be run
    NEED_TO_RUN_BASELINES=false
    ARCH_MODELS=("bert" "ctm" "btm" "rnd")
    
    for baseline_file in "${BASELINE_FILES[@]}"; do
        basename_file=$(basename "$baseline_file" .xml)
        for arch_model in "${ARCH_MODELS[@]}"; do
            cache_csv="${BASELINE_CACHE_DIR}/${basename_file}/${arch_model}/agg.ad.pred.eval.mean.csv"
            if [[ ! -f "$cache_csv" ]]; then
                NEED_TO_RUN_BASELINES=true
                break 2
            fi
        done
    done

    if [[ "$NEED_TO_RUN_BASELINES" == true ]]; then
        echo "🚀 Some baselines need to be computed. Launching baseline script..."
        echo ""
        
        # Call run_baselines.sh
        ./experiment_scripts/run_baselines.sh
        BASELINE_PID=$!
        
        echo ""
        echo "⏳ Waiting for baselines to complete before continuing with exp3..."
        echo "📊 You can monitor baseline progress in another terminal with:"
        echo "   tail -f experiment_datasets/semeval_baselines/output/baselines.log"
        echo ""
        
        # Launch exp3 in background after a delay to let baselines start
        sleep 5
        nohup bash "${BASH_SOURCE[0]}" --background -i "$INPUT_PATH" $([ "$TOY_MODE" == true ] && echo "--toy") > /dev/null 2>&1 &
        EXPERIMENT_PID=$!
        echo "✅ Experiment 3 launched in background!"
        echo "🆔 Process ID (PID): $EXPERIMENT_PID"
        echo ""
        echo "📊 Monitor exp3 progress with:"
        echo "   tail -f experiment_output/*exp3_*/experiments.log"
        echo ""
        echo "🛑 To stop exp3:"
        echo "   kill $EXPERIMENT_PID"
        exit 0
    else
        # All baselines are cached, just copy results
        echo "✅ All baselines are already cached. Copying results only..."
        # Continue to main execution without background mode
    fi
else
    validate_args
fi

#===================
# CONFIGURATION
#===================

# Note: GPU and experiment settings are handled by run_baselines.sh

# Directory Paths
export OUTPUT_DIR=$(get_unique_output_dir)
export LOG_DIR="${OUTPUT_DIR}/logs"
export BASELINE_DIR="experiment_datasets/semeval_baselines"
export BASELINE_CACHE_DIR="${BASELINE_DIR}/output"

# Architecture Models
ARCH_MODELS=("bert" "ctm" "btm" "rnd")

# Ensure we're in the project root directory
if [[ ! -f "src/main.py" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
    cd "${PROJECT_ROOT}" || exit 1
fi

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# No need to change directory as we're not running experiments directly

# Log experiment start
echo "==============================================================================="
echo "EXPERIMENT 3: Baseline Comparison and Results Collection"
echo "==============================================================================="
echo "Start Time: $(date)"
echo "Input Path: ${INPUT_PATH}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Baseline Directory: ${BASELINE_DIR}"
echo "Baseline Cache: ${BASELINE_CACHE_DIR}"
echo "==============================================================================="

log_message "Starting Experiment 3: Baseline Comparison and Results Collection"

# Find all baseline XML files
BASELINE_FILES=()
if [[ -d "$BASELINE_DIR" ]]; then
    while IFS= read -r -d '' file; do
        BASELINE_FILES+=("$file")
    done < <(find "$BASELINE_DIR" -maxdepth 1 -name "*.xml" -type f -print0 | sort -z)
fi

log_message "Found ${#BASELINE_FILES[@]} baseline dataset(s):"
for baseline_file in "${BASELINE_FILES[@]}"; do
    log_message "  - $(basename "$baseline_file")"
done

# Phase 1: Process baselines (check cache first, run if needed)
log_message ""
log_message "PHASE 1: Processing baseline results..."

BASELINES_TO_RUN=0
BASELINES_CACHED=0

for baseline_file in "${BASELINE_FILES[@]}"; do
    basename_file=$(basename "$baseline_file" .xml)
    
    for arch_model in "${ARCH_MODELS[@]}"; do
        cache_csv="${BASELINE_CACHE_DIR}/${basename_file}/${arch_model}/agg.ad.pred.eval.mean.csv"
        output_dir="${OUTPUT_DIR}/${arch_model}/baseline-${basename_file}"
        
        if [[ -f "$cache_csv" ]]; then
            # Copy from cache
            log_message "CACHE HIT: ${arch_model}/baseline-${basename_file} - copying from cache"
            mkdir -p "$output_dir"
            cp -r "${BASELINE_CACHE_DIR}/${basename_file}/${arch_model}"/* "$output_dir/" 2>/dev/null || true
            BASELINES_CACHED=$((BASELINES_CACHED + 1))
        else
            # Need to run this baseline
            log_message "CACHE MISS: ${arch_model}/baseline-${basename_file} - will run experiment"
            BASELINES_TO_RUN=$((BASELINES_TO_RUN + 1))
        fi
    done
done

log_message "Baselines status: ${BASELINES_CACHED} cached, ${BASELINES_TO_RUN} to run"

# If we need to run baselines, wait for them to complete
if [[ $BASELINES_TO_RUN -gt 0 ]]; then
    log_message "Waiting for baseline experiments to complete..."
    
    # Check if run_baselines.sh was called by the non-background process
    # If so, wait for it to complete
    baseline_status_file="${BASELINE_CACHE_DIR}/status.log"
    max_wait_time=3600  # 1 hour max wait
    wait_time=0
    check_interval=10
    
    while [[ $wait_time -lt $max_wait_time ]]; do
        # Re-check baseline status
        still_need_baselines=false
        for baseline_file in "${BASELINE_FILES[@]}"; do
            basename_file=$(basename "$baseline_file" .xml)
            for arch_model in "${ARCH_MODELS[@]}"; do
                cache_csv="${BASELINE_CACHE_DIR}/${basename_file}/${arch_model}/agg.ad.pred.eval.mean.csv"
                if [[ ! -f "$cache_csv" ]]; then
                    still_need_baselines=true
                    break 2
                fi
            done
        done
        
        if [[ "$still_need_baselines" == false ]]; then
            log_message "All baselines now available!"
            break
        fi
        
        # Check baseline status
        if [[ -f "$baseline_status_file" ]]; then
            status=$(grep "STATUS:" "$baseline_status_file" | tail -1 | awk '{print $3}')
            if [[ "$status" == "COMPLETED" ]] || [[ "$status" == "FAILED" ]]; then
                log_message "Baseline script finished with status: $status"
                break
            fi
        fi
        
        log_message "Still waiting for baselines... ($wait_time seconds elapsed)"
        sleep $check_interval
        wait_time=$((wait_time + check_interval))
    done
    
    if [[ $wait_time -ge $max_wait_time ]]; then
        log_message "WARNING: Timeout waiting for baselines, proceeding anyway"
    fi
    
    # Re-count cached baselines after waiting
    BASELINES_CACHED=0
    BASELINES_MISSING=0
    for baseline_file in "${BASELINE_FILES[@]}"; do
        basename_file=$(basename "$baseline_file" .xml)
        for arch_model in "${ARCH_MODELS[@]}"; do
            cache_csv="${BASELINE_CACHE_DIR}/${basename_file}/${arch_model}/agg.ad.pred.eval.mean.csv"
            output_dir="${OUTPUT_DIR}/${arch_model}/baseline-${basename_file}"
            
            if [[ -f "$cache_csv" ]]; then
                # Copy from cache
                mkdir -p "$output_dir"
                cp -r "${BASELINE_CACHE_DIR}/${basename_file}/${arch_model}"/* "$output_dir/" 2>/dev/null || true
                BASELINES_CACHED=$((BASELINES_CACHED + 1))
            else
                BASELINES_MISSING=$((BASELINES_MISSING + 1))
                log_message "WARNING: Missing baseline ${arch_model}/${basename_file}"
            fi
        done
    done
    
    log_message "After waiting: ${BASELINES_CACHED} baselines copied, ${BASELINES_MISSING} still missing"
else
    log_message "All baselines already cached, no experiments needed"
fi

# Phase 2: Copy exp2 results
log_message ""
log_message "PHASE 2: Copying exp2 results..."

EXP2_COPIED=0
EXP2_FAILED=0

# Copy all results from exp2
for arch_dir in "$INPUT_PATH"/*; do
    if [[ -d "$arch_dir" ]]; then
        arch_model=$(basename "$arch_dir")
        
        for model_dir in "$arch_dir"/*; do
            if [[ -d "$model_dir" ]]; then
                model_name=$(basename "$model_dir")
                csv_file="$model_dir/agg.ad.pred.eval.mean.csv"
                
                if [[ -f "$csv_file" ]]; then
                    output_dir="${OUTPUT_DIR}/${arch_model}/${model_name}"
                    mkdir -p "$output_dir"
                    cp -r "$model_dir"/* "$output_dir/" 2>/dev/null || true
                    log_message "Copied: ${arch_model}/${model_name}"
                    EXP2_COPIED=$((EXP2_COPIED + 1))
                else
                    log_message "WARNING: No CSV found for ${arch_model}/${model_name}"
                    EXP2_FAILED=$((EXP2_FAILED + 1))
                fi
            fi
        done
    fi
done

log_message "Exp2 results: ${EXP2_COPIED} copied, ${EXP2_FAILED} missing"

# Generate summary
summary_file="${OUTPUT_DIR}/baseline_comparison_summary.txt"
cat > "${summary_file}" << EOF
Experiment 3: Baseline Comparison Summary
Generated on: $(date)
===============================================================================

Data Sources:
- Baselines: ${BASELINE_DIR}
- Exp2 Results: ${INPUT_PATH}

Results Collected:
- Baseline datasets: ${#BASELINE_FILES[@]}
- Architecture models: ${#ARCH_MODELS[@]}
- Baselines cached: ${BASELINES_CACHED}
- Baselines computed: ${BASELINES_TO_RUN}
- Exp2 results copied: ${EXP2_COPIED}

Output Structure:
EOF

for arch_model in "${ARCH_MODELS[@]}"; do
    echo "" >> "${summary_file}"
    echo "${arch_model}/" >> "${summary_file}"
    
    # List baselines
    for baseline_file in "${BASELINE_FILES[@]}"; do
        basename_file=$(basename "$baseline_file" .xml)
        if [[ -d "${OUTPUT_DIR}/${arch_model}/baseline-${basename_file}" ]]; then
            echo "  baseline-${basename_file}/ (explicit baseline)" >> "${summary_file}"
        fi
    done
    
    # List LLM models
    for model_dir in "${OUTPUT_DIR}/${arch_model}"/*; do
        if [[ -d "$model_dir" ]]; then
            model_name=$(basename "$model_dir")
            if [[ ! "$model_name" =~ ^baseline- ]]; then
                echo "  ${model_name}/ (implicit LLM)" >> "${summary_file}"
            fi
        fi
    done
done

echo "" >> "${summary_file}"
echo "===============================================================================" >> "${summary_file}"
echo "Ready for data collection!" >> "${summary_file}"

# Final summary
echo ""
echo "==============================================================================="
echo "✅ EXPERIMENT 3 COMPLETED!"
echo "==============================================================================="
echo "End Time: $(date)"
echo "Output Directory: ${OUTPUT_DIR}"
echo ""
echo "📊 Results Summary:"
echo "  • Baseline datasets: ${#BASELINE_FILES[@]}"
echo "  • Baselines cached: ${BASELINES_CACHED}"
echo "  • Baselines computed: ${BASELINES_TO_RUN}"
echo "  • Exp2 results copied: ${EXP2_COPIED}"
echo ""
echo "📁 Output Structure:"
echo "  • Baselines: ${OUTPUT_DIR}/{arch}/baseline-*/"
echo "  • LLM results: ${OUTPUT_DIR}/{arch}/{model}/"
echo "  • Summary: ${OUTPUT_DIR}/baseline_comparison_summary.txt"
echo ""
echo "🚀 Ready for data collection!"
echo "==============================================================================="

log_message "Experiment 3 completed successfully!"