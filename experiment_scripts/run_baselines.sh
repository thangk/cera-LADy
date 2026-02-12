#!/bin/bash

#===============================================================================
# RUN BASELINES: Execute SemEval Baseline Experiments
# Purpose: Run baseline experiments for all architecture models and cache results
#
# Prerequisites:
#   1. cd /path/to/LADy-kap  
#   2. conda activate <your-env-name>
#   3. ./experiment_scripts/run_baselines.sh
#
# Note: Results are cached in experiment_datasets/semeval_baselines/output/
#===============================================================================

# Default values
BACKGROUND_MODE=false
FORCE_RERUN=false
PARALLEL_MODE=false

# Help menu
show_help() {
    cat << 'EOF'
RUN BASELINES: Execute SemEval Baseline Experiments

DESCRIPTION:
    Runs baseline experiments for all architecture models on SemEval explicit datasets.
    Results are cached for future use by exp3 and other scripts.

USAGE:
    ./experiment_scripts/run_baselines.sh [OPTIONS]

OPTIONS:
    -h, --help             Show this help message and exit
    --force                Force re-run of all baselines (ignore cache)
    --parallel             Run experiments in parallel (default: sequential)
    --background           Run in background mode (internal use)

PREREQUISITES:
    1. cd /path/to/LADy-kap
    2. conda activate <your-env-name>
    3. Baseline datasets in experiment_datasets/semeval_baselines/

WHAT THIS SCRIPT DOES:
    • Finds all baseline XML files in experiment_datasets/semeval_baselines/
    • Checks cache in experiment_datasets/semeval_baselines/output/
    • Runs missing baselines for all architecture models (BERT, CTM, BTM, RND)
    • Caches results for future use

OUTPUT:
    • Cache location: experiment_datasets/semeval_baselines/output/{dataset}/{arch}/
    • Logs: experiment_datasets/semeval_baselines/output/logs/
    • Status: experiment_datasets/semeval_baselines/output/status.log

MONITORING:
    • Process status: ps aux | grep 'python main.py'
    • Progress: tail -f experiment_datasets/semeval_baselines/output/logs/*.log
    • Summary: cat experiment_datasets/semeval_baselines/output/baselines_summary.txt

CONFIGURATION:
    • GPU_ID: Set in script (default: "1")
    • NUM_ASPECTS: 5
    • NUM_FOLDS: 5

EXAMPLES:
    # Run missing baselines sequentially
    ./experiment_scripts/run_baselines.sh
    
    # Force re-run all baselines
    ./experiment_scripts/run_baselines.sh --force
    
    # Run in parallel mode
    ./experiment_scripts/run_baselines.sh --parallel

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
            --force)
                FORCE_RERUN=true
                shift
                ;;
            --parallel)
                PARALLEL_MODE=true
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

# Main execution starts here
parse_args "$@"

# Skip checks if running in background mode
if [[ "$BACKGROUND_MODE" != true ]]; then
    # Check if we're in the right directory
    if [[ ! -f "src/main.py" ]] || [[ ! -d "experiment_scripts" ]]; then
        echo "❌ ERROR: Please run this script from the LADy-kap project root directory"
        echo "Usage:"
        echo "  cd /path/to/LADy-kap"
        echo "  conda activate <your-env-name>"
        echo "  ./experiment_scripts/run_baselines.sh"
        exit 1
    fi

    # Check if conda environment is activated
    if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
        echo "❌ ERROR: Please activate your conda environment for this project first"
        echo "Usage:"
        echo "  conda activate <your-env-name>"
        echo "  ./experiment_scripts/run_baselines.sh"
        exit 1
    fi

    # Launch in background
    echo "🚀 LAUNCHING BASELINE EXPERIMENTS"
    echo "⏰ Starting background execution..."
    nohup bash "${BASH_SOURCE[0]}" --background \
        $([ "$FORCE_RERUN" == true ] && echo "--force") \
        $([ "$PARALLEL_MODE" == true ] && echo "--parallel") \
        > /dev/null 2>&1 &
    EXPERIMENT_PID=$!
    echo "✅ Baselines launched in background!"
    echo "🆔 Process ID (PID): $EXPERIMENT_PID"
    echo ""
    echo "📊 Monitor progress with:"
    echo "   ps aux | grep 'python main.py'"
    echo "   tail -f experiment_datasets/semeval_baselines/output/logs/*.log"
    echo ""
    echo "🛑 To stop the baselines:"
    echo "   kill $EXPERIMENT_PID"
    exit 0
fi

#===================
# CONFIGURATION
#===================

# GPU Configuration
export GPU_ID="1"
export TOKENIZERS_PARALLELISM=false

# Experiment Settings
export NUM_ASPECTS="5"
export NUM_FOLDS="5"

# Directory Paths
export BASELINE_DIR="experiment_datasets/semeval_baselines"
export OUTPUT_DIR="${BASELINE_DIR}/output"
export LOG_DIR="${OUTPUT_DIR}/logs"
export SRC_DIR="src"

# Architecture Models
ARCH_MODELS=("bert" "ctm" "btm" "rnd")

# Create directories BEFORE any execution
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${OUTPUT_DIR}/baselines.log"
}

# Function to run baseline experiment
run_baseline_experiment() {
    local arch_model=$1
    local baseline_file="$2"
    local basename_file=$(basename "$baseline_file" .xml)
    local cache_output_path="${OUTPUT_DIR}/${basename_file}/${arch_model}"
    local experiment_log="${LOG_DIR}/${arch_model}_${basename_file}.log"
    
    # Check if baseline file exists
    if [[ ! -f "${baseline_file}" ]]; then
        log_message "ERROR: Baseline file not found: ${baseline_file}"
        return 1
    fi
    
    # Start timing
    local start_time=$(date +%s)
    log_message "START: ${arch_model}/${basename_file} at $(date)"
    
    # Create cache output directory
    mkdir -p "${cache_output_path}"
    
    # Build command - run from src directory
    local cmd="cd ${SRC_DIR} && python main.py"
    cmd+=" -am ${arch_model}"
    cmd+=" -data ../${baseline_file}"
    cmd+=" -output ../${cache_output_path}"
    cmd+=" -naspects ${NUM_ASPECTS}"
    cmd+=" -nfolds ${NUM_FOLDS}"
    
    # Add GPU parameter if specified
    if [[ -n "${GPU_ID}" ]]; then
        cmd+=" -gpu ${GPU_ID}"
    fi
    
    # Execute experiment
    log_message "CMD: ${cmd}"
    if bash -c "${cmd}" > "${experiment_log}" 2>&1; then
        # Calculate runtime
        local end_time=$(date +%s)
        local runtime=$((end_time - start_time))
        local hours=$((runtime / 3600))
        local minutes=$(((runtime % 3600) / 60))
        local seconds=$((runtime % 60))
        
        log_message "SUCCESS: ${arch_model}/${basename_file} completed in $(printf "%02dh %02dm %02ds" $hours $minutes $seconds)"
        
        # Check if aggregated results were generated
        if [[ -f "${cache_output_path}/agg.ad.pred.eval.mean.csv" ]]; then
            log_message "RESULTS: ${arch_model}/${basename_file} - CSV generated"
            # Extract key metric
            local p5_score=$(grep "P_5" "${cache_output_path}/agg.ad.pred.eval.mean.csv" 2>/dev/null | cut -d',' -f3 | head -1)
            if [[ -n "$p5_score" ]]; then
                log_message "METRIC: ${arch_model}/${basename_file} - P@5 = ${p5_score}"
            fi
        else
            log_message "WARNING: ${arch_model}/${basename_file} - No CSV found"
        fi
        
        return 0
    else
        local end_time=$(date +%s)
        local runtime=$((end_time - start_time))
        local hours=$((runtime / 3600))
        local minutes=$(((runtime % 3600) / 60))
        local seconds=$((runtime % 60))
        
        log_message "FAILED: ${arch_model}/${basename_file} after $(printf "%02dh %02dm %02ds" $hours $minutes $seconds)"
        log_message "ERROR: Check log file: ${experiment_log}"
        return 1
    fi
}

# Function to update runtime
update_runtime() {
    local start_time="$1"
    local status_file="$2"
    while true; do
        if ! kill -0 $$ 2>/dev/null; then break; fi
        
        current_time=$(date +%s)
        runtime_seconds=$((current_time - start_time))
        hours=$((runtime_seconds / 3600))
        minutes=$(((runtime_seconds % 3600) / 60))
        seconds=$((runtime_seconds % 60))
        runtime_formatted=$(printf "%03dh %02dm %02ds" $hours $minutes $seconds)
        
        sed -i "s/⏱️  [0-9]\{3\}h [0-9]\{2\}m [0-9]\{2\}s/⏱️  $runtime_formatted/" "$status_file" 2>/dev/null || true
        sleep 1
    done
}

# Create status file
status_file="${OUTPUT_DIR}/status.log"
experiment_start_time=$(date +%s)

# Log start
echo "===============================================================================" | tee "${status_file}"
echo "🔬 BASELINE EXPERIMENTS                                        ⏱️  000h 00m 00s" | tee -a "${status_file}"
echo "===============================================================================" | tee -a "${status_file}"
echo "🟡 STATUS: RUNNING" | tee -a "${status_file}"
echo "🆔 PID: $$" | tee -a "${status_file}"
echo "⏰ START_TIME: $(date)" | tee -a "${status_file}"
echo "🔧 MODE: $([ "$PARALLEL_MODE" == true ] && echo "PARALLEL" || echo "SEQUENTIAL")" | tee -a "${status_file}"
echo "♻️  FORCE_RERUN: ${FORCE_RERUN}" | tee -a "${status_file}"
echo "📂 OUTPUT_DIR: ${OUTPUT_DIR}" | tee -a "${status_file}"
echo "💀 KILL_COMMAND: kill $$" | tee -a "${status_file}"
echo "===============================================================================" | tee -a "${status_file}"

# Start runtime updater
update_runtime "$experiment_start_time" "$status_file" &
RUNTIME_UPDATER_PID=$!

log_message "Starting baseline experiments"

# Find all baseline XML files
BASELINE_FILES=()
if [[ -d "$BASELINE_DIR" ]]; then
    while IFS= read -r -d '' file; do
        BASELINE_FILES+=("$file")
    done < <(find "$BASELINE_DIR" -maxdepth 1 -name "*.xml" -type f -print0 | sort -z)
fi

if [[ ${#BASELINE_FILES[@]} -eq 0 ]]; then
    log_message "ERROR: No baseline datasets found in ${BASELINE_DIR}"
    exit 1
fi

log_message "Found ${#BASELINE_FILES[@]} baseline dataset(s):"
for baseline_file in "${BASELINE_FILES[@]}"; do
    log_message "  - $(basename "$baseline_file")"
done

# Count experiments to run
BASELINES_TO_RUN=0
BASELINES_CACHED=0

for baseline_file in "${BASELINE_FILES[@]}"; do
    basename_file=$(basename "$baseline_file" .xml)
    
    for arch_model in "${ARCH_MODELS[@]}"; do
        cache_csv="${OUTPUT_DIR}/${basename_file}/${arch_model}/agg.ad.pred.eval.mean.csv"
        
        if [[ -f "$cache_csv" ]] && [[ "$FORCE_RERUN" != true ]]; then
            BASELINES_CACHED=$((BASELINES_CACHED + 1))
        else
            BASELINES_TO_RUN=$((BASELINES_TO_RUN + 1))
        fi
    done
done

log_message "Baselines status: ${BASELINES_CACHED} cached, ${BASELINES_TO_RUN} to run"

if [[ $BASELINES_TO_RUN -eq 0 ]]; then
    log_message "All baselines already cached. Nothing to do!"
    echo ""
    echo "✅ All baselines already cached!"
    echo "📁 Cache location: ${OUTPUT_DIR}"
    echo "🔄 Use --force to re-run all baselines"
    exit 0
fi

# Run experiments
current_baseline=0
failed_baselines=0
successful_baselines=0
running_pids=()

for baseline_file in "${BASELINE_FILES[@]}"; do
    basename_file=$(basename "$baseline_file" .xml)
    
    for arch_model in "${ARCH_MODELS[@]}"; do
        cache_csv="${OUTPUT_DIR}/${basename_file}/${arch_model}/agg.ad.pred.eval.mean.csv"
        
        if [[ -f "$cache_csv" ]] && [[ "$FORCE_RERUN" != true ]]; then
            log_message "SKIP: ${arch_model}/${basename_file} (already cached)"
        else
            current_baseline=$((current_baseline + 1))
            log_message "Progress: ${current_baseline}/${BASELINES_TO_RUN} - Running ${arch_model}/${basename_file}"
            
            if [[ "$PARALLEL_MODE" == true ]]; then
                # Run in parallel
                run_baseline_experiment "${arch_model}" "${baseline_file}" &
                pid=$!
                running_pids+=($pid)
                log_message "PARALLEL: Started ${arch_model}/${basename_file} with PID $pid"
                sleep 2  # Small delay between launches
            else
                # Run sequentially
                if run_baseline_experiment "${arch_model}" "${baseline_file}"; then
                    successful_baselines=$((successful_baselines + 1))
                else
                    failed_baselines=$((failed_baselines + 1))
                fi
            fi
        fi
    done
done

# If running in parallel, wait for all processes to complete
if [[ "$PARALLEL_MODE" == true ]] && [[ ${#running_pids[@]} -gt 0 ]]; then
    log_message "PARALLEL: Waiting for ${#running_pids[@]} processes to complete..."
    
    for pid in "${running_pids[@]}"; do
        if wait $pid; then
            successful_baselines=$((successful_baselines + 1))
            log_message "PARALLEL: Process $pid completed successfully"
        else
            failed_baselines=$((failed_baselines + 1))
            log_message "PARALLEL: Process $pid failed"
        fi
    done
fi

# Stop runtime updater
if [[ -n "$RUNTIME_UPDATER_PID" ]] && kill -0 "$RUNTIME_UPDATER_PID" 2>/dev/null; then
    kill "$RUNTIME_UPDATER_PID" 2>/dev/null || true
fi

# Calculate final runtime
experiment_end_time=$(date +%s)
total_runtime=$((experiment_end_time - experiment_start_time))
hours=$((total_runtime / 3600))
minutes=$(((total_runtime % 3600) / 60))
seconds=$((total_runtime % 60))

# Update status file
cat >> "${status_file}" << EOF

===============================================================================
✅ BASELINES COMPLETED!
===============================================================================
🟢 STATUS: COMPLETED
⏰ END_TIME: $(date)
⏱️  TOTAL_RUNTIME: $(printf "%02dh %02dm %02ds" $hours $minutes $seconds)
📈 RESULTS:
   • Total Baselines: $((BASELINES_TO_RUN + BASELINES_CACHED))
   • Previously Cached: ${BASELINES_CACHED}
   • Newly Run: ${BASELINES_TO_RUN}
   • ✅ Successful: ${successful_baselines}
   • ❌ Failed: ${failed_baselines}
===============================================================================
EOF

# Generate summary
summary_file="${OUTPUT_DIR}/baselines_summary.txt"
cat > "${summary_file}" << EOF
Baseline Experiments Summary
Generated on: $(date)
===============================================================================

Configuration:
- Architecture Models: ${ARCH_MODELS[*]}
- Number of Aspects: ${NUM_ASPECTS}
- Number of Folds: ${NUM_FOLDS}
- GPU ID: ${GPU_ID}
- Execution Mode: $([ "$PARALLEL_MODE" == true ] && echo "PARALLEL" || echo "SEQUENTIAL")

Datasets:
EOF

for baseline_file in "${BASELINE_FILES[@]}"; do
    echo "- $(basename "$baseline_file")" >> "${summary_file}"
done

echo "" >> "${summary_file}"
echo "Results:" >> "${summary_file}"
echo "===============================================================================" >> "${summary_file}"

for baseline_file in "${BASELINE_FILES[@]}"; do
    basename_file=$(basename "$baseline_file" .xml)
    echo "" >> "${summary_file}"
    echo "Dataset: ${basename_file}" >> "${summary_file}"
    echo "----------------------------------------" >> "${summary_file}"
    
    for arch_model in "${ARCH_MODELS[@]}"; do
        cache_csv="${OUTPUT_DIR}/${basename_file}/${arch_model}/agg.ad.pred.eval.mean.csv"
        if [[ -f "$cache_csv" ]]; then
            p5_score=$(grep "P_5" "$cache_csv" 2>/dev/null | cut -d',' -f3 | head -1)
            if [[ -n "$p5_score" ]]; then
                echo "  ${arch_model}: P@5 = ${p5_score}" >> "${summary_file}"
            else
                echo "  ${arch_model}: Completed (no P@5 found)" >> "${summary_file}"
            fi
        else
            echo "  ${arch_model}: FAILED or NOT RUN" >> "${summary_file}"
        fi
    done
done

echo "" >> "${summary_file}"
echo "===============================================================================" >> "${summary_file}"
echo "Cache Location: ${OUTPUT_DIR}" >> "${summary_file}"

# Final output
echo ""
echo "==============================================================================="
echo "✅ BASELINE EXPERIMENTS COMPLETED!"
echo "==============================================================================="
echo "End Time: $(date)"
echo "Total Runtime: $(printf "%02dh %02dm %02ds" $hours $minutes $seconds)"
echo ""
echo "📊 Results Summary:"
echo "  • Total Baselines: $((BASELINES_TO_RUN + BASELINES_CACHED))"
echo "  • Previously Cached: ${BASELINES_CACHED}"
echo "  • Newly Run: ${BASELINES_TO_RUN}"
echo "  • Successful: ${successful_baselines}"
echo "  • Failed: ${failed_baselines}"
echo ""
echo "📁 Output:"
echo "  • Cache: ${OUTPUT_DIR}/{dataset}/{arch}/"
echo "  • Logs: ${OUTPUT_DIR}/logs/"
echo "  • Summary: ${OUTPUT_DIR}/baselines_summary.txt"
echo "==============================================================================="

log_message "Baseline experiments completed!"