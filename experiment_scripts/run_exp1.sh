#!/bin/bash

#===============================================================================
# EXPERIMENT 1: LLM Model Ranking
# Purpose: Test LLM models on architecture models (BERT, CTM, BTM, Random)
#          to identify top-performing LLM datasets for each architecture
#
# Prerequisites:
#   1. cd /path/to/LADy-kap  
#   2. conda activate <your-env-name>
#   3. ./experiment_scripts/run_exp1.sh -i <input>
#
# Note: Runs in background with auto-versioned output directories
#===============================================================================

# Default values
INPUT_PATH=""
TOY_MODE=false
BACKGROUND_MODE=false
CLEANUP_MODE=false

# Help menu
show_help() {
    cat << 'EOF'
EXPERIMENT 1: LLM Model Ranking

DESCRIPTION:
    Test LLM models on architecture models (BERT, CTM, BTM, Random)
    to identify top-performing LLM datasets for each architecture.

USAGE:
    ./experiment_scripts/run_exp1.sh -i <input> [OPTIONS]

REQUIRED:
    -i, --input PATH       Input dataset file or directory containing dataset files

OPTIONS:
    -h, --help             Show this help message and exit
    --toy                  Run in toy mode (prefix output with "toy_" and use 1 fold)
    --background           Run in background mode (internal use)
    --cleanup              Delete model files after successful aggregation to save disk space
                          Keeps: all CSV files, .ad.pred files, debug.txt, reviews.pkl, splits.json
                          Deletes: model checkpoint folders, .model files, cached files
    --gpu GPU_ID           Override default GPU selection (0-3, or empty for CPU)
                          The script checks GPU availability before each experiment

PREREQUISITES:
    1. cd /path/to/LADy-kap
    2. conda activate <your-env-name>
    3. Ensure GPU_ID is configured in the script (default: GPU 1)

WHAT THIS SCRIPT DOES:
    • Auto-detects LLM models from input datasets
    • Runs on 4 architecture models: BERT, CTM, BTM, Random
    • Single file: Runs all models on that dataset
    • Directory: Runs all models on all datasets found
    • Performs cross-validation with configurable aspects/folds
    • Generates P@5 rankings to identify top performers

OUTPUT:
    • Normal mode: experiment_output/exp1_llm_model_ranking[_runN]/
    • Toy mode: experiment_output/toy_exp1_llm_model_ranking[_runN]/
    • Logs saved to: {output_dir}/logs/
    • Summary: {output_dir}/experiment_summary.txt

MONITORING:
    • Process status: ps aux | grep run_exp1.sh
    • Progress logs: tail -f experiment_output/*exp1_*/experiments.log
    • Individual experiment logs: {output_dir}/logs/

CONFIGURATION:
    Edit the script to modify:
    • GPU_ID (default: "1")
    • NUM_ASPECTS (default: 5)
    • NUM_FOLDS (default: 5, or 1 in toy mode)

EXAMPLES:
    # Run single dataset
    ./experiment_scripts/run_exp1.sh -i experiment_datasets/semeval_implitcits/openai-gpt4o-700.xml
    
    # Run all datasets in directory  
    ./experiment_scripts/run_exp1.sh -i experiment_datasets/semeval_implitcits/
    
    # Run in toy mode
    ./experiment_scripts/run_exp1.sh -i experiment_datasets/semeval_implitcits_toys/ --toy
    
    # Run with cleanup to save disk space
    ./experiment_scripts/run_exp1.sh -i experiment_datasets/semeval_implitcits/ --cleanup

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
            --cleanup)
                CLEANUP_MODE=true
                shift
                ;;
            --gpu)
                GPU_ID="$2"
                shift 2
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
    if [[ ! -e "$INPUT_PATH" ]]; then
        echo "❌ ERROR: Input path does not exist: $INPUT_PATH"
        exit 1
    fi
}

# Function to get dataset files from input
get_dataset_files() {
    local input_path="$1"
    local files=()
    
    if [[ -f "$input_path" ]]; then
        # Single file
        if [[ "$input_path" == *.xml ]]; then
            files=("$input_path")
        else
            echo "❌ ERROR: Input file must be an XML file: $input_path"
            exit 1
        fi
    elif [[ -d "$input_path" ]]; then
        # Directory - find all XML files
        while IFS= read -r -d '' file; do
            files+=("$file")
        done < <(find "$input_path" -name "*.xml" -type f -print0)
        
        if [[ ${#files[@]} -eq 0 ]]; then
            echo "❌ ERROR: No XML files found in directory: $input_path"
            exit 1
        fi
    fi
    
    printf '%s\n' "${files[@]}"
}

# Function to extract LLM models from dataset files
extract_llm_models() {
    local dataset_files=("$@")
    local llm_models=()
    
    for file in "${dataset_files[@]}"; do
        local basename_file=$(basename "$file" .xml)
        # Extract LLM model name by removing size suffix (e.g., -700, -1300, -2000, -25)
        local llm_model=$(echo "$basename_file" | sed 's/-[0-9]*$//')
        
        # Add to array if not already present
        if [[ ! " ${llm_models[*]} " =~ " ${llm_model} " ]]; then
            llm_models+=("$llm_model")
        fi
    done
    
    printf '%s\n' "${llm_models[@]}"
}

# Main execution starts here
parse_args "$@"

# Skip checks if running in background mode
if [[ "$BACKGROUND_MODE" != true ]]; then
    validate_args
    
    # Check if we're in the right directory
    if [[ ! -f "src/main.py" ]] || [[ ! -d "experiment_scripts" ]]; then
        echo "❌ ERROR: Please run this script from the LADy-kap project root directory"
        echo "Usage:"
        echo "  cd /path/to/LADy-kap"
        echo "  conda activate <your-env-name>"
        echo "  ./experiment_scripts/run_exp1.sh -i <input>"
        exit 1
    fi

    # Check if conda environment is activated
    if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
        echo "❌ ERROR: Please activate your conda environment for this project first"
        echo "Usage:"
        echo "  conda activate <your-env-name>"
        echo "  ./experiment_scripts/run_exp1.sh -i <input>"
        exit 1
    fi

    # Launch in background
    if [[ "$TOY_MODE" == true ]]; then
        echo "🚀 LAUNCHING EXPERIMENT 1: LLM Model Ranking (TOY MODE)"
    else
        echo "🚀 LAUNCHING EXPERIMENT 1: LLM Model Ranking"
    fi
    if [[ "$CLEANUP_MODE" == true ]]; then
        echo "🧹 Cleanup mode enabled - will delete model files after successful aggregation"
    fi
    echo "⏰ Starting background execution..."
    nohup bash "${BASH_SOURCE[0]}" --background -i "$INPUT_PATH" $([ "$TOY_MODE" == true ] && echo "--toy") $([ "$CLEANUP_MODE" == true ] && echo "--cleanup") $([ -n "$GPU_ID" ] && echo "--gpu $GPU_ID") > /dev/null 2>&1 &
    EXPERIMENT_PID=$!
    echo "✅ Experiment launched in background!"
    echo "🆔 Process ID (PID): $EXPERIMENT_PID"
    echo ""
    echo "📊 Monitor progress with:"
    echo "   ps aux | grep run_exp1.sh"
    echo "   tail -f experiment_output/*exp1_*/experiments.log"
    echo ""
    echo "🛑 To stop the experiment:"
    echo "   kill $EXPERIMENT_PID"
    echo "   pkill -f run_exp1.sh"
    exit 0
else
    validate_args
fi

#===================
# CONFIGURATION
#===================

# GPU Configuration (set to desired GPU index: 0, 1, 2, 3, or leave empty for CPU)
# Can be overridden with --gpu command line option
if [[ -z "$GPU_ID" ]]; then
    export GPU_ID="1"
fi

# GPU Memory Safety Settings
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"  # Limit memory fragmentation
export CUDA_LAUNCH_BLOCKING="0"  # Non-blocking for better multi-user sharing

# Suppress HuggingFace tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# Experiment Settings - Adjust based on mode
if [[ "$TOY_MODE" == true ]]; then
    export EXPERIMENT_NAME="toy_exp1_llm_model_ranking"
    export NUM_FOLDS="1"  # Use 1 fold for toy mode
else
    export EXPERIMENT_NAME="exp1_llm_model_ranking"
    export NUM_FOLDS="5"
fi
export NUM_ASPECTS="5"

# Directory Paths (will be converted to absolute paths)
export SRC_DIR="src"

# Function to find next available directory name
get_unique_output_dir() {
    local base_dir="experiment_output/${EXPERIMENT_NAME}"
    local output_dir="${base_dir}"
    local counter=2
    
    # Check if base directory exists
    while [[ -d "${output_dir}" ]]; do
        output_dir="${base_dir}_run${counter}"
        counter=$((counter + 1))
    done
    
    echo "${output_dir}"
}

# Get unique output directory (will be converted to absolute path)
export OUTPUT_DIR_REL=$(get_unique_output_dir)
export LOG_DIR_REL="${OUTPUT_DIR_REL}/logs"

# Architecture Models to Test
ARCH_MODELS=("bert" "ctm" "btm" "rnd")

#===================
# SETUP
#===================

# Ensure we're in the project root directory first
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}" || exit 1

# Convert relative paths to absolute paths and resolve input path
if [[ "$INPUT_PATH" == /* ]]; then
    export DATA_INPUT="$INPUT_PATH"
else
    export DATA_INPUT="${PROJECT_ROOT}/${INPUT_PATH}"
fi

export OUTPUT_DIR="${PROJECT_ROOT}/${OUTPUT_DIR_REL}"
export LOG_DIR="${PROJECT_ROOT}/${LOG_DIR_REL}"
export SRC_DIR_ABS="${PROJECT_ROOT}/${SRC_DIR}"

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# Change to source directory
cd "${SRC_DIR_ABS}" || exit 1

# Log experiment start
echo "==============================================================================="
echo "EXPERIMENT 1: LLM Model Ranking"
echo "==============================================================================="
echo "Start Time: $(date)"
echo "GPU ID: ${GPU_ID}"
echo "Toy Mode: ${TOY_MODE}"
echo "Cleanup Mode: ${CLEANUP_MODE}"
echo "Input Path: ${DATA_INPUT}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Working Directory: $(pwd)"

# Get dataset files and extract LLM models
mapfile -t DATASET_FILES < <(get_dataset_files "$DATA_INPUT")
mapfile -t LLM_MODELS < <(extract_llm_models "${DATASET_FILES[@]}")

echo "Dataset Files Found: ${#DATASET_FILES[@]}"
echo "LLM Models Detected: ${#LLM_MODELS[@]} (${LLM_MODELS[*]})"
echo "Architecture Models: ${#ARCH_MODELS[@]} (${ARCH_MODELS[*]})"
echo "Total Experiments: $((${#DATASET_FILES[@]} * ${#ARCH_MODELS[@]}))"

# Extract and display run number
if [[ "${OUTPUT_DIR}" =~ _run([0-9]+)$ ]]; then
    echo "Run Number: ${BASH_REMATCH[1]}"
else
    echo "Run Number: 1 (initial run)"
fi

echo "==============================================================================="

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${OUTPUT_DIR}/experiments.log"
}

# Function to update progress in status file
update_progress() {
    local current_exp=$1
    local total_exp=$2
    local current_model=$3
    local status="${4:-running}"  # running, completed, failed
    local percent=$((current_exp * 100 / total_exp))
    
    # Calculate completed and failed counts
    local completed=$((current_exp - failed_experiments))
    
    # Calculate ETA based on average time per experiment
    local current_time=$(date +%s)
    local elapsed=$((current_time - start_time))
    if [[ $current_exp -gt 0 ]]; then
        local avg_time_per_exp=$((elapsed / current_exp))
        local remaining_exp=$((total_exp - current_exp))
        local eta_seconds=$((avg_time_per_exp * remaining_exp))
        local eta_hours=$((eta_seconds / 3600))
        local eta_minutes=$(((eta_seconds % 3600) / 60))
        local eta_formatted=$(printf "%02dh %02dm" $eta_hours $eta_minutes)
    else
        local eta_formatted="Calculating..."
    fi
    
    # Update the progress section in status file
    if [[ -f "$status_file" ]]; then
        # Create temporary file with updated progress
        awk -v progress="📈 PROGRESS: $current_exp/$total_exp ($percent%)" \
            -v current="🔄 CURRENT: $current_model" \
            -v eta="⏱️  ETA: $eta_formatted" \
            -v completed="✅ COMPLETED: $completed" \
            -v failed="❌ FAILED: $failed_experiments" '
        /^📈 PROGRESS:/ { print progress; next }
        /^🔄 CURRENT:/ { print current; next }
        /^⏱️  ETA:/ { print eta; next }
        /^✅ COMPLETED:/ { print completed; next }
        /^❌ FAILED:/ { print failed; next }
        { print }
        ' "$status_file" > "${status_file}.tmp" && mv "${status_file}.tmp" "$status_file"
    fi
}

# Function to add recent activity to status file
add_recent_activity() {
    local activity=$1
    local timestamp=$(date '+%H:%M:%S')
    
    if [[ -f "$status_file" ]]; then
        # Read existing file and add new activity
        awk -v new_activity="• [$timestamp] $activity" '
        BEGIN { found=0; count=0 }
        /^📋 RECENT ACTIVITY:/ { 
            print; 
            print new_activity;
            found=1; 
            next 
        }
        found && /^•/ { 
            count++;
            if (count < 5) print;  # Keep only last 5 activities
            next 
        }
        found && !/^•/ && !/^=$/ { found=0 }
        { print }
        ' "$status_file" > "${status_file}.tmp" && mv "${status_file}.tmp" "$status_file"
    fi
}

# Function to check GPU availability and memory
check_gpu_availability() {
    local gpu_id=$1
    
    # If no GPU specified or invalid, skip check
    if [[ -z "$gpu_id" ]] || ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
        return 0
    fi
    
    # Check if nvidia-smi is available
    if ! command -v nvidia-smi &> /dev/null; then
        log_message "WARNING: nvidia-smi not found. Cannot check GPU status."
        return 0
    fi
    
    # Get GPU memory usage
    local gpu_mem_info=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)
    
    if [[ -z "$gpu_mem_info" ]]; then
        log_message "WARNING: Cannot query GPU $gpu_id. It may not exist."
        return 1
    fi
    
    local mem_used=$(echo "$gpu_mem_info" | awk -F', ' '{print $1}')
    local mem_total=$(echo "$gpu_mem_info" | awk -F', ' '{print $2}')
    local mem_free=$((mem_total - mem_used))
    local mem_percent=$((mem_used * 100 / mem_total))
    
    log_message "GPU $gpu_id status: ${mem_used}MB/${mem_total}MB used (${mem_percent}%), ${mem_free}MB free"
    
    # Warn if GPU is heavily utilized
    if [[ $mem_percent -gt 80 ]]; then
        log_message "WARNING: GPU $gpu_id is ${mem_percent}% utilized. Experiments may fail due to OOM."
        log_message "Consider using a different GPU or waiting for resources to free up."
    fi
    
    return 0
}

# Function to run single experiment
run_experiment() {
    local arch_model=$1
    local dataset_file="$2"
    local basename_file=$(basename "$dataset_file" .xml)
    local output_path="${OUTPUT_DIR}/${arch_model}/${basename_file}"
    local experiment_log="${LOG_DIR}/${arch_model}_${basename_file}.log"
    
    # Check if experiment already completed
    if [[ -f "${output_path}/agg.ad.pred.eval.mean.csv" ]]; then
        log_message "SKIP: ${arch_model}/${basename_file} (already completed)"
        return 0
    fi
    
    # Check if dataset file exists
    if [[ ! -f "${dataset_file}" ]]; then
        log_message "ERROR: Dataset file not found: ${dataset_file}"
        return 1
    fi
    
    # Start timing
    local start_time=$(date +%s)
    log_message "START: ${arch_model}/${basename_file} at $(date)"
    
    # Check GPU availability before starting
    if [[ -n "${GPU_ID}" ]]; then
        check_gpu_availability "${GPU_ID}"
        
        # Wait if GPU is too busy (optional retry logic)
        local retry_count=0
        while [[ $retry_count -lt 3 ]]; do
            local gpu_mem_info=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)
            if [[ -n "$gpu_mem_info" ]]; then
                local mem_used=$(echo "$gpu_mem_info" | awk -F', ' '{print $1}')
                local mem_total=$(echo "$gpu_mem_info" | awk -F', ' '{print $2}')
                local mem_percent=$((mem_used * 100 / mem_total))
                
                if [[ $mem_percent -gt 90 ]]; then
                    log_message "GPU $GPU_ID is ${mem_percent}% full. Waiting 30s before retry..."
                    sleep 30
                    retry_count=$((retry_count + 1))
                else
                    break
                fi
            else
                break
            fi
        done
    fi
    
    # Create output directory
    mkdir -p "${output_path}"
    
    # Build command with GPU configuration
    local cmd="python main.py"
    cmd+=" -am ${arch_model}"
    cmd+=" -data ${dataset_file}"
    cmd+=" -output ${output_path}"
    cmd+=" -naspects ${NUM_ASPECTS}"
    cmd+=" -nfolds ${NUM_FOLDS}"
    
    # Add GPU parameter if specified
    if [[ -n "${GPU_ID}" ]]; then
        cmd+=" -gpu ${GPU_ID}"
    fi
    
    # Execute experiment
    log_message "CMD: ${cmd}"
    if eval "${cmd}" > "${experiment_log}" 2>&1; then
        # Calculate runtime
        local end_time=$(date +%s)
        local runtime=$((end_time - start_time))
        local hours=$((runtime / 3600))
        local minutes=$(((runtime % 3600) / 60))
        local seconds=$((runtime % 60))
        
        log_message "SUCCESS: ${arch_model}/${basename_file} completed in $(printf "%02dh %02dm %02ds" $hours $minutes $seconds)"
        
        # Check if aggregated results were generated
        if [[ -f "${output_path}/agg.ad.pred.eval.mean.csv" ]]; then
            log_message "RESULTS: ${arch_model}/${basename_file} - CSV generated"
            add_recent_activity "✅ ${arch_model}/${basename_file} completed"
            
            # Cleanup model files if requested
            if [[ "$CLEANUP_MODE" == true ]]; then
                log_message "CLEANUP: ${arch_model}/${basename_file} - Removing model files..."
                
                # Clean up inside the numbered directory (e.g., 5/)
                if [[ -d "${output_path}/${NUM_ASPECTS}" ]]; then
                    # Navigate to the architecture model subdirectory
                    local model_dir="${output_path}/${NUM_ASPECTS}/${arch_model}"
                    
                    if [[ -d "${model_dir}" ]]; then
                        # Delete fold directories (f0/, f1/, etc.) but keep CSV and model files
                        find "${model_dir}" -type d -name "f[0-9]*" -exec rm -rf {} + 2>/dev/null || true
                        
                        # Delete .model files (they're large binary files)
                        find "${model_dir}" -name "*.model" -type f -delete 2>/dev/null || true
                        
                        # Keep all CSV files and .ad.pred files
                        log_message "CLEANUP: ${arch_model}/${basename_file} - Model folders deleted, CSV files preserved"
                    fi
                fi
            fi
        else
            log_message "WARNING: ${arch_model}/${basename_file} - No CSV found"
        fi
        
        # Write runtime to temp file for total calculation
        runtime_file="/tmp/exp1_runtime_${arch_model}_${basename_file}.tmp"
        echo $runtime > "${runtime_file}"
        return 0
    else
        local end_time=$(date +%s)
        local runtime=$((end_time - start_time))
        local hours=$((runtime / 3600))
        local minutes=$(((runtime % 3600) / 60))
        local seconds=$((runtime % 60))
        
        log_message "FAILED: ${arch_model}/${basename_file} after $(printf "%02dh %02dm %02ds" $hours $minutes $seconds)"
        add_recent_activity "❌ ${arch_model}/${basename_file} failed"
        return 1
    fi
}

#===================
# MAIN EXECUTION
#===================

# Calculate total experiments
total_experiments=$((${#DATASET_FILES[@]} * ${#ARCH_MODELS[@]}))
current_experiment=0
failed_experiments=0
total_runtime=0
experiment_start_time=$(date +%s)

# Create status file to track experiment progress
status_file="${OUTPUT_DIR}/status.log"

# Set up signal handlers to update status on termination
cleanup_on_exit() {
    local exit_code=$?
    
    # Stop the runtime updater
    if [[ -n "$runtime_updater_pid" ]]; then
        kill "$runtime_updater_pid" 2>/dev/null
    fi
    
    local end_time=$(date +%s)
    local runtime=$((end_time - start_time))
    local hours=$((runtime / 3600))
    local minutes=$(((runtime % 3600) / 60))
    local seconds=$((runtime % 60))
    
    if [[ $exit_code -eq 0 ]]; then
        # Normal completion - status already updated
        return
    else
        # Abnormal termination - update final runtime
        sed -i "s/⏱️  [0-9]\{3\}h [0-9]\{2\}m [0-9]\{2\}s/⏱️  $(printf "%03dh %02dm %02ds" $hours $minutes $seconds)/" "$status_file"
        
        cat >> "${status_file}" << EOF

===============================================================================
🛑 EXPERIMENT TERMINATED!
===============================================================================
🔴 STATUS: TERMINATED
⏰ TERMINATED_TIME: $(date)
⏱️  FINAL_RUNTIME: $(printf "%03dh %02dm %02ds" $hours $minutes $seconds)
💀 EXIT_CODE: $exit_code
📈 PARTIAL_RESULTS: $current_experiment/${total_experiments} experiments completed
===============================================================================
EOF
    fi
}

# Register signal handlers
trap 'cleanup_on_exit' EXIT SIGTERM SIGINT

cat > "${status_file}" << EOF
===============================================================================
🧪 EXPERIMENT 1: LLM Model Ranking                           ⏱️  000h 00m 00s
===============================================================================
🟡 STATUS: RUNNING
🆔 PID: $$
⏰ START_TIME: $(date)
📊 TOTAL_EXPERIMENTS: ${total_experiments}
🏗️ ARCHITECTURES: ${#ARCH_MODELS[@]} (${ARCH_MODELS[*]})
🤖 LLM_MODELS: ${#DATASET_FILES[@]} datasets
📂 OUTPUT_DIR: ${OUTPUT_DIR}
💀 KILL_COMMAND: kill $$ (or pkill -f run_exp1.sh)
===============================================================================
📈 PROGRESS: 0/${total_experiments} (0%)
🔄 CURRENT: Starting...
⏱️  ETA: Calculating...
✅ COMPLETED: 0
❌ FAILED: 0
===============================================================================
📋 RECENT ACTIVITY:
• Initializing...
===============================================================================
EOF
log_message "Status file created: ${status_file} (PID: $$)"

# Function to update runtime in status file
update_runtime() {
    local start_time="$1"
    local status_file="$2"
    
    while true; do
        # Check if the main process is still running
        if ! kill -0 $$ 2>/dev/null; then
            break
        fi
        
        # Calculate runtime
        current_time=$(date +%s)
        runtime_seconds=$((current_time - start_time))
        
        hours=$((runtime_seconds / 3600))
        minutes=$(((runtime_seconds % 3600) / 60))
        seconds=$((runtime_seconds % 60))
        
        # Format runtime
        runtime_formatted=$(printf "%03dh %02dm %02ds" $hours $minutes $seconds)
        
        # Update the runtime line in status file
        if [[ -f "$status_file" ]]; then
            # Use sed to replace the runtime in the title line
            sed -i "s/⏱️  [0-9]\{3\}h [0-9]\{2\}m [0-9]\{2\}s/⏱️  $runtime_formatted/" "$status_file"
        fi
        
        sleep 1
    done
}

# Start runtime in background
start_time=$(date +%s)
update_runtime "$start_time" "$status_file" &
runtime_updater_pid=$!

# Check initial GPU status
if [[ -n "${GPU_ID}" ]]; then
    log_message "Checking initial GPU status..."
    check_gpu_availability "${GPU_ID}"
fi

log_message "Starting batch execution of ${total_experiments} experiments"
add_recent_activity "Starting ${total_experiments} experiments"

# Main execution loop
for arch_model in "${ARCH_MODELS[@]}"; do
    log_message "Processing architecture model: ${arch_model}"
    
    for dataset_file in "${DATASET_FILES[@]}"; do
        current_experiment=$((current_experiment + 1))
        basename_file=$(basename "$dataset_file" .xml)
        
        log_message "Progress: ${current_experiment}/${total_experiments} - ${arch_model}/${basename_file}"
        
        # Update status file with current progress
        update_progress $((current_experiment - 1)) "$total_experiments" "${arch_model}/${basename_file}"
        
        # Run experiment and capture runtime
        run_experiment "${arch_model}" "${dataset_file}"
        experiment_exit_code=$?
        
        # Update progress after experiment completes
        if [[ $experiment_exit_code -eq 0 ]]; then
            update_progress "$current_experiment" "$total_experiments" "${arch_model}/${basename_file}" "completed"
        else
            failed_experiments=$((failed_experiments + 1))
            update_progress "$current_experiment" "$total_experiments" "${arch_model}/${basename_file}" "failed"
        fi
        
        if [[ $experiment_exit_code -eq 0 ]]; then
            # Read the runtime from the temp file
            runtime_file="/tmp/exp1_runtime_${arch_model}_${basename_file}.tmp"
            if [[ -f "${runtime_file}" ]]; then
                experiment_runtime=$(cat "${runtime_file}")
                if [[ "${experiment_runtime}" =~ ^[0-9]+$ ]]; then
                    total_runtime=$((total_runtime + experiment_runtime))
                fi
                rm -f "${runtime_file}"
            fi
        fi
        
        # Small delay to prevent system overload and allow GPU memory cleanup
        if [[ -n "${GPU_ID}" ]]; then
            sleep 5  # Longer delay when using GPU to ensure memory is freed
        else
            sleep 2
        fi
    done
    
    log_message "Completed architecture model: ${arch_model}"
done

# Calculate total script runtime
experiment_end_time=$(date +%s)
script_total_runtime=$((experiment_end_time - experiment_start_time))
script_hours=$((script_total_runtime / 3600))
script_minutes=$(((script_total_runtime % 3600) / 60))
script_seconds=$((script_total_runtime % 60))

# Calculate total experiment runtime (excluding overhead)
exp_hours=$((total_runtime / 3600))
exp_minutes=$(((total_runtime % 3600) / 60))
exp_secs=$((total_runtime % 60))

#===================
# RESULTS SUMMARY
#===================

log_message "==============================================================================="
log_message "EXPERIMENT 1 COMPLETED"
log_message "==============================================================================="
log_message "End Time: $(date)"
log_message "Total Experiments: ${total_experiments}"
log_message "Failed Experiments: ${failed_experiments}"
log_message "Success Rate: $(( (total_experiments - failed_experiments) * 100 / total_experiments ))%"

# Stop the runtime updater
if [[ -n "$runtime_updater_pid" ]]; then
    kill "$runtime_updater_pid" 2>/dev/null
fi

# Update status file with completion info
experiment_end_time=$(date +%s)
total_script_runtime=$((experiment_end_time - start_time))
hours=$((total_script_runtime / 3600))
minutes=$(((total_script_runtime % 3600) / 60))
seconds=$((total_script_runtime % 60))

# Update the final runtime in the status file
sed -i "s/⏱️  [0-9]\{3\}h [0-9]\{2\}m [0-9]\{2\}s/⏱️  $(printf "%03dh %02dm %02ds" $hours $minutes $seconds)/" "$status_file"

cat >> "${status_file}" << EOF

===============================================================================
✅ EXPERIMENT COMPLETED!
===============================================================================
🟢 STATUS: COMPLETED
⏰ END_TIME: $(date)
⏱️  TOTAL_RUNTIME: $(printf "%03dh %02dm %02ds" $hours $minutes $seconds)
📈 RESULTS SUMMARY:
   • Total Experiments: ${total_experiments}
   • ✅ Completed: $((total_experiments - failed_experiments))
   • ❌ Failed: ${failed_experiments}
   • 📊 Success Rate: $(( (total_experiments - failed_experiments) * 100 / total_experiments ))%
📁 OUTPUTS:
   • CSV Results: ${OUTPUT_DIR}/*/agg.ad.pred.eval.mean.csv
   • Summary: ${OUTPUT_DIR}/experiment_summary.txt
   • Logs: ${OUTPUT_DIR}/logs/
===============================================================================
🎉 EXP1 READY FOR EXP2! Next: run_exp2.sh --toy -i ${OUTPUT_DIR}
===============================================================================
EOF
log_message "Status updated: COMPLETED (Runtime: $(printf "%03dh %02dm %02ds" $hours $minutes $seconds))"

# Generate results summary
summary_file="${OUTPUT_DIR}/experiment_summary.txt"
echo "Experiment 1: LLM Model Ranking Results Summary" > "${summary_file}"
echo "Generated on: $(date)" >> "${summary_file}"
echo "===============================================================================" >> "${summary_file}"

for arch_model in "${ARCH_MODELS[@]}"; do
    echo "" >> "${summary_file}"
    echo "Architecture Model: ${arch_model}" >> "${summary_file}"
    echo "----------------------------------------" >> "${summary_file}"
    
    # Look for actual LLM model directories instead of using dataset file names
    for llm_dir in "${OUTPUT_DIR}/${arch_model}"/*/; do
        if [[ -d "${llm_dir}" ]]; then
            llm_model=$(basename "${llm_dir}")
            result_file="${llm_dir}/agg.ad.pred.eval.mean.csv"
            if [[ -f "${result_file}" ]]; then
                # Extract P@5 score (column 3 contains the mean value)
                p5_score=$(grep "P_5" "${result_file}" 2>/dev/null | cut -d',' -f3 | head -1)
                if [[ -n "${p5_score}" ]]; then
                    if [[ "${p5_score}" == "0.0" ]]; then
                        echo "${llm_model}: P@5 = ${p5_score} (zero score)" >> "${summary_file}"
                    else
                        echo "${llm_model}: P@5 = ${p5_score}" >> "${summary_file}"
                    fi
                else
                    echo "${llm_model}: COMPLETED (no P@5 score found)" >> "${summary_file}"
                fi
            else
                echo "${llm_model}: FAILED" >> "${summary_file}"
            fi
        fi
    done
done

log_message "Results summary saved to: ${summary_file}"
log_message "Individual logs available in: ${LOG_DIR}"
log_message "Full results available in: ${OUTPUT_DIR}"

# Display quick results overview
echo ""
echo "Quick Results Overview:"
echo "======================"
find "${OUTPUT_DIR}" -name "agg.ad.pred.eval.mean.csv" | wc -l | xargs echo "Completed experiments:"
find "${LOG_DIR}" -name "*_*.log" | wc -l | xargs echo "Total experiment logs:"

log_message "Experiment 1 script completed successfully!"