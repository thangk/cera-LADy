#!/bin/bash

#===============================================================================
# EXPERIMENT 2: Scaling Analysis for Top LLM Models
# Purpose: Test top 3 LLM models from Exp1 across different dataset sizes
#          for each architecture model to analyze scaling patterns
#
# Prerequisites:
#   1. cd /path/to/LADy-kap  
#   2. conda activate <your-env-name>
#   3. ./experiment_scripts/run_exp2.sh -i <input>
#
# Note: Runs in background with auto-versioned output directories
#===============================================================================

# Default values
INPUT_PATH=""
TOY_MODE=false
BACKGROUND_MODE=false

# Help menu
show_help() {
    cat << 'EOF'
EXPERIMENT 2: Scaling Analysis for Top LLM Models

DESCRIPTION:
    Takes Exp1 output directory as input to extract top 3 LLM models per architecture.
    Analyzes performance scaling across different dataset sizes using those top models.

USAGE:
    ./experiment_scripts/run_exp2.sh -i <exp1_output_dir> [OPTIONS]

REQUIRED:
    -i, --input PATH       Path to Exp1 output directory (e.g., experiment_output/toy_exp1_llm_model_ranking)
                          Script extracts top 3 LLMs and finds corresponding datasets for scaling analysis

OPTIONS:
    -h, --help             Show this help message and exit
    --toy                  Run in toy mode (prefix output with "toy_" and use 1 fold)
    --background           Run in background mode (internal use)

PREREQUISITES:
    1. cd /path/to/LADy-kap
    2. conda activate <your-env-name>
    3. Complete run_exp1.sh first (for automatic top LLM detection)
    4. Ensure GPU_ID is configured in the script (default: GPU 1)

WHAT THIS SCRIPT DOES:
    • Takes Exp1 output directory as input
    • Extracts top 3 LLM models per architecture from Exp1 P@5 scores
    • Determines dataset sizes and LLM models from Exp1 structure
    • Finds corresponding datasets for scaling analysis
    • Runs on 4 architecture models: BERT, CTM, BTM, Random
    • Performs cross-validation with configurable aspects/folds
    • Analyzes performance trends as dataset size increases

OUTPUT:
    • Normal mode: experiment_output/exp2_scaling_analysis[_runN]/
    • Toy mode: experiment_output/toy_exp2_scaling_analysis[_runN]/
    • Logs saved to: {output_dir}/logs/
    • Summary: {output_dir}/scaling_analysis_summary.txt

MONITORING:
    • Process status: ps aux | grep run_exp2.sh
    • Progress logs: tail -f experiment_output/*exp2_*/experiments.log
    • Individual experiment logs: {output_dir}/logs/

CONFIGURATION:
    Edit the script to modify:
    • GPU_ID (default: "1")
    • NUM_ASPECTS (default: 5)
    • NUM_FOLDS (default: 5, or 1 in toy mode)

EXAMPLES:
    # Run scaling analysis using exp1 results (normal mode)
    ./experiment_scripts/run_exp2.sh -i experiment_output/exp1_llm_model_ranking
    
    # Run scaling analysis using toy exp1 results  
    ./experiment_scripts/run_exp2.sh --toy -i experiment_output/toy_exp1_llm_model_ranking
    
    # Run with specific exp1 run version
    ./experiment_scripts/run_exp2.sh -i experiment_output/toy_exp1_llm_model_ranking_run2

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

# Function to extract LLM models and sizes from dataset files
extract_llm_models_and_sizes() {
    local dataset_files=("$@")
    declare -A llm_sizes
    
    for file in "${dataset_files[@]}"; do
        local basename_file=$(basename "$file" .xml)
        # Extract LLM model name and size
        if [[ "$basename_file" =~ ^(.+)-([0-9]+)$ ]]; then
            local llm_model="${BASH_REMATCH[1]}"
            local size="${BASH_REMATCH[2]}"
            
            if [[ -z "${llm_sizes[$llm_model]}" ]]; then
                llm_sizes[$llm_model]="$size"
            else
                # Add size if not already present
                if [[ ! " ${llm_sizes[$llm_model]} " =~ " $size " ]]; then
                    llm_sizes[$llm_model]+=" $size"
                fi
            fi
        fi
    done
    
    # Output in format: llm_model:size1,size2,size3
    for llm_model in "${!llm_sizes[@]}"; do
        echo "$llm_model:${llm_sizes[$llm_model]// /,}"
    done
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
        echo "  ./experiment_scripts/run_exp2.sh -i <input>"
        exit 1
    fi

    # Check if conda environment is activated
    if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
        echo "❌ ERROR: Please activate your conda environment for this project first"
        echo "Usage:"
        echo "  conda activate <your-env-name>"
        echo "  ./experiment_scripts/run_exp2.sh -i <input>"
        exit 1
    fi

    # Launch in background
    if [[ "$TOY_MODE" == true ]]; then
        echo "🚀 LAUNCHING EXPERIMENT 2: Scaling Analysis (TOY MODE)"
    else
        echo "🚀 LAUNCHING EXPERIMENT 2: Scaling Analysis"
    fi
    echo "⏰ Starting background execution..."
    nohup bash "${BASH_SOURCE[0]}" --background -i "$INPUT_PATH" $([ "$TOY_MODE" == true ] && echo "--toy") > /dev/null 2>&1 &
    EXPERIMENT_PID=$!
    echo "✅ Experiment launched in background!"
    echo "🆔 Process ID (PID): $EXPERIMENT_PID"
    echo ""
    echo "📊 Monitor progress with:"
    echo "   ps aux | grep run_exp2.sh"
    echo "   tail -f experiment_output/*exp2_*/experiments.log"
    echo ""
    echo "🛑 To stop the experiment:"
    echo "   kill $EXPERIMENT_PID"
    echo "   pkill -f run_exp2.sh"
    exit 0
else
    validate_args
fi

#===================
# CONFIGURATION
#===================

# GPU Configuration (set to desired GPU index: 0, 1, 2, 3, or leave empty for CPU)
export GPU_ID="1"

# Suppress HuggingFace tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# Experiment Settings - Adjust based on mode
if [[ "$TOY_MODE" == true ]]; then
    export EXPERIMENT_NAME="toy_exp2_scaling_analysis"
    export NUM_FOLDS="1"  # Use 1 fold for toy mode
else
    export EXPERIMENT_NAME="exp2_scaling_analysis"
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

# Find most recent exp1 directory for reference
get_latest_exp1_dir() {
    # First determine the matching exp1 pattern based on toy mode
    if [[ "$TOY_MODE" == true ]]; then
        local exp1_pattern="toy_exp1_llm_model_ranking"
    else
        local exp1_pattern="exp1_llm_model_ranking"
    fi
    
    local exp1_base="experiment_output/${exp1_pattern}"
    
    # Check base directory first
    if [[ -d "${exp1_base}" ]]; then
        echo "${exp1_base}"
        return
    fi
    
    # Look for numbered runs with matching pattern
    local latest_dir=""
    local highest_num=0
    for dir in experiment_output/${exp1_pattern}_run*; do
        if [[ -d "${dir}" ]]; then
            local num=$(echo "${dir}" | grep -o 'run[0-9]*$' | sed 's/run//')
            if [[ ${num} -gt ${highest_num} ]]; then
                highest_num=${num}
                latest_dir="${dir}"
            fi
        fi
    done
    
    if [[ -n "${latest_dir}" ]]; then
        echo "${latest_dir}"
    else
        echo "${exp1_base}"  # fallback to base name even if it doesn't exist
    fi
}

export EXP1_DIR="$EXP1_INPUT"

# Architecture Models
ARCH_MODELS=("bert" "ctm" "btm" "rnd")

# Top LLM Models Configuration (fallback defaults)
declare -A TOP_LLMS
TOP_LLMS[bert]="openai-gpt4o anthropic-sonnet4 xai-grok4"
TOP_LLMS[ctm]="anthropic-sonnet4 openai-gpt3.5-turbo google-gemini2.5pro"  
TOP_LLMS[btm]="xai-grok3 anthropic-haiku3.5 google-gemini2.5flash"
TOP_LLMS[rnd]="anthropic-sonnet4 openai-gpt3.5-turbo xai-grok4"

#===================
# FUNCTIONS
#===================

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${OUTPUT_DIR}/experiments.log"
}

# Function to automatically determine top LLMs from Exp1 results
# Uses P@5 metric for ranking (Precision at 5)
determine_top_llms() {
    local arch_model=$1
    local exp1_arch_dir="${EXP1_DIR}/${arch_model}"
    
    if [[ ! -d "${exp1_arch_dir}" ]]; then
        log_message "WARNING: Exp1 results not found for ${arch_model}, using all detected LLMs from input"
        return
    fi
    
    # Find all completed experiments and extract P@5 scores
    local temp_file="/tmp/exp2_top_llms_${arch_model}.tmp"
    > "${temp_file}"
    
    for llm_dir in "${exp1_arch_dir}"/*; do
        if [[ -d "${llm_dir}" ]]; then
            local result_file="${llm_dir}/agg.ad.pred.eval.mean.csv"
            local llm_name=$(basename "${llm_dir}" | sed 's/-[0-9]*$//')
            
            if [[ -f "${result_file}" ]]; then
                # Extract P@5 score (Precision at 5) - use column 3 for mean
                local p5_score=$(grep "P_5" "${result_file}" 2>/dev/null | cut -d',' -f3 | head -1)
                if [[ -n "${p5_score}" && "${p5_score}" != "0.0" ]]; then
                    echo "${p5_score} ${llm_name}" >> "${temp_file}"
                fi
            fi
        fi
    done
    
    # Sort by P@5 score (descending) and get top models (up to 3, but could be fewer)
    if [[ -s "${temp_file}" ]]; then
        local num_llms=$(wc -l < "${temp_file}")
        local max_top=$(( num_llms < 3 ? num_llms : 3 ))
        local top_llms=$(sort -nr "${temp_file}" | head -${max_top} | awk '{print $2}' | tr '\n' ' ')
        TOP_LLMS[${arch_model}]="${top_llms% }"  # Remove trailing space
        log_message "AUTO-DETECTED top ${max_top} LLM(s) for ${arch_model} (ranked by P@5): ${TOP_LLMS[${arch_model}]}"
        
        # Show the actual P@5 scores for transparency
        log_message "  P@5 scores for ${arch_model}:"
        sort -nr "${temp_file}" | head -${max_top} | while read score llm; do
            log_message "    ${llm}: ${score}"
        done
    else
        log_message "WARNING: No valid Exp1 results for ${arch_model}, using all detected LLMs from input"
    fi
    
    rm -f "${temp_file}"
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
        else
            log_message "WARNING: ${arch_model}/${basename_file} - No CSV found"
        fi
        
        # Write runtime to temp file for total calculation
        runtime_file="/tmp/exp2_runtime_${arch_model}_${basename_file}.tmp"
        echo $runtime > "${runtime_file}"
        return 0
    else
        local end_time=$(date +%s)
        local runtime=$((end_time - start_time))
        local hours=$((runtime / 3600))
        local minutes=$(((runtime % 3600) / 60))
        local seconds=$((runtime % 60))
        
        log_message "FAILED: ${arch_model}/${basename_file} after $(printf "%02dh %02dm %02ds" $hours $minutes $seconds)"
        return 1
    fi
}

#===================
# SETUP
#===================

# Ensure we're in the project root directory first
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}" || exit 1

# Convert relative paths to absolute paths and resolve exp1 input path
if [[ "$INPUT_PATH" == /* ]]; then
    export EXP1_INPUT="$INPUT_PATH"
else
    export EXP1_INPUT="${PROJECT_ROOT}/${INPUT_PATH}"
fi

# Validate exp1 input directory
if [[ ! -d "$EXP1_INPUT" ]]; then
    echo "❌ Error: Exp1 input directory does not exist: $EXP1_INPUT"
    exit 1
fi

# Check for basic exp1 structure (should have architecture model directories)
expected_dirs=("bert" "ctm" "btm" "rnd")
missing_dirs=()
for dir in "${expected_dirs[@]}"; do
    if [[ ! -d "$EXP1_INPUT/$dir" ]]; then
        missing_dirs+=("$dir")
    fi
done

if [[ ${#missing_dirs[@]} -gt 0 ]]; then
    echo "❌ Error: Input directory doesn't appear to be a valid exp1 output directory"
    echo "Missing architecture directories: ${missing_dirs[*]}"
    echo "Expected structure: $EXP1_INPUT/{bert,ctm,btm,rnd}/"
    exit 1
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
echo "EXPERIMENT 2: Scaling Analysis for Top LLM Models"
echo "==============================================================================="
echo "Start Time: $(date)"
echo "GPU ID: ${GPU_ID}"
echo "Toy Mode: ${TOY_MODE}"
echo "Input Path: ${EXP1_INPUT}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Working Directory: $(pwd)"
echo "==============================================================================="

log_message "Starting Experiment 2: Scaling Analysis"

# Check exp1 status before starting
exp1_status_file="$EXP1_INPUT/status.log"
if [[ -f "$exp1_status_file" ]]; then
    exp1_status=$(grep "STATUS:" "$exp1_status_file" | tail -1 | cut -d' ' -f2)
    if [[ "$exp1_status" == "RUNNING" ]]; then
        exp1_pid=$(grep "PID:" "$exp1_status_file" | cut -d' ' -f2)
        echo "⚠️  WARNING: Exp1 is still RUNNING (PID: $exp1_pid)"
        echo "📁 Exp1 Status: $exp1_status_file"
        echo "🛑 To stop exp1: kill $exp1_pid"
        echo "⏳ Exp2 will wait for exp1 to complete..."
        
        # Wait for exp1 to complete
        while [[ -f "$exp1_status_file" ]]; do
            current_status=$(grep "STATUS:" "$exp1_status_file" | tail -1 | cut -d' ' -f2)
            if [[ "$current_status" != "RUNNING" ]]; then
                break
            fi
            echo "⏳ Still waiting for exp1... (checking every 30s)"
            sleep 30
        done
        echo "✅ Exp1 completed! Proceeding with exp2..."
    elif [[ "$exp1_status" == "COMPLETED" ]]; then
        echo "✅ Exp1 status: COMPLETED - proceeding with exp2"
    else
        echo "❌ Warning: Exp1 status is '$exp1_status' - proceeding anyway"
    fi
else
    echo "⚠️  Warning: No exp1 status file found at $exp1_status_file - proceeding anyway"
fi

# Extract LLM models from exp1 directory structure
log_message "Analyzing exp1 directory structure: $EXP1_INPUT"

# Extract all LLM models from exp1 directory
declare -a ALL_LLMS_FROM_EXP1
for arch_dir in "$EXP1_INPUT"/*; do
    if [[ -d "$arch_dir" ]]; then
        arch_model=$(basename "$arch_dir")
        
        for llm_dir in "$arch_dir"/*; do
            if [[ -d "$llm_dir" ]]; then
                llm_full_name=$(basename "$llm_dir")
                # Extract base LLM name (remove size suffix like -25)
                llm_base_name=$(echo "$llm_full_name" | sed 's/-[0-9]*$//')
                # Add to array if not already present
                found=false
                for existing in "${ALL_LLMS_FROM_EXP1[@]}"; do
                    if [[ "$existing" == "$llm_base_name" ]]; then
                        found=true
                        break
                    fi
                done
                if [[ "$found" == false ]]; then
                    ALL_LLMS_FROM_EXP1+=("$llm_base_name")
                fi
            fi
        done
    fi
done

log_message "LLM models found in exp1: ${ALL_LLMS_FROM_EXP1[*]}"

# Determine dataset directory based on toy mode (use absolute path)
# Get the actual project root by going up from current working directory to find the git repo
PROJECT_ROOT="/home/thangk/msc/LADy-kap"

if [[ "$TOY_MODE" == true ]]; then
    DATASET_DIR="$PROJECT_ROOT/experiment_datasets/semeval_implitcits_toys"
else
    DATASET_DIR="$PROJECT_ROOT/experiment_datasets/semeval_implitcits"
fi

log_message "DEBUG: PROJECT_ROOT=$PROJECT_ROOT"
log_message "DEBUG: DATASET_DIR=$DATASET_DIR"

# Find corresponding dataset files for detected LLM models
declare -a DATASET_FILES
log_message "Looking for datasets in: $DATASET_DIR"

for llm_model in "${ALL_LLMS_FROM_EXP1[@]}"; do
    # Look for all size variants of this LLM model
    shopt -s nullglob  # Enable nullglob so empty globs return nothing
    for dataset_file in "$DATASET_DIR"/${llm_model}-*.xml; do
        if [[ -f "$dataset_file" ]]; then
            DATASET_FILES+=("$dataset_file")
            log_message "Found dataset: $(basename "$dataset_file")"
        fi
    done
    shopt -u nullglob  # Disable nullglob
done

if [[ ${#DATASET_FILES[@]} -eq 0 ]]; then
    echo "❌ Error: No matching dataset files found in $DATASET_DIR"
    echo "Expected files like: ${ALL_LLMS_FROM_EXP1[0]}-15.xml, ${ALL_LLMS_FROM_EXP1[0]}-25.xml, etc."
    echo "Available LLMs: ${ALL_LLMS_FROM_EXP1[*]}"
    echo "Checking directory: $DATASET_DIR"
    echo "Files in directory:"
    ls -la "$DATASET_DIR"/*.xml 2>/dev/null || echo "No XML files found"
    exit 1
fi

log_message "Found ${#DATASET_FILES[@]} dataset files for scaling analysis"

# Determine top LLMs from Exp1 results if available
log_message "Determining top LLMs for each architecture model from Exp1 results..."
for arch_model in "${ARCH_MODELS[@]}"; do
    determine_top_llms "${arch_model}"
    
    # If no Exp1 results were found, use all LLMs from exp1 
    if [[ -z "${TOP_LLMS[${arch_model}]}" ]]; then
        TOP_LLMS[${arch_model}]="${ALL_LLMS_FROM_EXP1[*]}"
        log_message "FALLBACK: Using all LLMs from exp1 for ${arch_model}: ${TOP_LLMS[${arch_model}]}"
    fi
done

# Save top LLM selections to file for transparency
top_llms_file="${OUTPUT_DIR}/top_llm_selections.txt"
echo "Experiment 2: Top LLM Model Selections for Scaling Analysis" > "${top_llms_file}"
echo "Generated on: $(date)" >> "${top_llms_file}"
echo "Based on P@5 scores from: ${EXP1_DIR}" >> "${top_llms_file}"
echo "===============================================================================" >> "${top_llms_file}"

# Display final LLM selections
log_message "Final LLM selections for scaling analysis:"
for arch_model in "${ARCH_MODELS[@]}"; do
    log_message "  ${arch_model}: ${TOP_LLMS[${arch_model}]}"
    echo "" >> "${top_llms_file}"
    echo "Architecture Model: ${arch_model}" >> "${top_llms_file}"
    echo "Selected LLMs: ${TOP_LLMS[${arch_model}]}" >> "${top_llms_file}"
done

log_message "Top LLM selections saved to: ${top_llms_file}"

#===================
# MAIN EXECUTION
#===================

# Filter dataset files to only include those with LLMs in the top 3 for at least one architecture
# OR use all files if no Exp1 results are available
FILTERED_DATASET_FILES=()
for dataset_file in "${DATASET_FILES[@]}"; do
    basename_file=$(basename "$dataset_file" .xml)
    llm_model=$(echo "$basename_file" | sed 's/-[0-9]*$//')
    
    # Check if this LLM is in any of the top 3 lists
    is_top_llm=false
    for arch_model in "${ARCH_MODELS[@]}"; do
        if [[ " ${TOP_LLMS[${arch_model}]} " =~ " ${llm_model} " ]]; then
            is_top_llm=true
            break
        fi
    done
    
    # If no Exp1 results were found or LLM is in top 3, include it
    if [[ "$is_top_llm" == true ]] || [[ ! -d "$EXP1_DIR" ]]; then
        FILTERED_DATASET_FILES+=("$dataset_file")
    fi
done

if [[ ${#FILTERED_DATASET_FILES[@]} -eq 0 ]]; then
    log_message "WARNING: No datasets match top LLM models, using all datasets"
    FILTERED_DATASET_FILES=("${DATASET_FILES[@]}")
fi

total_experiments=$((${#FILTERED_DATASET_FILES[@]} * ${#ARCH_MODELS[@]}))
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
📈 EXPERIMENT 2: Scaling Analysis                              ⏱️  000h 00m 00s
===============================================================================
🟡 STATUS: RUNNING
🆔 PID: $$
⏰ START_TIME: $(date)
📊 TOTAL_EXPERIMENTS: ${total_experiments}
🏗️ ARCHITECTURES: ${#ARCH_MODELS[@]} (${ARCH_MODELS[*]})
🤖 TOP_LLMS: ${#FILTERED_DATASET_FILES[@]} datasets from exp1
📂 EXP1_INPUT: ${EXP1_INPUT}
📂 OUTPUT_DIR: ${OUTPUT_DIR}
💀 KILL_COMMAND: kill $$ (or pkill -f run_exp2.sh)
===============================================================================
EOF
log_message "Status file created: ${status_file} (PID: $$)"

# Start runtime updater in background
update_runtime "$experiment_start_time" "$status_file" &
runtime_updater_pid=$!
log_message "Runtime updater started (PID: $runtime_updater_pid)"

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

log_message "Starting batch execution of ${total_experiments} experiments"
log_message "Using ${#FILTERED_DATASET_FILES[@]} datasets (filtered from ${#DATASET_FILES[@]} total)"

# Main execution loop
for arch_model in "${ARCH_MODELS[@]}"; do
    log_message "Processing architecture model: ${arch_model}"
    
    for dataset_file in "${FILTERED_DATASET_FILES[@]}"; do
        current_experiment=$((current_experiment + 1))
        basename_file=$(basename "$dataset_file" .xml)
        
        log_message "Progress: ${current_experiment}/${total_experiments} - ${arch_model}/${basename_file}"
        
        # Run experiment and capture runtime
        run_experiment "${arch_model}" "${dataset_file}"
        experiment_exit_code=$?
        if [[ $experiment_exit_code -eq 0 ]]; then
            # Read the runtime from the temp file
            runtime_file="/tmp/exp2_runtime_${arch_model}_${basename_file}.tmp"
            if [[ -f "${runtime_file}" ]]; then
                experiment_runtime=$(cat "${runtime_file}")
                if [[ "${experiment_runtime}" =~ ^[0-9]+$ ]]; then
                    total_runtime=$((total_runtime + experiment_runtime))
                fi
                rm -f "${runtime_file}"
            fi
        else
            failed_experiments=$((failed_experiments + 1))
        fi
        
        # Small delay to prevent system overload
        sleep 2
    done
    
    log_message "Completed architecture model: ${arch_model}"
done

# Stop the runtime updater
if [[ -n "$runtime_updater_pid" ]]; then
    kill "$runtime_updater_pid" 2>/dev/null
fi

# Calculate total script runtime
experiment_end_time=$(date +%s)
script_total_runtime=$((experiment_end_time - start_time))
script_hours=$((script_total_runtime / 3600))
script_minutes=$(((script_total_runtime % 3600) / 60))
script_seconds=$((script_total_runtime % 60))

# Update the final runtime in the status file
sed -i "s/⏱️  [0-9]\{3\}h [0-9]\{2\}m [0-9]\{2\}s/⏱️  $(printf "%03dh %02dm %02ds" $script_hours $script_minutes $script_seconds)/" "$status_file"

# Calculate total experiment runtime (excluding overhead)
exp_hours=$((total_runtime / 3600))
exp_minutes=$(((total_runtime % 3600) / 60))
exp_secs=$((total_runtime % 60))

#===================
# RESULTS SUMMARY AND ANALYSIS
#===================

log_message "==============================================================================="
log_message "EXPERIMENT 2 COMPLETED"
log_message "==============================================================================="
log_message "End Time: $(date)"
log_message "Total Experiments: ${total_experiments}"
log_message "Failed Experiments: ${failed_experiments}"
log_message "Success Rate: $(( (total_experiments - failed_experiments) * 100 / total_experiments ))%"
log_message "Total Script Runtime: $(printf "%02dh %02dm %02ds" $script_hours $script_minutes $script_seconds)"
log_message "Total Experiment Runtime: $(printf "%02dh %02dm %02ds" $exp_hours $exp_minutes $exp_secs)"

# Update status file with completion info
cat >> "${status_file}" << EOF

===============================================================================
✅ EXPERIMENT COMPLETED!
===============================================================================
🟢 STATUS: COMPLETED
⏰ END_TIME: $(date)
⏱️  TOTAL_RUNTIME: $(printf "%03dh %02dm %02ds" $script_hours $script_minutes $script_seconds)
📈 RESULTS SUMMARY:
   • Total Experiments: ${total_experiments}
   • ✅ Completed: $((total_experiments - failed_experiments))
   • ❌ Failed: ${failed_experiments}
   • 📊 Success Rate: $(( (total_experiments - failed_experiments) * 100 / total_experiments ))%
📁 OUTPUTS:
   • CSV Results: ${OUTPUT_DIR}/*/agg.ad.pred.eval.mean.csv
   • Top LLM List: ${OUTPUT_DIR}/top_llm_selections.txt
   • Summary: ${OUTPUT_DIR}/scaling_analysis_summary.txt
   • Logs: ${OUTPUT_DIR}/logs/
===============================================================================
🎉 EXP2 READY FOR EXP3! Next: run_exp3.sh --toy -i ${OUTPUT_DIR}
===============================================================================
EOF
# Stop runtime updater
if [[ -n "$runtime_updater_pid" ]] && kill -0 "$runtime_updater_pid" 2>/dev/null; then
    kill "$runtime_updater_pid" 2>/dev/null || true
    log_message "Runtime updater stopped"
fi

log_message "Status updated: COMPLETED (Runtime: $(printf "%02dh %02dm %02ds" $script_hours $script_minutes $script_seconds))"

# Generate scaling analysis summary
summary_file="${OUTPUT_DIR}/scaling_analysis_summary.txt"
echo "Experiment 2: Scaling Analysis Results Summary" > "${summary_file}"
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
                # Extract key metrics (column 3 contains the mean value)
                p5_score=$(grep "P_5" "${result_file}" 2>/dev/null | cut -d',' -f3 | head -1)
                recall5_score=$(grep "recall_5" "${result_file}" 2>/dev/null | cut -d',' -f3 | head -1)
                ndcg5_score=$(grep "ndcg_cut_5" "${result_file}" 2>/dev/null | cut -d',' -f3 | head -1)
                
                if [[ -n "${p5_score}" ]]; then
                    echo "  ${llm_model}: P@5=${p5_score}, R@5=${recall5_score}, NDCG@5=${ndcg5_score}" >> "${summary_file}"
                else
                    echo "  ${llm_model}: COMPLETED (no scores found)" >> "${summary_file}"
                fi
            else
                echo "  ${llm_model}: FAILED" >> "${summary_file}"
            fi
        fi
    done
done

log_message "Scaling analysis summary saved to: ${summary_file}"
log_message "Individual logs available in: ${LOG_DIR}"
log_message "Full results available in: ${OUTPUT_DIR}"

# Display quick results overview
echo ""
echo "Quick Results Overview:"
echo "======================"
find "${OUTPUT_DIR}" -name "agg.ad.pred.eval.mean.csv" | wc -l | xargs echo "Completed experiments:"
find "${LOG_DIR}" -name "*_*.log" | wc -l | xargs echo "Total experiment logs:"

log_message "Experiment 2 script completed successfully!"