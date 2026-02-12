#!/bin/bash

#===============================================================================
# Selective Re-run Script for Failed Experiments
# Purpose: Re-run only specific failed experiments to save computation resources
#
# Usage:
#   Auto-detect failures: ./experiment_scripts/rerun_exp.sh <exp_number>
#   Manual mode: ./experiment_scripts/rerun_exp.sh <exp_number> <arch_model> <dataset1> <dataset2> ...
#   
# Examples:
#   ./experiment_scripts/rerun_exp.sh exp1
#   ./experiment_scripts/rerun_exp.sh exp1 bert google-gemini2.5pro-700 xai-grok3-700
#   ./experiment_scripts/rerun_exp.sh exp2 ctm anthropic-sonnet4-1300
#   ./experiment_scripts/rerun_exp.sh exp3 lda openai-gpt4o-2000
#===============================================================================

# Help menu
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    cat << 'EOF'
SELECTIVE RE-RUN SCRIPT FOR FAILED EXPERIMENTS

DESCRIPTION:
    Re-run only specific failed experiments to save computation resources.
    Automatically detects missing results or allows manual specification.

USAGE:
    ./experiment_scripts/rerun_exp.sh <exp_number> [arch_model] [datasets...]

ARGUMENTS:
    exp_number     Experiment to check: exp1, exp2, or exp3
    arch_model     (Optional) Architecture model: bert, ctm, lda
    datasets...    (Optional) Specific dataset names to re-run

OPTIONS:
    -h, --help     Show this help message and exit

MODES:

  AUTO-DETECT MODE:
    Automatically scans latest experiment results and identifies missing outputs
    ./experiment_scripts/rerun_failed_experiments.sh exp1
    ./experiment_scripts/rerun_failed_experiments.sh exp2
    ./experiment_scripts/rerun_failed_experiments.sh exp3

  MANUAL MODE:
    Specify exact architecture and datasets to re-run
    ./experiment_scripts/rerun_failed_experiments.sh exp1 bert google-gemini2.5pro-700 xai-grok3-700
    ./experiment_scripts/rerun_failed_experiments.sh exp2 ctm anthropic-sonnet4-1300
    ./experiment_scripts/rerun_failed_experiments.sh exp3 lda openai-gpt4o-2000

EXPERIMENT TYPES:
    exp1    LLM Model Ranking (700-sentence datasets)
    exp2    Scaling Analysis (700, 1300, 2000-sentence datasets)
    exp3    Baseline Comparison (1300, 2000 + SemEval baselines)

ARCHITECTURE MODELS:
    bert    BERT-E2E-ABSA architecture
    ctm     Contextualized Topic Model
    lda     Latent Dirichlet Allocation
    btm     Biterm Topic Model
    rnd     Random baseline model

WHAT THIS SCRIPT DOES:
    • Scans experiment output directories for missing agg.ad.pred.eval.mean.csv files
    • Re-runs only failed/missing experiments to save time
    • Uses same configuration as original experiments
    • Creates timestamped output directories (rerun_expN_YYYYMMDD_HHMMSS)
    • Provides detailed progress and error reporting

CONFIGURATION:
    • GPU_ID: "1" (edit script to change)
    • NUM_ASPECTS: 5
    • NUM_FOLDS: 5
    • TOKENIZERS_PARALLELISM: false

OUTPUT:
    • Results saved to: experiment_output/rerun_expN_YYYYMMDD_HHMMSS/
    • Individual logs for each re-run experiment
    • Success/failure status with P@5 scores when available

MONITORING:
    • Real-time progress output to console
    • Individual experiment logs saved
    • Last 10 lines of failed experiments displayed

NEXT STEPS:
    After successful re-runs:
    • Copy results back to original experiment directories if needed
    • Continue with subsequent experiments (exp1 → exp2 → exp3)
    • Analyze results and generate reports

EOF
    exit 0
fi

# Check arguments
if [[ $# -eq 0 ]]; then
    echo "❌ Usage: $0 <exp_number> [arch_model] [datasets...]"
    echo ""
    echo "📋 Examples:"
    echo "   Auto-detect exp1 failures: $0 exp1"
    echo "   Manual exp1 reruns: $0 exp1 bert google-gemini2.5pro-700 xai-grok3-700"
    echo "   Auto-detect exp2 failures: $0 exp2" 
    echo "   Manual exp2 reruns: $0 exp2 ctm anthropic-sonnet4-1300"
    echo "   Auto-detect exp3 failures: $0 exp3"
    echo "   Manual exp3 reruns: $0 exp3 lda openai-gpt4o-2000"
    echo ""
    echo "🧪 Experiment types: exp1, exp2, exp3"
    echo "🏗️ Architecture models: bert, ctm, lda, btm, rnd"
    echo ""
    echo "For detailed help: $0 --help"
    exit 1
fi

# Configuration
export GPU_ID="1"
export NUM_ASPECTS="5"
export NUM_FOLDS="5"
export TOKENIZERS_PARALLELISM=false

EXP_NUMBER=$1
shift

# Validate experiment number
if [[ ! "${EXP_NUMBER}" =~ ^exp[123]$ ]]; then
    echo "❌ Invalid experiment number: ${EXP_NUMBER}"
    echo "   Valid options: exp1, exp2, exp3"
    exit 1
fi

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}" || exit 1

# Experiment-specific configurations
case "${EXP_NUMBER}" in
    "exp1")
        EXP_PATTERN="exp1_llm_model_ranking*"
        EXP_NAME="LLM Model Ranking"
        DEFAULT_DATASETS=("anthropic-haiku3.5-700" "anthropic-sonnet4-700" "google-gemini2.5flash-700" "google-gemini2.5pro-700" "openai-gpt3.5-turbo-700" "openai-gpt4o-700" "xai-grok3-700" "xai-grok4-700")
        DATA_DIR="${PROJECT_ROOT}/experiment_datasets/semeval_implitcits"
        DATASET_SUFFIX=".xml"
        ;;
    "exp2")
        EXP_PATTERN="exp2_scaling_analysis*"
        EXP_NAME="Scaling Analysis"
        DEFAULT_DATASETS=("anthropic-haiku3.5-700" "anthropic-haiku3.5-1300" "anthropic-haiku3.5-2000" "openai-gpt4o-700" "openai-gpt4o-1300" "openai-gpt4o-2000" "xai-grok4-700" "xai-grok4-1300" "xai-grok4-2000")
        DATA_DIR="${PROJECT_ROOT}/experiment_datasets/semeval_implitcits"
        DATASET_SUFFIX=".xml"
        ;;
    "exp3")
        EXP_PATTERN="exp3_*baseline*"
        EXP_NAME="Baseline Comparison"
        DEFAULT_DATASETS=("anthropic-haiku3.5-1300" "anthropic-haiku3.5-2000" "openai-gpt4o-1300" "openai-gpt4o-2000" "semeval-baseline-1300" "semeval-baseline-2000")
        DATA_DIR="${PROJECT_ROOT}/experiment_datasets"
        DATASET_SUFFIX=".xml"
        ;;
esac

# Function to get correct data file path
get_data_file() {
    local dataset=$1
    
    case "${EXP_NUMBER}" in
        "exp1"|"exp2")
            echo "${DATA_DIR}/${dataset}${DATASET_SUFFIX}"
            ;;
        "exp3")
            if [[ "${dataset}" == semeval-baseline-* ]]; then
                local size=$(echo "${dataset}" | grep -o '[0-9]\+')
                if [[ "${size}" == "1300" ]]; then
                    echo "${DATA_DIR}/semeval_baselines/SemEval-15-res-1315.xml"
                elif [[ "${size}" == "2000" ]]; then
                    echo "${DATA_DIR}/semeval_baselines/SemEval-16-res-2000.xml"
                else
                    echo "${DATA_DIR}/semeval_baselines/${dataset}${DATASET_SUFFIX}"
                fi
            else
                echo "${DATA_DIR}/semeval_implitcits/${dataset}${DATASET_SUFFIX}"
            fi
            ;;
    esac
}

# Function to run single experiment
run_experiment() {
    local arch_model=$1
    local dataset_name=$2
    local data_file=$(get_data_file "${dataset_name}")
    local output_dir="${PROJECT_ROOT}/experiment_output/rerun_${EXP_NUMBER}_$(date +%Y%m%d_%H%M%S)"
    local log_file="${output_dir}/logs/${arch_model}_${dataset_name}.log"
    
    echo "🔄 Re-running ${EXP_NAME}: ${arch_model} on ${dataset_name}"
    
    # Check if dataset file exists
    if [[ ! -f "${data_file}" ]]; then
        echo "❌ ERROR: Dataset file not found: ${data_file}"
        if [[ "${EXP_NUMBER}" == "exp1" || "${EXP_NUMBER}" == "exp2" ]]; then
            echo "   This might be due to XML corruption. Please check the dataset."
        fi
        return 1
    fi
    
    # Create output directory
    mkdir -p "${output_dir}/${arch_model}/${dataset_name}"
    mkdir -p "$(dirname "${log_file}")"
    
    # Build command
    local cmd="python main.py -am ${arch_model} -data ${data_file} -output ${output_dir}/${arch_model}/${dataset_name} -naspects ${NUM_ASPECTS} -nfolds ${NUM_FOLDS}"
    
    # Add GPU if specified
    if [[ -n "${GPU_ID}" ]]; then
        cmd+=" -gpu ${GPU_ID}"
    fi
    
    echo "🚀 CMD: ${cmd}"
    echo "📝 Log: ${log_file}"
    
    # Change to src directory and run
    cd "${PROJECT_ROOT}/src" || exit 1
    
    # Execute experiment with timestamp
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting rerun: ${arch_model}/${dataset_name}" | tee "${log_file}"
    
    if eval "${cmd}" >> "${log_file}" 2>&1; then
        echo "✅ SUCCESS: ${arch_model}/${dataset_name}"
        
        # Check for result file
        local result_file="${output_dir}/${arch_model}/${dataset_name}/agg.ad.pred.eval.mean.csv"
        if [[ -f "${result_file}" ]]; then
            echo "📊 Results available: ${result_file}"
            
            # Extract P@5 score for quick verification
            local p5_score=$(grep "P_5" "${result_file}" 2>/dev/null | cut -d',' -f2 | head -1)
            if [[ -n "${p5_score}" ]]; then
                echo "🎯 P@5 Score: ${p5_score}"
            fi
        else
            echo "⚠️  Warning: No aggregated results found, but command succeeded"
        fi
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed successfully: ${arch_model}/${dataset_name}" | tee -a "${log_file}"
        return 0
    else
        echo "❌ FAILED: ${arch_model}/${dataset_name}"
        echo "📝 Check log: ${log_file}"
        echo ""
        echo "📄 Last 10 lines of log:"
        tail -10 "${log_file}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed: ${arch_model}/${dataset_name}" | tee -a "${log_file}"
        return 1
    fi
    
    cd "${PROJECT_ROOT}"
}

# Main execution
echo "🔧 ${EXP_NAME} Failure Rerun Tool"
echo "$(printf '=%.0s' {1..40})"

if [[ $# -eq 0 ]]; then
    echo "🔍 Auto-detecting failed experiments from ${EXP_NUMBER} results..."
    
    # Find the latest experiment run
    latest_exp=$(find experiment_output -name "${EXP_PATTERN}" -type d | sort | tail -1)
    
    if [[ -z "${latest_exp}" ]]; then
        echo "❌ No ${EXP_NUMBER} results found."
        echo ""
        echo "💡 Usage examples:"
        echo "   Auto-detect failures: $0 ${EXP_NUMBER}"
        echo "   Manual failures: $0 ${EXP_NUMBER} bert google-gemini2.5pro-700 xai-grok3-700"
        echo "   Single failure: $0 ${EXP_NUMBER} ctm anthropic-sonnet4-700"
        exit 1
    fi
    
    echo "📁 Found latest ${EXP_NUMBER}: ${latest_exp}"
    
    # Auto-detect failed experiments
    failed_experiments=()
    
    echo ""
    echo "🔍 Scanning for missing results..."
    for arch_model in bert ctm lda btm rnd; do
        echo "  Checking ${arch_model}..."
        for dataset in "${DEFAULT_DATASETS[@]}"; do
            result_file="${latest_exp}/${arch_model}/${dataset}/agg.ad.pred.eval.mean.csv"
            if [[ ! -f "${result_file}" ]]; then
                failed_experiments+=("${arch_model}:${dataset}")
                echo "    ❌ Missing: ${arch_model}/${dataset}"
            else
                echo "    ✅ Present: ${arch_model}/${dataset}"
            fi
        done
    done
    
    if [[ ${#failed_experiments[@]} -eq 0 ]]; then
        echo ""
        echo "🎉 No failed experiments detected! All ${EXP_NUMBER} results are present."
        exit 0
    fi
    
    echo ""
    echo "📋 Found ${#failed_experiments[@]} failed experiments. Re-running..."
    echo ""
    
    for exp in "${failed_experiments[@]}"; do
        IFS=':' read -r arch_model dataset <<< "$exp"
        run_experiment "${arch_model}" "${dataset}"
        echo ""
        sleep 2  # Small delay between experiments
    done
    
else
    # Manual mode: specify arch_model and datasets
    arch_model=$1
    shift
    datasets=("$@")
    
    # Validate architecture model
    if [[ ! "${arch_model}" =~ ^(bert|ctm|lda|btm|rnd)$ ]]; then
        echo "❌ Invalid architecture model: ${arch_model}"
        echo "   Valid options: bert, ctm, lda, btm, rnd"
        exit 1
    fi
    
    echo "🎯 Manual rerun mode for ${EXP_NAME}"
    echo "Architecture model: ${arch_model}"
    echo "Datasets: ${datasets[*]}"
    echo ""
    
    for dataset in "${datasets[@]}"; do
        run_experiment "${arch_model}" "${dataset}"
        echo ""
        sleep 2
    done
fi

echo "🏁 ${EXP_NAME} rerun complete!"
echo ""
echo "💡 Next steps:"
case "${EXP_NUMBER}" in
    "exp1")
        echo "   1. Check results in experiment_output/rerun_exp1_*/"
        echo "   2. Run exp2 scaling analysis: ./experiment_scripts/exp2_scaling_analysis.sh"
        echo "   3. Run exp3 baseline comparison: ./experiment_scripts/exp3_baseline_comparison.sh"
        ;;
    "exp2")
        echo "   1. Check results in experiment_output/rerun_exp2_*/"
        echo "   2. Analyze scaling patterns across dataset sizes"
        echo "   3. Continue with exp3 baseline comparison: ./experiment_scripts/exp3_baseline_comparison.sh"
        ;;
    "exp3")
        echo "   1. Check results in experiment_output/rerun_exp3_*/"
        echo "   2. Analyze LLM vs SemEval baseline comparisons"
        echo "   3. Generate final research conclusions"
        ;;
esac