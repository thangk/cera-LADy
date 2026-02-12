#!/bin/bash

#===============================================================================
# EXPERIMENT 2: Copy Top 3 LLM Models from Exp1 Based on Average P@5 Scores
# Purpose: Analyze exp1 results and copy top 3 LLM models for each architecture
#          based on average P@5 scores across all sizes
#
# Prerequisites:
#   1. cd /path/to/LADy-kap  
#   2. ./experiment_scripts/run_exp2.sh -i <exp1_output>
#
# Note: This script does NOT run experiments, it only analyzes and copies results
#===============================================================================

# Default values
INPUT_PATH=""
TOY_MODE=false

# Help menu
show_help() {
    cat << 'EOF'
EXPERIMENT 2: Copy Top 3 LLM Models from Exp1

DESCRIPTION:
    Analyzes exp1 experiment_summary.txt to calculate average P@5 scores for each
    LLM model across all sizes, selects top 3 models per architecture, and copies
    their folders to exp2 output directory.

USAGE:
    ./experiment_scripts/run_exp2.sh -i <exp1_output_dir> [OPTIONS]

REQUIRED:
    -i, --input PATH       Path to Exp1 output directory containing experiment_summary.txt
                          (e.g., experiment_output/toy_exp1_llm_model_ranking)

OPTIONS:
    -h, --help             Show this help message and exit
    --toy                  Run in toy mode (prefix output with "toy_")

PREREQUISITES:
    1. cd /path/to/LADy-kap
    2. Complete run_exp1.sh first

WHAT THIS SCRIPT DOES:
    • Reads experiment_summary.txt from exp1 output
    • Calculates average P@5 scores for each LLM model across all sizes
    • Selects top 3 LLM models for each architecture model
    • Copies selected model folders to exp2 output directory
    • If models have equal scores, selects alphabetically

OUTPUT:
    • Normal mode: experiment_output/exp2_top_models[_runN]/
    • Toy mode: experiment_output/toy_exp2_top_models[_runN]/
    • Summary: {output_dir}/top_models_summary.txt

EXAMPLES:
    # Analyze and copy top models from exp1 results
    ./experiment_scripts/run_exp2.sh -i experiment_output/exp1_llm_model_ranking
    
    # Toy mode
    ./experiment_scripts/run_exp2.sh --toy -i experiment_output/toy_exp1_llm_model_ranking

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
    
    # Check if experiment_summary.txt exists
    if [[ ! -f "$INPUT_PATH/experiment_summary.txt" ]]; then
        echo "❌ ERROR: experiment_summary.txt not found in: $INPUT_PATH"
        exit 1
    fi
}

# Function to find next available directory name
get_unique_output_dir() {
    local base_name="exp2_top_models"
    if [[ "$TOY_MODE" == true ]]; then
        base_name="toy_exp2_top_models"
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

# Main execution starts here
parse_args "$@"
validate_args

# Ensure we're in the project root directory
if [[ ! -f "src/main.py" ]] || [[ ! -d "experiment_scripts" ]]; then
    echo "❌ ERROR: Please run this script from the LADy-kap project root directory"
    exit 1
fi

# Set up paths
OUTPUT_DIR=$(get_unique_output_dir)
SUMMARY_FILE="${OUTPUT_DIR}/top_models_summary.txt"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "==============================================================================="
echo "EXPERIMENT 2: Analyzing Top LLM Models from Exp1"
echo "==============================================================================="
echo "Start Time: $(date)"
echo "Toy Mode: ${TOY_MODE}"
echo "Input Path: ${INPUT_PATH}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "==============================================================================="

# Architecture models
ARCH_MODELS=("bert" "ctm" "btm" "rnd")

# Parse experiment_summary.txt and calculate averages
echo "📊 Parsing experiment_summary.txt..."

# Create summary file header
cat > "${SUMMARY_FILE}" << EOF
Experiment 2: Top LLM Models Selection Summary
Generated on: $(date)
Based on average P@5 scores from: ${INPUT_PATH}/experiment_summary.txt
===============================================================================

Selection Criteria:
- Calculate average P@5 score for each LLM model across all sizes
- Select top 3 models for each architecture
- If scores are equal, select alphabetically

EOF

# Process each architecture model
for arch_model in "${ARCH_MODELS[@]}"; do
    echo ""
    echo "🔍 Processing architecture: ${arch_model}"
    echo "" >> "${SUMMARY_FILE}"
    echo "Architecture Model: ${arch_model}" >> "${SUMMARY_FILE}"
    echo "----------------------------------------" >> "${SUMMARY_FILE}"
    
    # Create architecture directory in output
    arch_output_dir="${OUTPUT_DIR}/${arch_model}"
    mkdir -p "${arch_output_dir}"
    
    # Extract scores for this architecture from experiment_summary.txt
    awk -v arch="$arch_model" '
    BEGIN { in_section = 0; }
    /^Architecture Model: / { 
        if ($3 == arch) {
            in_section = 1;
        } else {
            in_section = 0;
        }
    }
    in_section && /: P@5 = / {
        # Extract LLM model name and score
        split($1, parts, ":");
        model_size = parts[1];
        
        # Extract score value
        score = $4;
        
        # Extract base model name (remove size suffix)
        match(model_size, /(.+)-([0-9]+)$/, arr);
        if (arr[1] != "") {
            model = arr[1];
            size = arr[2];
            print model, size, score;
        }
    }
    ' "${INPUT_PATH}/experiment_summary.txt" > "/tmp/exp2_${arch_model}_scores.tmp"
    
    # Calculate averages for each LLM model
    declare -A model_scores
    declare -A model_counts
    declare -A model_sizes
    
    while read -r model size score; do
        if [[ -n "$model" && -n "$score" ]]; then
            # Add score to sum
            if [[ -z "${model_scores[$model]}" ]]; then
                model_scores[$model]="0"
                model_counts[$model]="0"
                model_sizes[$model]=""
            fi
            model_scores[$model]=$(awk "BEGIN {print ${model_scores[$model]} + $score}")
            model_counts[$model]=$((${model_counts[$model]} + 1))
            model_sizes[$model]="${model_sizes[$model]} ${size}"
        fi
    done < "/tmp/exp2_${arch_model}_scores.tmp"
    
    # Calculate averages and prepare for sorting
    > "/tmp/exp2_${arch_model}_averages.tmp"
    for model in "${!model_scores[@]}"; do
        if [[ ${model_counts[$model]} -gt 0 ]]; then
            avg=$(awk "BEGIN {print ${model_scores[$model]} / ${model_counts[$model]}}")
            echo "$avg $model" >> "/tmp/exp2_${arch_model}_averages.tmp"
            echo "  ${model}: Average P@5 = ${avg} (sizes:${model_sizes[$model]})" >> "${SUMMARY_FILE}"
        fi
    done
    
    # Sort by average score (descending) and then by name (ascending)
    # Get top 3 models
    echo "" >> "${SUMMARY_FILE}"
    echo "Top 3 LLM Models (by average P@5):" >> "${SUMMARY_FILE}"
    
    sort -k1,1nr -k2,2 "/tmp/exp2_${arch_model}_averages.tmp" | head -3 | while read -r avg model; do
        echo "  ${model} = ${avg}" >> "${SUMMARY_FILE}"
        
        # Copy all size folders for this model from exp1 to exp2
        echo "📁 Copying folders for ${model}..."
        
        # Find and copy all folders for this model
        for size_folder in "${INPUT_PATH}/${arch_model}/${model}"-*; do
            if [[ -d "$size_folder" ]]; then
                folder_name=$(basename "$size_folder")
                echo "   Copying ${folder_name}..."
                cp -r "$size_folder" "${arch_output_dir}/"
            fi
        done
    done
    
    # List excluded models
    echo "" >> "${SUMMARY_FILE}"
    echo "Excluded models:" >> "${SUMMARY_FILE}"
    sort -k1,1nr -k2,2 "/tmp/exp2_${arch_model}_averages.tmp" | tail -n +4 | while read -r avg model; do
        echo "  ${model} = ${avg}" >> "${SUMMARY_FILE}"
    done
    
    # Clean up temp files
    rm -f "/tmp/exp2_${arch_model}_scores.tmp"
    rm -f "/tmp/exp2_${arch_model}_averages.tmp"
    
    # Clear associative arrays for next iteration
    unset model_scores
    unset model_counts
    unset model_sizes
done

# Final summary
echo ""
echo "==============================================================================="
echo "✅ EXPERIMENT 2 COMPLETED!"
echo "==============================================================================="
echo "End Time: $(date)"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Summary File: ${SUMMARY_FILE}"
echo ""
echo "📊 Results Summary:"
echo "  • Analyzed P@5 scores from: ${INPUT_PATH}/experiment_summary.txt"
echo "  • Selected top 3 LLM models for each architecture"
echo "  • Copied model folders to: ${OUTPUT_DIR}"
echo ""
echo "📁 Directory Structure:"
for arch_model in "${ARCH_MODELS[@]}"; do
    num_models=$(ls -d "${OUTPUT_DIR}/${arch_model}"/* 2>/dev/null | wc -l)
    echo "  • ${arch_model}/: ${num_models} model folders"
done
echo ""
echo "🚀 Ready for data collection!"
echo "==============================================================================="

# Append completion info to summary
cat >> "${SUMMARY_FILE}" << EOF

===============================================================================
Experiment 2 Completed Successfully
End Time: $(date)
Output Directory: ${OUTPUT_DIR}
===============================================================================
EOF

echo "✅ Top models successfully copied from exp1 to exp2!"