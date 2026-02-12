#!/bin/bash

#===============================================================================
# Experiment Stopper Script
# Purpose: Easily stop running experiment scripts
#
# Usage:
#   ./stop_exp.sh           # Stop all experiments
#   ./stop_exp.sh exp1      # Stop only exp1
#   ./stop_exp.sh exp2      # Stop only exp2  
#   ./stop_exp.sh exp3      # Stop only exp3
#===============================================================================

# Help menu
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    cat << 'EOF'
EXPERIMENT STOPPER SCRIPT

DESCRIPTION:
    Easily stop running experiment scripts and related processes.

USAGE:
    ./stop_exp.sh [EXPERIMENT]

ARGUMENTS:
    EXPERIMENT    (Optional) Specific experiment to stop: exp1, exp2, exp3
                 If not provided, stops all experiments

OPTIONS:
    -h, --help   Show this help message and exit

EXAMPLES:
    ./stop_exp.sh           # Stop all running experiments
    ./stop_exp.sh exp1      # Stop only experiment 1
    ./stop_exp.sh exp2      # Stop only experiment 2
    ./stop_exp.sh exp3      # Stop only experiment 3

WHAT THIS SCRIPT DOES:
    • Finds and kills experiment script processes
    • Stops any related python main.py processes
    • Shows status of stopped processes
    • Provides confirmation of actions taken

EOF
    exit 0
fi

EXPERIMENT_TYPE="$1"

echo "🛑 EXPERIMENT STOPPER"
echo "===================="

if [[ -z "$EXPERIMENT_TYPE" ]]; then
    echo "🎯 Stopping ALL running experiments..."
    PATTERNS=("exp1_llm_model_ranking.sh" "exp2_scaling_analysis.sh" "exp3_baseline_comparison.sh")
else
    case "$EXPERIMENT_TYPE" in
        "exp1")
            echo "🎯 Stopping Experiment 1 (LLM Model Ranking)..."
            PATTERNS=("exp1_llm_model_ranking.sh")
            ;;
        "exp2")
            echo "🎯 Stopping Experiment 2 (Scaling Analysis)..."
            PATTERNS=("exp2_scaling_analysis.sh")
            ;;
        "exp3")
            echo "🎯 Stopping Experiment 3 (Baseline Comparison)..."
            PATTERNS=("exp3_baseline_comparison.sh")
            ;;
        *)
            echo "❌ Invalid experiment type: $EXPERIMENT_TYPE"
            echo "   Valid options: exp1, exp2, exp3"
            exit 1
            ;;
    esac
fi

STOPPED_COUNT=0

# Stop experiment scripts
for pattern in "${PATTERNS[@]}"; do
    echo ""
    echo "🔍 Looking for processes matching: $pattern"
    
    # Find processes
    PIDS=$(pgrep -f "$pattern" 2>/dev/null)
    
    if [[ -n "$PIDS" ]]; then
        echo "📋 Found processes: $PIDS"
        
        for pid in $PIDS; do
            echo "🔪 Killing process $pid ($pattern)"
            kill "$pid" 2>/dev/null
            
            # Wait a moment and check if it's really dead
            sleep 1
            if kill -0 "$pid" 2>/dev/null; then
                echo "💀 Force killing process $pid"
                kill -9 "$pid" 2>/dev/null
            fi
            
            STOPPED_COUNT=$((STOPPED_COUNT + 1))
        done
    else
        echo "✅ No processes found for: $pattern"
    fi
done

# Also stop any related python main.py processes that might be running experiments
echo ""
echo "🔍 Looking for related Python experiment processes..."
PYTHON_PIDS=$(pgrep -f "python.*main.py.*-am.*(bert|ctm|lda)" 2>/dev/null)

if [[ -n "$PYTHON_PIDS" ]]; then
    echo "📋 Found Python experiment processes: $PYTHON_PIDS"
    
    for pid in $PYTHON_PIDS; do
        echo "🔪 Killing Python experiment process $pid"
        kill "$pid" 2>/dev/null
        
        # Wait a moment and check if it's really dead
        sleep 1
        if kill -0 "$pid" 2>/dev/null; then
            echo "💀 Force killing Python process $pid"
            kill -9 "$pid" 2>/dev/null
        fi
        
        STOPPED_COUNT=$((STOPPED_COUNT + 1))
    done
else
    echo "✅ No Python experiment processes found"
fi

echo ""
echo "=" * 40
if [[ $STOPPED_COUNT -eq 0 ]]; then
    echo "😌 No running experiments found to stop"
else
    echo "✅ Stopped $STOPPED_COUNT processes"
fi

echo ""
echo "📊 Current experiment-related processes:"
pgrep -f "(exp[123]|main.py.*-am)" -l || echo "   (none found)"

echo ""
echo "💡 To check if experiments are still running:"
echo "   ps aux | grep -E '(exp[123]|main.py.*-am)'"