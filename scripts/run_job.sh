#!/bin/bash
# Run LADy aspect detection benchmarks on a CERA job output directory.
# Usage: run_job.sh type={real|cera|heuristic} dir=<job-name> ac=<categories> [options]
# Run from inside the Docker container: docker exec -it cera-lady-cli bash /app/scripts/run_job.sh ...

set -eo pipefail
export TOKENIZERS_PARALLELISM=false
SCRIPT_START=$(date +%s)

# ─── Defaults ────────────────────────────────────────────────────────────────

TYPE=""
DIR=""
AC=""
MODELS="rnd,btm,ctm,bert"
TARGETS=""
PARALLEL="none"

JOBS_ROOT="/app/jobs"
CATEGORIES_ROOT="/app/datasets/categories"
OUTPUT_ROOT="/app/output"

# ─── Help ────────────────────────────────────────────────────────────────────

show_help() {
    cat <<'HELP'
run_job.sh — Run LADy aspect detection benchmarks on CERA job output

USAGE
  run_job.sh type=<type> dir=<job-dir> ac=<categories> [options]

REQUIRED
  type=       Dataset type: real, cera, or heuristic
  dir=        Job directory name (resolved under /app/jobs/)
              or an absolute path to a job directory
  ac=         Aspect categories: name of a .csv file in /app/datasets/categories/
              (e.g., laptops, restaurants, hotels) or a full path to a .csv file.
              The number of categories in the file is used as the naspects value.

OPTIONS
  models=     Comma-separated list of LADy architecture models to run
              Available: rnd, btm, ctm, bert
              Use models=all for all four (default: all)
  targets=    Comma-separated target sizes to evaluate
              These correspond to folder names inside the job's datasets/ dir
              (default: auto-discover all numeric folders)
  parallel=   Parallelization mode (default: none)
              none    - Sequential execution (default)
              models  - Run all models in parallel for each run
              targets - Run all targets in parallel (models sequential)
              all     - Run both targets and models in parallel
  -h, --help  Show this help message

XML DISCOVERY
  For type=cera or type=heuristic:
    Looks for XML files inside datasets/{target}/run{N}/ (recursive).
    Prefers *explicit.xml over *implicit.xml.

  For type=real:
    Looks for XML files directly inside datasets/{target}/ (recursive).
    Each XML found counts as one "run".

OUTPUT
  Results are written to /app/output/{type}/ with collision avoidance.
  If /app/output/cera/ already exists, uses cera-1/, cera-2/, etc.

  Per-target structure:
    output/{type}/target-{size}/run{N}/   LADy results per run
    output/{type}/target-{size}/aggregate.csv   Cross-run mean +/- std

EXAMPLES
  # Full benchmark: all models, all targets, laptop categories
  run_job.sh type=cera dir=j9715-rq1-cera-laptops-5-targets ac=laptops

  # Single model, specific targets
  run_job.sh type=heuristic dir=j972a-rq1-heuristic ac=laptops models=bert targets=100,500

  # Real baseline with hotel categories
  run_job.sh type=real dir=rq3-real-hotels ac=hotels

  # Full path and custom categories
  run_job.sh type=cera dir=/data/my-job ac=/data/custom-categories.csv

  # Parallel: run all models simultaneously
  run_job.sh type=cera dir=my-job ac=laptops parallel=models

  # Parallel: run all targets simultaneously
  run_job.sh type=cera dir=my-job ac=laptops parallel=targets

  # Parallel: run both targets and models simultaneously
  run_job.sh type=cera dir=my-job ac=laptops parallel=all
HELP
}

# ─── Time Formatting ────────────────────────────────────────────────────────

fmt_duration() {
    local total_seconds="$1"
    local hours=$((total_seconds / 3600))
    local minutes=$(( (total_seconds % 3600) / 60 ))
    local seconds=$((total_seconds % 60))
    printf "%02dh %02dm %02ds" "$hours" "$minutes" "$seconds"
}

fmt_timestamp() {
    date -d "@$1" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || date -r "$1" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "N/A"
}

# ─── Functions ───────────────────────────────────────────────────────────────

resolve_categories() {
    local ac_arg="$1"

    if [[ "$ac_arg" == /* || "$ac_arg" == *.csv ]]; then
        CAT_PATH="$ac_arg"
    else
        CAT_PATH="${CATEGORIES_ROOT}/${ac_arg}.csv"
    fi

    if [[ ! -f "$CAT_PATH" ]]; then
        echo "ERROR: Categories file not found: $CAT_PATH"
        exit 1
    fi

    NASPECTS=$(tail -n +2 "$CAT_PATH" | grep -c -v '^[[:space:]]*$' || true)
    if [[ "$NASPECTS" -eq 0 ]]; then
        echo "ERROR: No aspect categories found in $CAT_PATH"
        exit 1
    fi
}

resolve_dir() {
    local dir_arg="$1"

    if [[ "$dir_arg" == /* ]]; then
        JOB_DIR="$dir_arg"
    else
        JOB_DIR="${JOBS_ROOT}/${dir_arg}"
    fi

    if [[ ! -d "$JOB_DIR" ]]; then
        echo "ERROR: Job directory not found: $JOB_DIR"
        exit 1
    fi

    if [[ ! -d "$JOB_DIR/datasets" ]]; then
        echo "ERROR: No datasets/ subfolder in $JOB_DIR"
        exit 1
    fi
}

discover_targets() {
    if [[ -n "$TARGETS" ]]; then
        IFS=',' read -ra TARGET_LIST <<< "$TARGETS"
    else
        TARGET_LIST=()
        for d in "$JOB_DIR/datasets"/*/; do
            [[ -d "$d" ]] || continue
            local name
            name=$(basename "$d")
            if [[ "$name" =~ ^[0-9]+$ ]]; then
                TARGET_LIST+=("$name")
            fi
        done
        IFS=$'\n' TARGET_LIST=($(sort -n <<<"${TARGET_LIST[*]}")); unset IFS
    fi

    if [[ ${#TARGET_LIST[@]} -eq 0 ]]; then
        echo "ERROR: No targets found in $JOB_DIR/datasets/"
        exit 1
    fi
}

# Find XMLs for cera/heuristic jobs: datasets/{target}/run{N}/{model-slug}/*.xml
discover_xmls_generated() {
    local target_dir="$1"
    local -n _result=$2

    _result=()
    local run_dirs=()
    for rd in "$target_dir"/run*/; do
        [[ -d "$rd" ]] || continue
        run_dirs+=("$rd")
    done

    # Sort run dirs numerically by run number
    IFS=$'\n' run_dirs=($(printf '%s\n' "${run_dirs[@]}" | sort -t'n' -k2 -n)); unset IFS

    for run_dir in "${run_dirs[@]}"; do
        local xml=""

        # Prefer explicit
        xml=$(find "$run_dir" -name '*explicit.xml' -type f 2>/dev/null | head -1)

        # Fallback to implicit
        if [[ -z "$xml" ]]; then
            xml=$(find "$run_dir" -name '*implicit.xml' -type f 2>/dev/null | head -1)
        fi

        # Fallback to any XML
        if [[ -z "$xml" ]]; then
            xml=$(find "$run_dir" -name '*.xml' -type f 2>/dev/null | head -1)
        fi

        if [[ -n "$xml" ]]; then
            _result+=("$xml")
        else
            echo "  WARNING: No XML found in $run_dir"
        fi
    done
}

# Find XMLs for real jobs: datasets/{target}/*.xml (flat, no run subdirs)
discover_xmls_real() {
    local target_dir="$1"
    local -n _result=$2

    _result=()
    while IFS= read -r -d '' xml; do
        _result+=("$xml")
    done < <(find "$target_dir" -name '*.xml' -type f -print0 2>/dev/null | sort -z)
}

resolve_output_dir() {
    local base_name="$1"

    if [[ ! -d "$OUTPUT_ROOT/$base_name" ]]; then
        OUTPUT_DIR="$OUTPUT_ROOT/$base_name"
    else
        local i=1
        while [[ -d "$OUTPUT_ROOT/${base_name}-${i}" ]]; do
            ((i++))
        done
        OUTPUT_DIR="$OUTPUT_ROOT/${base_name}-${i}"
    fi

    mkdir -p "$OUTPUT_DIR"
}

run_experiment() {
    local xml_path="$1"
    local model="$2"
    local run_output="$3"

    mkdir -p "$run_output"
    local log_file="${run_output}/${model}_log.txt"
    local start_time
    start_time=$(date +%s)

    if python -u main.py \
        -am "$model" \
        -data "$xml_path" \
        -output "$run_output" \
        -naspects "$NASPECTS" \
        -categories "$CAT_PATH" \
        2>&1 | tee "$log_file"; then

        local end_time
        end_time=$(date +%s)
        local elapsed=$(( end_time - start_time ))
        echo "  DONE: $model ($(fmt_duration $elapsed))"
        return 0
    else
        local end_time
        end_time=$(date +%s)
        local elapsed=$(( end_time - start_time ))
        echo "  FAILED: $model ($(fmt_duration $elapsed))"
        return 1
    fi
}

aggregate_cross_runs() {
    local target="$1"
    local target_output="$OUTPUT_DIR/target-${target}"

    # Collect agg CSVs from all runs
    local agg_files=()
    for run_dir in "$target_output"/run*/; do
        [[ -d "$run_dir" ]] || continue
        local agg="$run_dir/agg.ad.pred.eval.mean.csv"
        if [[ -f "$agg" ]]; then
            agg_files+=("$agg")
        fi
    done

    if [[ ${#agg_files[@]} -eq 0 ]]; then
        echo "  WARNING: No aggregation files found for target $target"
        return
    fi

    python3 -c "
import pandas as pd
import sys
import os

files = sys.argv[1:]
dfs = []
for i, f in enumerate(files):
    try:
        df = pd.read_csv(f, index_col=0)
        # Use the 'mean' column if it exists, otherwise last column
        if 'mean' in df.columns:
            col = df[['mean']].rename(columns={'mean': f'run{i+1}'})
        else:
            col = df.iloc[:, [-1]].rename(columns={df.columns[-1]: f'run{i+1}'})
        dfs.append(col)
    except Exception as e:
        print(f'  WARNING: Could not read {f}: {e}', file=sys.stderr)

if not dfs:
    print('  WARNING: No valid aggregation data found', file=sys.stderr)
    sys.exit(0)

merged = pd.concat(dfs, axis=1)
run_cols = [c for c in merged.columns if c.startswith('run')]
merged['mean'] = merged[run_cols].mean(axis=1)
merged['std'] = merged[run_cols].std(axis=1)

out_path = os.path.join('${target_output}', 'aggregate.csv')
merged.to_csv(out_path)
print(f'  Aggregated {len(files)} run(s) -> {out_path}')
" "${agg_files[@]}"
}

# ─── Process a single target (used by both sequential and parallel modes) ───

process_target() {
    local target="$1"
    local target_dir="$JOB_DIR/datasets/$target"
    local target_start
    target_start=$(date +%s)

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Target: $target"
    echo "  Started: $(fmt_timestamp $target_start)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Discover XMLs
    local XML_LIST=()
    if [[ "$TYPE" == "real" ]]; then
        discover_xmls_real "$target_dir" XML_LIST
    else
        discover_xmls_generated "$target_dir" XML_LIST
    fi

    echo "  Found ${#XML_LIST[@]} run(s)"

    if [[ ${#XML_LIST[@]} -eq 0 ]]; then
        echo "  WARNING: No XMLs found for target $target, skipping"
        return
    fi

    local target_failures=()

    for run_idx in "${!XML_LIST[@]}"; do
        local run_num=$((run_idx + 1))
        local xml="${XML_LIST[$run_idx]}"
        local run_output="$OUTPUT_DIR/target-${target}/run${run_num}"

        echo ""
        echo "  ── Run $run_num: $(basename "$xml")"
        echo "     XML: $xml"

        if [[ "$PARALLEL" == "models" || "$PARALLEL" == "all" ]]; then
            # Run all models in parallel for this run
            # Each subshell writes to its own log file to avoid shared-pipe contention
            local model_pids=()
            local model_names=()
            local model_logs=()
            mkdir -p "$run_output"

            for model in "${MODEL_LIST[@]}"; do
                local mlog="$run_output/.parallel_${model}.log"
                model_logs+=("$mlog")
                echo ""
                echo "  [parallel] target=$target run=$run_num model=$model"
                (
                    set +e  # Prevent pipe errors from killing subshell
                    run_experiment "$xml" "$model" "$run_output"
                    exit $?
                ) > "$mlog" 2>&1 &
                model_pids+=($!)
                model_names+=("$model")
            done

            # Wait for all model processes, replay logs, and collect failures
            for i in "${!model_pids[@]}"; do
                if ! wait "${model_pids[$i]}"; then
                    target_failures+=("target=$target run=$run_num model=${model_names[$i]}")
                fi
                # Replay model log to main output
                if [[ -f "${model_logs[$i]}" ]]; then
                    cat "${model_logs[$i]}"
                    rm -f "${model_logs[$i]}"
                fi
            done
        else
            # Sequential model execution
            for model in "${MODEL_LIST[@]}"; do
                echo ""
                echo "  target=$target run=$run_num model=$model"

                if ! run_experiment "$xml" "$model" "$run_output"; then
                    target_failures+=("target=$target run=$run_num model=$model")
                fi
            done
        fi
    done

    echo ""
    echo "  Aggregating runs for target $target..."
    aggregate_cross_runs "$target"

    local target_end
    target_end=$(date +%s)
    local target_elapsed=$(( target_end - target_start ))

    echo ""
    echo "  ── Target $target complete"
    echo "     Started:  $(fmt_timestamp $target_start)"
    echo "     Finished: $(fmt_timestamp $target_end)"
    echo "     Duration: $(fmt_duration $target_elapsed)"

    # Write failures to a temp file so the parent can collect them
    if [[ ${#target_failures[@]} -gt 0 ]]; then
        local fail_file="$OUTPUT_DIR/.failures_target_${target}"
        printf '%s\n' "${target_failures[@]}" > "$fail_file"
    fi
}

# ─── Argument Parsing ────────────────────────────────────────────────────────

if [[ $# -eq 0 ]]; then
    show_help
    exit 0
fi

for arg in "$@"; do
    case "$arg" in
        -h|--help)  show_help; exit 0 ;;
        type=*)     TYPE="${arg#type=}" ;;
        dir=*)      DIR="${arg#dir=}" ;;
        ac=*)       AC="${arg#ac=}" ;;
        models=*)   MODELS="${arg#models=}" ;;
        targets=*)  TARGETS="${arg#targets=}" ;;
        parallel=*) PARALLEL="${arg#parallel=}" ;;
        *)          echo "ERROR: Unknown argument: $arg"; echo ""; show_help; exit 1 ;;
    esac
done

# ─── Validation ──────────────────────────────────────────────────────────────

if [[ -z "$TYPE" ]]; then
    echo "ERROR: type= is required (real, cera, or heuristic)"
    exit 1
fi
if [[ "$TYPE" != "real" && "$TYPE" != "cera" && "$TYPE" != "heuristic" ]]; then
    echo "ERROR: type= must be one of: real, cera, heuristic (got: $TYPE)"
    exit 1
fi
if [[ -z "$DIR" ]]; then
    echo "ERROR: dir= is required"
    exit 1
fi
if [[ -z "$AC" ]]; then
    echo "ERROR: ac= is required (e.g., laptops, restaurants, hotels)"
    exit 1
fi
if [[ "$PARALLEL" != "none" && "$PARALLEL" != "models" && "$PARALLEL" != "targets" && "$PARALLEL" != "all" ]]; then
    echo "ERROR: parallel= must be one of: none, models, targets, all (got: $PARALLEL)"
    exit 1
fi

# ─── Resolve inputs ──────────────────────────────────────────────────────────

resolve_dir "$DIR"
resolve_categories "$AC"

# Expand models
if [[ "$MODELS" == "all" ]]; then
    MODELS="rnd,btm,ctm,bert"
fi
IFS=',' read -ra MODEL_LIST <<< "$MODELS"

discover_targets
resolve_output_dir "$TYPE"

# Tee all subsequent output to logs.log in the output directory
LOG_FILE="$OUTPUT_DIR/logs.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# ─── Count total experiments ─────────────────────────────────────────────────

TOTAL=0
for target in "${TARGET_LIST[@]}"; do
    target_dir="$JOB_DIR/datasets/$target"
    if [[ "$TYPE" == "real" ]]; then
        discover_xmls_real "$target_dir" _count_xmls
    else
        discover_xmls_generated "$target_dir" _count_xmls
    fi
    TOTAL=$(( TOTAL + ${#_count_xmls[@]} * ${#MODEL_LIST[@]} ))
done

FAILED=()

# ─── Banner ──────────────────────────────────────────────────────────────────

echo ""
echo "============================================"
echo "  run_job.sh"
echo "  Type:       $TYPE"
echo "  Job dir:    $JOB_DIR"
echo "  Categories: $CAT_PATH ($NASPECTS aspects)"
echo "  Models:     ${MODEL_LIST[*]}"
echo "  Targets:    ${TARGET_LIST[*]}"
echo "  Parallel:   $PARALLEL"
echo "  Output:     $OUTPUT_DIR"
echo "  Total:      $TOTAL experiments"
echo "  Started:    $(fmt_timestamp $SCRIPT_START)"
echo "============================================"
echo ""

# ─── Main Loop ───────────────────────────────────────────────────────────────

cd /app/src

if [[ "$PARALLEL" == "targets" || "$PARALLEL" == "all" ]]; then
    # Run all targets in parallel
    # Each target writes to its own log file to avoid shared-pipe contention
    target_pids=()
    target_logs=()

    for target in "${TARGET_LIST[@]}"; do
        tlog="$OUTPUT_DIR/.parallel_target_${target}.log"
        target_logs+=("$tlog")
        (
            set +e  # Prevent pipe errors from killing subshell
            process_target "$target"
        ) > "$tlog" 2>&1 &
        target_pids+=($!)
    done

    # Wait for all target processes and replay logs
    for i in "${!target_pids[@]}"; do
        wait "${target_pids[$i]}" || true
        if [[ -f "${target_logs[$i]}" ]]; then
            cat "${target_logs[$i]}"
            rm -f "${target_logs[$i]}"
        fi
    done
else
    # Sequential target execution
    for target in "${TARGET_LIST[@]}"; do
        process_target "$target"
    done
fi

# Collect failures from temp files
for target in "${TARGET_LIST[@]}"; do
    local_fail_file="$OUTPUT_DIR/.failures_target_${target}"
    if [[ -f "$local_fail_file" ]]; then
        while IFS= read -r line; do
            FAILED+=("$line")
        done < "$local_fail_file"
        rm -f "$local_fail_file"
    fi
done

# ─── Summary ─────────────────────────────────────────────────────────────────

SCRIPT_END=$(date +%s)
TOTAL_TIME=$(( SCRIPT_END - SCRIPT_START ))

echo ""
echo "============================================"
echo "  Job complete: $TYPE"
echo "  Started:    $(fmt_timestamp $SCRIPT_START)"
echo "  Finished:   $(fmt_timestamp $SCRIPT_END)"
echo "  Duration:   $(fmt_duration $TOTAL_TIME)"
echo "============================================"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo "  FAILURES (${#FAILED[@]}):"
    for f in "${FAILED[@]}"; do
        echo "    - $f"
    done
fi

echo ""
echo "  Results:"
for target in "${TARGET_LIST[@]}"; do
    agg="$OUTPUT_DIR/target-${target}/aggregate.csv"
    if [[ -f "$agg" ]]; then
        echo "    target=$target: $agg"
    else
        echo "    target=$target: (no aggregate)"
    fi
done

echo ""
echo "  Output root: $OUTPUT_DIR"
echo ""

if [[ ${#FAILED[@]} -gt 0 ]]; then
    exit 1
fi
