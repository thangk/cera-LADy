#!/bin/bash
# Run all dataset × model combinations for cera-LADy benchmarking
# Usage: ./run_all.sh
# Run from inside the Docker container: docker exec -it cera-lady-cli bash

set -e

export TOKENIZERS_PARALLELISM=false

MODELS=("rnd" "btm" "ctm" "bert")
NASPECTS=5
CATEGORIES="../datasets/categories/laptop.csv"

declare -A DATASETS
DATASETS["cera"]="../datasets/cera/reviews-run1-explicit.xml"
DATASETS["heuristic"]="../datasets/heuristic/reviews-explicit-run1.xml"
DATASETS["real-baselines"]="../datasets/real-baselines/real-laptops-1500-sents.xml"

TOTAL=$(( ${#DATASETS[@]} * ${#MODELS[@]} ))
CURRENT=0
FAILED=()

echo "============================================"
echo "  cera-LADy Benchmark: $TOTAL experiments"
echo "  Datasets: ${!DATASETS[*]}"
echo "  Models:   ${MODELS[*]}"
echo "  Aspects:  $NASPECTS"
echo "============================================"
echo ""

cd /app/src

# Create output directories upfront
for DATASET_NAME in "${!DATASETS[@]}"; do
    mkdir -p "../output/${DATASET_NAME}"
done

for DATASET_NAME in "${!DATASETS[@]}"; do
    DATA_PATH="${DATASETS[$DATASET_NAME]}"
    OUTPUT_DIR="../output/${DATASET_NAME}"

    for MODEL in "${MODELS[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "--------------------------------------------"
        echo "  [$CURRENT/$TOTAL] $DATASET_NAME × $MODEL"
        echo "  Data:   $DATA_PATH"
        echo "  Output: $OUTPUT_DIR"
        echo "--------------------------------------------"

        START_TIME=$(date +%s)

        if python -u main.py \
            -am "$MODEL" \
            -data "$DATA_PATH" \
            -output "$OUTPUT_DIR" \
            -naspects "$NASPECTS" \
            -categories "$CATEGORIES" \
            2>&1 | tee "${OUTPUT_DIR}/${MODEL}_log.txt"; then

            END_TIME=$(date +%s)
            ELAPSED=$(( END_TIME - START_TIME ))
            echo ""
            echo "  DONE: $DATASET_NAME × $MODEL (${ELAPSED}s)"
        else
            END_TIME=$(date +%s)
            ELAPSED=$(( END_TIME - START_TIME ))
            FAILED+=("$DATASET_NAME × $MODEL")
            echo ""
            echo "  FAILED: $DATASET_NAME × $MODEL (${ELAPSED}s)"
        fi
    done
done

echo ""
echo "============================================"
echo "  All experiments complete"
echo "============================================"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "  FAILURES (${#FAILED[@]}):"
    for F in "${FAILED[@]}"; do
        echo "    - $F"
    done
    echo ""
    exit 1
else
    echo "  All $TOTAL experiments succeeded."
    echo ""
    echo "  Results in:"
    for DATASET_NAME in "${!DATASETS[@]}"; do
        echo "    output/${DATASET_NAME}/5.na/*/agg.ad.pred.eval.mean.csv"
    done
    echo ""
fi
