#!/usr/bin/env bash
# ============================================================================
# SVote-2S Ablation Runner — E5 Random Filtering / E7 Gold-Only Oracle
# ============================================================================
# Usage:
#   bash scripts/run_ablation.sh <MODE> <MODEL> <DATASET> [NUM_GPUS]
#
# MODE:
#   random   — E5: Randomly select K paragraphs, then re-inference
#   gold     — E7: Use gold support, then re-inference (oracle upper bound)
#
# Examples:
#   bash scripts/run_ablation.sh random qwen2_5_7b_instruct hotpotqa 4
#   bash scripts/run_ablation.sh gold   gemma2_9b_it        musique  4
#   bash scripts/run_ablation.sh random llama3_1_8b_instruct 2wiki 4
#
# Prerequisites:
#   - Stage 0 (SC sampling) must have been run (samples.jsonl exists)
#   - For MODE=random with K_PARA matching SVote, Stage 1 CISC is NOT needed
#   - For MODE=gold, no voting data is needed
# ============================================================================
set -euo pipefail

MODE="${1:?Usage: $0 <MODE=random|gold> <MODEL> <DATASET> [NUM_GPUS]}"
MODEL="${2:?Usage: $0 <MODE> <MODEL> <DATASET> [NUM_GPUS]}"
DATASET="${3:?Usage: $0 <MODE> <MODEL> <DATASET> [NUM_GPUS]}"
NUM_GPUS="${4:-4}"

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
CONFIG="$PROJECT/configs/models.yaml"
EVAL_CONFIG="${EVAL_CONFIG:-$PROJECT/configs/eval_default.yaml}"

# ---------------------------------------------------------------------------
# Read eval hyperparameters
# ---------------------------------------------------------------------------
yq() {
    $PYTHON -c "
import yaml
with open('$EVAL_CONFIG') as f: cfg = yaml.safe_load(f)
keys = '$1'.split('.')
v = cfg
for k in keys: v = v[k]
print(v)
"
}

# Model-specific
declare -A DTYPE_MAP=( [qwen2_5_7b_instruct]=float16 [gemma2_9b_it]=bfloat16 [llama3_1_8b_instruct]=float16 )
DTYPE="${DTYPE_MAP[$MODEL]:-float16}"
MAX_NEW_TOKENS=$(yq "max_new_tokens.$MODEL")

# Sampling
TEMPERATURE=$(yq "sampling.temperature")
TOP_P=$(yq "sampling.top_p")
MAX_MODEL_LEN=$(yq "sampling.max_model_len")
SEED=$(yq "sampling.seed")

# SVote Stage-2
N_RESAMPLE=$(yq "svote_stage2.n_resample")
CONTEXT_MODE=$(yq "svote_stage1.context_mode")

# Random ablation: match SVote avg K_para ≈ 2
RANDOM_K_PARA="${RANDOM_K_PARA:-2}"
RANDOM_N_TRIALS="${RANDOM_N_TRIALS:-3}"

# Directories
OUT_ROOT="$PROJECT/outputs/$MODEL/$DATASET"
P0_DIR="$OUT_ROOT/stage0_samples"
LOG_ROOT="$OUT_ROOT/logs"
mkdir -p "$LOG_ROOT"

echo "=================================================================="
echo " Ablation: $MODE | model=$MODEL dataset=$DATASET gpus=$NUM_GPUS"
echo " N'=$N_RESAMPLE  context=$CONTEXT_MODE  dtype=$DTYPE"
if [ "$MODE" = "random" ]; then
    echo " random_k_para=$RANDOM_K_PARA  n_trials=$RANDOM_N_TRIALS"
fi
echo "=================================================================="

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
wait_all() {
    local pids=("$@")
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "[ERROR] GPU $i FAILED (exit $?)"; exit 1; }
    done
}

merge_jsonl() {
    local pattern="$1" outfile="$2" sortkey="${3:-id}"
    $PYTHON -c "
import glob, json
rows = []
for p in sorted(glob.glob('$pattern')):
    with open(p) as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
rows.sort(key=lambda r: r.get('$sortkey',''))
with open('$outfile', 'w') as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'Merged {len(rows)} records -> $outfile')
"
}

# ---------------------------------------------------------------------------
# Validate prerequisites
# ---------------------------------------------------------------------------
validate_samples() {
    local gpu=$1
    local shard_samples="$P0_DIR/shard${gpu}/${DATASET}_shard${gpu}/${MODEL}/fullctx/samples.jsonl"
    if [ ! -f "$shard_samples" ]; then
        echo "[ERROR] $shard_samples not found."
        echo "  Run: bash scripts/run_stage.sh sc $MODEL $DATASET $NUM_GPUS"
        exit 1
    fi
    echo "$shard_samples"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
case "$MODE" in
# ====================================================================
random)
    ABL_DIR="$OUT_ROOT/ablation_random_k${RANDOM_K_PARA}"
    LOG_DIR="$LOG_ROOT/ablation_random"; mkdir -p "$LOG_DIR"

    echo "[$(date)] E5: Random Filtering (K_para=$RANDOM_K_PARA, trials=$RANDOM_N_TRIALS) ..."
    mkdir -p "$ABL_DIR"

    PIDS=()
    for GPU in $(seq 0 $((NUM_GPUS-1))); do
        SHARD_SAMPLES=$(validate_samples $GPU)
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$PROJECT/src/svote/eval_svote_ablation_vllm.py" \
            --selection_mode random \
            --samples_jsonl "$SHARD_SAMPLES" \
            --dataset "${DATASET}_shard${GPU}" \
            --model "$MODEL" \
            --config "$CONFIG" \
            --output_dir "$ABL_DIR/shard${GPU}" \
            --context_mode $CONTEXT_MODE \
            --random_k_para $RANDOM_K_PARA \
            --random_n_trials $RANDOM_N_TRIALS \
            --random_seed $SEED \
            --n_resample $N_RESAMPLE \
            --temperature $TEMPERATURE \
            --top_p $TOP_P \
            --max_new_tokens $MAX_NEW_TOKENS \
            --max_model_len $MAX_MODEL_LEN \
            --dtype "$DTYPE" \
            --seed $SEED \
            --overwrite \
            > "$LOG_DIR/shard${GPU}.log" 2>&1 &
        PIDS+=($!)
    done
    wait_all "${PIDS[@]}"

    # Merge
    mkdir -p "$ABL_DIR/merged"
    merge_jsonl "$ABL_DIR/shard*/per_question.jsonl" "$ABL_DIR/merged/per_question.jsonl"

    # Aggregate metrics
    $PYTHON -c "
import json, glob
metrics_files = sorted(glob.glob('$ABL_DIR/shard*/metrics.json'))
total_em = 0; total_n = 0
for p in metrics_files:
    d = json.load(open(p))
    total_em += d['em'] * d['num_evaluated']
    total_n += d['num_evaluated']
print(f'Random Filter (K_para=$RANDOM_K_PARA): EM = {total_em/total_n:.4f} ({total_n} questions)')
with open('$ABL_DIR/merged/metrics.json', 'w') as f:
    json.dump({'method': 'ablation_random', 'random_k_para': $RANDOM_K_PARA,
               'n_trials': $RANDOM_N_TRIALS, 'em': total_em/total_n,
               'num_evaluated': total_n}, f, indent=2)
"
    echo "[$(date)] E5 Random Filtering complete. -> $ABL_DIR/merged/"
    ;;

# ====================================================================
gold)
    ABL_DIR="$OUT_ROOT/ablation_gold"
    LOG_DIR="$LOG_ROOT/ablation_gold"; mkdir -p "$LOG_DIR"

    echo "[$(date)] E7: Gold-Only Oracle ..."
    mkdir -p "$ABL_DIR"

    PIDS=()
    for GPU in $(seq 0 $((NUM_GPUS-1))); do
        SHARD_SAMPLES=$(validate_samples $GPU)
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$PROJECT/src/svote/eval_svote_ablation_vllm.py" \
            --selection_mode gold \
            --samples_jsonl "$SHARD_SAMPLES" \
            --dataset "${DATASET}_shard${GPU}" \
            --model "$MODEL" \
            --config "$CONFIG" \
            --output_dir "$ABL_DIR/shard${GPU}" \
            --context_mode $CONTEXT_MODE \
            --n_resample $N_RESAMPLE \
            --temperature $TEMPERATURE \
            --top_p $TOP_P \
            --max_new_tokens $MAX_NEW_TOKENS \
            --max_model_len $MAX_MODEL_LEN \
            --dtype "$DTYPE" \
            --seed $SEED \
            --overwrite \
            > "$LOG_DIR/shard${GPU}.log" 2>&1 &
        PIDS+=($!)
    done
    wait_all "${PIDS[@]}"

    # Merge
    mkdir -p "$ABL_DIR/merged"
    merge_jsonl "$ABL_DIR/shard*/per_question.jsonl" "$ABL_DIR/merged/per_question.jsonl"

    # Aggregate
    $PYTHON -c "
import json, glob
metrics_files = sorted(glob.glob('$ABL_DIR/shard*/metrics.json'))
total_em = 0; total_af1 = 0; total_n = 0
for p in metrics_files:
    d = json.load(open(p))
    total_em  += d['em']  * d['num_evaluated']
    total_af1 += d['af1'] * d['num_evaluated']
    total_n   += d['num_evaluated']
print(f'Gold Oracle: EM = {total_em/total_n:.4f}  AF1 = {total_af1/total_n:.4f} ({total_n} questions)')
with open('$ABL_DIR/merged/metrics.json', 'w') as f:
    json.dump({'method': 'ablation_gold', 'em': total_em/total_n,
               'af1': total_af1/total_n, 'num_evaluated': total_n}, f, indent=2)
"
    echo "[$(date)] E7 Gold-Only Oracle complete. -> $ABL_DIR/merged/"
    ;;

# ====================================================================
*)
    echo "Unknown mode: $MODE"
    echo "Available: random, gold"
    exit 1
    ;;
esac

echo "=================================================================="
echo " Done: ablation/$MODE | $MODEL / $DATASET"
echo "=================================================================="
