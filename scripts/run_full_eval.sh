#!/usr/bin/env bash
# ============================================================================
# SVote-2S Full Evaluation Pipeline
# ============================================================================
# Usage:
#   bash scripts/run_full_eval.sh <MODEL_KEY> <DATASET_KEY> [NUM_GPUS]
#
# Example:
#   bash scripts/run_full_eval.sh qwen2_5_7b_instruct hotpotqa 4
#   bash scripts/run_full_eval.sh gemma2_9b_it musique 4
#   bash scripts/run_full_eval.sh llama3_1_8b_instruct 2wiki_dev12576 4
#
# Pipeline stages (all use vLLM):
#   P0. SC Sampling   — N=20 stochastic CoT samples per question
#   P1. CISC P(True)  — confidence scoring on SC samples
#   P2. RASC          — reasoning-aware reweighting (CPU only, no LLM)
#   P3. SVote Stage-1 — weighted voting → compressed evidence K∈[2,4]
#       SVote Stage-2 — re-inference with filtered context, N'=10
#   P4. CISC on SVote Stage-2 — confidence scoring for fusion
#   P5. SVote-2S Fusion — grid search over (T, α) hyperparameters
#   P6. CoT            — deterministic baseline (temperature=0)
#   P7. RVSC           — ranked voting SC baseline
#
# Outputs go to: outputs/<MODEL>/<DATASET>/<stage>/
# ============================================================================
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/GPUdata/student/asc002/ouyangguo/anaconda/envs/vetagent/bin/python}"
CONFIG="$PROJECT/configs/models.yaml"
EVAL_CONFIG="${EVAL_CONFIG:-$PROJECT/configs/eval_default.yaml}"

MODEL="${1:?Usage: $0 <MODEL_KEY> <DATASET_KEY> [NUM_GPUS]}"
DATASET="${2:?Usage: $0 <MODEL_KEY> <DATASET_KEY> [NUM_GPUS]}"
NUM_GPUS="${3:-4}"

# ---------------------------------------------------------------------------
# Read eval hyperparameters from eval_default.yaml
# ---------------------------------------------------------------------------
read_yaml() {
    $PYTHON -c "
import yaml, sys
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
MAX_NEW_TOKENS=$(read_yaml "max_new_tokens.$MODEL")

# Sampling
N_SAMPLES=$(read_yaml "sampling.n_samples")
TEMPERATURE=$(read_yaml "sampling.temperature")
TOP_P=$(read_yaml "sampling.top_p")
MAX_MODEL_LEN=$(read_yaml "sampling.max_model_len")
SEED=$(read_yaml "sampling.seed")

# SVote
K_MIN=$(read_yaml "svote_stage1.k_min")
K_MAX=$(read_yaml "svote_stage1.k_max")
N_RESAMPLE=$(read_yaml "svote_stage2.n_resample")
WEIGHT_MODE=$(read_yaml "svote_stage1.weight_mode")
CONTEXT_MODE=$(read_yaml "svote_stage1.context_mode")

# RVSC
RVSC_TOP_K=$(read_yaml "rvsc.top_k")

# Output root
OUT_ROOT="$PROJECT/outputs/$MODEL/$DATASET"
LOG_ROOT="$OUT_ROOT/logs"
mkdir -p "$LOG_ROOT"

echo "=================================================================="
echo " SVote-2S Pipeline: model=$MODEL dataset=$DATASET gpus=$NUM_GPUS"
echo " max_new_tokens=$MAX_NEW_TOKENS dtype=$DTYPE"
echo " output=$OUT_ROOT"
echo "=================================================================="

# ====================================================================
# P0. SC Sampling (N=20, 4-GPU parallel shards)
# ====================================================================
echo "[$(date)] P0: SC Sampling (N=$N_SAMPLES) ..."
P0_DIR="$OUT_ROOT/stage0_samples"
P0_LOG="$LOG_ROOT/p0_sc_sampling"
mkdir -p "$P0_DIR" "$P0_LOG"

PIDS=()
for GPU in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$PROJECT/src/eval/sample_pc_score_vllm.py" \
        --dataset "${DATASET}_shard${GPU}" \
        --model "$MODEL" \
        --setting fullctx \
        --run_id "shard${GPU}" \
        --config "$CONFIG" \
        --output_root "$P0_DIR" \
        --n_samples $N_SAMPLES \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --max_new_tokens $MAX_NEW_TOKENS \
        --max_model_len $MAX_MODEL_LEN \
        --dtype "$DTYPE" \
        --seed $SEED \
        --overwrite \
        > "$P0_LOG/shard${GPU}.log" 2>&1 &
    PIDS+=($!)
done
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || { echo "[ERROR] P0 shard $i FAILED"; exit 1; }
done
echo "[$(date)] P0: SC Sampling complete."

# Merge shards
MERGED="$P0_DIR/merged"
mkdir -p "$MERGED"
$PYTHON -c "
import glob, json
files = sorted(glob.glob('$P0_DIR/shard*/${DATASET}_shard*/${MODEL}/fullctx/samples.jsonl'))
rows = []
for p in files:
    with open(p) as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
rows.sort(key=lambda r: r['id'])
with open('$MERGED/samples.jsonl', 'w') as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'Merged {len(rows)} questions from {len(files)} shards')
"

# ====================================================================
# P1. CISC P(True) on SC Samples
# ====================================================================
echo "[$(date)] P1: CISC P(True) scoring ..."
P1_DIR="$OUT_ROOT/cisc_stage0"
P1_LOG="$LOG_ROOT/p1_cisc"
mkdir -p "$P1_DIR" "$P1_LOG"

PIDS=()
for GPU in $(seq 0 $((NUM_GPUS-1))); do
    SHARD_SAMPLES="$P0_DIR/shard${GPU}/${DATASET}_shard${GPU}/${MODEL}/fullctx/samples.jsonl"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$PROJECT/src/eval/eval_cisc_ptrue_vllm.py" \
        --samples_jsonl "$SHARD_SAMPLES" \
        --dataset "${DATASET}_shard${GPU}" \
        --model "$MODEL" \
        --config "$CONFIG" \
        --output_dir "$P1_DIR/shard${GPU}" \
        --max_model_len $MAX_MODEL_LEN \
        --dtype "$DTYPE" \
        --overwrite \
        > "$P1_LOG/shard${GPU}.log" 2>&1 &
    PIDS+=($!)
done
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || { echo "[ERROR] P1 shard $i FAILED"; exit 1; }
done
echo "[$(date)] P1: CISC complete."

# Merge CISC
$PYTHON -c "
import glob, json
merged_perq = []
for p in sorted(glob.glob('$P1_DIR/shard*/per_question_cisc.jsonl')):
    with open(p) as f:
        for line in f:
            if line.strip(): merged_perq.append(json.loads(line))
merged_perq.sort(key=lambda r: r['id'])
with open('$P1_DIR/merged_per_question_cisc.jsonl', 'w') as f:
    for r in merged_perq: f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'Merged CISC: {len(merged_perq)} questions')
"

# ====================================================================
# P2. RASC (CPU-only, no LLM)
# ====================================================================
echo "[$(date)] P2: RASC ..."
P2_DIR="$OUT_ROOT/rasc"
$PYTHON "$PROJECT/src/baselines/rasc/eval_rasc.py" \
    --samples_jsonl "$MERGED/samples.jsonl" \
    --output_dir "$P2_DIR" \
    --overwrite \
    > "$LOG_ROOT/p2_rasc.log" 2>&1
echo "[$(date)] P2: RASC complete."

# ====================================================================
# P3. SVote Stage-1 + Stage-2 (vLLM)
# ====================================================================
echo "[$(date)] P3: SVote Stage-1 → Stage-2 ..."
P3_DIR="$OUT_ROOT/svote_stage12"
P3_LOG="$LOG_ROOT/p3_svote"
mkdir -p "$P3_DIR" "$P3_LOG"

PIDS=()
for GPU in $(seq 0 $((NUM_GPUS-1))); do
    SHARD_SAMPLES="$P0_DIR/shard${GPU}/${DATASET}_shard${GPU}/${MODEL}/fullctx/samples.jsonl"
    SHARD_CISC="$P1_DIR/shard${GPU}/per_question_cisc.jsonl"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$PROJECT/src/svote/eval_svote_vllm.py" \
        --samples_jsonl "$SHARD_SAMPLES" \
        --cisc_perq "$SHARD_CISC" \
        --dataset "${DATASET}_shard${GPU}" \
        --model "$MODEL" \
        --config "$CONFIG" \
        --output_dir "$P3_DIR/shard${GPU}" \
        --weight_mode $WEIGHT_MODE \
        --k_min $K_MIN --k_max $K_MAX \
        --context_mode $CONTEXT_MODE \
        --n_resample $N_RESAMPLE \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --max_new_tokens $MAX_NEW_TOKENS \
        --max_model_len $MAX_MODEL_LEN \
        --dtype "$DTYPE" \
        --seed $SEED \
        --overwrite \
        > "$P3_LOG/shard${GPU}.log" 2>&1 &
    PIDS+=($!)
done
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || { echo "[ERROR] P3 shard $i FAILED"; exit 1; }
done
echo "[$(date)] P3: SVote Stage-1/2 complete."

# Merge SVote stage-2
$PYTHON -c "
import glob, json
merged = []
for p in sorted(glob.glob('$P3_DIR/shard*/per_question.jsonl')):
    with open(p) as f:
        for line in f:
            if line.strip(): merged.append(json.loads(line))
merged.sort(key=lambda r: r['id'])
out_dir = '$P3_DIR/merged'
import os; os.makedirs(out_dir, exist_ok=True)
with open(f'{out_dir}/per_question.jsonl', 'w') as f:
    for r in merged: f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'Merged SVote stage-2: {len(merged)} questions')
"

# ====================================================================
# P4. CISC on SVote Stage-2 samples (for fusion)
# ====================================================================
echo "[$(date)] P4: CISC on SVote Stage-2 ..."
P4_DIR="$OUT_ROOT/cisc_stage2"
P4_LOG="$LOG_ROOT/p4_cisc_s2"
mkdir -p "$P4_DIR" "$P4_LOG"

# Build stage-2 samples.jsonl from per_question.jsonl for each shard
PIDS=()
for GPU in $(seq 0 $((NUM_GPUS-1))); do
    # Convert per_question.jsonl -> samples.jsonl format for CISC
    S2_SAMPLES="$P4_DIR/shard${GPU}_samples.jsonl"
    $PYTHON -c "
import json
with open('$P3_DIR/shard${GPU}/per_question.jsonl') as f:
    rows = [json.loads(l) for l in f if l.strip()]
with open('$S2_SAMPLES', 'w') as f:
    for r in rows:
        out = {'id': r['id'], 'gold_answer': r['gold_answer'],
               'samples': r.get('samples', []), 'question': ''}
        f.write(json.dumps(out, ensure_ascii=False) + '\n')
print(f'Prepared {len(rows)} stage-2 questions for CISC shard $GPU')
"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$PROJECT/src/eval/eval_cisc_ptrue_vllm.py" \
        --samples_jsonl "$S2_SAMPLES" \
        --dataset "${DATASET}_shard${GPU}" \
        --model "$MODEL" \
        --config "$CONFIG" \
        --output_dir "$P4_DIR/shard${GPU}" \
        --max_model_len $MAX_MODEL_LEN \
        --dtype "$DTYPE" \
        --overwrite \
        > "$P4_LOG/shard${GPU}.log" 2>&1 &
    PIDS+=($!)
done
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || { echo "[ERROR] P4 shard $i FAILED"; exit 1; }
done
echo "[$(date)] P4: CISC Stage-2 complete."

# Merge CISC stage-2
$PYTHON -c "
import glob, json
merged = []
for p in sorted(glob.glob('$P4_DIR/shard*/per_question_cisc.jsonl')):
    with open(p) as f:
        for line in f:
            if line.strip(): merged.append(json.loads(line))
merged.sort(key=lambda r: r['id'])
with open('$P4_DIR/merged_per_question_cisc.jsonl', 'w') as f:
    for r in merged: f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'Merged CISC stage-2: {len(merged)} questions')
"

# ====================================================================
# P5. SVote-2S Fusion Grid Search
# ====================================================================
echo "[$(date)] P5: Fusion grid search ..."
P5_DIR="$OUT_ROOT/fusion"
mkdir -p "$P5_DIR"

T_VALUES=$(read_yaml "fusion.T_values" | tr -d '[],' | tr ',' ' ')
ALPHA_VALUES=$(read_yaml "fusion.alpha_values" | tr -d '[],' | tr ',' ' ')
BEST_EM=0; BEST_T=""; BEST_ALPHA=""

for T in $T_VALUES; do
    for ALPHA in $ALPHA_VALUES; do
        OUT="$P5_DIR/T${T}_alpha${ALPHA}.json"
        $PYTHON "$PROJECT/src/svote/fusion_cisc2.py" \
            --stage1_samples "$MERGED/samples.jsonl" \
            --stage2_perq "$P3_DIR/merged/per_question.jsonl" \
            --stage1_cisc "$P1_DIR/merged_per_question_cisc.jsonl" \
            --stage2_cisc "$P4_DIR/merged_per_question_cisc.jsonl" \
            --T_s1 "$T" --T_s2 "$T" --alpha_s2 "$ALPHA" \
            --output "$OUT" > /dev/null 2>&1

        EM=$($PYTHON -c "import json; d=json.load(open('$OUT')); print(d['em'])")
        IS_BETTER=$($PYTHON -c "print(1 if $EM > $BEST_EM else 0)")
        if [ "$IS_BETTER" = "1" ]; then
            BEST_EM="$EM"; BEST_T="$T"; BEST_ALPHA="$ALPHA"
        fi
    done
done

echo "[$(date)] P5: Best fusion: T=$BEST_T alpha=$BEST_ALPHA EM=$BEST_EM"
cp "$P5_DIR/T${BEST_T}_alpha${BEST_ALPHA}.json" "$P5_DIR/best.json"
echo "{\"best_T\": $BEST_T, \"best_alpha\": $BEST_ALPHA, \"best_em\": $BEST_EM}" > "$P5_DIR/best_params.json"

# ====================================================================
# P6. CoT (temperature=0, deterministic single decode)
# ====================================================================
echo "[$(date)] P6: CoT ..."
P6_DIR="$OUT_ROOT/cot"
P6_LOG="$LOG_ROOT/p6_cot"
mkdir -p "$P6_DIR" "$P6_LOG"

PIDS=()
for GPU in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$PROJECT/src/eval/sample_pc_score_vllm.py" \
        --dataset "${DATASET}_shard${GPU}" \
        --model "$MODEL" \
        --setting fullctx \
        --run_id "shard${GPU}" \
        --config "$CONFIG" \
        --output_root "$P6_DIR" \
        --n_samples 1 \
        --temperature 0.0 \
        --top_p 1.0 \
        --max_new_tokens $MAX_NEW_TOKENS \
        --max_model_len $MAX_MODEL_LEN \
        --dtype "$DTYPE" \
        --seed $SEED \
        --overwrite \
        > "$P6_LOG/shard${GPU}.log" 2>&1 &
    PIDS+=($!)
done
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || { echo "[ERROR] P6 shard $i FAILED"; exit 1; }
done
echo "[$(date)] P6: CoT complete."

# ====================================================================
# P7. Ranked Voting SC (RVSC)
# ====================================================================
echo "[$(date)] P7: RVSC ..."
P7_DIR="$OUT_ROOT/rvsc"
P7_LOG="$LOG_ROOT/p7_rvsc"
mkdir -p "$P7_DIR" "$P7_LOG"

PIDS=()
for GPU in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$PROJECT/src/baselines/rvsc/eval_rvsc_vllm.py" \
        --dataset "${DATASET}_shard${GPU}" \
        --model "$MODEL" \
        --setting fullctx \
        --run_id "shard${GPU}" \
        --config "$CONFIG" \
        --output_root "$P7_DIR" \
        --n_samples $N_SAMPLES \
        --top_k $RVSC_TOP_K \
        --max_new_tokens $MAX_NEW_TOKENS \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --max_model_len $MAX_MODEL_LEN \
        --dtype "$DTYPE" \
        --seed $SEED \
        --overwrite \
        > "$P7_LOG/shard${GPU}.log" 2>&1 &
    PIDS+=($!)
done
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || { echo "[ERROR] P7 shard $i FAILED"; exit 1; }
done
echo "[$(date)] P7: RVSC complete."

# ====================================================================
# Summary
# ====================================================================
echo "=================================================================="
echo " Pipeline complete for $MODEL / $DATASET"
echo " Results: $OUT_ROOT/"
echo "=================================================================="
echo ""
echo " Stage 0 (SC samples):   $P0_DIR/merged/samples.jsonl"
echo " CISC stage-0:           $P1_DIR/"
echo " RASC:                   $P2_DIR/"
echo " SVote stage-1/2:        $P3_DIR/"
echo " CISC stage-2:           $P4_DIR/"
echo " SVote-2S fusion best:   $P5_DIR/best.json"
echo " CoT:                    $P6_DIR/"
echo " RVSC:                   $P7_DIR/"
echo "=================================================================="
