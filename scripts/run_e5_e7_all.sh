#!/usr/bin/env bash
# ============================================================================
# Run E5 (Random Filtering) and E7 (Gold-Only Oracle) for all model×dataset
# ============================================================================
# Usage:
#   bash scripts/run_e5_e7_all.sh [NUM_GPUS]
#
# This runs 18 jobs sequentially (3 models × 3 datasets × 2 ablation modes).
# Each job uses NUM_GPUS GPUs in parallel sharding.
#
# To run a single configuration:
#   bash scripts/run_ablation.sh random qwen2_5_7b_instruct hotpotqa 4
#   bash scripts/run_ablation.sh gold   qwen2_5_7b_instruct hotpotqa 4
# ============================================================================
set -euo pipefail

NUM_GPUS="${1:-4}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODELS=(qwen2_5_7b_instruct gemma2_9b_it llama3_1_8b_instruct)
DATASETS=(hotpotqa musique 2wiki)

echo "=================================================================="
echo " SVote-2S: E5 + E7 Ablation (all configurations)"
echo " GPUs per job: $NUM_GPUS"
echo " Total jobs: ${#MODELS[@]} × ${#DATASETS[@]} × 2 = $((${#MODELS[@]} * ${#DATASETS[@]} * 2))"
echo "=================================================================="
echo ""

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        echo ""
        echo "============================================================"
        echo " [E5] Random: $MODEL / $DATASET"
        echo "============================================================"
        bash "$SCRIPT_DIR/run_ablation.sh" random "$MODEL" "$DATASET" "$NUM_GPUS"

        echo ""
        echo "============================================================"
        echo " [E7] Gold:   $MODEL / $DATASET"
        echo "============================================================"
        bash "$SCRIPT_DIR/run_ablation.sh" gold "$MODEL" "$DATASET" "$NUM_GPUS"
    done
done

echo ""
echo "=================================================================="
echo " All E5/E7 ablations complete!"
echo "=================================================================="

# Collect results summary
"${PYTHON:-python}" -c "
import json, os, glob

PROJECT = '$SCRIPT_DIR/..'
models = ['qwen2_5_7b_instruct', 'gemma2_9b_it', 'llama3_1_8b_instruct']
datasets = ['hotpotqa', 'musique', '2wiki']

print()
print('=' * 80)
print(' E5/E7 Results Summary')
print('=' * 80)
print(f'{\"Model\":<25} {\"Dataset\":<15} {\"Random EM\":>10} {\"Gold EM\":>10} {\"SVote EM\":>10}')
print('-' * 80)

for m in models:
    for d in datasets:
        rand_p = f'{PROJECT}/outputs/{m}/{d}/ablation_random_k2/merged/metrics.json'
        gold_p = f'{PROJECT}/outputs/{m}/{d}/ablation_gold/merged/metrics.json'
        svote_p = f'{PROJECT}/outputs/{m}/{d}/svote_stage12/merged/per_question.jsonl'

        rand_em = '---'
        if os.path.exists(rand_p):
            rand_em = f'{json.load(open(rand_p))[\"em\"]*100:.2f}'
        gold_em = '---'
        if os.path.exists(gold_p):
            gold_em = f'{json.load(open(gold_p))[\"em\"]*100:.2f}'
        # SVote EM from existing per_question
        svote_em = '---'
        if os.path.exists(svote_p):
            from collections import Counter
            correct = 0; total = 0
            with open(svote_p) as f:
                for line in f:
                    if not line.strip(): continue
                    r = json.loads(line)
                    chosen = r.get('chosen', {})
                    if chosen.get('em', 0) == 1: correct += 1
                    total += 1
            if total: svote_em = f'{correct/total*100:.2f}'
        print(f'{m:<25} {d:<15} {rand_em:>10} {gold_em:>10} {svote_em:>10}')
print()
print('Expected: Gold >> SVote >> Random >> SC')
"
