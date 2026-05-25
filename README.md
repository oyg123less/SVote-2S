# SVote-2S: Two-Stage Support Voting for Self-Consistency in Multi-hop QA

This repository contains the evaluation code and baselines for **SVote-2S**, a two-stage support-voting framework that improves self-consistency (SC) for multi-hop question answering. All LLM inference is accelerated via [vLLM](https://github.com/vllm-project/vllm).

## Overview

SVote-2S improves upon standard Self-Consistency (SC) by:

1. **Stage-1 (Support Voting):** Aggregates weighted votes over N sampled CoT traces to identify the most relevant evidence passages (K ∈ [2, 4]).
2. **Stage-2 (Re-inference):** Re-prompts the LLM with filtered context (only selected evidence) and performs SC on N' new samples.
3. **Fusion:** Combines confidence-weighted votes from both stages via P(True) scoring.

---

## Project Structure

```
SVote-2S/
├── configs/
│   ├── models.yaml            # Model paths, dtypes & dataset definitions
│   └── eval_default.yaml      # All evaluation hyperparameters (centralized)
├── src/
│   ├── eval/                   # Core evaluation infrastructure
│   │   ├── eval_llm_baseline.py       # Prompt building, answer parsing, metrics (EM/F1)
│   │   ├── sample_pc_score_vllm.py    # vLLM-based N-sample SC generation
│   │   └── eval_cisc_ptrue_vllm.py    # CISC P(True) confidence scoring (vLLM)
│   ├── svote/                  # SVote-2S core modules
│   │   ├── voting.py           # Weighted support voting + bridging selection
│   │   ├── eval_svote_vllm.py  # Stage-1 voting + Stage-2 re-inference (vLLM)
│   │   └── fusion_cisc2.py     # Two-stage fusion with P(True) weighting
│   └── baselines/              # Baseline method reproductions
│       ├── rasc/               # RASC — Reasoning-Aware SC (CPU post-processing)
│       │   ├── eval_rasc.py
│       │   ├── features.py
│       │   ├── scoring.py
│       │   └── voting.py
│       └── rvsc/               # RVSC — Ranked Voting SC (vLLM)
│           ├── eval_rvsc_vllm.py
│           ├── prompt.py
│           ├── parser.py
│           └── voting.py
├── scripts/
│   ├── run_full_eval.sh        # One-click full pipeline (all stages)
│   ├── run_stage.sh            # Run a single stage independently
│   ├── aggregate_results.py    # Collect & print results summary table
│   └── shard_dataset.py        # Split dataset into N shards for multi-GPU
├── data/                       # Dataset files (not tracked in git)
│   ├── hotpotqa/               # HotpotQA distractor dev (7,405 questions)
│   ├── musique/                # MuSiQue ans-dev (2,417 questions)
│   └── 2wiki/                  # 2WikiMultihopQA dev (12,576 questions)
├── outputs/                    # Evaluation outputs (auto-generated, see below)
├── requirements.txt
├── LICENSE                     # Apache 2.0
└── README.md
```

---

## Setup

### 1. Environment

```bash
conda create -n svote python=3.10 -y
conda activate svote
pip install -r requirements.txt
```

Key dependencies: `torch>=2.1`, `vllm>=0.4`, `transformers>=4.40`, `pyyaml`.

### 2. Models

Download the following models from HuggingFace and update paths in `configs/models.yaml`:

| Model | HuggingFace Link | Default dtype |
|---|---|---|
| Qwen2.5-7B-Instruct | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | float16 |
| Gemma-2-9B-IT | [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it) | bfloat16 |
| Llama-3.1-8B-Instruct | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | float16 |

### 3. Datasets

Download and place in `data/`:

- **HotpotQA**: [hotpot_dev_distractor_v1.json](https://hotpotqa.github.io/)
- **MuSiQue**: [musique_ans_v1.0_dev.jsonl](https://github.com/StonyBrookNLP/musique)
- **2WikiMultihopQA**: [dev.json](https://github.com/Alab-NII/2wikimultihop)

For multi-GPU parallel evaluation, split into shards:

```bash
python scripts/shard_dataset.py --input data/hotpotqa/dev_full.json --num_shards 4
python scripts/shard_dataset.py --input data/musique/musique_ans_v1.0_dev.jsonl --num_shards 4
python scripts/shard_dataset.py --input data/2wiki/dev_full.json --num_shards 4
```

---

## Configuration

All hyperparameters are managed via two YAML files in `configs/`:

### `configs/models.yaml` — Model & Dataset Definitions

Defines model paths, dtypes, and dataset file locations. Example:

```yaml
models:
  qwen2_5_7b_instruct:
    path: /path/to/Qwen2.5-7B-Instruct
    dtype: float16
    model_family: qwen

datasets:
  hotpotqa_full:
    path: data/hotpotqa/dev_full.json
    file_type: json
    adapter: hotpotqa
```

### `configs/eval_default.yaml` — Evaluation Hyperparameters

Centralizes all experimental settings. No hardcoded values in scripts.

```yaml
sampling:                          # SC Sampling (Stage-0)
  n_samples: 20
  temperature: 0.7
  top_p: 0.95
  seed: 42
  max_model_len: 8192

max_new_tokens:                    # Per-model generation length
  qwen2_5_7b_instruct: 512
  gemma2_9b_it: 512
  llama3_1_8b_instruct: 768

svote_stage1:                      # SVote Stage-1 (Support Voting)
  weight_mode: combined            # uniform | cisc | cluster | combined
  k_min: 2
  k_max: 4
  context_mode: paragraph

svote_stage2:                      # SVote Stage-2 (Re-inference)
  n_resample: 10

fusion:                            # SVote-2S Fusion Grid Search
  T_values: [1.0, 2.0, 5.0, 8.0, 15.0, 20.0]
  alpha_values: [1.0, 1.5, 2.0, 2.5, 3.0]

cisc:                              # CISC P(True)
  T_conf: 2.0

rvsc:                              # Ranked Voting SC
  top_k: 3
```

To run ablation experiments, copy the file and override:

```bash
cp configs/eval_default.yaml configs/eval_ablation_k1_6.yaml
# edit k_min=1, k_max=6
EVAL_CONFIG=configs/eval_ablation_k1_6.yaml bash scripts/run_stage.sh svote qwen2_5_7b_instruct hotpotqa 4
```

---

## Usage

### Option A: Full Pipeline (all stages, one command)

```bash
bash scripts/run_full_eval.sh <MODEL> <DATASET> [NUM_GPUS]
```

Example:

```bash
bash scripts/run_full_eval.sh qwen2_5_7b_instruct hotpotqa 4
```

This sequentially runs: SC → CISC → RASC → SVote → CISC-S2 → Fusion → CoT → RVSC.

### Option B: Single Stage

```bash
bash scripts/run_stage.sh <STAGE> <MODEL> <DATASET> [NUM_GPUS]
```

Available stages and their dependencies:

| Stage | Command | Requires | Description |
|---|---|---|---|
| `sc` | `bash scripts/run_stage.sh sc ...` | — | SC sampling (N=20, vLLM) |
| `cisc` | `bash scripts/run_stage.sh cisc ...` | `sc` | CISC P(True) confidence scoring |
| `rasc` | `bash scripts/run_stage.sh rasc ...` | `sc` | RASC post-processing (CPU only) |
| `svote` | `bash scripts/run_stage.sh svote ...` | `sc` + `cisc` | SVote Stage-1 voting + Stage-2 re-inference |
| `cisc_s2` | `bash scripts/run_stage.sh cisc_s2 ...` | `svote` | CISC on Stage-2 samples (for fusion) |
| `fusion` | `bash scripts/run_stage.sh fusion ...` | `sc` + `cisc` + `svote` + `cisc_s2` | Fusion grid search over (T, α) |
| `cot` | `bash scripts/run_stage.sh cot ...` | — | CoT (temperature=0, deterministic) |
| `rvsc` | `bash scripts/run_stage.sh rvsc ...` | — | Ranked Voting SC |

The script checks for required upstream outputs and exits with an informative error if missing.

Examples:

```bash
# Run only SC sampling for Qwen on HotpotQA (4 GPUs)
bash scripts/run_stage.sh sc qwen2_5_7b_instruct hotpotqa 4

# Then run CISC
bash scripts/run_stage.sh cisc qwen2_5_7b_instruct hotpotqa 4

# Run CoT for Gemma on MuSiQue (independent, no prerequisites)
bash scripts/run_stage.sh cot gemma2_9b_it musique 4

# Run RVSC for Llama on 2Wiki
bash scripts/run_stage.sh rvsc llama3_1_8b_instruct 2wiki 4
```

### Option C: Run All Models × All Datasets

```bash
for MODEL in qwen2_5_7b_instruct gemma2_9b_it llama3_1_8b_instruct; do
    for DATASET in hotpotqa musique 2wiki; do
        bash scripts/run_full_eval.sh $MODEL $DATASET 4
    done
done
```

---

## Pipeline Stages

```
                ┌──────────┐
                │  sc (P0) │  N=20 stochastic CoT samples
                └────┬─────┘
            ┌────────┼────────┐
            ▼        ▼        ▼
      ┌──────────┐ ┌──────┐ ┌──────────┐
      │ cisc(P1) │ │rasc  │ │(independent)
      └────┬─────┘ │(P2)  │ │
           │       └──────┘ │
           ▼                │
      ┌──────────┐          │
      │svote(P3) │          │
      └────┬─────┘          │
           ▼                │
      ┌──────────┐          │
      │cisc_s2   │          │
      │  (P4)    │          │
      └────┬─────┘          │
           ▼                │
      ┌──────────┐   ┌──────────┐  ┌──────────┐
      │fusion(P5)│   │ cot (P6) │  │rvsc (P7) │
      └──────────┘   └──────────┘  └──────────┘
```

---

## Output Structure

All outputs are organized under `outputs/<MODEL>/<DATASET>/`:

```
outputs/qwen2_5_7b_instruct/hotpotqa/
├── stage0_samples/              # P0: SC sampling
│   ├── shard0/.../samples.jsonl
│   ├── shard1/.../samples.jsonl
│   ├── ...
│   └── merged/samples.jsonl     # Merged across all shards
├── cisc_stage0/                 # P1: CISC on SC samples
│   ├── shard0/per_question_cisc.jsonl
│   └── merged_per_question_cisc.jsonl
├── rasc/                        # P2: RASC
│   └── metrics.json
├── svote_stage12/               # P3: SVote Stage-1 + Stage-2
│   ├── shard0/per_question.jsonl
│   └── merged/per_question.jsonl
├── cisc_stage2/                 # P4: CISC on Stage-2
│   └── merged_per_question_cisc.jsonl
├── fusion/                      # P5: Fusion grid search
│   ├── T2.0_alpha2.5.json       # Each (T, α) combination
│   ├── ...
│   ├── best.json                # Best combination result
│   └── best_params.json         # {"best_T": ..., "best_alpha": ..., "best_em": ...}
├── cot/                         # P6: CoT (deterministic)
│   └── shard0/.../samples.jsonl
├── rvsc/                        # P7: Ranked Voting SC
│   └── shard0/.../samples.jsonl
└── logs/                        # Execution logs for all stages
    ├── p0_sc_sampling/shard0.log
    ├── p1_cisc/shard0.log
    ├── p2_rasc.log
    ├── p3_svote/shard0.log
    ├── p6_cot/shard0.log
    ├── ...
```

### View Aggregated Results

```bash
# All models × all datasets
python scripts/aggregate_results.py

# Single model × dataset
python scripts/aggregate_results.py --model qwen2_5_7b_instruct --dataset hotpotqa
```

Output example:

```
======================================================================
 qwen2_5_7b_instruct / hotpotqa
======================================================================
Method                       EM     AF1     JF1     ΔSC
-------------------------------------------------------
CoT                        50.12   56.34   42.11       -
SC                         52.50   58.21   44.30   +0.00
CISC (T=2.0)               53.10   59.02   45.01   +0.60
RASC                       52.80   58.50   44.60   +0.30
RVSC (BCV)                 53.40   59.30   45.20   +0.90
SVote (input-vote)         54.20   60.10   46.50   +1.70
SVote-2S                   55.00   61.00   47.80   +2.50
Oracle (BoN)               72.30       -       -  +19.80
```

---

## Key Hyperparameters

All values are defined in `configs/eval_default.yaml`:

| Parameter | Value | Config Key |
|---|---|---|
| SC samples (N) | 20 | `sampling.n_samples` |
| Temperature | 0.7 | `sampling.temperature` |
| top_p | 0.95 | `sampling.top_p` |
| max_new_tokens (Qwen/Gemma) | 512 | `max_new_tokens.qwen2_5_7b_instruct` |
| max_new_tokens (Llama) | 768 | `max_new_tokens.llama3_1_8b_instruct` |
| max_model_len | 8192 | `sampling.max_model_len` |
| SVote K range | [2, 4] | `svote_stage1.k_min` / `k_max` |
| SVote weight mode | combined | `svote_stage1.weight_mode` |
| SVote Stage-2 N' | 10 | `svote_stage2.n_resample` |
| CISC T_conf | 2.0 | `cisc.T_conf` |
| RVSC top_k | 3 | `rvsc.top_k` |
| Fusion T search | [1, 2, 5, 8, 15, 20] | `fusion.T_values` |
| Fusion α search | [1.0, 1.5, 2.0, 2.5, 3.0] | `fusion.alpha_values` |

---

## Methods

| Method | Type | Description |
|---|---|---|
| **CoT** | Baseline | Deterministic chain-of-thought decoding (temperature=0) |
| **SC** | Baseline | Majority vote over N=20 stochastic CoT samples |
| **CISC** | Baseline | P(True) confidence-weighted SC (Portillo Wightman et al., 2023) |
| **RASC** | Baseline | Reasoning-Aware SC with logistic regression (Wan et al., NAACL 2025) |
| **RVSC** | Baseline | Ranked-answer voting via IRV/BCV/MRRV (Wang et al., ACL 2025) |
| **SVote (input-vote)** | Ours | Stage-2 SC on evidence-filtered context |
| **SVote-2S** | Ours | Two-stage fusion combining Stage-1 and Stage-2 with P(True) weighting |
| **Oracle (BoN)** | Upper bound | Best-of-N answer matched against gold |

## Metrics

- **EM**: Exact Match (after answer normalization)
- **A-F1**: Answer F1 (token-level overlap with gold answer)
- **J-F1**: Joint F1 = Answer-F1 × Support-F1
- **ΔSC**: EM gain relative to standard Self-Consistency

---

## Citation

```bibtex
@article{svote2s2025,
  title={SVote-2S: Two-Stage Support Voting for Self-Consistency in Multi-hop Question Answering},
  author={...},
  year={2025}
}
```

## License

Apache 2.0
