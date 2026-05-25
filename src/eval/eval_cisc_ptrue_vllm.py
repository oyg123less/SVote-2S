#!/usr/bin/env python3
"""CISC (Confidence-Improves Self-Consistency) — P(True) variant on HotpotQA.

Reproduces Taubenfeld et al. 2025 (ACL Findings) for the P(True) confidence:
  - Read N pre-generated samples per question from `samples.jsonl`.
  - For each (question, sample.raw) ask the same model:
        Is the proposed answer:
        (A) True
        (B) False
        The proposed answer is: (
    take logprobs of " A" vs " B" at the next token.
  - confidence c_i = logp_A - logp_B
  - CISC: w_i = softmax(c_i / T); score(A) = Σ w_i, argmax A.
  - Compare to SC (uniform majority).

Output for one shard:
  - per_question_cisc.jsonl: per-question SC/CISC results (multiple T)
  - metrics.json: aggregated EM/F1 for SC, BoN, and CISC@T-list
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import multiprocessing as _mp
try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "eval"))
import eval_llm_baseline as base  # noqa: E402


def _norm(s: str) -> str:
    """Same as eval_llm_baseline.normalize_answer."""
    return base.normalize_answer(s) if s else ""


def build_ptrue_prompt(question: str, sample_raw: str, gold_answer_unused: str = "") -> str:
    """P(True) prompt à la Kadavath et al. 2022, used by CISC.

    We feed (question, the sample's reasoning trace + answer) and ask the model
    to label it True/False. We deliberately omit the original long context to
    keep prompts short — the reasoning typically already cites the evidence.
    """
    # Trim the sample raw to keep prompt short and stable.
    raw = sample_raw.strip()
    if len(raw) > 1500:
        raw = raw[:1500]
    return (
        "You will be given a question and a proposed reasoning trace with a final answer.\n"
        "Your job is to decide whether the final answer is correct.\n\n"
        f"Question: {question}\n\n"
        f"Proposed reasoning and answer:\n{raw}\n\n"
        "Is the proposed final answer correct?\n"
        "(A) True\n"
        "(B) False\n\n"
        "Reply with ONLY the single letter A or B. No other text."
    )


def _find_token_logprob(top_logprobs: dict, target_letters: tuple[str, ...]) -> float | None:
    """Search for a token whose decoded text matches one of target_letters
    (case-insensitive, leading-space tolerant)."""
    best = None
    for tid, lp in top_logprobs.items():
        # vLLM's Logprob has .decoded_token attribute
        tok_text = getattr(lp, "decoded_token", None) or ""
        tok_clean = tok_text.strip().upper()
        if tok_clean in target_letters:
            v = lp.logprob
            if best is None or v > best:
                best = v
    return best


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CISC P(True) evaluator on HotpotQA samples.jsonl")
    p.add_argument("--samples_jsonl", required=True,
                   help="Path to samples.jsonl produced by the SC sampling run")
    p.add_argument("--dataset", default=None,
                   help="Dataset key (informational, for logging)")
    p.add_argument("--model", required=True, help="Model key in YAML config")
    p.add_argument("--config", default="configs/models.yaml")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dtype", default=None,
                   help="Override model dtype (float16/bfloat16/auto)")
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--top_logprobs", type=int, default=20,
                   help="Number of top logprobs to request at the next token")
    p.add_argument("--temperatures", default="0.1,0.3,0.5,1.0,2.0",
                   help="Comma-separated CISC softmax temperatures")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = base.load_config(str(config_path))
    models = config.get("models", {})
    if args.model not in models:
        raise ValueError(f"Unknown model: {args.model}")
    model_cfg = dict(models[args.model])
    model_path = str(model_cfg["path"])
    dtype = args.dtype if args.dtype else str(model_cfg.get("dtype", "float16"))
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    metrics_path = output_dir / "metrics.json"
    perq_path = output_dir / "per_question_cisc.jsonl"
    log_path = output_dir / "run.log"

    if output_dir.exists() and not args.overwrite:
        if any(p.exists() for p in [metrics_path, perq_path, log_path]):
            raise FileExistsError(f"Output exists: {output_dir}. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = base.RunLogger(log_path)
    start_iso = datetime.now().isoformat()

    Ts = [float(t) for t in args.temperatures.split(",") if t.strip()]

    try:
        # ---- Load samples.jsonl ----
        samples_path = Path(args.samples_jsonl)
        questions: list[dict[str, Any]] = []
        with samples_path.open() as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
        if args.limit is not None:
            questions = questions[: args.limit]
        logger.log(f"[cisc] loaded {len(questions)} questions from {samples_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)

        # ---- Build all P(True) prompts ----
        prompts: list[str] = []
        index: list[tuple[int, int]] = []  # (q_index, sample_index)
        for qi, q in enumerate(questions):
            for si, s in enumerate(q["samples"]):
                pr = build_ptrue_prompt(q["question"], s["raw"])
                # Use plain prompt (no chat template) since we want a constrained
                # next-token. But Qwen instruct models do behave better with chat
                # templating; we wrap as user message and let assistant continue.
                model_input = base.build_generation_input(tokenizer, pr)
                prompts.append(model_input)
                index.append((qi, si))
        logger.log(f"[cisc] built {len(prompts)} P(True) prompts")

        # ---- vLLM init ----
        from vllm import LLM, SamplingParams
        logger.log(f"[cisc] loading model dtype={dtype} max_model_len={args.max_model_len} "
                   f"tp={args.tensor_parallel_size} mem_util={args.gpu_memory_utilization}")
        llm = LLM(
            model=model_path,
            tokenizer=model_path,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            enforce_eager=False,
            disable_log_stats=True,
            enable_prefix_caching=True,
        )
        sp = SamplingParams(
            temperature=0.0, top_p=1.0,
            max_tokens=1,
            logprobs=args.top_logprobs,
        )

        logger.log(f"[cisc] generating P(True) for {len(prompts)} prompts ...")
        t0 = time.time()
        outputs = llm.generate(prompts, sp)
        elapsed = time.time() - t0
        logger.log(f"[cisc] done in {elapsed:.1f}s ({len(prompts)/elapsed:.2f} prompts/s)")

        # ---- Parse logprobs to get c_i = logp_A - logp_B ----
        logprob_records: list[dict[str, Any]] = [None] * len(prompts)
        miss_a = miss_b = 0
        for i, out in enumerate(outputs):
            gen = out.outputs[0]
            # gen.logprobs is a list of dicts {token_id: Logprob} per generated token.
            # We have max_tokens=1 → one dict.
            top = gen.logprobs[0] if gen.logprobs else {}
            lp_a = _find_token_logprob(top, ("A", "TRUE", "YES"))
            lp_b = _find_token_logprob(top, ("B", "FALSE", "NO"))
            if lp_a is None: miss_a += 1
            if lp_b is None: miss_b += 1
            # Substitute very-negative defaults if a side is missing
            if lp_a is None: lp_a = -50.0
            if lp_b is None: lp_b = -50.0
            logprob_records[i] = {
                "logp_A": lp_a, "logp_B": lp_b,
                "first_token": gen.text,
                "conf": lp_a - lp_b,
            }
        logger.log(f"[cisc] missed-A={miss_a}/{len(prompts)} missed-B={miss_b}/{len(prompts)}")

        # ---- Aggregate per question ----
        # Reorganize by question
        per_q_conf: list[list[dict]] = [[] for _ in questions]
        for i, (qi, si) in enumerate(index):
            per_q_conf[qi].append({"sample_index": si, **logprob_records[i]})

        rows_out: list[dict[str, Any]] = []

        # Stats accumulators
        stats = {
            "sc": {"em": 0.0, "af1": 0.0, "sf1": 0.0, "jf1": 0.0},
            "bon": {"em": 0.0, "af1": 0.0},
            "sampled_mean": {"em": 0.0, "af1": 0.0},
        }
        cisc_stats: dict[float, dict[str, float]] = {T: {"em": 0.0, "af1": 0.0, "sf1": 0.0, "jf1": 0.0} for T in Ts}

        n_q = len(questions)

        for qi, q in enumerate(questions):
            samples = q["samples"]
            n_s = len(samples)
            confs = [None] * n_s
            for rec in per_q_conf[qi]:
                confs[rec["sample_index"]] = rec["conf"]

            # Build (norm_answer -> [(idx, em, af1, sf1, ...), ...])
            groups: dict[str, list[int]] = {}
            for si, s in enumerate(samples):
                key = _norm(s["pred_answer"])
                groups.setdefault(key, []).append(si)

            # ---- SC: uniform vote ----
            sc_winner = max(groups.items(), key=lambda kv: (len(kv[1]), -min(kv[1])))[1]
            sc_repr_idx = sc_winner[0]  # representative sample
            sc_em = samples[sc_repr_idx]["em"]
            sc_af1 = samples[sc_repr_idx]["answer_f1"]
            sc_sf1 = samples[sc_repr_idx]["support_f1"]
            sc_jf1 = sc_af1 * sc_sf1 if sc_af1 and sc_sf1 else 0.0  # joint approximation

            # ---- BoN oracle: any sample correct ----
            bon_em = max(s["em"] for s in samples)
            bon_af1 = max(s["answer_f1"] for s in samples)

            # ---- Sampled mean ----
            mean_em = sum(s["em"] for s in samples) / n_s
            mean_af1 = sum(s["answer_f1"] for s in samples) / n_s

            stats["sc"]["em"] += sc_em
            stats["sc"]["af1"] += sc_af1
            stats["sc"]["sf1"] += sc_sf1
            stats["sc"]["jf1"] += sc_jf1
            stats["bon"]["em"] += bon_em
            stats["bon"]["af1"] += bon_af1
            stats["sampled_mean"]["em"] += mean_em
            stats["sampled_mean"]["af1"] += mean_af1

            # ---- CISC over multiple T ----
            cisc_results: dict[str, dict[str, float]] = {}
            for T in Ts:
                # softmax(conf/T)
                cs = [c / T for c in confs]
                m = max(cs)
                exps = [math.exp(c - m) for c in cs]
                Z = sum(exps)
                ws = [e / Z for e in exps]
                # score per group
                best_key, best_score = None, -1.0
                for key, idxs in groups.items():
                    sc_w = sum(ws[i] for i in idxs)
                    if sc_w > best_score:
                        best_score = sc_w
                        best_key = key
                # representative sample = highest-confidence sample in winning group
                winner_idxs = groups[best_key]
                repr_idx = max(winner_idxs, key=lambda i: confs[i])
                cisc_em = samples[repr_idx]["em"]
                cisc_af1 = samples[repr_idx]["answer_f1"]
                cisc_sf1 = samples[repr_idx]["support_f1"]
                cisc_jf1 = cisc_af1 * cisc_sf1 if cisc_af1 and cisc_sf1 else 0.0
                cisc_results[f"T={T}"] = {
                    "em": cisc_em, "af1": cisc_af1, "sf1": cisc_sf1, "jf1": cisc_jf1,
                    "winning_group": best_key, "winning_score": best_score,
                    "winning_repr_sample": repr_idx,
                }
                cisc_stats[T]["em"] += cisc_em
                cisc_stats[T]["af1"] += cisc_af1
                cisc_stats[T]["sf1"] += cisc_sf1
                cisc_stats[T]["jf1"] += cisc_jf1

            rows_out.append({
                "id": q["id"],
                "n_samples": n_s,
                "gold_answer": q["gold_answer"],
                "sc_em": sc_em, "sc_af1": sc_af1,
                "bon_em": bon_em, "bon_af1": bon_af1,
                "sampled_mean_em": mean_em, "sampled_mean_af1": mean_af1,
                "confidences": confs,
                "groups": {k: idxs for k, idxs in groups.items()},
                "cisc": cisc_results,
            })

        # ---- Persist ----
        with perq_path.open("w") as f:
            for r in rows_out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        def avgd(d):
            return {k: v / n_q for k, v in d.items()}

        metrics = {
            "method": "cisc_ptrue_vllm",
            "samples_jsonl": str(samples_path),
            "model": args.model,
            "model_path": model_path,
            "num_questions": n_q,
            "n_samples_per_q": questions[0]["n_samples"] if questions else 0,
            "sc": avgd(stats["sc"]),
            "bon": avgd(stats["bon"]),
            "sampled_mean": avgd(stats["sampled_mean"]),
            "cisc_by_T": {f"T={T}": avgd(cisc_stats[T]) for T in Ts},
            "p_true": {
                "missed_A": miss_a,
                "missed_B": miss_b,
                "wall_clock_seconds": elapsed,
                "throughput": (len(prompts) / elapsed) if elapsed > 0 else None,
            },
            "vllm": {
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "tensor_parallel_size": args.tensor_parallel_size,
                "dtype": dtype,
                "top_logprobs": args.top_logprobs,
            },
            "start_time": start_iso,
        }
        with metrics_path.open("w") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        # Print summary
        logger.log("[cisc] === summary ===")
        logger.log(f"  SC  EM={metrics['sc']['em']:.4f}  A-F1={metrics['sc']['af1']:.4f}  S-F1={metrics['sc']['sf1']:.4f}")
        logger.log(f"  BoN EM={metrics['bon']['em']:.4f}  A-F1={metrics['bon']['af1']:.4f}")
        for T in Ts:
            d = metrics["cisc_by_T"][f"T={T}"]
            logger.log(f"  CISC T={T:<4}  EM={d['em']:.4f}  A-F1={d['af1']:.4f}  S-F1={d['sf1']:.4f}")

    finally:
        logger.close()


if __name__ == "__main__":
    main()
