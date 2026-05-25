#!/usr/bin/env python3
"""SVote ablation — Random / Gold-Only / SVote selection + Stage-2 re-inference.

Extends eval_svote_vllm.py with --selection_mode:
  svote   : normal SVote voting (default, identical to eval_svote_vllm.py)
  random  : randomly select K paragraphs (E5 ablation)
  gold    : use gold support sentences (E7 oracle upper bound)

Usage:
  python src/svote/eval_svote_ablation_vllm.py \
      --selection_mode random \
      --samples_jsonl outputs/MODEL/DATASET/stage0_samples/merged/samples.jsonl \
      --dataset hotpotqa --model qwen2_5_7b_instruct \
      --output_dir outputs/MODEL/DATASET/ablation_random \
      --n_resample 10 ...

  python src/svote/eval_svote_ablation_vllm.py \
      --selection_mode gold \
      --samples_jsonl outputs/MODEL/DATASET/stage0_samples/merged/samples.jsonl \
      --dataset hotpotqa --model qwen2_5_7b_instruct \
      --output_dir outputs/MODEL/DATASET/ablation_gold \
      --n_resample 10 ...
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import multiprocessing as _mp
try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "eval"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import eval_llm_baseline as base  # noqa: E402
from svote.voting import vote_supports, select_supports, build_sid_to_para_map  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--selection_mode", required=True,
                   choices=["svote", "random", "gold"],
                   help="Evidence selection strategy")
    p.add_argument("--samples_jsonl", required=True)
    p.add_argument("--cisc_perq", default=None)
    p.add_argument("--dataset", default="hotpotqa_full")
    p.add_argument("--model", default="qwen2_5_7b_instruct")
    p.add_argument("--config", default="configs/models.yaml")
    p.add_argument("--output_dir", required=True)
    # Voting (only used when selection_mode=svote)
    p.add_argument("--weight_mode", default="combined",
                   choices=["uniform", "cisc", "cluster", "combined"])
    p.add_argument("--ptrue_T", type=float, default=1.0)
    # Selection
    p.add_argument("--k_min", type=int, default=2)
    p.add_argument("--k_max", type=int, default=4)
    p.add_argument("--score_thresh_continue", type=float, default=0.5)
    p.add_argument("--score_thresh_same_para", type=float, default=0.7)
    p.add_argument("--no_bridging", action="store_true")
    p.add_argument("--context_mode", default="paragraph",
                   choices=["paragraph", "sentence"])
    # Random-specific
    p.add_argument("--random_k_para", type=int, default=None,
                   help="Number of paragraphs for random selection "
                        "(default: match SVote avg K_para ≈ 2)")
    p.add_argument("--svote_perq_jsonl", default=None,
                   help="Path to SVote default per_question.jsonl for per-question "
                        "K-matched random selection (overrides --random_k_para)")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--random_n_trials", type=int, default=3,
                   help="Number of random trials (results averaged)")
    # Re-inference
    p.add_argument("--n_resample", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=384)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", default="float16")
    # Sharding
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _norm(s: str) -> str:
    return base.normalize_answer(s) if s else ""


def select_random_paragraphs(flat_rows, sid_to_para, k_para, rng):
    """Randomly select k_para distinct paragraphs, return their sids."""
    all_paras = sorted(set(sid_to_para.values()))
    if k_para >= len(all_paras):
        return [r["sid"] for r in flat_rows]
    chosen_paras = set(rng.sample(all_paras, k_para))
    return [r["sid"] for r in flat_rows if sid_to_para.get(r["sid"], "") in chosen_paras]


def select_gold_support(flat_rows, gold_support, sid_to_para, context_mode):
    """Select context rows containing gold support sentences."""
    gs = set(gold_support) if isinstance(gold_support, list) else gold_support
    if context_mode == "sentence":
        return [r for r in flat_rows if r["sid"] in gs], sorted(gs)
    else:
        gold_paras = {sid_to_para.get(s, "") for s in gs}
        rows_kept = [r for r in flat_rows if sid_to_para.get(r["sid"], "") in gold_paras]
        return rows_kept, sorted(gs)


def run_single_trial(args, questions, gold_lookup, cisc_logits_by_id,
                     tokenizer, rng_seed=None):
    """Build prompts for one trial. Returns (all_prompts, ctx_list)."""
    if rng_seed is not None:
        rng = random.Random(rng_seed)
    else:
        rng = random.Random(args.random_seed)

    model_cfg_path = PROJECT_ROOT / args.config
    config = base.load_config(str(model_cfg_path))
    model_cfg = dict(config["models"][args.model])
    dataset_cfg = dict(config["datasets"][args.dataset])
    support_unit = str(dataset_cfg.get("support_unit", "sentence"))
    model_family = model_cfg.get("model_family", None)

    all_prompts = []
    ctx = []
    skipped = 0

    for q in questions:
        qid = q["id"]
        gold = gold_lookup.get(qid)
        if gold is None:
            skipped += 1
            continue
        flat_rows = gold["flat_rows"]
        sid_to_para = build_sid_to_para_map(flat_rows)
        samples = q.get("samples", [])

        if args.selection_mode == "gold":
            rows_kept, selected = select_gold_support(
                flat_rows, gold["gold_support"], sid_to_para, args.context_mode)
        elif args.selection_mode == "random":
            if hasattr(args, '_svote_k_by_id') and args._svote_k_by_id and qid in args._svote_k_by_id:
                k_para = args._svote_k_by_id[qid]
            else:
                k_para = args.random_k_para or 2
            selected_sids = select_random_paragraphs(flat_rows, sid_to_para, k_para, rng)
            selected = selected_sids
            if args.context_mode == "sentence":
                rows_kept = [r for r in flat_rows if r["sid"] in set(selected)]
            else:
                keep_paras = {sid_to_para.get(s, "") for s in selected}
                rows_kept = [r for r in flat_rows if sid_to_para.get(r["sid"], "") in keep_paras]
        else:  # svote
            if not samples:
                skipped += 1
                continue
            supports = [set(s.get("pred_support", [])) for s in samples]
            ans_norm = [_norm(s.get("pred_answer", "") or "") for s in samples]
            logits = cisc_logits_by_id.get(qid) if cisc_logits_by_id else None
            scores = vote_supports(supports, ans_norm,
                                   ptrue_logits=logits,
                                   ptrue_T=args.ptrue_T,
                                   weight_mode=args.weight_mode)
            if not scores:
                selected = list({r["sid"] for r in flat_rows})
            else:
                selected = select_supports(
                    scores, sid_to_para,
                    k_min=args.k_min, k_max=args.k_max,
                    score_thresh_continue=args.score_thresh_continue,
                    score_thresh_same_para=args.score_thresh_same_para,
                    require_bridging=not args.no_bridging,
                )
            if args.context_mode == "sentence":
                rows_kept = [r for r in flat_rows if r["sid"] in set(selected)]
            else:
                keep_paras = {sid_to_para.get(s, "") for s in selected}
                rows_kept = [r for r in flat_rows if sid_to_para.get(r["sid"], "") in keep_paras]

        ctx_str = base.render_context_lines(rows_kept)
        prompt = base.build_prompt(gold["question"], ctx_str, "fullctx", support_unit,
                                   model_family=model_family)
        prompt_text = base.build_generation_input(tokenizer, prompt)
        all_prompts.append(prompt_text)
        ctx.append({
            "id": qid, "gold_answer": gold["gold_answer"],
            "gold_support": gold["gold_support"],
            "selected": sorted(set(selected)) if isinstance(selected, list) else selected,
            "selected_paras": sorted({sid_to_para.get(s, "") for s in selected}),
            "rows_kept_count": len(rows_kept),
            "rows_total_count": len(flat_rows),
            "question": gold["question"],
        })

    return all_prompts, ctx


def main() -> None:
    args = parse_args()

    cfg_path = PROJECT_ROOT / args.config
    config = base.load_config(str(cfg_path))
    model_cfg = dict(config["models"][args.model])
    dataset_cfg = dict(config["datasets"][args.dataset])
    support_unit = str(dataset_cfg.get("support_unit", "sentence"))
    adapter_name = str(dataset_cfg.get("adapter", args.dataset))
    id_field = str(dataset_cfg.get("id_field", "id"))
    model_family = model_cfg.get("model_family", None)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    perq_path = output_dir / "per_question.jsonl"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metrics_path} exists; use --overwrite")
    logger = base.RunLogger(output_dir / "run.log")
    start_iso = datetime.now().isoformat()

    logger.log(f"[ablation] selection_mode={args.selection_mode}")

    # ---- Load dataset ----
    dpath = PROJECT_ROOT / dataset_cfg["path"]
    adapter_map = {"hotpotqa": base.adapt_hotpotqa_sample,
                   "musique": base.adapt_musique_sample,
                   "2wiki": base.adapt_2wiki_sample}
    adapter_key = next((k for k in adapter_map if adapter_name.startswith(k)), None)
    adapter_fn = adapter_map[adapter_key]
    diagnostics = base.init_diagnostics()
    raw_data = base.load_dataset_samples(dpath, dataset_cfg.get("file_type", "json"))
    logger.log(f"[ablation] loaded {len(raw_data)} dataset rows from {dpath}")

    gold_lookup = {}
    for idx, sample in enumerate(raw_data, start=1):
        ad = adapter_fn(sample, idx, id_field, diagnostics)
        gold_lookup[ad["id"]] = {
            "flat_rows": ad["flattened"],
            "gold_support": ad["gold_support"],
            "gold_answer": ad["gold_answer"],
            "question": ad["question"],
        }

    # ---- Load samples.jsonl ----
    questions = []
    with open(args.samples_jsonl) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    logger.log(f"[ablation] loaded {len(questions)} sample rows")

    # ---- Shard ----
    if args.num_shards > 1:
        questions.sort(key=lambda q: q["id"])
        questions = [q for i, q in enumerate(questions)
                     if i % args.num_shards == args.shard_id]
        logger.log(f"[ablation] shard {args.shard_id}/{args.num_shards}: "
                   f"{len(questions)} questions")

    # ---- Load SVote per-question K map (for K-matched random) ----
    if args.svote_perq_jsonl and args.selection_mode == "random":
        args._svote_k_by_id = {}
        with open(args.svote_perq_jsonl) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    args._svote_k_by_id[r["id"]] = len(r.get("selected_paras", []))
        logger.log(f"[ablation] loaded SVote K map for {len(args._svote_k_by_id)} questions "
                   f"(avg K={sum(args._svote_k_by_id.values())/len(args._svote_k_by_id):.2f})")
    else:
        args._svote_k_by_id = None

    # ---- CISC (only for svote mode) ----
    cisc_logits_by_id = {}
    if args.cisc_perq:
        with open(args.cisc_perq) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    cisc_logits_by_id[r["id"]] = r.get("confidences", [])
        logger.log(f"[ablation] loaded CISC logits for {len(cisc_logits_by_id)} questions")

    # ---- Tokenizer ----
    from transformers import AutoTokenizer
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["path"], trust_remote_code=trust_remote_code)

    # ---- Build all trial prompts up front (batch all trials together) ----
    if args.selection_mode == "random" and args.random_n_trials > 1:
        n_trials = args.random_n_trials
    else:
        n_trials = 1

    # trials_data[t] = (prompts, ctx) for trial t
    trials_data = []
    for t in range(n_trials):
        prompts_t, ctx_t = run_single_trial(
            args, questions, gold_lookup, cisc_logits_by_id, tokenizer,
            rng_seed=args.random_seed + t)
        trials_data.append((prompts_t, ctx_t))
        logger.log(f"[ablation] trial {t+1}/{n_trials}: {len(prompts_t)} prompts")

    if not trials_data[0][0]:
        logger.log("[ablation] nothing to evaluate; exiting")
        logger.close()
        return

    # Concatenate ALL trials into one flat batch for a single llm.generate() call
    batch_prompts = []
    batch_offsets = []   # (trial_idx, start, end) for slicing outputs back
    for t, (prompts_t, _) in enumerate(trials_data):
        start = len(batch_prompts)
        batch_prompts.extend(prompts_t)
        batch_offsets.append((t, start, len(batch_prompts)))

    logger.log(f"[ablation] total batch: {len(batch_prompts)} prompts "
               f"({n_trials} trial(s) × ~{len(trials_data[0][0])} questions × n={args.n_resample})")

    # ---- vLLM: single model load, single generate() call ----
    from vllm import LLM, SamplingParams
    logger.log(f"[ablation] loading {model_cfg['path']} ...")
    t0 = time.time()
    llm = LLM(
        model=str(model_cfg["path"]),
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=1,
        trust_remote_code=trust_remote_code,
        seed=args.seed,
        disable_log_stats=True,
        enable_prefix_caching=True,
    )
    logger.log(f"[ablation] loaded in {time.time()-t0:.1f}s")
    stop_strs = ["</answer>"] if model_family == "llama" else None
    if args.n_resample == 1:
        sp = SamplingParams(n=1, temperature=0.0, max_tokens=args.max_new_tokens,
                            seed=args.seed, stop=stop_strs)
    else:
        sp = SamplingParams(
            n=args.n_resample, temperature=args.temperature, top_p=args.top_p,
            max_tokens=args.max_new_tokens, seed=args.seed, stop=stop_strs)

    logger.log(f"[ablation] generating {len(batch_prompts)} × n={args.n_resample} in ONE batch ...")
    t1 = time.time()
    all_outputs = llm.generate(batch_prompts, sp)
    logger.log(f"[ablation] generation done in {time.time()-t1:.1f}s")

    # ---- Evaluate: slice outputs back per trial ----
    trial_ems = []
    last_agg = None
    last_ctx = None

    for trial_idx, start, end in batch_offsets:
        prompts_t, ctx = trials_data[trial_idx]
        outputs = all_outputs[start:end]

        agg = {"em": 0.0, "af1": 0.0, "sf1": 0.0, "jf1": 0.0,
               "selected_recall_gold": 0.0, "selected_f1_gold": 0.0}
        n = len(ctx)

        write_perq = (trial_idx == n_trials - 1)
        if write_perq:
            pf = perq_path.open("w", encoding="utf-8")

        for i, c in enumerate(ctx):
            sample_results = []
            for comp in outputs[i].outputs:
                _, pred_support, pred_ans, _, _, parse_err = base.parse_model_output(comp.text)
                em = base.answer_em(pred_ans, c["gold_answer"])
                _, _, af1 = base.answer_scores(pred_ans, c["gold_answer"])
                _, _, sf1 = base.support_scores(pred_support, c["gold_support"])
                sample_results.append({
                    "raw": comp.text, "pred_answer": pred_ans,
                    "pred_support": base._sorted_support_ids(pred_support),
                    "em": em, "answer_f1": af1, "support_f1": sf1,
                    "parse_error_type": parse_err,
                })

            ans_counter = Counter(_norm(r["pred_answer"]) for r in sample_results if r["pred_answer"])
            if ans_counter:
                winner_an = ans_counter.most_common(1)[0][0]
                repr_idx = next((j for j, r in enumerate(sample_results)
                                 if _norm(r["pred_answer"]) == winner_an), 0)
            else:
                repr_idx = 0
            chosen = sample_results[repr_idx]
            em, af1, sf1 = chosen["em"], chosen["answer_f1"], chosen["support_f1"]
            jf1 = af1 * sf1
            agg["em"]  += em
            agg["af1"] += af1
            agg["sf1"] += sf1
            agg["jf1"] += jf1

            gs = set(c["gold_support"])
            sel = set(c["selected"])
            if gs and sel:
                tp = len(gs & sel); prec = tp/len(sel); rec = tp/len(gs)
                sel_f1 = 2*prec*rec/(prec+rec) if tp else 0
            else:
                sel_f1 = 1.0 if not gs and not sel else 0.0
            sel_rec = (len(sel & gs) / len(gs)) if gs else 1.0
            agg["selected_recall_gold"] += sel_rec
            agg["selected_f1_gold"]     += sel_f1

            if write_perq:
                pf.write(json.dumps({
                    "id": c["id"], "gold_answer": c["gold_answer"],
                    "selected": c["selected"], "selected_paras": c["selected_paras"],
                    "rows_kept_count": c["rows_kept_count"],
                    "rows_total_count": c["rows_total_count"],
                    "selected_recall_gold": sel_rec, "selected_f1_gold": sel_f1,
                    "n_resample": args.n_resample,
                    "samples": sample_results,
                    "chosen": {"em": em, "af1": af1, "sf1": sf1, "jf1": jf1,
                               "repr_index": repr_idx},
                }, ensure_ascii=False) + "\n")

        if write_perq:
            pf.close()

        trial_em = agg["em"] / n
        trial_ems.append(trial_em)
        last_agg = agg
        last_ctx = ctx
        logger.log(f"[ablation] trial {trial_idx+1}/{n_trials}: EM={trial_em:.4f}")

    # ---- Final metrics (average over trials for random) ----
    ctx = last_ctx
    agg = last_agg
    avg_em = sum(trial_ems) / len(trial_ems)
    sel_counts = [len(c["selected"]) for c in ctx]
    para_counts = [len(c["selected_paras"]) for c in ctx]
    rows_kept_arr = [c["rows_kept_count"] for c in ctx]
    rows_tot_arr = [c["rows_total_count"] for c in ctx]

    metrics = {
        "method": f"ablation_{args.selection_mode}",
        "selection_mode": args.selection_mode,
        "context_mode": args.context_mode,
        "n_resample": args.n_resample, "temperature": args.temperature,
        "num_evaluated": len(ctx),
        "n_trials": n_trials,
        "em": avg_em,
        "trial_ems": [round(e, 6) for e in trial_ems],
        "af1": agg["af1"]/len(ctx), "sf1": agg["sf1"]/len(ctx), "jf1": agg["jf1"]/len(ctx),
        "selected_recall_gold": agg["selected_recall_gold"]/len(ctx),
        "selected_f1_gold":     agg["selected_f1_gold"]/len(ctx),
        "avg_K_sentences": sum(sel_counts)/len(sel_counts),
        "avg_K_paragraphs": sum(para_counts)/len(para_counts),
        "avg_context_kept_rows": sum(rows_kept_arr)/len(rows_kept_arr),
        "avg_context_total_rows": sum(rows_tot_arr)/len(rows_tot_arr),
        "start_time": start_iso,
    }
    if args.selection_mode == "random":
        metrics["random_k_para"] = "per-question" if args._svote_k_by_id else (args.random_k_para or 2)
        metrics["random_k_matched"] = bool(args._svote_k_by_id)
        metrics["random_seed"] = args.random_seed
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.log("[ablation] === results ===")
    logger.log(f"  mode={args.selection_mode}  ctx={args.context_mode}  N'={args.n_resample}  trials={n_trials}")
    logger.log(f"  EM    = {metrics['em']:.4f}" + (f"  (trials: {metrics['trial_ems']})" if n_trials > 1 else ""))
    logger.log(f"  A-F1  = {metrics['af1']:.4f}")
    logger.log(f"  gold recall = {metrics['selected_recall_gold']:.4f}")
    logger.log(f"  avg K_para={metrics['avg_K_paragraphs']:.2f}  ctx_kept={metrics['avg_context_kept_rows']:.1f}/{metrics['avg_context_total_rows']:.1f}")
    logger.close()


if __name__ == "__main__":
    main()
