#!/usr/bin/env python3
"""SVote — Support-voting + re-inference 2-stage pipeline (fullctx eval).

Stage 1 (CPU): aggregate weighted votes over N=20 ballots' cited supports,
              apply bridging-aware greedy selection -> K supports per question.
Stage 2 (LLM): re-prompt with filtered context restricted to selected supports
              (sentence mode) or their containing paragraphs (paragraph mode);
              vLLM N' generations -> SC vote -> EM/F1.

Eval setting: fullctx (the original samples were sampled in fullctx mode; we
compare end-to-end EM/F1 against the fullctx SC baseline).
"""
from __future__ import annotations

import argparse
import json
import os
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
    p.add_argument("--samples_jsonl", required=True)
    p.add_argument("--cisc_perq", default=None,
                   help="Optional CISC per_question file with 'confidences'")
    p.add_argument("--dataset", default="hotpotqa_full")
    p.add_argument("--model", default="qwen2_5_7b_instruct")
    p.add_argument("--config", default="configs/models.yaml")
    p.add_argument("--output_dir", required=True)
    # Voting
    p.add_argument("--weight_mode", default="combined",
                   choices=["uniform", "cisc", "cluster", "combined"])
    p.add_argument("--ptrue_T", type=float, default=1.0,
                   help="Temperature for sigmoid(L/T) on CISC logits")
    # Selection
    p.add_argument("--k_min", type=int, default=2)
    p.add_argument("--k_max", type=int, default=4)
    p.add_argument("--score_thresh_continue", type=float, default=0.5)
    p.add_argument("--score_thresh_same_para", type=float, default=0.7)
    p.add_argument("--no_bridging", action="store_true",
                   help="Disable paragraph-bridging constraint (ablation)")
    p.add_argument("--context_mode", default="paragraph",
                   choices=["paragraph", "sentence"],
                   help="Stage-2 context: keep paragraphs of selected sids or sentences only")
    # Re-inference
    p.add_argument("--n_resample", type=int, default=5,
                   help="N' samples on filtered context (1 = deterministic mode)")
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

    # ---- Load dataset (to reconstruct context + sid->paragraph map) ----
    dpath = PROJECT_ROOT / dataset_cfg["path"]
    adapter_map = {"hotpotqa": base.adapt_hotpotqa_sample,
                   "musique": base.adapt_musique_sample,
                   "2wiki": base.adapt_2wiki_sample}
    adapter_key = next((k for k in adapter_map if adapter_name.startswith(k)), None)
    adapter_fn = adapter_map[adapter_key]
    diagnostics = base.init_diagnostics()
    raw_data = base.load_dataset_samples(dpath, dataset_cfg.get("file_type", "json"))
    logger.log(f"[svote] loaded {len(raw_data)} dataset rows from {dpath}")

    gold_lookup: dict[str, dict] = {}
    for idx, sample in enumerate(raw_data, start=1):
        ad = adapter_fn(sample, idx, id_field, diagnostics)
        gold_lookup[ad["id"]] = {
            "flat_rows": ad["flattened"],
            "gold_support": ad["gold_support"],
            "gold_answer": ad["gold_answer"],
            "question": ad["question"],
        }

    # ---- Load samples.jsonl ----
    questions: list[dict] = []
    with open(args.samples_jsonl) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    logger.log(f"[svote] loaded {len(questions)} sample rows")

    # ---- Load CISC confidences (optional) ----
    cisc_logits_by_id: dict = {}
    if args.cisc_perq:
        with open(args.cisc_perq) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    cisc_logits_by_id[r["id"]] = r.get("confidences", [])
        logger.log(f"[svote] loaded CISC logits for {len(cisc_logits_by_id)} questions")
    elif args.weight_mode in ("cisc", "combined"):
        raise ValueError(f"--weight_mode={args.weight_mode} requires --cisc_perq")

    # ---- Shard ----
    if args.num_shards > 1:
        questions.sort(key=lambda q: q["id"])
        questions = [q for i, q in enumerate(questions)
                     if i % args.num_shards == args.shard_id]
        logger.log(f"[svote] shard {args.shard_id}/{args.num_shards}: "
                   f"{len(questions)} questions")

    # ---- Tokenizer + prompts ----
    from transformers import AutoTokenizer
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["path"], trust_remote_code=trust_remote_code,
    )

    all_prompts: list[str] = []
    ctx: list[dict] = []  # per-question metadata
    skipped_no_gold = 0
    skipped_no_supports = 0
    for q in questions:
        qid = q["id"]
        gold = gold_lookup.get(qid)
        if gold is None:
            skipped_no_gold += 1
            continue
        flat_rows = gold["flat_rows"]
        sid_to_para = build_sid_to_para_map(flat_rows)

        samples = q.get("samples", [])
        if not samples:
            skipped_no_supports += 1
            continue
        supports = [set(s.get("pred_support", [])) for s in samples]
        ans_norm = [_norm(s.get("pred_answer", "") or "") for s in samples]
        logits = cisc_logits_by_id.get(qid) if cisc_logits_by_id else None

        scores = vote_supports(supports, ans_norm,
                               ptrue_logits=logits,
                               ptrue_T=args.ptrue_T,
                               weight_mode=args.weight_mode)
        if not scores:
            # No supports cited by any ballot; fall back to full context
            selected = list({r["sid"] for r in flat_rows})
        else:
            selected = select_supports(
                scores, sid_to_para,
                k_min=args.k_min, k_max=args.k_max,
                score_thresh_continue=args.score_thresh_continue,
                score_thresh_same_para=args.score_thresh_same_para,
                require_bridging=not args.no_bridging,
            )

        # Build filtered context rows
        if args.context_mode == "sentence":
            keep_set = set(selected)
            rows_kept = [r for r in flat_rows if r["sid"] in keep_set]
        else:  # paragraph mode
            keep_paras = {sid_to_para.get(s, "") for s in selected}
            rows_kept = [r for r in flat_rows
                         if sid_to_para.get(r["sid"], "") in keep_paras]

        ctx_str = base.render_context_lines(rows_kept)
        prompt = base.build_prompt(gold["question"], ctx_str, "fullctx", support_unit, model_family=model_family)
        prompt_text = base.build_generation_input(tokenizer, prompt)
        all_prompts.append(prompt_text)
        ctx.append({
            "id": qid, "gold_answer": gold["gold_answer"],
            "gold_support": gold["gold_support"],
            "selected": selected,
            "selected_paras": sorted({sid_to_para.get(s, "") for s in selected}),
            "rows_kept_count": len(rows_kept),
            "rows_total_count": len(flat_rows),
            "question": gold["question"],
        })
    logger.log(f"[svote] built {len(all_prompts)} prompts "
               f"(skipped no-gold={skipped_no_gold}, no-samples={skipped_no_supports})")

    if not all_prompts:
        logger.log("[svote] nothing to evaluate; exiting")
        logger.close()
        return

    # Context size stats
    rows_kept_arr = [c["rows_kept_count"] for c in ctx]
    rows_tot_arr = [c["rows_total_count"] for c in ctx]
    sel_counts = [len(c["selected"]) for c in ctx]
    para_counts = [len(c["selected_paras"]) for c in ctx]
    logger.log(f"[svote] selection avg: K_sentences={sum(sel_counts)/len(sel_counts):.2f}, "
               f"K_paragraphs={sum(para_counts)/len(para_counts):.2f}, "
               f"context_rows kept={sum(rows_kept_arr)/len(rows_kept_arr):.1f} / "
               f"{sum(rows_tot_arr)/len(rows_tot_arr):.1f} "
               f"({sum(rows_kept_arr)/sum(rows_tot_arr):.1%})")

    # ---- vLLM ----
    from vllm import LLM, SamplingParams
    logger.log(f"[svote] loading {model_cfg['path']} ...")
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
    logger.log(f"[svote] loaded in {time.time()-t0:.1f}s")
    stop_strs = ["</answer>"] if model_family == "llama" else None
    if args.n_resample == 1:
        sp = SamplingParams(n=1, temperature=0.0, max_tokens=args.max_new_tokens, seed=args.seed, stop=stop_strs)
    else:
        sp = SamplingParams(
            n=args.n_resample, temperature=args.temperature, top_p=args.top_p,
            max_tokens=args.max_new_tokens, seed=args.seed, stop=stop_strs,
        )
    logger.log(f"[svote] generating {len(all_prompts)} × n={args.n_resample} ...")
    t1 = time.time()
    outputs = llm.generate(all_prompts, sp)
    logger.log(f"[svote] generation done in {time.time()-t1:.1f}s")

    # ---- Aggregate ----
    agg = {"em": 0.0, "af1": 0.0, "sf1": 0.0, "jf1": 0.0,
           "selected_recall_gold": 0.0, "selected_f1_gold": 0.0}
    n = len(ctx)
    pf = perq_path.open("w", encoding="utf-8")
    for i, c in enumerate(ctx):
        # SC over n_resample
        sample_results = []
        for comp in outputs[i].outputs:
            _, pred_support, pred_ans, _, _, parse_err = base.parse_model_output(comp.text)
            an = _norm(pred_ans)
            em = base.answer_em(pred_ans, c["gold_answer"])
            _, _, af1 = base.answer_scores(pred_ans, c["gold_answer"])
            _, _, sf1 = base.support_scores(pred_support, c["gold_support"])
            sample_results.append({
                "raw": comp.text, "pred_answer": pred_ans,
                "pred_support": base._sorted_support_ids(pred_support),
                "em": em, "answer_f1": af1, "support_f1": sf1,
                "parse_error_type": parse_err,
            })
        # SC majority
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
        # selected vs gold support F1
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
    pf.close()

    metrics = {
        "method": "svote_2stage",
        "weight_mode": args.weight_mode, "ptrue_T": args.ptrue_T,
        "context_mode": args.context_mode,
        "k_min": args.k_min, "k_max": args.k_max,
        "no_bridging": args.no_bridging,
        "n_resample": args.n_resample, "temperature": args.temperature,
        "num_evaluated": n,
        "em": agg["em"]/n, "af1": agg["af1"]/n, "sf1": agg["sf1"]/n, "jf1": agg["jf1"]/n,
        "selected_recall_gold": agg["selected_recall_gold"]/n,
        "selected_f1_gold":     agg["selected_f1_gold"]/n,
        "avg_K_sentences": sum(sel_counts)/len(sel_counts),
        "avg_K_paragraphs": sum(para_counts)/len(para_counts),
        "avg_context_kept_rows": sum(rows_kept_arr)/len(rows_kept_arr),
        "avg_context_total_rows": sum(rows_tot_arr)/len(rows_tot_arr),
        "start_time": start_iso,
    }
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.log("[svote] === results ===")
    logger.log(f"  weight={args.weight_mode}  ctx={args.context_mode}  K=[{args.k_min},{args.k_max}]  bridging={'on' if not args.no_bridging else 'off'}  N'={args.n_resample}")
    logger.log(f"  EM    = {metrics['em']:.4f}")
    logger.log(f"  A-F1  = {metrics['af1']:.4f}")
    logger.log(f"  S-F1  = {metrics['sf1']:.4f}")
    logger.log(f"  J-F1  = {metrics['jf1']:.4f}")
    logger.log(f"  selected vs gold support: F1={metrics['selected_f1_gold']:.4f}  Recall={metrics['selected_recall_gold']:.4f}")
    logger.log(f"  avg K_sentences={metrics['avg_K_sentences']:.2f}  K_paragraphs={metrics['avg_K_paragraphs']:.2f}  ctx_kept={metrics['avg_context_kept_rows']:.1f}/{metrics['avg_context_total_rows']:.1f}")
    logger.close()


if __name__ == "__main__":
    main()
