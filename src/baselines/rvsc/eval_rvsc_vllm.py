#!/usr/bin/env python3
"""Ranked-Voting Self-Consistency (RVSC) evaluator on HotpotQA — vLLM backend.

Pipeline per shard:
  1. Load dataset, build a *ranked-answer* prompt for each question.
  2. vLLM N=20 stochastic sampling per question.
  3. Parse each completion into a ranked list of K=3 candidate answers.
  4. Build N ballots (rank-1..rank-K) per question; normalize answers.
  5. Vote with IRV / BCV / MRRV; also report:
       - Plain CoT  (k=0 deterministic temperature=0 first sample, optional)
       - SC = uniform majority over rank-1 only
       - BoN oracle (any rank in any ballot matches gold)
  6. Compute EM / answer-F1 / support-F1 / joint-F1 against gold.

Output (per shard):
  - samples.jsonl         : raw N samples per question (for re-aggregation)
  - per_question.jsonl    : voting outputs per question
  - metrics.json          : aggregated EM/F1 by method
  - run.log
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import multiprocessing as _mp
try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "eval"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import eval_llm_baseline as base  # noqa: E402
from baselines.rvsc.prompt import build_ranked_prompt  # noqa: E402
from baselines.rvsc.parser import parse_ranked_output  # noqa: E402
from baselines.rvsc.voting import IRV, BCV, MRRV  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--setting", default="fullctx",
                   choices=["fullctx", "goldonly", "nogold", "questiononly"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--run_id", required=True)
    p.add_argument("--config", default="configs/models.yaml")
    p.add_argument("--output_root", default="outputs/eval_rvsc")
    p.add_argument("--n_samples", type=int, default=20)
    p.add_argument("--top_k", type=int, default=3,
                   help="Number of ranked candidate answers each ballot should contain")
    p.add_argument("--max_new_tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "auto"])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _norm(s: str) -> str:
    return base.normalize_answer(s) if s else ""


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    config = base.load_config(str(cfg_path))
    models = config.get("models", {})
    datasets = config.get("datasets", {})
    if args.model not in models:
        raise ValueError(f"Unknown model: {args.model}")
    if args.dataset not in datasets:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    model_cfg = dict(models[args.model])
    dataset_cfg = dict(datasets[args.dataset])

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_dir = output_root / args.run_id / args.dataset / args.model / args.setting
    samples_path = output_dir / "samples.jsonl"
    perq_path = output_dir / "per_question.jsonl"
    metrics_path = output_dir / "metrics.json"
    args_path = output_dir / "args.json"
    log_path = output_dir / "run.log"
    if output_dir.exists() and not args.overwrite:
        if any(p.exists() for p in [samples_path, perq_path, metrics_path]):
            raise FileExistsError(f"Output exists: {output_dir}. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = base.RunLogger(log_path)
    start_iso = datetime.now().isoformat()

    try:
        dpath = Path(dataset_cfg["path"])
        if not dpath.is_absolute():
            dpath = PROJECT_ROOT / dpath
        file_type = str(dataset_cfg.get("file_type", "json"))
        support_unit = str(dataset_cfg.get("support_unit", "sentence"))
        adapter_name = str(dataset_cfg.get("adapter", args.dataset))
        id_field = str(dataset_cfg.get("id_field", "id"))

        logger.log(f"[rvsc] run_id={args.run_id} N={args.n_samples} K={args.top_k}")

        adapter_map = {
            "hotpotqa": base.adapt_hotpotqa_sample,
            "musique": base.adapt_musique_sample,
            "2wiki": base.adapt_2wiki_sample,
        }
        # Find adapter by stem (hotpotqa_shardX -> hotpotqa)
        adapter_key = next((k for k in adapter_map if adapter_name.startswith(k)), None)
        if adapter_key is None:
            raise ValueError(f"No adapter for {adapter_name}")
        adapter_fn = adapter_map[adapter_key]
        data = base.load_dataset_samples(dpath, file_type)
        if args.limit is not None:
            data = data[: args.limit]

        from transformers import AutoTokenizer
        trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["path"], trust_remote_code=trust_remote_code,
        )

        diagnostics = base.init_diagnostics()
        all_prompts: list[str] = []
        prompt_token_ids: list[list[int]] = []
        ctx: list[dict[str, Any]] = []
        for idx, sample in enumerate(data, start=1):
            ad = adapter_fn(sample, idx, id_field, diagnostics)
            sid = ad["id"]; q = ad["question"]; ga = ad["gold_answer"]
            flat = ad["flattened"]; gsup = ad["gold_support"]
            if args.setting == "fullctx":
                rows = flat
            elif args.setting == "goldonly":
                rows = base.keep_context_by_ids(flat, gsup)
            elif args.setting == "nogold":
                rows = base.drop_context_by_ids(flat, gsup)
            else:
                rows = []
            visible = {r["sid"] for r in rows}
            ctx_str = (base.render_context_lines(rows)
                       if args.setting != "questiononly" else None)
            prompt = build_ranked_prompt(q, ctx_str, args.setting, support_unit, k=args.top_k)
            mi = base.build_generation_input(tokenizer, prompt)
            tids = tokenizer(mi, add_special_tokens=False)["input_ids"]
            all_prompts.append(mi)
            prompt_token_ids.append(tids)
            ctx.append({
                "sample_id": sid, "question": q, "gold_answer": ga,
                "gold_support": gsup, "visible_support_ids": visible,
            })
        n = len(all_prompts)
        N = args.n_samples
        K = args.top_k

        budget = args.max_model_len - args.max_new_tokens - 8
        is_skip = [len(t) > budget for t in prompt_token_ids]
        n_skip = sum(is_skip)
        valid_indices = [i for i, sk in enumerate(is_skip) if not sk]
        valid_prompts = [all_prompts[i] for i in valid_indices]
        logger.log(f"[rvsc] built {n} prompts; effective ballots={n*N}; skip={n_skip}")

        with args_path.open("w", encoding="utf-8") as f:
            json.dump({
                "method": "rvsc_vllm", "run_id": args.run_id,
                "dataset": args.dataset, "model": args.model,
                "setting": args.setting, "n_samples": N, "top_k": K,
                "temperature": args.temperature, "top_p": args.top_p,
                "seed": args.seed, "limit": args.limit,
                "max_new_tokens": args.max_new_tokens,
                "max_model_len": args.max_model_len,
                "tensor_parallel_size": args.tensor_parallel_size,
                "dtype": args.dtype, "config_path": str(cfg_path),
                "model_config": model_cfg, "dataset_config": dataset_cfg,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "start_time": start_iso,
            }, f, ensure_ascii=False, indent=2)

        # ---- vLLM ----
        from vllm import LLM, SamplingParams
        logger.log(f"[rvsc] loading model {model_cfg.get('path')} ...")
        t0 = time.time()
        llm = LLM(
            model=str(model_cfg["path"]),
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            seed=args.seed,
            disable_log_stats=True,
            enable_prefix_caching=True,
        )
        logger.log(f"[rvsc] loaded in {time.time()-t0:.1f}s")

        sp = SamplingParams(
            n=N, temperature=args.temperature, top_p=args.top_p,
            max_tokens=args.max_new_tokens, seed=args.seed,
        )
        n_valid = len(valid_prompts)
        logger.log(f"[rvsc] generating {n_valid} valid prompts × n={N} samples ...")
        t_start = time.time()
        outputs = llm.generate(valid_prompts, sp) if n_valid > 0 else []
        wall_gen = time.time() - t_start
        logger.log(f"[rvsc] generation finished in {wall_gen:.1f}s")
        idx_to_output = {valid_indices[k]: outputs[k] for k in range(n_valid)}

        # ---- Parse + vote + score ----
        sf = samples_path.open("w", encoding="utf-8")
        pf = perq_path.open("w", encoding="utf-8")

        # Aggregators
        agg = {
            "sc":   {"em": 0.0, "af1": 0.0, "sf1": 0.0, "jf1": 0.0},
            "irv":  {"em": 0.0, "af1": 0.0, "sf1": 0.0, "jf1": 0.0},
            "bcv":  {"em": 0.0, "af1": 0.0, "sf1": 0.0, "jf1": 0.0},
            "mrrv": {"em": 0.0, "af1": 0.0, "sf1": 0.0, "jf1": 0.0},
            "bon":  {"em": 0.0, "af1": 0.0},
        }
        sampled_mean_em = 0.0
        sampled_mean_af1 = 0.0
        n_skipped = 0
        format_ok_count = 0
        strict_ok_count = 0
        ballot_count = 0

        for i in range(n):
            c = ctx[i]
            gold = c["gold_answer"]
            gold_sup = c["gold_support"]
            visible = c["visible_support_ids"]
            row_samples: list[dict[str, Any]] = []
            ballots: list[list[str]] = []          # normalized for voting
            ballot_displays: list[list[str]] = []  # original strings (for repr)
            sample_supports: list[set[str]] = []   # parsed support per sample

            if is_skip[i]:
                n_skipped += 1
            else:
                req_out = idx_to_output[i]
                for k_idx, comp in enumerate(req_out.outputs):
                    raw = comp.text
                    parsed = parse_ranked_output(raw, k=K)
                    sup = parsed["support_raw"] & visible
                    sample_supports.append(sup)
                    ranked = parsed["ranked_answers"]
                    norm = [_norm(a) for a in ranked]
                    if parsed["format_ok"]:
                        format_ok_count += 1
                    if parsed["format_ok_strict"]:
                        strict_ok_count += 1
                    # rank-1 metrics for sampled-mean
                    rank1 = ranked[0] if ranked else ""
                    em1 = base.answer_em(rank1, gold)
                    _, _, af1_1 = base.answer_scores(rank1, gold)
                    row_samples.append({
                        "k": k_idx, "raw": raw,
                        "ranked_answers": ranked,
                        "normalized": norm,
                        "support": base._sorted_support_ids(sup),
                        "format_ok": parsed["format_ok"],
                        "format_ok_strict": parsed["format_ok_strict"],
                        "parse_error_type": parsed["parse_error_type"],
                        "rank1_em": em1,
                        "rank1_af1": af1_1,
                        "generated_tokens": int(len(comp.token_ids)),
                    })
                    ballots.append(norm)
                    ballot_displays.append(ranked)
                    ballot_count += 1
                    sampled_mean_em += em1
                    sampled_mean_af1 += af1_1

            # ---- Voting ----
            sc_winner = bcv_winner = irv_winner = mrrv_winner = ""
            if ballots:
                # SC = uniform majority on rank-1 only
                from collections import Counter as _C
                rank1_norm = [b[0] if b else "" for b in ballots]
                sc_winner = _C([x for x in rank1_norm if x]).most_common(1)[0][0] if any(rank1_norm) else ""
                irv_winner = IRV(ballots).run()
                bcv_winner = BCV(ballots).run()
                mrrv_winner = MRRV(ballots).run()

            # ---- Lookup display string + best support per winner ----
            def lookup(winner_norm: str) -> tuple[str, set[str]]:
                """Find a ballot whose any rank matches winner_norm, return its
                rank-1 display string (closest representative) + its support."""
                if not winner_norm:
                    return "", set()
                # Prefer ballots where rank-1 normalizes to winner
                for j, b in enumerate(ballots):
                    if b and b[0] == winner_norm:
                        disp = ballot_displays[j][0]
                        return disp, sample_supports[j]
                # Otherwise find any ballot where winner appears, use that disp
                for j, b in enumerate(ballots):
                    if winner_norm in b:
                        idx_in_b = b.index(winner_norm)
                        disp = ballot_displays[j][idx_in_b]
                        return disp, sample_supports[j]
                return winner_norm, set()

            # Compute metrics
            def score_method(winner_norm: str) -> dict[str, float]:
                disp, sup = lookup(winner_norm)
                em = base.answer_em(disp, gold)
                _, _, af1 = base.answer_scores(disp, gold)
                _, _, sf1 = base.support_scores(sup, gold_sup)
                jf1 = af1 * sf1
                return {"winner": disp, "winner_norm": winner_norm,
                        "em": em, "af1": af1, "sf1": sf1, "jf1": jf1,
                        "support": base._sorted_support_ids(sup)}

            results = {
                "sc":   score_method(sc_winner),
                "irv":  score_method(irv_winner),
                "bcv":  score_method(bcv_winner),
                "mrrv": score_method(mrrv_winner),
            }

            # BoN: any rank-1 across N samples matches gold?
            bon_em = 0
            bon_af1 = 0.0
            for b_norm in ballots:
                for cand in b_norm:
                    if cand == _norm(gold):
                        bon_em = 1
                    _, _, af1c = base.answer_scores(cand, gold)
                    if af1c > bon_af1:
                        bon_af1 = af1c

            for k_, d_ in results.items():
                agg[k_]["em"]  += d_["em"]
                agg[k_]["af1"] += d_["af1"]
                agg[k_]["sf1"] += d_["sf1"]
                agg[k_]["jf1"] += d_["jf1"]
            agg["bon"]["em"]  += bon_em
            agg["bon"]["af1"] += bon_af1

            # Persist
            sf.write(json.dumps({
                "id": c["sample_id"], "dataset": args.dataset,
                "model": args.model, "setting": args.setting,
                "question": c["question"], "gold_answer": gold,
                "gold_support": base._sorted_support_ids(gold_sup),
                "visible_support_count": len(visible),
                "n_samples": len(row_samples), "samples": row_samples,
                "skipped": is_skip[i],
            }, ensure_ascii=False) + "\n")

            pf.write(json.dumps({
                "id": c["sample_id"],
                "gold_answer": gold,
                "gold_support_count": len(gold_sup),
                "n_ballots": len(ballots),
                "ballots_norm": ballots,
                "results": results,
                "bon": {"em": bon_em, "af1": bon_af1},
                "skipped": is_skip[i],
            }, ensure_ascii=False) + "\n")

        sf.close(); pf.close()

        # ---- Aggregate ----
        denom = max(1, n - n_skipped)
        def avgd(d): return {k: v / denom for k, v in d.items()}

        metrics = {
            "method": "rvsc_vllm",
            "model": args.model,
            "dataset": args.dataset,
            "setting": args.setting,
            "num_questions": n,
            "num_skipped_long": n_skipped,
            "num_evaluated": denom,
            "n_samples_per_q": N,
            "top_k": K,
            "format_ok_rate": format_ok_count / max(1, ballot_count),
            "strict_format_ok_rate": strict_ok_count / max(1, ballot_count),
            "sampled_mean_em":  sampled_mean_em  / max(1, ballot_count),
            "sampled_mean_af1": sampled_mean_af1 / max(1, ballot_count),
            "sc":   avgd(agg["sc"]),
            "irv":  avgd(agg["irv"]),
            "bcv":  avgd(agg["bcv"]),
            "mrrv": avgd(agg["mrrv"]),
            "bon":  avgd(agg["bon"]),
            "wall_clock_seconds": wall_gen,
            "start_time": start_iso,
        }
        with metrics_path.open("w") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        logger.log("[rvsc] === summary ===")
        logger.log(f"  format_ok_rate = {metrics['format_ok_rate']:.4f} "
                   f"(strict={metrics['strict_format_ok_rate']:.4f})")
        logger.log(f"  Sampled-mean rank1 EM = {metrics['sampled_mean_em']:.4f}")
        logger.log(f"  SC   EM={metrics['sc']['em']:.4f} A-F1={metrics['sc']['af1']:.4f} "
                   f"S-F1={metrics['sc']['sf1']:.4f} J-F1={metrics['sc']['jf1']:.4f}")
        logger.log(f"  IRV  EM={metrics['irv']['em']:.4f} A-F1={metrics['irv']['af1']:.4f} "
                   f"S-F1={metrics['irv']['sf1']:.4f} J-F1={metrics['irv']['jf1']:.4f}")
        logger.log(f"  BCV  EM={metrics['bcv']['em']:.4f} A-F1={metrics['bcv']['af1']:.4f} "
                   f"S-F1={metrics['bcv']['sf1']:.4f} J-F1={metrics['bcv']['jf1']:.4f}")
        logger.log(f"  MRRV EM={metrics['mrrv']['em']:.4f} A-F1={metrics['mrrv']['af1']:.4f} "
                   f"S-F1={metrics['mrrv']['sf1']:.4f} J-F1={metrics['mrrv']['jf1']:.4f}")
        logger.log(f"  BoN  EM={metrics['bon']['em']:.4f} A-F1={metrics['bon']['af1']:.4f}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
