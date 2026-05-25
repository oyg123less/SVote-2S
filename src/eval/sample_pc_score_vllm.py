#!/usr/bin/env python3
"""vLLM-based N-sample stochastic generation for PC-Score (drop-in replacement
for sample_pc_score.py).

Same prompt construction, parser, adapter, metrics, and JSONL output schema as
sample_pc_score.py — only the generation backend switches from HF transformers
batched generate() to vLLM with SamplingParams(n=N).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "eval"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import eval_llm_baseline as base  # noqa: E402

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="vLLM N-sample generator for PC-Score")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--setting", required=True,
                   choices=["fullctx", "goldonly", "nogold", "questiononly"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--run_id", required=True)
    p.add_argument("--config", default="configs/models.yaml")
    p.add_argument("--output_root", default="outputs/eval")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--n_samples", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    # vLLM-specific knobs
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--enforce_eager", action="store_true",
                   help="Disable CUDA graph capture (faster startup, slower inference)")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "auto"])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


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
    args_path = output_dir / "args.json"
    log_path = output_dir / "run.log"
    if output_dir.exists() and not args.overwrite:
        if any(p.exists() for p in [samples_path, args_path, log_path]):
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

        logger.log(f"[pc-sample-vllm] run_id={args.run_id} N={args.n_samples} "
                   f"T={args.temperature} top_p={args.top_p}")

        adapter_map = {
            "hotpotqa": base.adapt_hotpotqa_sample,
            "musique": base.adapt_musique_sample,
            "2wiki": base.adapt_2wiki_sample,
        }
        adapter_fn = adapter_map[adapter_name]
        data = base.load_dataset_samples(dpath, file_type)
        if args.limit is not None:
            data = data[: args.limit]

        # --- Tokenizer (only for chat-template prompt building) ---
        from transformers import AutoTokenizer
        trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
        model_family = model_cfg.get("model_family", None)
        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["path"], trust_remote_code=trust_remote_code,
        )

        # --- Build all prompts up-front ---
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
            prompt = base.build_prompt(q, ctx_str, args.setting, support_unit, model_family=model_family)
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

        # --- Length guard: skip prompts that won't fit max_model_len - max_new_tokens ---
        # Leave a small safety margin (8 tokens) for any internal vllm bookkeeping.
        budget = args.max_model_len - args.max_new_tokens - 8
        is_skip = [len(t) > budget for t in prompt_token_ids]
        n_skip = sum(is_skip)
        valid_indices = [i for i, sk in enumerate(is_skip) if not sk]
        valid_prompts = [all_prompts[i] for i in valid_indices]
        logger.log(f"[pc-sample-vllm] built {n} prompts; effective samples = {n*N}")
        logger.log(f"[pc-sample-vllm] length guard: budget={budget} "
                   f"(max_model_len={args.max_model_len} \u2212 max_new_tokens={args.max_new_tokens} \u2212 8); "
                   f"skipping {n_skip}/{n} prompts that exceed budget")
        if n_skip > 0:
            offenders = [(ctx[i]["sample_id"], len(prompt_token_ids[i]))
                         for i, sk in enumerate(is_skip) if sk]
            logger.log(f"[pc-sample-vllm] skipped ids (id, n_tokens): {offenders[:20]}"
                       + (" ..." if len(offenders) > 20 else ""))

        with args_path.open("w", encoding="utf-8") as f:
            json.dump({
                "method": "pc_sample_vllm",
                "run_id": args.run_id, "dataset": args.dataset,
                "model": args.model, "setting": args.setting,
                "n_samples": N, "temperature": args.temperature,
                "top_p": args.top_p, "seed": args.seed,
                "limit": args.limit, "max_new_tokens": args.max_new_tokens,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_model_len": args.max_model_len,
                "tensor_parallel_size": args.tensor_parallel_size,
                "enforce_eager": args.enforce_eager,
                "dtype": args.dtype,
                "config_path": str(cfg_path),
                "model_config": model_cfg, "dataset_config": dataset_cfg,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "start_time": start_iso,
            }, f, ensure_ascii=False, indent=2)

        # --- vLLM init ---
        from vllm import LLM, SamplingParams
        logger.log(f"[pc-sample-vllm] loading model {model_cfg.get('path')} ...")
        t0 = time.time()
        llm = LLM(
            model=str(model_cfg["path"]),
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            enforce_eager=args.enforce_eager,
            trust_remote_code=trust_remote_code,
            seed=args.seed,
        )
        logger.log(f"[pc-sample-vllm] loaded in {time.time()-t0:.1f}s")

        stop_strs = ["</answer>"] if model_family == "llama" else None
        sp = SamplingParams(
            n=N,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            seed=args.seed,
            stop=stop_strs,
        )

        # --- Generate (vLLM batches everything internally) ---
        n_valid = len(valid_prompts)
        logger.log(f"[pc-sample-vllm] generating {n_valid} valid prompts \u00d7 n={N} samples ...")
        t_start = time.time()
        outputs = llm.generate(valid_prompts, sp) if n_valid > 0 else []
        wall_gen = time.time() - t_start
        if n_valid > 0:
            logger.log(f"[pc-sample-vllm] generation finished in {wall_gen:.1f}s "
                       f"({n_valid/wall_gen:.2f} q/s, {n_valid*N/wall_gen:.1f} samples/s)")
        else:
            logger.log(f"[pc-sample-vllm] no valid prompts to generate")

        if len(outputs) != n_valid:
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs but expected {n_valid}")
        # Map valid_index -> output
        idx_to_output = {valid_indices[k]: outputs[k] for k in range(n_valid)}

        # --- Parse + score + write JSONL (preserves input order) ---
        out_f = samples_path.open("w", encoding="utf-8")
        pbar = tqdm(total=n, desc=f"PC-Sample-vLLM-{args.dataset}-{args.setting}-N{N}",
                    unit="q") if tqdm else None
        t_parse = time.time()

        for i in range(n):
            c = ctx[i]
            samples = []
            if is_skip[i]:
                # Skipped due to over-length prompt: emit row with zero samples
                row = {
                    "id": c["sample_id"],
                    "dataset": args.dataset,
                    "model": args.model,
                    "setting": args.setting,
                    "question": c["question"],
                    "gold_answer": c["gold_answer"],
                    "gold_support": base._sorted_support_ids(c["gold_support"]),
                    "visible_support_count": len(c["visible_support_ids"]),
                    "input_tokens": len(prompt_token_ids[i]),
                    "n_samples": 0,
                    "samples": [],
                    "skip_reason": f"prompt_too_long ({len(prompt_token_ids[i])} > budget={budget})",
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if pbar is not None:
                    pbar.update(1)
                continue
            req_out = idx_to_output[i]
            # vLLM gives len(req_out.outputs) == N CompletionOutput objects
            for k, comp in enumerate(req_out.outputs):
                raw = comp.text
                gen_count = len(comp.token_ids)
                # finish_reason "stop" indicates EOS / stop token reached
                ended = comp.finish_reason == "stop"
                (_, sup_raw, ans, fmt_ok, fmt_strict, perr) = \
                    base.parse_model_output(raw)
                sup = sup_raw & c["visible_support_ids"]
                em = base.answer_em(ans, c["gold_answer"])
                ap, ar, af1 = base.answer_scores(ans, c["gold_answer"])
                sp_p, sp_r, sf1 = base.support_scores(sup, c["gold_support"])
                samples.append({
                    "k": k,
                    "raw": raw,
                    "pred_answer": ans,
                    "pred_support": base._sorted_support_ids(sup),
                    "pred_support_raw": base._sorted_support_ids(sup_raw),
                    "em": em,
                    "answer_f1": af1,
                    "support_f1": sf1,
                    "format_ok_strict": fmt_strict,
                    "parse_error_type": perr,
                    "generated_tokens": int(gen_count),
                    "ended_with_eos": ended,
                })
            row = {
                "id": c["sample_id"],
                "dataset": args.dataset,
                "model": args.model,
                "setting": args.setting,
                "question": c["question"],
                "gold_answer": c["gold_answer"],
                "gold_support": base._sorted_support_ids(c["gold_support"]),
                "visible_support_count": len(c["visible_support_ids"]),
                "input_tokens": len(req_out.prompt_token_ids),
                "n_samples": N,
                "samples": samples,
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if pbar is not None:
                pbar.update(1)

        out_f.close()
        if pbar is not None:
            pbar.close()
        wall = time.time() - t_start
        logger.log(f"[pc-sample-vllm] done; {n} questions × {N} samples in "
                   f"{wall:.1f}s wall (gen {wall_gen:.1f}s + "
                   f"parse {time.time()-t_parse:.1f}s) "
                   f"({n/wall:.2f} q/s, {n*N/wall:.1f} samples/s)")
        print(json.dumps({"questions": n, "samples_each": N,
                          "wall_seconds": wall,
                          "questions_per_sec": n/wall if wall > 0 else None}))
    except Exception as e:
        logger.log(f"[pc-sample-vllm] ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
