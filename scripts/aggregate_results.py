#!/usr/bin/env python3
"""Aggregate evaluation results from all pipeline stages into a summary table.

Usage:
    python scripts/aggregate_results.py --model qwen2_5_7b_instruct --dataset hotpotqa
    python scripts/aggregate_results.py  # all models x all datasets
"""
from __future__ import annotations

import argparse
import json
import glob
import os
from pathlib import Path
from collections import Counter

PROJECT = Path(__file__).resolve().parents[1]


def load_json(path: str) -> dict | None:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def compute_sc_metrics(samples_path: str) -> dict:
    """Compute SC majority-vote metrics from samples.jsonl."""
    rows = load_jsonl(samples_path)
    if not rows:
        return {}

    n = 0
    em_sum = af1_sum = sf1_sum = jf1_sum = 0.0
    bon_em = 0

    for row in rows:
        samples = row.get("samples", [])
        if not samples:
            continue
        n += 1
        gold = row["gold_answer"]
        gold_sup = set(row.get("gold_support", []))

        # SC majority vote
        ans_counter = Counter()
        for s in samples:
            an = s.get("pred_answer", "").lower().strip()
            if an:
                ans_counter[an] += 1
        if ans_counter:
            winner = ans_counter.most_common(1)[0][0]
            rep = next(s for s in samples
                       if s.get("pred_answer", "").lower().strip() == winner)
        else:
            rep = samples[0]

        em_sum += rep.get("em", 0)
        af1_sum += rep.get("answer_f1", 0)
        sf1_sum += rep.get("support_f1", 0)
        jf1_sum += rep.get("answer_f1", 0) * rep.get("support_f1", 0)

        # BoN oracle
        if any(s.get("em", 0) for s in samples):
            bon_em += 1

    if n == 0:
        return {}
    return {
        "method": "SC",
        "n": n,
        "em": em_sum / n * 100,
        "af1": af1_sum / n * 100,
        "jf1": jf1_sum / n * 100,
        "bon_em": bon_em / n * 100,
    }


def compute_cot_metrics(dir_path: str) -> dict:
    """Compute CoT metrics from sharded output."""
    all_rows = []
    for p in sorted(glob.glob(f"{dir_path}/shard*/*/fullctx/samples.jsonl")):
        all_rows.extend(load_jsonl(p))
    if not all_rows:
        return {}

    n = len(all_rows)
    em_sum = af1_sum = jf1_sum = 0.0
    for row in all_rows:
        samples = row.get("samples", [])
        if samples:
            s = samples[0]
            em_sum += s.get("em", 0)
            af1_sum += s.get("answer_f1", 0)
            jf1_sum += s.get("answer_f1", 0) * s.get("support_f1", 0)
    return {
        "method": "CoT",
        "n": n,
        "em": em_sum / n * 100,
        "af1": af1_sum / n * 100,
        "jf1": jf1_sum / n * 100,
    }


def get_svote_metrics(dir_path: str) -> dict:
    """Weighted average from SVote shards."""
    total_n = 0
    em_sum = af1_sum = jf1_sum = 0.0
    for mp in sorted(glob.glob(f"{dir_path}/shard*/metrics.json")):
        m = load_json(mp)
        if m:
            nn = m.get("num_evaluated", 0)
            total_n += nn
            em_sum += m["em"] * nn
            af1_sum += m["af1"] * nn
            jf1_sum += m["jf1"] * nn
    if total_n == 0:
        return {}
    return {
        "method": "SVote (input-vote)",
        "n": total_n,
        "em": em_sum / total_n * 100,
        "af1": af1_sum / total_n * 100,
        "jf1": jf1_sum / total_n * 100,
    }


def summarize(model: str, dataset: str) -> list[dict]:
    base = f"{PROJECT}/outputs/{model}/{dataset}"
    results = []

    # CoT
    r = compute_cot_metrics(f"{base}/cot")
    if r:
        results.append(r)

    # SC
    r = compute_sc_metrics(f"{base}/stage0_samples/merged/samples.jsonl")
    if r:
        sc_em = r["em"]
        results.append(r)
        results.append({"method": "Oracle (BoN)", "n": r["n"],
                        "em": r["bon_em"], "af1": "-", "jf1": "-"})
    else:
        sc_em = None

    # CISC
    for sp in sorted(glob.glob(f"{base}/cisc_stage0/shard*/metrics.json")):
        m = load_json(sp)
        if m and "cisc_by_T" in m:
            cisc_t2 = m["cisc_by_T"].get("T=2.0", {})
            if cisc_t2:
                results.append({
                    "method": "CISC (T=2.0)",
                    "n": m.get("num_evaluated", 0),
                    "em": cisc_t2.get("em", 0) * 100,
                    "af1": cisc_t2.get("af1", 0) * 100,
                    "jf1": cisc_t2.get("jf1", 0) * 100,
                })
            break

    # RASC
    rasc = load_json(f"{base}/rasc/metrics.json")
    if rasc:
        best = rasc.get("best", {})
        results.append({
            "method": "RASC",
            "n": rasc.get("num_questions", 0),
            "em": best.get("em", 0) * 100,
            "af1": best.get("af1", 0) * 100,
            "jf1": best.get("jf1", 0) * 100,
        })

    # RVSC
    for mp in sorted(glob.glob(f"{base}/rvsc/shard*/metrics.json"))[:1]:
        m = load_json(mp)
        if m and "methods" in m:
            best_method = max(m["methods"].items(),
                              key=lambda kv: kv[1].get("em", 0))
            results.append({
                "method": f"RVSC ({best_method[0]})",
                "n": m.get("num_evaluated", 0),
                "em": best_method[1].get("em", 0) * 100,
                "af1": best_method[1].get("af1", 0) * 100,
                "jf1": best_method[1].get("jf1", 0) * 100,
            })

    # SVote (input-vote only)
    r = get_svote_metrics(f"{base}/svote_stage12")
    if r:
        results.append(r)

    # SVote-2S Fusion
    fusion = load_json(f"{base}/fusion/best.json")
    if fusion:
        results.append({
            "method": "SVote-2S",
            "n": fusion.get("num_questions", 0),
            "em": fusion.get("em", 0) * 100,
            "af1": fusion.get("af1", 0) * 100,
            "jf1": fusion.get("jf1", 0) * 100,
        })

    # Add ΔSC
    if sc_em is not None:
        for r in results:
            if isinstance(r.get("em"), (int, float)):
                r["delta_sc"] = r["em"] - sc_em

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    models = [args.model] if args.model else [
        "qwen2_5_7b_instruct", "gemma2_9b_it", "llama3_1_8b_instruct"]
    datasets = [args.dataset] if args.dataset else [
        "hotpotqa", "musique", "2wiki_dev12576"]

    for model in models:
        for dataset in datasets:
            results = summarize(model, dataset)
            if not results:
                continue
            print(f"\n{'='*70}")
            print(f" {model} / {dataset}")
            print(f"{'='*70}")
            print(f"{'Method':<25} {'EM':>7} {'AF1':>7} {'JF1':>7} {'ΔSC':>7}")
            print("-" * 55)
            for r in results:
                em = f"{r['em']:.2f}" if isinstance(r['em'], float) else str(r['em'])
                af1 = f"{r['af1']:.2f}" if isinstance(r['af1'], float) else str(r['af1'])
                jf1 = f"{r['jf1']:.2f}" if isinstance(r['jf1'], float) else str(r['jf1'])
                dsc = f"{r.get('delta_sc', 0):+.2f}" if 'delta_sc' in r else "-"
                print(f"{r['method']:<25} {em:>7} {af1:>7} {jf1:>7} {dsc:>7}")


if __name__ == "__main__":
    main()
