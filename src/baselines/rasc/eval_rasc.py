#!/usr/bin/env python3
"""Reasoning-Aware Self-Consistency (RASC) evaluator on HotpotQA.

Reuses the *existing* `samples.jsonl` produced by the SC sampling pass
(7405 questions × N=20 samples). No new LLM calls — pure CPU post-processing:
  1. Extract 10 features per sample (length / error / similarity etc.).
  2. Score each sample with the paper's logistic-regression coefficients.
  3. Apply Algorithm 1 (high-quality-buffer early stopping) for a grid of
     (threshold T, buffer capacity N).
  4. Report EM/A-F1/S-F1/J-F1 and average #samples used per question.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "eval"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import eval_llm_baseline as base  # noqa: E402

from baselines.rasc.features import extract_features_per_question  # noqa: E402
from baselines.rasc.scoring import (  # noqa: E402
    score_paper_coefficients, score_with_coefficients,
    PAPER_FEATURE_ORDER,
)
from baselines.rasc.voting import rasc_decide, sc_decide  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--samples_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7",
                   help="Comma-separated sufficiency thresholds T to sweep")
    p.add_argument("--buffer_sizes", default="3,5,7,10",
                   help="Comma-separated buffer capacities N to sweep")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _norm(s: str) -> str:
    return base.normalize_answer(s) if s else ""


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    perq_path = output_dir / "per_question_rasc.jsonl"
    log_path = output_dir / "run.log"
    if not args.overwrite and metrics_path.exists():
        raise FileExistsError(f"{metrics_path} exists; use --overwrite")
    logger = base.RunLogger(log_path)
    start_iso = datetime.now().isoformat()

    Ts = [float(t) for t in args.thresholds.split(",") if t.strip()]
    Ns = [int(n) for n in args.buffer_sizes.split(",") if n.strip()]

    samples_path = Path(args.samples_jsonl)
    questions: list[dict[str, Any]] = []
    with samples_path.open() as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    n_q = len(questions)
    logger.log(f"[rasc] loaded {n_q} questions from {samples_path}")

    # ---- Aggregators ----
    sc_agg  = {"em": 0.0, "af1": 0.0, "sf1": 0.0, "jf1": 0.0}
    bon_agg = {"em": 0.0, "af1": 0.0}
    rasc_aggs: dict[tuple[float, int], dict[str, float]] = {
        (T, N): {"em": 0.0, "af1": 0.0, "sf1": 0.0, "jf1": 0.0,
                 "steps": 0.0, "fallback": 0}
        for T in Ts for N in Ns
    }
    # All-samples weighted vote (no stopping)
    all_agg = {"em": 0.0, "af1": 0.0, "sf1": 0.0, "jf1": 0.0}

    pf = perq_path.open("w", encoding="utf-8")

    for q in questions:
        samples = q["samples"]
        if not samples:
            continue
        gold = q["gold_answer"]
        gold_sup = set(q.get("gold_support", []))

        # Feature extraction + scoring
        feats = extract_features_per_question(q["question"], samples)
        scores = score_paper_coefficients(feats)
        ans_norm = [_norm(s.get("pred_answer", "")) for s in samples]

        # ---- SC (uniform majority on all N) ----
        sc_winner = sc_decide(ans_norm)
        sc_repr = next((i for i, a in enumerate(ans_norm) if a == sc_winner), 0)
        sc_em  = samples[sc_repr]["em"]
        sc_af1 = samples[sc_repr]["answer_f1"]
        sc_sf1 = samples[sc_repr]["support_f1"]
        sc_jf1 = sc_af1 * sc_sf1
        sc_agg["em"]  += sc_em
        sc_agg["af1"] += sc_af1
        sc_agg["sf1"] += sc_sf1
        sc_agg["jf1"] += sc_jf1

        # ---- BoN oracle ----
        bon_em  = max(s["em"] for s in samples)
        bon_af1 = max(s["answer_f1"] for s in samples)
        bon_agg["em"]  += bon_em
        bon_agg["af1"] += bon_af1

        # ---- RASC weighted vote over ALL samples (no stopping) ----
        from collections import Counter as _C
        all_votes: _C = _C()
        for i, a in enumerate(ans_norm):
            if a:
                all_votes[a] += scores[i]
        if all_votes:
            all_winner = max(all_votes.items(), key=lambda kv: kv[1])[0]
            supporting = [i for i, a in enumerate(ans_norm) if a == all_winner]
            all_repr = max(supporting, key=lambda i: scores[i])
            all_agg["em"]  += samples[all_repr]["em"]
            all_agg["af1"] += samples[all_repr]["answer_f1"]
            all_agg["sf1"] += samples[all_repr]["support_f1"]
            all_agg["jf1"] += samples[all_repr]["answer_f1"] * samples[all_repr]["support_f1"]

        # ---- RASC with early-stopping over grid ----
        per_q_rasc = {}
        for T in Ts:
            for N in Ns:
                d = rasc_decide(ans_norm, scores, threshold=T, buffer_size=N)
                if d["winner_norm"] and d["repr_index"] >= 0:
                    r = samples[d["repr_index"]]
                    em, af1, sf1 = r["em"], r["answer_f1"], r["support_f1"]
                else:
                    em = af1 = sf1 = 0
                jf1 = af1 * sf1
                rasc_aggs[(T, N)]["em"]    += em
                rasc_aggs[(T, N)]["af1"]   += af1
                rasc_aggs[(T, N)]["sf1"]   += sf1
                rasc_aggs[(T, N)]["jf1"]   += jf1
                rasc_aggs[(T, N)]["steps"] += d["steps_used"]
                rasc_aggs[(T, N)]["fallback"] += int(d["fallback_used"])
                per_q_rasc[f"T={T}_N={N}"] = {
                    "em": em, "af1": af1, "sf1": sf1, "jf1": jf1,
                    "steps_used": d["steps_used"],
                    "stop_reason": d["stop_reason"],
                    "fallback_used": d["fallback_used"],
                    "winner_norm": d["winner_norm"],
                    "repr_index": d["repr_index"],
                    "buffer_size": len(d["buffer"]),
                }

        pf.write(json.dumps({
            "id": q["id"], "gold": gold, "scores": scores,
            "answers_norm": ans_norm, "sc_winner": sc_winner,
            "sc_em": sc_em, "bon_em": bon_em,
            "rasc": per_q_rasc,
        }, ensure_ascii=False) + "\n")
    pf.close()

    def avg(d, denom=n_q):
        return {k: (v / denom if k != "fallback" else v) for k, v in d.items()}

    metrics = {
        "method": "rasc",
        "samples_jsonl": str(samples_path),
        "num_questions": n_q,
        "n_samples_per_q": questions[0]["n_samples"] if questions else 0,
        "feature_order": list(PAPER_FEATURE_ORDER),
        "sc":   avg(sc_agg),
        "bon":  avg(bon_agg),
        "rasc_all_weighted": avg(all_agg),
        "rasc_by_T_N": {
            f"T={T}_N={N}": {
                "em": rasc_aggs[(T, N)]["em"] / n_q,
                "af1": rasc_aggs[(T, N)]["af1"] / n_q,
                "sf1": rasc_aggs[(T, N)]["sf1"] / n_q,
                "jf1": rasc_aggs[(T, N)]["jf1"] / n_q,
                "avg_steps_used": rasc_aggs[(T, N)]["steps"] / n_q,
                "fallback_count": rasc_aggs[(T, N)]["fallback"],
                "fallback_rate":  rasc_aggs[(T, N)]["fallback"] / n_q,
            }
            for T in Ts for N in Ns
        },
        "start_time": start_iso,
    }
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Print summary
    logger.log("[rasc] === summary ===")
    logger.log(f"  SC  EM={metrics['sc']['em']:.4f}  A-F1={metrics['sc']['af1']:.4f}  S-F1={metrics['sc']['sf1']:.4f}  J-F1={metrics['sc']['jf1']:.4f}")
    logger.log(f"  BoN EM={metrics['bon']['em']:.4f}  A-F1={metrics['bon']['af1']:.4f}")
    logger.log(f"  RASC-all (no stop) EM={metrics['rasc_all_weighted']['em']:.4f}  A-F1={metrics['rasc_all_weighted']['af1']:.4f}")
    logger.log("  --- RASC early-stopping grid ---")
    for T in Ts:
        for N in Ns:
            d = metrics["rasc_by_T_N"][f"T={T}_N={N}"]
            logger.log(
                f"    T={T}  N={N:<2}  EM={d['em']:.4f} ({d['em']-metrics['sc']['em']:+.4f})  "
                f"A-F1={d['af1']:.4f}  steps_avg={d['avg_steps_used']:.2f}  "
                f"fallback={d['fallback_rate']:.2%}"
            )
    logger.close()


if __name__ == "__main__":
    main()
