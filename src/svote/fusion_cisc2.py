#!/usr/bin/env python3
"""Fusion v2: stage-1 (N=20 fullctx) + stage-2 (N'=10 filtered),
both weighted by their own CISC P(True). Pure post-processing.

For each ballot b at stage k with logit L_b, weight = alpha_k * sigmoid(L_b / T_k)
where alpha_k is a per-stage scaling factor (alpha_s1 = 1.0 fixed).
"""
from __future__ import annotations
import argparse, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src" / "eval"))
import eval_llm_baseline as base


def sigmoid(x: float) -> float:
    if x >= 0: z = math.exp(-x); return 1.0/(1.0+z)
    z = math.exp(x); return z/(1.0+z)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_samples", required=True)
    p.add_argument("--stage2_perq", required=True,
                   help="SVote stage-2 per_question.jsonl with samples")
    p.add_argument("--stage1_cisc", required=True)
    p.add_argument("--stage2_cisc", required=True)
    p.add_argument("--T_s1", type=float, default=2.0)
    p.add_argument("--T_s2", type=float, default=2.0)
    p.add_argument("--alpha_s2", type=float, default=1.5)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)

    s1 = {}
    with open(args.stage1_samples) as f:
        for line in f:
            if line.strip():
                r = json.loads(line); s1[r["id"]] = r
    s2 = {}
    with open(args.stage2_perq) as f:
        for line in f:
            if line.strip():
                r = json.loads(line); s2[r["id"]] = r
    cisc1 = {}
    with open(args.stage1_cisc) as f:
        for line in f:
            if line.strip():
                r = json.loads(line); cisc1[r["id"]] = r.get("confidences", [])
    cisc2 = {}
    with open(args.stage2_cisc) as f:
        for line in f:
            if line.strip():
                r = json.loads(line); cisc2[r["id"]] = r.get("confidences", [])

    n=0; em_s=af1_s=sf1_s=jf1_s=0.0
    fused_correct=s1_only=s2_only=both_agree=res_s2=res_s1=0
    for qid, s1q in s1.items():
        n += 1
        gold = s1q["gold_answer"]
        gold_n = base.normalize_answer(gold)
        gold_sup = set(s1q["gold_support"])
        ballots = []
        # Stage-1
        logits1 = cisc1.get(qid, [])
        for i, smp in enumerate(s1q.get("samples", [])):
            an = base.normalize_answer(smp.get("pred_answer","") or "")
            if not an: continue
            w = sigmoid(logits1[i]/args.T_s1) if i < len(logits1) else 1.0
            ballots.append({"an":an, "support":list(smp.get("pred_support",[])),
                            "ans_text":smp.get("pred_answer",""), "w":w, "src":"s1"})
        # Stage-2
        s2q = s2.get(qid)
        if s2q:
            logits2 = cisc2.get(qid, [])
            for i, smp in enumerate(s2q.get("samples", [])):
                an = base.normalize_answer(smp.get("pred_answer","") or "")
                if not an: continue
                w = args.alpha_s2 * (sigmoid(logits2[i]/args.T_s2) if i < len(logits2) else 1.0)
                ballots.append({"an":an, "support":list(smp.get("pred_support",[])),
                                "ans_text":smp.get("pred_answer",""), "w":w, "src":"s2"})

        if not ballots: continue
        score = defaultdict(float)
        for b in ballots: score[b["an"]] += b["w"]
        winner = max(score.items(), key=lambda kv: kv[1])[0]
        rep = max([b for b in ballots if b["an"]==winner],
                  key=lambda b: (b["w"], b["src"]=="s2"))
        em = base.answer_em(rep["ans_text"], gold)
        _,_,af1 = base.answer_scores(rep["ans_text"], gold)
        _,_,sf1 = base.support_scores(set(rep["support"]), gold_sup)
        em_s += em; af1_s += af1; sf1_s += sf1; jf1_s += af1*sf1

        s1_w = Counter(b["an"] for b in ballots if b["src"]=="s1").most_common(1)
        s2_w = Counter(b["an"] for b in ballots if b["src"]=="s2").most_common(1)
        s1_winner = s1_w[0][0] if s1_w else None
        s2_winner = s2_w[0][0] if s2_w else None
        if winner==gold_n: fused_correct += 1
        if s1_winner==gold_n: s1_only += 1
        if s2_winner==gold_n: s2_only += 1
        if s1_winner==s2_winner: both_agree += 1
        if s1_winner!=gold_n and winner==gold_n and s2_winner==gold_n: res_s2 += 1
        if s2_winner!=gold_n and winner==gold_n and s1_winner==gold_n: res_s1 += 1

    metrics = {
        "method":"svote_fusion_cisc2",
        "T_s1":args.T_s1, "T_s2":args.T_s2, "alpha_s2":args.alpha_s2,
        "num_questions": n,
        "em": em_s/n, "af1": af1_s/n, "sf1": sf1_s/n, "jf1": jf1_s/n,
        "fused_correct": fused_correct, "s1_only_correct": s1_only,
        "s2_only_correct": s2_only, "stage_winners_agree": both_agree,
        "rescued_by_s2": res_s2, "rescued_by_s1": res_s1,
    }
    with out.open("w") as f: json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
