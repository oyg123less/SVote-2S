"""Per-sample feature extraction for RASC.

Reproduces the 10 features defined in the original RASC paper
(`CS_feature_extractor.py`) but operates on the project's per-sample JSON
schema (samples produced by `sample_pc_score_vllm.py`):

  sample = {
    'k': int,
    'raw': str,               # full <reason><support><answer> text
    'pred_answer': str,       # parsed final answer
    'parse_error_type': str,
    ...
  }

The 10 features (ordered to match paper coefficients):
  LEN             — length category (1 if reason has > 4 sentences else 0)
  QUA_IM          — error-admitting regex hit (1/0)
  DIF_IV          — parsing-error flag (1 if parse_error_type != strict_ok)
  SIM_COT_BIGRAM  — jaccard sim between current CoT and the PREVIOUS one
  SIM_COT_AGG     — jaccard sim between current CoT and concatenation of all previous
  SIM_AC_BIGRAM   — 1 if current normalized answer == previous one
  SIM_AC_AGG      — 1 if current answer == most-frequent of previous answers
  SIM_INPUT       — jaccard sim between current CoT and the question
  STEP_COUNT      — number of "Step N:" markers (typically 0 for HotpotQA CoT)
  STEP_COHERENCE  — average jaccard sim between consecutive sentences in reason
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "eval"))
import eval_llm_baseline as base  # noqa: E402

ERROR_ADMIT_RE = re.compile(
    r"(a mistake)|(an error)|(not solvable)|(not enough information)|(apologize)",
    re.IGNORECASE,
)
STEP_RE = re.compile(r"Step\s+\d+\s*:")
REASON_RE = re.compile(r"<reason>(.*?)</reason>", re.IGNORECASE | re.DOTALL)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text or "")]


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:])) if len(tokens) >= 2 else set()


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _bigram_jaccard(text_a: str, text_b: str) -> float:
    return _jaccard(_bigrams(_tokens(text_a)), _bigrams(_tokens(text_b)))


def _extract_reason(raw: str) -> str:
    m = REASON_RE.search(raw or "")
    return m.group(1).strip() if m else (raw or "")


def _sentence_count(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    parts = [p for p in SENT_SPLIT_RE.split(text) if p.strip()]
    return len(parts)


def _step_count(text: str) -> int:
    return len(STEP_RE.findall(text or ""))


def _step_coherence(text: str) -> float:
    """Mean jaccard (word level) between consecutive Step blocks; if no
    explicit steps, use consecutive sentences."""
    steps = STEP_RE.split(text or "")[1:]  # drop preamble
    if len(steps) < 2:
        steps = [p for p in SENT_SPLIT_RE.split(text or "") if p.strip()]
    if len(steps) < 2:
        return 0.0
    scores: list[float] = []
    for i in range(len(steps) - 1):
        scores.append(_jaccard(set(_tokens(steps[i])), set(_tokens(steps[i + 1]))))
    return sum(scores) / len(scores)


FEATURE_ORDER = (
    "LEN", "QUA_IM", "DIF_IV",
    "SIM_COT_BIGRAM", "SIM_COT_AGG",
    "SIM_AC_BIGRAM", "SIM_AC_AGG",
    "SIM_INPUT", "STEP_COUNT", "STEP_COHERENCE",
)


def extract_features_per_question(question: str,
                                  samples: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Return one dict-of-features per sample (in input order).

    Sequential features (BIGRAM / AGG) use *order* in `samples` as the sampling
    timeline — this matches the paper's setup where N=40 CoTs are sampled one
    after another.
    """
    n = len(samples)
    if n == 0:
        return []

    # Pre-compute per-sample primitives
    reasons = [_extract_reason(s.get("raw", "")) for s in samples]
    raw_norm_ans = [base.normalize_answer(s.get("pred_answer", "") or "") for s in samples]
    qa_tokens = set(_tokens(question))

    feats: list[dict[str, float]] = []
    for i, s in enumerate(samples):
        r_i = reasons[i]
        # LEN: > 4 sentences (matches the original >5 sentences-2 rule loosely)
        sc = _sentence_count(r_i)
        len_bin = 1 if sc > 4 else 0

        # QUA_IM
        qua_im = 1 if ERROR_ADMIT_RE.search(r_i) else 0

        # DIF_IV: parsing-error flag
        dif_iv = 0 if s.get("parse_error_type") == "strict_ok" else 1

        # SIM_COT_BIGRAM: bigram jaccard with previous reason
        if i == 0:
            sim_cot_bi = 0.0
        else:
            sim_cot_bi = _bigram_jaccard(r_i, reasons[i - 1])

        # SIM_COT_AGG: bigram jaccard with concatenation of all previous reasons
        if i == 0:
            sim_cot_agg = 0.0
        else:
            agg_prev = " ".join(reasons[:i])
            sim_cot_agg = _bigram_jaccard(r_i, agg_prev)

        # SIM_AC_BIGRAM: 1 if current answer == previous answer
        if i == 0:
            sim_ac_bi = 0
        else:
            sim_ac_bi = 1 if raw_norm_ans[i] and raw_norm_ans[i] == raw_norm_ans[i - 1] else 0

        # SIM_AC_AGG: 1 if current answer == most-frequent previous answer
        if i == 0:
            sim_ac_agg = 0
        else:
            counter = Counter(a for a in raw_norm_ans[:i] if a)
            if counter:
                most_common, _ = counter.most_common(1)[0]
                sim_ac_agg = 1 if raw_norm_ans[i] == most_common else 0
            else:
                sim_ac_agg = 0

        # SIM_INPUT: token jaccard between reason and question
        sim_input = _jaccard(set(_tokens(r_i)), qa_tokens)

        # STEP_COUNT
        step_cnt = _step_count(r_i)

        # STEP_COHERENCE
        step_coh = _step_coherence(r_i)

        feats.append({
            "LEN": float(len_bin),
            "QUA_IM": float(qua_im),
            "DIF_IV": float(dif_iv),
            "SIM_COT_BIGRAM": float(sim_cot_bi),
            "SIM_COT_AGG": float(sim_cot_agg),
            "SIM_AC_BIGRAM": float(sim_ac_bi),
            "SIM_AC_AGG": float(sim_ac_agg),
            "SIM_INPUT": float(sim_input),
            "STEP_COUNT": float(step_cnt),
            "STEP_COHERENCE": float(step_coh),
        })
    return feats
