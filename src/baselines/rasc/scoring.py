"""Sufficiency scoring for RASC.

Implements the paper's logistic regression scorer with the *exact* coefficients
released in `CS_based_early_stopping.py` (custom mode).

  P(correct | features) = sigmoid(b + Σ w_i * f_i)

These coefficients were trained on math reasoning datasets (GSM8K/MATH/AQuA
across multiple LLMs) — we apply them out-of-domain to HotpotQA as a baseline,
then optionally train a fresh LR on our own data for an upper bound.
"""
from __future__ import annotations

import math
from typing import Iterable

# Order MUST match features.FEATURE_ORDER
PAPER_FEATURE_ORDER = (
    "LEN", "QUA_IM", "DIF_IV",
    "SIM_COT_BIGRAM", "SIM_COT_AGG",
    "SIM_AC_BIGRAM", "SIM_AC_AGG",
    "SIM_INPUT", "STEP_COUNT", "STEP_COHERENCE",
)
PAPER_COEFFICIENTS = (
    -0.17887917, -2.47526597, 2.57520725,
    0.68997781, 1.65216567,
    -2.61836719, -0.04469021,
    3.54958297, 0.0, 0.0,
)
PAPER_INTERCEPT = -0.6


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def score_with_coefficients(feature_dicts: Iterable[dict[str, float]],
                            coefficients: Iterable[float],
                            intercept: float,
                            order: tuple[str, ...] = PAPER_FEATURE_ORDER
                            ) -> list[float]:
    coefs = list(coefficients)
    assert len(coefs) == len(order), (
        f"Got {len(coefs)} coefs, expected {len(order)}")
    scores: list[float] = []
    for f in feature_dicts:
        lin = intercept
        for w, name in zip(coefs, order):
            lin += w * float(f.get(name, 0.0))
        scores.append(_sigmoid(lin))
    return scores


def score_paper_coefficients(feature_dicts: Iterable[dict[str, float]]
                             ) -> list[float]:
    return score_with_coefficients(
        feature_dicts, PAPER_COEFFICIENTS, PAPER_INTERCEPT, PAPER_FEATURE_ORDER,
    )
