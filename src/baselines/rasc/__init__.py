"""Reasoning-Aware Self-Consistency (RASC) for free-text QA.

Adapted from Wan et al. 2025 "Reasoning Aware Self-Consistency: Leveraging
Reasoning Paths for Efficient LLM Sampling" (NAACL 2025).
Original repo (math-oriented): RASC-Submission_Code.

We port the *core algorithm* — feature extraction + logistic-regression
sufficiency scoring + early-stopping buffer + weighted majority vote — to
HotpotQA / multi-hop free-text QA, reusing existing `samples.jsonl` produced
by the SC sampling pass (no extra LLM calls).
"""
from .features import extract_features_per_question  # noqa: F401
from .scoring import score_paper_coefficients, score_with_coefficients  # noqa: F401
from .voting import rasc_decide, sc_decide  # noqa: F401
