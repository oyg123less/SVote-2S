"""Ranked-Voting Self-Consistency (RVSC) for open-ended QA.

Adapted from Wang et al. 2025 "Ranked Voting based Self-Consistency of Large
Language Models" (ACL 2025 Findings).

Original repo (multiple-choice oriented):
  https://github.com/szu-tera/RankedVotingSC

This package generalizes IRV / Borda Count / MRR voting to **free-text answers**
(HotpotQA-style) by:
  1. Prompting the model to emit a top-K ranked list of candidate answers
     instead of a single answer.
  2. Normalizing each candidate (lowercase, strip articles/punct) so that
     surface variants collapse to the same equivalence class for voting.
  3. Running IRV / BCV / MRRV over the N ballots × K ranks.
"""

from .voting import IRV, BCV, MRRV, run_all  # noqa: F401
