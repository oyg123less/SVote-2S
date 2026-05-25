"""RASC voting / early stopping — ported from `CS_based_early_stopping.py`.

Algorithm 1 of the RASC paper:
  Iterate samples in order. Each sample has a sufficiency score s_k ∈ [0,1].
    - If s_k >= T, add to "high-quality buffer".
    - Stop when buffer reaches capacity N.
  Final answer = argmax_a Σ_{k in buffer} s_k * 1[answer_k == a].
  Best rationale = highest-score buffered sample whose answer matches winner.
  If buffer is empty at the end, fall back to weighted vote over ALL samples.
"""
from __future__ import annotations

from collections import Counter
from typing import Any


def sc_decide(answers_norm: list[str]) -> str:
    """Plain uniform majority vote (no scores)."""
    valid = [a for a in answers_norm if a]
    if not valid:
        return ""
    return Counter(valid).most_common(1)[0][0]


def rasc_decide(answers_norm: list[str],
                scores: list[float],
                threshold: float,
                buffer_size: int,
                ) -> dict[str, Any]:
    """Returns dict with:
        winner_norm   : str, final voted answer (normalized)
        repr_index    : int, index of representative sample (best rationale)
        buffer        : list[int], indices of samples in buffer
        steps_used    : int, number of samples processed before stopping
        stop_reason   : 'buffer_full' or 'exhausted'
        fallback_used : bool
    """
    assert len(answers_norm) == len(scores)
    n = len(answers_norm)
    buffer: list[int] = []
    steps = 0
    for i in range(n):
        steps = i + 1
        if scores[i] >= threshold:
            buffer.append(i)
            if len(buffer) >= buffer_size:
                break
    stop_reason = "buffer_full" if len(buffer) >= buffer_size else "exhausted"

    # Use buffer if non-empty, else fallback to all samples
    fallback = False
    candidates = buffer
    if not candidates:
        candidates = list(range(n))
        fallback = True

    # Weighted vote
    votes: Counter = Counter()
    for j in candidates:
        a = answers_norm[j]
        if not a:
            continue
        votes[a] += scores[j]

    if not votes:
        return {"winner_norm": "", "repr_index": -1, "buffer": buffer,
                "steps_used": steps, "stop_reason": stop_reason,
                "fallback_used": fallback}

    winner = max(votes.items(), key=lambda kv: kv[1])[0]
    supporting = [j for j in candidates if answers_norm[j] == winner]
    repr_idx = max(supporting, key=lambda j: scores[j]) if supporting else candidates[0]

    return {
        "winner_norm": winner,
        "repr_index": repr_idx,
        "buffer": buffer,
        "steps_used": steps,
        "stop_reason": stop_reason,
        "fallback_used": fallback,
    }
