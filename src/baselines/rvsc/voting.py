"""IRV / Borda / MRR voting — ported and generalized from the original
RankedVotingSC repo (RankedVotingSC/RankBasedSC.py).

Original assumed multiple-choice (single character per slot). We keep the
algorithms identical but accept arbitrary string keys (normalized free-text
answers) so the same code works for HotpotQA / MuSiQue / 2WikiMultihop.

A "ballot" is a list of candidate answer strings ordered from most-preferred
(rank 1) to least-preferred (rank K). Empty / "invalid" tokens are filtered.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

INVALID = "invalid"


def _filter_ballot(ballot: list[str]) -> list[str]:
    """Drop empty / 'invalid' tokens but preserve order and duplicates among
    valid entries (we deduplicate at the ballot level later for IRV)."""
    return [c for c in ballot if c and c != INVALID]


def _dedupe_keep_order(seq: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# --------------------------- IRV --------------------------- #
class IRV:
    """Instant-Runoff Voting.

    Round 1: count each ballot's top choice. If any candidate has a strict
    majority, return it. Otherwise eliminate the *least-voted* candidate and
    re-tally each ballot's highest non-eliminated choice. Repeat.
    """

    def __init__(self, ballots: list[list[str]]):
        self.ballots = [_dedupe_keep_order(_filter_ballot(b)) for b in ballots]
        # Discard fully-empty ballots
        self.ballots = [b for b in self.ballots if b]

    def run(self) -> str:
        if not self.ballots:
            return ""
        # Universe of candidates that ever appeared
        eliminated: set[str] = set()
        while True:
            # Compute current top choice of each ballot
            tops: list[str] = []
            for b in self.ballots:
                pick = next((c for c in b if c not in eliminated), None)
                if pick is not None:
                    tops.append(pick)
            if not tops:
                return ""
            counts = Counter(tops)
            total = sum(counts.values())
            # Strict majority
            for cand, cnt in counts.items():
                if cnt > total / 2:
                    return cand
            # Eliminate the least-voted (ties → drop one with smallest count and
            # lexicographically smallest key for determinism). If only one
            # candidate remains, return it (no majority is possible otherwise).
            if len(counts) == 1:
                return next(iter(counts))
            min_count = min(counts.values())
            losers = [c for c, n in counts.items() if n == min_count]
            losers.sort()
            eliminated.add(losers[0])


# --------------------------- BCV (Borda) --------------------------- #
class BCV:
    """Borda Count Voting.

    For each ballot of length L, give score (L - rank) to the candidate at
    `rank` (rank=0 → score L, rank=1 → L-1, …). Sum across ballots, return
    argmax.

    NOTE: This matches the original repo's implementation:
        scores[c] += len(resp) - i  for i, c in enumerate(resp)
    so a length-3 ballot awards 3/2/1 to ranks 1/2/3.
    """

    def __init__(self, ballots: list[list[str]]):
        self.ballots = [_dedupe_keep_order(_filter_ballot(b)) for b in ballots]
        self.ballots = [b for b in self.ballots if b]

    def run(self) -> str:
        if not self.ballots:
            return ""
        scores: Counter = Counter()
        for b in self.ballots:
            L = len(b)
            for i, c in enumerate(b):
                scores[c] += L - i
        return scores.most_common(1)[0][0]


# --------------------------- MRRV --------------------------- #
class MRRV:
    """Mean Reciprocal Rank Voting.

    For each ballot, the candidate at rank `r` (1-indexed) contributes 1/r to
    its score. Sum across ballots, return argmax.
    """

    def __init__(self, ballots: list[list[str]]):
        self.ballots = [_dedupe_keep_order(_filter_ballot(b)) for b in ballots]
        self.ballots = [b for b in self.ballots if b]

    def run(self) -> str:
        if not self.ballots:
            return ""
        scores: Counter = Counter()
        for b in self.ballots:
            for r, c in enumerate(b, start=1):
                scores[c] += 1.0 / r
        # Counter.most_common is stable; tie-break by first-seen order is OK.
        return max(scores.items(), key=lambda kv: kv[1])[0]


def run_all(ballots: list[list[str]]) -> dict[str, str]:
    """Convenience wrapper. Returns {'irv': ..., 'bcv': ..., 'mrrv': ...}."""
    return {
        "irv": IRV(ballots).run(),
        "bcv": BCV(ballots).run(),
        "mrrv": MRRV(ballots).run(),
    }
