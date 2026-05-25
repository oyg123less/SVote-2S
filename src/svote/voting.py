"""SVote weighted support voting + bridging selection."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x); return 1.0 / (1.0 + z)
    z = math.exp(x); return z / (1.0 + z)


def vote_supports(supports: list[set[str]],
                  ans_norm: list[str],
                  ptrue_logits: list[float] | None = None,
                  ptrue_T: float = 1.0,
                  weight_mode: str = "combined",
                  ) -> dict[str, float]:
    """Aggregate support-sentence votes across N ballots.

    weight_mode:
      'uniform'  -> w_i = 1
      'cisc'     -> w_i = sigmoid(L_i / T)            [needs ptrue_logits]
      'cluster'  -> w_i = |C(a_i)| / N
      'combined' -> w_i = cisc * cluster              [needs ptrue_logits]

    Returns: dict[sid] -> raw aggregated score (un-normalized).
    """
    n = len(supports)
    cluster_size = Counter(a for a in ans_norm if a)
    weights: list[float] = []
    for i in range(n):
        a = ans_norm[i]
        cl = cluster_size.get(a, 0) / n if n else 0.0
        cisc = 1.0
        if ptrue_logits is not None and i < len(ptrue_logits):
            cisc = _sigmoid(ptrue_logits[i] / max(1e-6, ptrue_T))
        if weight_mode == "uniform":
            w = 1.0
        elif weight_mode == "cisc":
            w = cisc
        elif weight_mode == "cluster":
            w = cl
        else:  # combined
            w = cisc * cl
        weights.append(w)

    scores: dict[str, float] = defaultdict(float)
    for i, S in enumerate(supports):
        for sid in S:
            scores[sid] += weights[i]
    return dict(scores)


def _sid_to_para(sid: str, sid_to_para_map: dict[str, str]) -> str:
    """Return paragraph id for a sid. Falls back to sid if no mapping."""
    return sid_to_para_map.get(sid, sid)


def select_supports(scores: dict[str, float],
                    sid_to_para_map: dict[str, str],
                    k_min: int = 2,
                    k_max: int = 4,
                    score_thresh_top: float = 0.5,
                    score_thresh_continue: float = 0.5,
                    score_thresh_same_para: float = 0.7,
                    require_bridging: bool = True,
                    ) -> list[str]:
    """Greedy selection with paragraph-bridging constraint.

    Algorithm:
      1. Sort sids by score desc.
      2. Pick top-1 (highest score).
      3. While have < k_max and remaining candidates:
         - prefer a sid from a NEW paragraph if its score >= top * score_thresh_continue
         - else accept same-paragraph sid only if score >= top * score_thresh_same_para
         - skip if score < top * 0.3 (hard cutoff)
      4. If fewer than k_min selected after bridging, relax bridging requirement
         and take top-k_min by raw score.
    """
    if not scores:
        return []
    sorted_sids = sorted(scores.items(), key=lambda kv: -kv[1])
    top_score = sorted_sids[0][1]
    if top_score <= 0:
        return []

    selected: list[str] = []
    selected_paras: set[str] = set()
    min_score = top_score * 0.3

    for sid, sc in sorted_sids:
        if len(selected) >= k_max:
            break
        if sc < min_score:
            break
        para = _sid_to_para(sid, sid_to_para_map)
        norm_score = sc / top_score

        # First sentence always accepted
        if not selected:
            selected.append(sid)
            selected_paras.add(para)
            continue

        # Bridging-favored: accept readily if new paragraph
        if para not in selected_paras:
            if norm_score >= score_thresh_continue:
                selected.append(sid)
                selected_paras.add(para)
            elif len(selected) < k_min and norm_score >= 0.3:
                # need K_min, take with lower threshold
                selected.append(sid)
                selected_paras.add(para)
        else:
            # same paragraph: stricter threshold (HotpotQA usually wants bridge)
            if norm_score >= score_thresh_same_para and not require_bridging:
                selected.append(sid)

    # Fallback: if we got fewer than k_min, relax bridging
    if len(selected) < k_min:
        already = set(selected)
        for sid, sc in sorted_sids:
            if len(selected) >= k_min:
                break
            if sid in already:
                continue
            if sc < min_score:
                break
            selected.append(sid)

    return selected


# ============================================================
# Sid <-> paragraph mapping inferred from flattened context rows
# (each row has 'sid' and optionally 'para_idx' / 'paragraph_id')
# ============================================================

_PARA_NUM_RE = re.compile(r"\d+")


def build_sid_to_para_map(flat_rows: list[dict]) -> dict[str, str]:
    """Build sid -> paragraph identifier mapping.

    HotpotQA-style flat rows are produced by base.adapt_hotpotqa_sample and
    contain a 'para_idx' (or 'title') field per sentence row. We use whichever
    is available; fall back to the first integer in `sid` itself.
    """
    out: dict[str, str] = {}
    for r in flat_rows:
        sid = str(r.get("sid", ""))
        if not sid:
            continue
        pid = r.get("para_idx", None)
        if pid is None:
            pid = r.get("title", None)
        if pid is None:
            pid = r.get("paragraph_id", None)
        if pid is None:
            # fallback: assume one paragraph; not ideal but safe
            pid = "P0"
        out[sid] = str(pid)
    return out
