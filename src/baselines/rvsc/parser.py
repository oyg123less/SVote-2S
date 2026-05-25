"""Parse a ranked-answer model output.

Expected format (relaxed):
  <reason>...</reason>
  <support>[S?], [S?]</support>
  <answer1>...</answer1>
  <answer2>...</answer2>
  <answer3>...</answer3>

Returns:
  reason_raw    (str)
  support_set   (set[str])  — same as base.parse_model_output
  ranked_answers list[str] of length up to K (only non-empty ones)
  format_ok     (bool)
  parse_error_type (str)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "eval"))
import eval_llm_baseline as base  # type: ignore  # noqa: E402

_REASON_RE = re.compile(r"<reason>(.*?)</reason>", re.DOTALL | re.IGNORECASE)
_SUPPORT_RE = re.compile(r"<support>(.*?)</support>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer(\d+)>(.*?)</answer\1>", re.DOTALL | re.IGNORECASE)


def parse_ranked_output(raw: str, k: int = 3) -> dict[str, Any]:
    rm = _REASON_RE.search(raw)
    sm = _SUPPORT_RE.search(raw)
    reason = rm.group(1).strip() if rm else ""

    if sm:
        support_block = sm.group(1)
        support_raw = base.extract_support_ids(support_block)
    else:
        support_raw = base.extract_support_ids(raw)

    answers: list[tuple[int, str]] = []
    for m in _ANSWER_RE.finditer(raw):
        idx = int(m.group(1))
        text = m.group(2).strip()
        if 1 <= idx <= k:
            answers.append((idx, text))
    answers.sort(key=lambda kv: kv[0])
    # Build ordered list, fill missing with ""
    ranked: list[str] = []
    seen_ranks: set[int] = set()
    for r, t in answers:
        if r in seen_ranks:
            continue
        seen_ranks.add(r)
        ranked.append(t)

    # Fallback: if no <answerN> tags, try to find a single <answer> tag and
    # use it as rank-1.
    if not ranked:
        m = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL | re.IGNORECASE)
        if m:
            ranked = [m.group(1).strip()]

    fmt_ok = bool(rm and sm and len(ranked) >= 1)
    fmt_strict = bool(rm and sm and len(ranked) == k and
                      all(t for t in ranked))
    perr = "strict_ok" if fmt_strict else (
        "missing_some_ranks" if (rm and sm and ranked) else "malformed"
    )
    return {
        "reason": reason,
        "support_raw": support_raw,
        "ranked_answers": ranked,
        "format_ok": fmt_ok,
        "format_ok_strict": fmt_strict,
        "parse_error_type": perr,
    }
