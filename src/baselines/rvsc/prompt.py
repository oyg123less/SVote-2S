"""Ranked-answer prompt builder for HotpotQA-style multi-hop QA.

Asks the model to output its top-K candidate answers in ranked order, in
addition to the usual reasoning + support fields. The XML schema is a
strict superset of the project's <reason>/<support>/<answer> contract so
that downstream parsing still works for SC (rank-1) baseline.
"""
from __future__ import annotations

DEFAULT_K = 3


def build_ranked_prompt(question: str, context_str: str | None,
                        setting: str, support_unit: str,
                        k: int = DEFAULT_K) -> str:
    answers_block = "\n".join(
        f"<answer{i}>Candidate answer ranked #{i} (most likely correct first).</answer{i}>"
        for i in range(1, k + 1)
    )
    rules = (
        "Rules:\n"
        "1. Always close every XML tag.\n"
        f"2. Output exactly {k} answer tags <answer1>..</answer{k}>, "
        "ordered from MOST LIKELY correct to LEAST LIKELY correct.\n"
        "3. Each candidate must be a different plausible final answer "
        "(short noun phrase, person name, year, etc.). Do NOT repeat the same answer.\n"
        f"4. The final output must end with </answer{k}>.\n"
        "5. Do not output any text before <reason>."
    )

    if setting == "questiononly":
        return (
            "Answer the following question.\n\n"
            f"Question:\n{question}\n\n"
            "You will give a ranked list of your top candidate answers, with the "
            "most likely answer ranked first.\n\n"
            "Please output exactly the following XML fields in this order:\n\n"
            "<reason>Explain the reasoning briefly.</reason>\n"
            "<support>None</support>\n"
            f"{answers_block}\n\n"
            f"{rules}"
        )

    if support_unit == "paragraph":
        unit = "paragraph"
        unit_id_example = "[S3], [S7]"
    else:
        unit = "sentence"
        unit_id_example = "[S3], [S7]"

    return (
        f"You are given a question and a set of context {unit}s.\n\n"
        f"Answer the question using only the provided context.\n"
        f"You must cite the {unit} IDs that directly support your reasoning.\n"
        "You will give a ranked list of your top candidate answers, with the "
        "most likely answer ranked first.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context_str}\n\n"
        "Please output exactly the following XML fields in this order:\n\n"
        f"<reason>Use at most 5 concise reasoning sentences. Cite supporting {unit} IDs when needed.</reason>\n"
        f"<support>List only the supporting {unit} IDs, such as {unit_id_example}.</support>\n"
        f"{answers_block}\n\n"
        f"{rules}"
    )
