#!/usr/bin/env python3
import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
try:
    import yaml
except ModuleNotFoundError:
    print("PyYAML is required. Please run: pip install pyyaml", file=sys.stderr)
    raise
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
SUPPORT_TAG_PATTERN = re.compile(r"<support>\s*(.*?)\s*</support>", re.IGNORECASE | re.DOTALL)
REASON_TAG_PATTERN = re.compile(r"<reason>\s*(.*?)\s*</reason>", re.IGNORECASE | re.DOTALL)
SUPPORT_ID_PATTERN = re.compile(r"\bS\d+\b", re.IGNORECASE)
OPEN_ANSWER_PATTERN = re.compile(r"<answer>\s*(.*)$", re.IGNORECASE | re.DOTALL)
OPEN_SUPPORT_PATTERN = re.compile(r"<support>\s*(.*?)(?:<answer>|$)", re.IGNORECASE | re.DOTALL)
ARTICLES_PATTERN = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
SID_NUMBER_PATTERN = re.compile(r"S(\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified LLM baseline evaluator")
    parser.add_argument("--dataset", required=True, help="Dataset key in YAML config")
    parser.add_argument("--model", required=True, help="Model key in YAML config")
    parser.add_argument(
        "--setting",
        required=True,
        choices=["fullctx", "goldonly", "nogold", "questiononly"],
        help="Evaluation setting",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate first N samples")
    parser.add_argument("--run_id", required=True, help="Run identifier for output isolation")
    parser.add_argument("--config", default="config/eval_baseline.yaml", help="Path to YAML config")
    parser.add_argument("--output_root", default="outputs/eval", help="Root output directory")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0")
    parser.add_argument("--tau", type=float, default=0.5, help="Support-F1 threshold for OCU")
    parser.add_argument("--log_every", type=int, default=5, help="Print live status every N samples")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing run outputs")
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return cfg


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = ARTICLES_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def answer_em(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def answer_scores(pred: str, gold: str) -> tuple[float, float, float]:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    common = pred_counter & gold_counter
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def support_scores(pred_support: set[str], gold_support: set[str]) -> tuple[float, float, float]:
    hit = pred_support & gold_support
    precision = len(hit) / len(pred_support) if pred_support else 0.0
    recall = len(hit) / len(gold_support) if gold_support else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def joint_scores(
    answer_precision: float,
    answer_recall: float,
    support_precision: float,
    support_recall: float,
) -> tuple[float, float, float]:
    joint_precision = answer_precision * support_precision
    joint_recall = answer_recall * support_recall
    if joint_precision + joint_recall == 0:
        return joint_precision, joint_recall, 0.0
    joint_f1 = 2 * joint_precision * joint_recall / (joint_precision + joint_recall)
    return joint_precision, joint_recall, joint_f1


def _normalize_support_id(token: str) -> str | None:
    match = SID_NUMBER_PATTERN.search(token)
    if not match:
        return None
    return f"S{int(match.group(1))}"


def _sorted_support_ids(ids: set[str]) -> list[str]:
    return sorted(ids, key=lambda x: int(x[1:]))


def extract_support_ids(text: str) -> set[str]:
    ids: set[str] = set()
    for raw_id in SUPPORT_ID_PATTERN.findall(text):
        norm_id = _normalize_support_id(raw_id)
        if norm_id is not None:
            ids.add(norm_id)
    return ids


def parse_model_output(text: str) -> tuple[str, set[str], str, bool, bool, str]:
    reason_match = REASON_TAG_PATTERN.search(text)
    support_match = SUPPORT_TAG_PATTERN.search(text)
    answer_match = ANSWER_TAG_PATTERN.search(text)
    open_support_match = OPEN_SUPPORT_PATTERN.search(text)
    open_answer_match = OPEN_ANSWER_PATTERN.search(text)

    reason_text = reason_match.group(1).strip() if reason_match else ""
    support_text = ""
    answer_text = ""
    parse_error_type = "missing_answer"

    if support_match:
        support_text = support_match.group(1).strip()
    elif open_support_match:
        support_text = open_support_match.group(1).strip()
        parse_error_type = "unclosed_support"

    if answer_match:
        answer_text = answer_match.group(1).strip()
    elif open_answer_match:
        answer_text = open_answer_match.group(1).strip()
        parse_error_type = "unclosed_answer"

    if not answer_text:
        fallback = re.findall(r"answer\s*[:：]\s*(.+)", text, flags=re.IGNORECASE)
        if fallback:
            answer_text = fallback[-1].strip()
            parse_error_type = "fallback_answer_line"
        else:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            answer_text = lines[-1] if lines else ""
            parse_error_type = "fallback_last_line" if answer_text else parse_error_type

    if support_text:
        pred_support = extract_support_ids(support_text)
    else:
        pred_support = extract_support_ids(text)

    format_ok_strict = bool(reason_match and support_match and answer_match)
    format_ok = bool(reason_text and support_text and answer_text)

    has_any_tag = bool(
        REASON_TAG_PATTERN.search(text)
        or SUPPORT_TAG_PATTERN.search(text)
        or ANSWER_TAG_PATTERN.search(text)
        or re.search(r"<reason>|<support>|<answer>", text, re.IGNORECASE)
    )

    if format_ok_strict:
        parse_error_type = "strict_ok"
    elif not has_any_tag:
        parse_error_type = "no_tags"
    elif reason_match and not support_text and not answer_text:
        parse_error_type = "reason_only"
    elif not support_text and (reason_text or answer_text):
        parse_error_type = "missing_support_tag"
    elif not answer_text:
        parse_error_type = "missing_answer"

    return reason_text, pred_support, answer_text, format_ok, format_ok_strict, parse_error_type


def build_generation_input(tokenizer: AutoTokenizer, prompt: str) -> str:
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def build_prompt(question: str, context_str: str | None, setting: str, support_unit: str,
                 model_family: str | None = None) -> str:
    if setting == "questiononly":
        return (
            "Answer the following question.\n\n"
            f"Question:\n{question}\n\n"
            "Please output exactly three XML fields in this order:\n\n"
            "<reason>Explain the reasoning briefly.</reason>\n"
            "<support>None</support>\n"
            "<answer>Give the shortest possible final answer.</answer>\n\n"
            "Rules:\n"
            "1. Always close every XML tag.\n"
            "2. The final output must end with </answer>.\n"
            "3. Do not output any text before <reason> or after </answer>."
        )

    unit = "paragraph" if support_unit == "paragraph" else "sentence"

    # --- Llama-specific: add 1-shot example for better format compliance ---
    if model_family == "llama":
        fewshot_example = (
            "Here is an example of the expected output format:\n\n"
            f"<reason>[{unit[0].upper()}3] states that Alice was born in 1910. "
            f"[{unit[0].upper()}7] states that Bob was born in 1923. "
            f"Therefore Alice was born first.</reason>\n"
            f"<support>[S3], [S7]</support>\n"
            "<answer>Alice</answer>\n\n"
            "---\n\n"
            "Now answer the following question using the same XML format.\n\n"
        )
    else:
        fewshot_example = ""

    return (
        f"You are given a question and a set of context {unit}s.\n\n"
        "Answer the question using only the provided context.\n"
        f"You must cite the {unit} IDs that directly support your reasoning.\n\n"
        f"{fewshot_example}"
        f"Question:\n{question}\n\n"
        f"Context:\n{context_str}\n\n"
        "Please output exactly three XML fields in this order:\n\n"
        f"<reason>Use at most 5 concise reasoning sentences. Cite supporting {unit} IDs when needed.</reason>\n"
        f"<support>List only the supporting {unit} IDs, such as [S3], [S7].</support>\n"
        "<answer>Give the shortest possible final answer.</answer>\n\n"
        "Rules:\n"
        "1. Always close every XML tag.\n"
        "2. The final output must end with </answer>.\n"
        "3. Do not output any text before <reason> or after </answer>.\n"
        "4. Keep the reasoning concise.\n"
        "5. Output ONLY the three XML fields. No other text."
    )


def render_context_lines(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(No context provided)"
    return "\n".join(f"[{r['sid']}] {r['title']}: {r['text']}" for r in rows)


def keep_context_by_ids(flattened: list[dict[str, Any]], keep_ids: set[str]) -> list[dict[str, Any]]:
    return [row for row in flattened if row["sid"] in keep_ids]


def drop_context_by_ids(flattened: list[dict[str, Any]], drop_ids: set[str]) -> list[dict[str, Any]]:
    return [row for row in flattened if row["sid"] not in drop_ids]


def _to_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def flatten_sentence_context(context: Any) -> tuple[list[dict[str, Any]], dict[tuple[str, int], str], int]:
    flattened: list[dict[str, Any]] = []
    index_map: dict[tuple[str, int], str] = {}
    bad_context_item_count = 0

    if not isinstance(context, list):
        return flattened, index_map, 0

    sid = 1
    for item in context:
        title = ""
        sentences: Any = None

        if isinstance(item, list) and len(item) >= 2:
            title = str(item[0])
            sentences = item[1]
        elif isinstance(item, dict):
            title = str(item.get("title") or item.get("name") or "")
            sentences = item.get("sentences")
            if sentences is None:
                sentences = item.get("context")
        else:
            bad_context_item_count += 1
            continue

        if isinstance(sentences, str):
            sentences = [sentences]

        if not isinstance(sentences, list):
            bad_context_item_count += 1
            continue

        for local_idx, sent in enumerate(sentences):
            if sent is None:
                bad_context_item_count += 1
                continue
            if not isinstance(sent, str):
                sent = str(sent)

            sid_token = f"S{sid}"
            flattened.append(
                {
                    "sid": sid_token,
                    "title": title,
                    "text": sent,
                    "local_idx": local_idx,
                }
            )
            index_map[(title, local_idx)] = sid_token
            sid += 1

    return flattened, index_map, bad_context_item_count


def map_supporting_facts(
    supporting_facts: Any,
    index_map: dict[tuple[str, int], str],
) -> tuple[set[str], int, int]:
    gold_support: set[str] = set()
    missing = 0
    bad_support_fact_count = 0

    if not isinstance(supporting_facts, list):
        return gold_support, missing, 0

    for sf in supporting_facts:
        title: str | None = None
        sent_idx: int | None = None

        if isinstance(sf, list) and len(sf) >= 2:
            title = str(sf[0])
            sent_idx = _to_int(sf[1])
        elif isinstance(sf, dict):
            raw_title = sf.get("title") or sf.get("page")
            if raw_title is not None:
                title = str(raw_title)
            sent_idx = _to_int(sf.get("sent_id"))
            if sent_idx is None:
                sent_idx = _to_int(sf.get("sentence_id"))
            if sent_idx is None:
                sent_idx = _to_int(sf.get("index"))
        else:
            bad_support_fact_count += 1
            continue

        if title is None or sent_idx is None:
            bad_support_fact_count += 1
            continue

        sid = index_map.get((title, sent_idx))
        if sid is None:
            missing += 1
            continue
        gold_support.add(sid)

    return gold_support, missing, bad_support_fact_count


def extract_2wiki_supporting_facts(sample: dict[str, Any], diagnostics: dict[str, int]) -> list[Any]:
    sf = sample.get("supporting_facts")
    if sf is None:
        sf = sample.get("supportingFacts")
    if sf is not None:
        if isinstance(sf, list):
            return sf
        diagnostics["bad_supporting_facts_count"] += 1
        return []

    ev = sample.get("evidences")
    if ev is None:
        ev = sample.get("evidence")
    if ev is None:
        return []
    if not isinstance(ev, list):
        diagnostics["bad_supporting_facts_count"] += 1
        return []

    recovered: list[Any] = []

    def try_add_pair(obj: Any) -> bool:
        if isinstance(obj, list) and len(obj) >= 2:
            title = obj[0]
            idx = _to_int(obj[1])
            if isinstance(title, str) and idx is not None:
                recovered.append([title, idx])
                return True
        if isinstance(obj, dict):
            title = obj.get("title") or obj.get("page")
            idx = _to_int(obj.get("sent_id"))
            if idx is None:
                idx = _to_int(obj.get("sentence_id"))
            if idx is None:
                idx = _to_int(obj.get("index"))
            if title is not None and idx is not None:
                recovered.append([str(title), idx])
                return True
        return False

    for item in ev:
        if try_add_pair(item):
            continue
        if isinstance(item, list):
            added = False
            for sub in item:
                if try_add_pair(sub):
                    added = True
            if not added:
                diagnostics["bad_support_fact_count"] += 1
        else:
            diagnostics["bad_support_fact_count"] += 1

    return recovered


def init_diagnostics() -> dict[str, int]:
    return {
        "missing_gold_support_mappings": 0,
        "empty_paragraph_count": 0,
        "num_no_gold_support": 0,
        "bad_context_count": 0,
        "bad_context_item_count": 0,
        "bad_supporting_facts_count": 0,
        "bad_support_fact_count": 0,
        "parsing_failure_count": 0,
    }


def adapt_hotpotqa_sample(sample: Any, idx: int, id_field: str, diagnostics: dict[str, int]) -> dict[str, Any]:
    if not isinstance(sample, dict):
        diagnostics["bad_context_count"] += 1
        sample = {}

    sample_id = sample.get(id_field) or sample.get("id") or sample.get("_id") or str(idx)
    question = str(sample.get("question", ""))
    gold_answer = str(sample.get("answer", ""))

    context = sample.get("context", [])
    if not isinstance(context, list):
        diagnostics["bad_context_count"] += 1
        context = []

    supporting_facts = sample.get("supporting_facts", [])
    if not isinstance(supporting_facts, list):
        diagnostics["bad_supporting_facts_count"] += 1
        supporting_facts = []

    flattened, index_map, bad_ctx_items = flatten_sentence_context(context)
    diagnostics["bad_context_item_count"] += bad_ctx_items

    gold_support, missing, bad_sf = map_supporting_facts(supporting_facts, index_map)
    diagnostics["missing_gold_support_mappings"] += missing
    diagnostics["bad_support_fact_count"] += bad_sf

    if len(gold_support) == 0:
        diagnostics["num_no_gold_support"] += 1

    return {
        "id": str(sample_id),
        "question": question,
        "gold_answer": gold_answer,
        "flattened": flattened,
        "gold_support": gold_support,
        "diagnostics": {},
    }


def adapt_musique_sample(sample: Any, idx: int, id_field: str, diagnostics: dict[str, int]) -> dict[str, Any]:
    if not isinstance(sample, dict):
        diagnostics["bad_context_count"] += 1
        sample = {}

    sample_id = sample.get(id_field) or sample.get("_id") or sample.get("id") or str(idx)
    question = str(sample.get("question", ""))
    gold_answer = str(sample.get("answer", ""))

    paragraphs = sample.get("paragraphs", [])
    if not isinstance(paragraphs, list):
        diagnostics["bad_context_count"] += 1
        paragraphs = []

    flattened: list[dict[str, Any]] = []
    gold_support: set[str] = set()

    sid = 1
    for para in paragraphs:
        if not isinstance(para, dict):
            diagnostics["bad_context_item_count"] += 1
            continue

        title = str(para.get("title") or "")
        paragraph_text = (
            para.get("paragraph_text")
            or para.get("text")
            or para.get("paragraph")
            or ""
        )
        if not str(paragraph_text).strip():
            diagnostics["empty_paragraph_count"] += 1

        sid_token = f"S{sid}"
        flattened.append(
            {
                "sid": sid_token,
                "title": title,
                "text": str(paragraph_text),
                "local_idx": _to_int(para.get("idx")),
            }
        )

        if bool(para.get("is_supporting", False)):
            gold_support.add(sid_token)

        sid += 1

    if len(gold_support) == 0:
        diagnostics["num_no_gold_support"] += 1

    return {
        "id": str(sample_id),
        "question": question,
        "gold_answer": gold_answer,
        "flattened": flattened,
        "gold_support": gold_support,
        "diagnostics": {},
    }


def adapt_2wiki_sample(sample: Any, idx: int, id_field: str, diagnostics: dict[str, int]) -> dict[str, Any]:
    if not isinstance(sample, dict):
        diagnostics["bad_context_count"] += 1
        sample = {}

    sample_id = sample.get(id_field) or sample.get("id") or sample.get("_id") or str(idx)
    question = str(sample.get("question", ""))
    gold_answer = str(sample.get("answer", ""))

    context = sample.get("context")
    if context is None:
        context = sample.get("contexts", [])
    if not isinstance(context, list):
        diagnostics["bad_context_count"] += 1
        context = []

    supporting_facts = extract_2wiki_supporting_facts(sample, diagnostics)

    flattened, index_map, bad_ctx_items = flatten_sentence_context(context)
    diagnostics["bad_context_item_count"] += bad_ctx_items

    gold_support, missing, bad_sf = map_supporting_facts(supporting_facts, index_map)
    diagnostics["missing_gold_support_mappings"] += missing
    diagnostics["bad_support_fact_count"] += bad_sf

    if len(gold_support) == 0:
        diagnostics["num_no_gold_support"] += 1

    return {
        "id": str(sample_id),
        "question": question,
        "gold_answer": gold_answer,
        "flattened": flattened,
        "gold_support": gold_support,
        "diagnostics": {},
    }


def load_dataset_samples(path: Path, file_type: str) -> list[Any]:
    if file_type == "json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"JSON dataset must be a list: {path}")
        return data

    if file_type == "jsonl":
        data: list[Any] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL parse error at line {line_no}: {path} ({exc})") from exc
        return data

    raise ValueError(f"Unsupported file_type: {file_type}")


def parse_dtype(dtype_name: str) -> torch.dtype:
    table = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in table:
        raise ValueError(f"Unsupported dtype in config: {dtype_name}")
    return table[dtype_name]


def resolve_dtype_and_device(dtype_name: str, device: str) -> tuple[torch.dtype, str | dict[str, str]]:
    cfg_dtype = parse_dtype(dtype_name)

    if device == "cpu":
        return torch.float32, "cpu"

    if device == "auto":
        if torch.cuda.is_available():
            return cfg_dtype, "auto"
        return torch.float32, "cpu"

    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device cuda requested but CUDA is not available")
        return cfg_dtype, "auto"

    if re.fullmatch(r"cuda:\d+", device):
        if not torch.cuda.is_available():
            raise ValueError(f"--device {device} requested but CUDA is not available")
        idx = int(device.split(":", maxsplit=1)[1])
        visible = torch.cuda.device_count()
        if idx >= visible:
            raise ValueError(
                f"Requested device {device}, but only {visible} CUDA devices are visible. "
                "If CUDA_VISIBLE_DEVICES is set, cuda:0 refers to the first visible GPU."
            )
        return cfg_dtype, {"": device}

    raise ValueError(f"Unsupported --device value: {device}")


def get_input_device(model: AutoModelForCausalLM, device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if re.fullmatch(r"cuda:\d+", device_arg):
        return torch.device(device_arg)
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        self.f = path.open("w", encoding="utf-8")

    def log(self, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        print(line)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    config = load_config(str(config_path))

    models = config.get("models", {})
    datasets = config.get("datasets", {})

    if args.model not in models:
        available = ", ".join(sorted(models.keys()))
        raise ValueError(f"Unknown model: {args.model}; available models: {available}")

    if args.dataset not in datasets:
        available = ", ".join(sorted(datasets.keys()))
        raise ValueError(f"Unknown dataset: {args.dataset}; available datasets: {available}")

    model_cfg = dict(models[args.model])
    dataset_cfg = dict(datasets[args.dataset])

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root

    output_dir = output_root / args.run_id / args.dataset / args.model / args.setting
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    args_path = output_dir / "args.json"
    log_path = output_dir / "run.log"
    existing_outputs = [
        predictions_path,
        metrics_path,
        args_path,
        log_path,
    ]

    if output_dir.exists() and not args.overwrite:
        if any(p.exists() for p in existing_outputs):
            raise FileExistsError(
                f"Output already exists: {output_dir}. Use a different --run_id or pass --overwrite."
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(log_path)
    start_time_iso = datetime.now().isoformat()

    try:
        dataset_path = Path(dataset_cfg["path"])
        if not dataset_path.is_absolute():
            dataset_path = project_root / dataset_path

        file_type = str(dataset_cfg.get("file_type", "json"))
        support_unit = str(dataset_cfg.get("support_unit", "sentence"))
        adapter_name = str(dataset_cfg.get("adapter", args.dataset))
        id_field = str(dataset_cfg.get("id_field", "id"))
        model_family = model_cfg.get("model_family", None)

        logger.log("Starting baseline evaluation")
        logger.log(f"run_id={args.run_id} dataset={args.dataset} model={args.model} setting={args.setting}")
        logger.log(f"dataset_path={dataset_path}")
        logger.log(f"model_path={model_cfg.get('path')}")

        adapter_map = {
            "hotpotqa": adapt_hotpotqa_sample,
            "musique": adapt_musique_sample,
            "2wiki": adapt_2wiki_sample,
        }
        if adapter_name not in adapter_map:
            raise ValueError(f"Unsupported adapter: {adapter_name}")
        adapter_fn = adapter_map[adapter_name]

        data = load_dataset_samples(dataset_path, file_type)
        if args.limit is not None:
            data = data[: args.limit]

        dtype, device_map = resolve_dtype_and_device(str(model_cfg.get("dtype", "float16")), args.device)

        trust_remote_code = bool(model_cfg.get("trust_remote_code", False))

        tokenizer = AutoTokenizer.from_pretrained(model_cfg["path"], trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["path"],
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        input_device = get_input_device(model, args.device)

        logger.log(f"requested_device={args.device}")
        logger.log(f"resolved_dtype={dtype}")
        logger.log(f"device_map={device_map}")
        logger.log(f"input_device={input_device}")
        logger.log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        logger.log(f"cuda_device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")

        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        run_args_dump = {
            "dataset": args.dataset,
            "model": args.model,
            "setting": args.setting,
            "limit": args.limit,
            "run_id": args.run_id,
            "config_path": str(config_path),
            "output_root": str(output_root),
            "output_dir": str(output_dir),
            "max_new_tokens": args.max_new_tokens,
            "device": args.device,
            "resolved_dtype": str(dtype),
            "device_map": str(device_map),
            "input_device": str(input_device),
            "tau": args.tau,
            "log_every": args.log_every,
            "overwrite": args.overwrite,
            "model_config": model_cfg,
            "dataset_config": dataset_cfg,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "start_time": start_time_iso,
        }
        with args_path.open("w", encoding="utf-8") as f:
            json.dump(run_args_dump, f, ensure_ascii=False, indent=2)

        rows_out: list[dict[str, Any]] = []
        diagnostics = init_diagnostics()

        running_em_sum = 0.0
        running_answer_f1_sum = 0.0
        running_support_f1_sum = 0.0
        running_joint_f1_sum = 0.0
        running_format_ok = 0
        running_strict_format_ok = 0
        running_em_correct = 0
        running_ocu = 0
        running_invalid_support_samples = 0

        t0 = time.time()
        pbar = tqdm(total=len(data), desc=f"Eval-{args.dataset}-{args.setting}", unit="sample") if tqdm else None

        for idx, sample in enumerate(data, start=1):
            adapted = adapter_fn(sample, idx, id_field, diagnostics)

            sample_id = adapted["id"]
            question = adapted["question"]
            gold_answer = adapted["gold_answer"]
            flattened = adapted["flattened"]
            gold_support = adapted["gold_support"]

            if args.setting == "fullctx":
                context_rows = flattened
            elif args.setting == "goldonly":
                context_rows = keep_context_by_ids(flattened, gold_support)
            elif args.setting == "nogold":
                context_rows = drop_context_by_ids(flattened, gold_support)
            elif args.setting == "questiononly":
                context_rows = []
            else:
                raise ValueError(f"Unsupported setting: {args.setting}")

            visible_support_ids = {row["sid"] for row in context_rows}

            context_str = render_context_lines(context_rows) if args.setting != "questiononly" else None
            prompt = build_prompt(question, context_str, args.setting, support_unit, model_family=model_family)
            model_input = build_generation_input(tokenizer, prompt)

            inputs = tokenizer(model_input, return_tensors="pt")
            input_len = inputs["input_ids"].shape[-1]
            inputs = {k: v.to(input_device) for k, v in inputs.items()}

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            gen_ids = generated[0][input_len:]
            raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True)
            generated_tokens = int(gen_ids.shape[-1])
            reached_max_new_tokens = generated_tokens >= args.max_new_tokens

            ended_with_eos = False
            if tokenizer.eos_token_id is not None:
                last_token_id = int(generated[0][-1].item())
                if isinstance(tokenizer.eos_token_id, int):
                    ended_with_eos = last_token_id == tokenizer.eos_token_id
                elif isinstance(tokenizer.eos_token_id, list):
                    ended_with_eos = last_token_id in tokenizer.eos_token_id

            _, pred_support_raw, pred_answer, format_ok, format_ok_strict, parse_error_type = parse_model_output(
                raw_output
            )

            if args.setting == "questiononly":
                pred_support = set()
                invalid_pred_support = pred_support_raw
            else:
                pred_support = pred_support_raw & visible_support_ids
                invalid_pred_support = pred_support_raw - visible_support_ids

            em = answer_em(pred_answer, gold_answer)
            answer_p, answer_r, ans_f1 = answer_scores(pred_answer, gold_answer)
            sup_p, sup_r, sup_f1 = support_scores(pred_support, gold_support)
            joint_p, joint_r, joint_f1 = joint_scores(answer_p, answer_r, sup_p, sup_r)
            ocu = int(em == 1 and sup_f1 < args.tau)

            invalid_pred_support_count = len(invalid_pred_support)

            running_em_sum += em
            running_answer_f1_sum += ans_f1
            running_support_f1_sum += sup_f1
            running_joint_f1_sum += joint_f1
            running_format_ok += int(format_ok)
            running_strict_format_ok += int(format_ok_strict)
            running_em_correct += em
            running_ocu += ocu
            running_invalid_support_samples += int(invalid_pred_support_count > 0)

            rows_out.append(
                {
                    "id": sample_id,
                    "dataset": args.dataset,
                    "model": args.model,
                    "setting": args.setting,
                    "question": question,
                    "gold_answer": gold_answer,
                    "pred_answer": pred_answer,
                    "gold_support": _sorted_support_ids(gold_support),
                    "pred_support_raw": _sorted_support_ids(pred_support_raw),
                    "pred_support": _sorted_support_ids(pred_support),
                    "invalid_pred_support": _sorted_support_ids(invalid_pred_support),
                    "em": em,
                    "answer_precision": answer_p,
                    "answer_recall": answer_r,
                    "answer_f1": ans_f1,
                    "support_precision": sup_p,
                    "support_recall": sup_r,
                    "support_f1": sup_f1,
                    "joint_precision": joint_p,
                    "joint_recall": joint_r,
                    "joint_f1": joint_f1,
                    "ocu": ocu,
                    "visible_support_count": len(visible_support_ids),
                    "gold_support_count": len(gold_support),
                    "pred_support_raw_count": len(pred_support_raw),
                    "pred_support_count": len(pred_support),
                    "invalid_pred_support_count": invalid_pred_support_count,
                    "input_tokens": input_len,
                    "format_ok": format_ok,
                    "format_ok_strict": format_ok_strict,
                    "parse_error_type": parse_error_type,
                    "generated_tokens": generated_tokens,
                    "reached_max_new_tokens": reached_max_new_tokens,
                    "ended_with_eos": ended_with_eos,
                    "raw_output": raw_output,
                }
            )

            if pbar is not None:
                pbar.update(1)

            if args.log_every > 0 and (idx % args.log_every == 0 or idx == len(data)):
                elapsed = time.time() - t0
                speed = idx / elapsed if elapsed > 0 else 0.0
                eta_sec = ((len(data) - idx) / speed) if speed > 0 else 0.0

                live_em = running_em_sum / idx
                live_ans_f1 = running_answer_f1_sum / idx
                live_sup_f1 = running_support_f1_sum / idx
                live_joint_f1 = running_joint_f1_sum / idx
                live_fmt_ok = running_format_ok / idx
                live_strict_ok = running_strict_format_ok / idx
                live_ocu_rate = (running_ocu / running_em_correct) if running_em_correct > 0 else None
                live_invalid_rate = running_invalid_support_samples / idx
                ocu_text = f"{live_ocu_rate:.4f}" if live_ocu_rate is not None else "None"

                if pbar is not None:
                    pbar.set_postfix(
                        **{
                            "A-F1": f"{live_ans_f1:.4f}",
                            "S-F1": f"{live_sup_f1:.4f}",
                            "J-F1": f"{live_joint_f1:.4f}",
                        },
                        EM=f"{live_em:.4f}",
                        OCU=ocu_text,
                        FmtOK=f"{live_fmt_ok:.4f}",
                        StrictOK=f"{live_strict_ok:.4f}",
                        InvSup=f"{live_invalid_rate:.4f}",
                        Speed=f"{speed:.2f}/s",
                        ETA=f"{eta_sec:.1f}s",
                    )

                logger.log(
                    f"[live] {idx}/{len(data)} | A-F1={live_ans_f1:.4f} S-F1={live_sup_f1:.4f} "
                    f"J-F1={live_joint_f1:.4f} EM={live_em:.4f} OCU={ocu_text} "
                    f"FmtOK={live_fmt_ok:.4f} StrictOK={live_strict_ok:.4f} "
                    f"InvSup={live_invalid_rate:.4f} "
                    f"Speed={speed:.2f}/s ETA={eta_sec:.1f}s"
                )

        if pbar is not None:
            pbar.close()

        num_samples = len(rows_out)
        em_avg = sum(r["em"] for r in rows_out) / num_samples if num_samples else 0.0
        answer_p_avg = sum(r["answer_precision"] for r in rows_out) / num_samples if num_samples else 0.0
        answer_r_avg = sum(r["answer_recall"] for r in rows_out) / num_samples if num_samples else 0.0
        ans_f1_avg = sum(r["answer_f1"] for r in rows_out) / num_samples if num_samples else 0.0
        sup_p_avg = sum(r["support_precision"] for r in rows_out) / num_samples if num_samples else 0.0
        sup_r_avg = sum(r["support_recall"] for r in rows_out) / num_samples if num_samples else 0.0
        sup_f1_avg = sum(r["support_f1"] for r in rows_out) / num_samples if num_samples else 0.0
        joint_p_avg = sum(r["joint_precision"] for r in rows_out) / num_samples if num_samples else 0.0
        joint_r_avg = sum(r["joint_recall"] for r in rows_out) / num_samples if num_samples else 0.0
        joint_f1_avg = sum(r["joint_f1"] for r in rows_out) / num_samples if num_samples else 0.0
        format_ok_rate = sum(1 for r in rows_out if r["format_ok"]) / num_samples if num_samples else 0.0
        strict_format_ok_rate = sum(1 for r in rows_out if r["format_ok_strict"]) / num_samples if num_samples else 0.0

        avg_input_tokens = sum(r["input_tokens"] for r in rows_out) / num_samples if num_samples else 0.0
        max_input_tokens = max((r["input_tokens"] for r in rows_out), default=0)
        avg_generated_tokens = sum(r["generated_tokens"] for r in rows_out) / num_samples if num_samples else 0.0
        max_generated_tokens = max((r["generated_tokens"] for r in rows_out), default=0)
        truncation_rate = sum(1 for r in rows_out if r["reached_max_new_tokens"]) / num_samples if num_samples else 0.0
        ended_with_eos_rate = sum(1 for r in rows_out if r["ended_with_eos"]) / num_samples if num_samples else 0.0
        parse_error_distribution = dict(Counter(r["parse_error_type"] for r in rows_out))

        avg_gold_support_count = sum(r["gold_support_count"] for r in rows_out) / num_samples if num_samples else 0.0
        avg_visible_support_count = (
            sum(r["visible_support_count"] for r in rows_out) / num_samples if num_samples else 0.0
        )
        avg_pred_support_raw_count = (
            sum(r["pred_support_raw_count"] for r in rows_out) / num_samples if num_samples else 0.0
        )
        avg_pred_support_count = sum(r["pred_support_count"] for r in rows_out) / num_samples if num_samples else 0.0
        avg_invalid_pred_support_count = (
            sum(r["invalid_pred_support_count"] for r in rows_out) / num_samples if num_samples else 0.0
        )
        invalid_support_rate = (
            sum(1 for r in rows_out if r["invalid_pred_support_count"] > 0) / num_samples if num_samples else 0.0
        )

        num_em_correct = sum(r["em"] for r in rows_out)
        num_ocu = sum(r["ocu"] for r in rows_out)
        ocu_rate = (num_ocu / num_em_correct) if num_em_correct > 0 else None

        diagnostics["parsing_failure_count"] = (
            diagnostics["bad_context_count"]
            + diagnostics["bad_context_item_count"]
            + diagnostics["bad_supporting_facts_count"]
            + diagnostics["bad_support_fact_count"]
        )

        metrics = {
            "run_id": args.run_id,
            "dataset": args.dataset,
            "model": args.model,
            "setting": args.setting,
            "num_samples": num_samples,
            "em": em_avg,
            "answer_precision": answer_p_avg,
            "answer_recall": answer_r_avg,
            "answer_f1": ans_f1_avg,
            "support_precision": sup_p_avg,
            "support_recall": sup_r_avg,
            "support_f1": sup_f1_avg,
            "joint_precision": joint_p_avg,
            "joint_recall": joint_r_avg,
            "joint_f1": joint_f1_avg,
            "main_metrics": {
                "A-F1": ans_f1_avg,
                "S-F1": sup_f1_avg,
                "J-F1": joint_f1_avg,
            },
            "ocu_rate": ocu_rate,
            "num_em_correct": num_em_correct,
            "num_ocu": num_ocu,
            "format_ok_rate": format_ok_rate,
            "strict_format_ok_rate": strict_format_ok_rate,
            "avg_input_tokens": avg_input_tokens,
            "max_input_tokens": max_input_tokens,
            "avg_generated_tokens": avg_generated_tokens,
            "max_generated_tokens": max_generated_tokens,
            "truncation_rate": truncation_rate,
            "ended_with_eos_rate": ended_with_eos_rate,
            "parse_error_distribution": parse_error_distribution,
            "avg_gold_support_count": avg_gold_support_count,
            "avg_visible_support_count": avg_visible_support_count,
            "avg_pred_support_raw_count": avg_pred_support_raw_count,
            "avg_pred_support_count": avg_pred_support_count,
            "avg_invalid_pred_support_count": avg_invalid_pred_support_count,
            "invalid_support_rate": invalid_support_rate,
            "tau": args.tau,
            "diagnostics": diagnostics,
        }

        with predictions_path.open("w", encoding="utf-8") as f:
            for row in rows_out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        logger.log("Evaluation finished")
        logger.log(json.dumps(metrics, ensure_ascii=False))

    finally:
        logger.close()


if __name__ == "__main__":
    main()
