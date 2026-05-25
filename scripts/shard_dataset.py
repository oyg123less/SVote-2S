#!/usr/bin/env python3
"""Split a JSON/JSONL dataset into N equal shards for multi-GPU evaluation.

Usage:
    python scripts/shard_dataset.py --input data/hotpotqa/dev_full.json --num_shards 4
    python scripts/shard_dataset.py --input data/musique/musique_ans_v1.0_dev.jsonl --num_shards 4
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON or JSONL file")
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: same as input)")
    args = parser.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.output_dir) if args.output_dir else inp.parent

    # Load
    if inp.suffix == ".jsonl":
        with inp.open() as f:
            data = [json.loads(line) for line in f if line.strip()]
        ext = ".jsonl"
    else:
        with inp.open() as f:
            data = json.load(f)
        ext = ".json"

    n = len(data)
    shard_size = (n + args.num_shards - 1) // args.num_shards
    stem = inp.stem.replace("_full", "")

    for i in range(args.num_shards):
        shard = data[i * shard_size : (i + 1) * shard_size]
        out_path = out_dir / f"{stem}_shard{i}{ext}"
        with out_path.open("w") as f:
            if ext == ".jsonl":
                for row in shard:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                json.dump(shard, f, ensure_ascii=False, indent=None)
        print(f"  Shard {i}: {len(shard)} items -> {out_path}")

    print(f"Split {n} items into {args.num_shards} shards.")


if __name__ == "__main__":
    main()
