#!/usr/bin/env python3
"""Create deterministic 0.25%-style samples from official DPR train/dev/test data.

Train/dev inputs are DPR JSON arrays and are sampled only from examples that
contain at least one positive context, matching the existing project sampler.
Official retrieval test inputs are tab-delimited question/answer files.
"""

import argparse
import ast
import csv
import gzip
import json
import math
import random
from pathlib import Path
from typing import Iterable

import ijson


SPLITS = {
    "nq": {"train": "nq-train.json", "dev": "nq-dev.json", "test": "nq-test.csv"},
    "trivia": {"train": "trivia-train.json.gz", "dev": "trivia-dev.json.gz", "test": "trivia-test.csv"},
    "webq": {"train": "webq-train.json.gz", "dev": "webq-dev.json.gz", "test": "webq-test.csv"},
    "curatedtrec": {"train": "curatedtrec-train.json.gz", "dev": "curatedtrec-dev.json.gz", "test": "curatedtrec-test.csv"},
    "squad1": {"train": "squad1-train.json.gz", "dev": "squad1-dev.json.gz", "test": "squad1-test.csv"},
}


def iter_json_examples(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as source:
        yield from ijson.items(source, "item", use_float=True)


def sample_json(path: Path, count: int, seed: int) -> tuple[list, int]:
    rng = random.Random(seed)
    selected = []
    eligible = 0
    for example in iter_json_examples(path):
        if not isinstance(example, dict) or not isinstance(example.get("positive_ctxs"), list) or not example["positive_ctxs"]:
            continue
        eligible += 1
        if len(selected) < count:
            selected.append(example)
        else:
            j = rng.randrange(eligible)
            if j < count:
                selected[j] = example
    rng.shuffle(selected)
    return selected, eligible


def sample_test_csv(path: Path, count: int, seed: int) -> tuple[list[list[str]], int]:
    rng = random.Random(seed)
    selected = []
    total = 0
    with path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.reader(source, delimiter="\t"):
            if len(row) < 2:
                raise ValueError(f"Malformed test row {total + 1} in {path}: expected question and answers")
            answers = ast.literal_eval(row[1])
            if not isinstance(answers, list):
                raise ValueError(f"Malformed answers at row {total + 1} in {path}")
            total += 1
            if len(selected) < count:
                selected.append(row)
            else:
                j = rng.randrange(total)
                if j < count:
                    selected[j] = row
    rng.shuffle(selected)
    return selected, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/downloads/data/retriever"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/raw/downloads/data/retriever/qas"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/paper_samples_025pct"))
    parser.add_argument("--fraction", type=float, default=0.0025)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        parser.error("--fraction must be in (0, 1]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": args.seed, "fraction": args.fraction, "datasets": {}}

    for dataset_index, (dataset, filenames) in enumerate(SPLITS.items()):
        manifest["datasets"][dataset] = {}
        for split_index, split in enumerate(("train", "dev")):
            source = args.input_dir / filenames[split]
            if not source.is_file():
                raise FileNotFoundError(source)
            # First pass counts eligible records, so the second-pass reservoir has
            # an exact, reproducible size without loading the large JSON into RAM.
            eligible = sum(
                1 for ex in iter_json_examples(source)
                if isinstance(ex, dict) and isinstance(ex.get("positive_ctxs"), list) and ex["positive_ctxs"]
            )
            sample_count = max(1, math.floor(eligible * args.fraction + 0.5))
            sampled, verified_eligible = sample_json(
                source, sample_count, args.seed + dataset_index * 100 + split_index
            )
            if verified_eligible != eligible or len(sampled) != sample_count:
                raise RuntimeError(f"Source changed or sampling failed for {dataset} {split}")
            output = args.output_dir / f"{dataset}-{split}-0p25.json"
            output.write_text(json.dumps(sampled, ensure_ascii=False) + "\n", encoding="utf-8")
            manifest["datasets"][dataset][split] = {
                "source": str(source), "source_bytes": source.stat().st_size,
                "eligible_examples": eligible, "sample_count": len(sampled),
                "sample_file": str(output),
            }
            print(f"{dataset} {split}: {len(sampled)}/{eligible} -> {output}", flush=True)

        test_source = args.test_dir / filenames["test"]
        if not test_source.is_file():
            raise FileNotFoundError(test_source)
        # Count once to choose the exact rounded fraction; the sampler then reads
        # the small official QA CSV and validates its question/answer fields.
        with test_source.open("r", encoding="utf-8", newline="") as stream:
            total_test = sum(1 for _ in csv.reader(stream, delimiter="\t"))
        test_count = max(1, math.floor(total_test * args.fraction + 0.5))
        test_rows, verified_total = sample_test_csv(
            test_source, test_count, args.seed + dataset_index * 100 + 50
        )
        if verified_total != total_test or len(test_rows) != test_count:
            raise RuntimeError(f"Source changed or sampling failed for {dataset} test")
        test_output = args.output_dir / f"{dataset}-test-0p25.tsv"
        with test_output.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream, delimiter="\t", lineterminator="\n").writerows(test_rows)
        manifest["datasets"][dataset]["test"] = {
            "source": str(test_source), "source_bytes": test_source.stat().st_size,
            "source_examples": total_test, "sample_count": len(test_rows),
            "sample_file": str(test_output),
        }
        print(f"{dataset} test: {len(test_rows)}/{total_test} -> {test_output}", flush=True)

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
