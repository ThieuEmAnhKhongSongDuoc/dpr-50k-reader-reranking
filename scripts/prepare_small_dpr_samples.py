#!/usr/bin/env python3
"""Create deterministic, memory-bounded subsets from official DPR JSON arrays."""

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import ijson


DATASETS = {
    "nq": ("nq-train.json", "nq-dev.json"),
    "trivia": ("trivia-train.json.gz", "trivia-dev.json.gz"),
    "webq": ("webq-train.json.gz", "webq-dev.json.gz"),
    "curatedtrec": ("curatedtrec-train.json.gz", "curatedtrec-dev.json.gz"),
    "squad1": ("squad1-train.json.gz", "squad1-dev.json.gz"),
}


def iter_examples(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as source:
        yield from ijson.items(source, "item", use_float=True)


def make_reservoir(path: Path, sample_size: int, seed: int) -> Tuple[list, Dict[str, int]]:
    rng = random.Random(seed)
    selected = []
    scanned = 0
    with_positive = 0

    for example in iter_examples(path):
        scanned += 1
        if not isinstance(example, dict):
            continue
        if not isinstance(example.get("positive_ctxs"), list) or not example["positive_ctxs"]:
            continue

        with_positive += 1
        if len(selected) < sample_size:
            selected.append(example)
        else:
            replace_at = rng.randrange(with_positive)
            if replace_at < sample_size:
                selected[replace_at] = example

    rng.shuffle(selected)
    return selected, {"examples_scanned": scanned, "examples_with_positive": with_positive}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/downloads/data/retriever"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/paper_samples"))
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--dev-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    if args.train_samples < 1 or args.dev_samples < 1:
        parser.error("sample counts must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": args.seed,
        "train_sample_limit": args.train_samples,
        "dev_sample_limit": args.dev_samples,
        "datasets": {},
    }

    for dataset, (train_name, dev_name) in DATASETS.items():
        manifest["datasets"][dataset] = {}
        for split, filename, sample_size in (
            ("train", train_name, args.train_samples),
            ("dev", dev_name, args.dev_samples),
        ):
            source = args.input_dir / filename
            if not source.is_file():
                raise FileNotFoundError(f"Missing official DPR resource: {source}")
            samples, counts = make_reservoir(source, sample_size, args.seed)
            output = args.output_dir / f"{dataset}-{split}-sample{len(samples)}.json"
            with output.open("w", encoding="utf-8", newline="\n") as target:
                json.dump(samples, target, ensure_ascii=False)
                target.write("\n")
            manifest["datasets"][dataset][split] = {
                "source": str(source),
                "sample_file": str(output),
                "source_bytes": source.stat().st_size,
                "sample_count": len(samples),
                **counts,
            }
            print(
                f"{dataset} {split}: scanned={counts['examples_scanned']} "
                f"with_positive={counts['examples_with_positive']} "
                f"sampled={len(samples)} -> {output}",
                flush=True,
            )

    multi_dev_examples = []
    for dataset in ("nq", "trivia", "webq", "curatedtrec"):
        sample_path = Path(manifest["datasets"][dataset]["dev"]["sample_file"])
        multi_dev_examples.extend(json.loads(sample_path.read_text(encoding="utf-8")))
    multi_dev_path = args.output_dir / f"multi-dev-sample{len(multi_dev_examples)}.json"
    with multi_dev_path.open("w", encoding="utf-8", newline="\n") as target:
        json.dump(multi_dev_examples, target, ensure_ascii=False)
        target.write("\n")
    manifest["multi_dev_file"] = str(multi_dev_path)
    manifest["multi_dev_count"] = len(multi_dev_examples)
    print(f"multi dev: combined={len(multi_dev_examples)} -> {multi_dev_path}", flush=True)

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote sampling manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
