"""Paired answer-hit comparison from per-query DPR and Reader outputs."""
import argparse
import gzip
import json
import math
from pathlib import Path


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def key(row):
    return (row["dataset"], int(row["query_index"]))


def hit_by_cutoff(row, cutoff):
    rank = row.get("first_answer_rank")
    return rank is not None and int(rank) <= cutoff


def exact_mcnemar_p(improved, regressed):
    """Two-sided exact binomial test for the discordant paired outcomes."""
    n = improved + regressed
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(improved, regressed) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def compare(baseline_rows, reranked_rows, cutoffs):
    baseline = {key(row): row for row in baseline_rows}
    reranked = {key(row): row for row in reranked_rows}
    if set(baseline) != set(reranked):
        raise ValueError(
            f"Question ID sets differ: baseline={len(baseline)}, Reader={len(reranked)}, "
            f"intersection={len(set(baseline) & set(reranked))}"
        )
    results = {}
    for cutoff in cutoffs:
        improved = regressed = both_hit = both_miss = 0
        for qid in baseline:
            before = hit_by_cutoff(baseline[qid], cutoff)
            after = hit_by_cutoff(reranked[qid], cutoff)
            if before and after:
                both_hit += 1
            elif before:
                regressed += 1
            elif after:
                improved += 1
            else:
                both_miss += 1
        n = len(baseline)
        results[str(cutoff)] = {
            "questions": n,
            "baseline_hits": both_hit + regressed,
            "reader_hits": both_hit + improved,
            "both_hit": both_hit,
            "both_miss": both_miss,
            "improved": improved,
            "regressed": regressed,
            "net_hit_change": improved - regressed,
            "exact_mcnemar_p": exact_mcnemar_p(improved, regressed),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="DPR per-query JSONL or JSONL.GZ")
    parser.add_argument("--reader", type=Path, required=True, help="Reader per-query JSONL or JSONL.GZ")
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 5, 20, 100])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "metric": "answer-containing-passage hit",
        "baseline": str(args.baseline),
        "reader": str(args.reader),
        "paired_test": "two-sided exact McNemar/binomial test on discordant question outcomes",
        "results": compare(list(rows(args.baseline)), list(rows(args.reader)), args.cutoffs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
