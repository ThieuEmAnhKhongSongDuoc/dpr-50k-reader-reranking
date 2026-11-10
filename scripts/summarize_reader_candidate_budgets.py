"""Summarize Reader reranking at smaller candidate budgets from a larger run.

For example, a single K=200 development run contains Reader scores for the
original DPR top 200. Filtering to retriever_rank <= 100 and sorting by the
same Reader scores gives the corresponding K=100 result without re-scoring.
"""
import argparse
import gzip
import json
from pathlib import Path


def open_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def summarize(rows, budgets):
    counts = {budget: {cutoff: 0 for cutoff in (1, 5, 20, 100)} for budget in budgets}
    exact_match_counts = {budget: 0 for budget in budgets}
    exact_match_available = {budget: False for budget in budgets}
    n = 0
    for row in rows:
        n += 1
        for budget in budgets:
            candidates = [
                candidate for candidate in row["candidates"]
                if int(candidate["retriever_rank"]) <= budget
            ]
            candidates.sort(key=lambda candidate: float(candidate["reader_relevance_score"]), reverse=True)
            if candidates and candidates[0].get("reader_exact_match") is not None:
                exact_match_available[budget] = True
                exact_match_counts[budget] += int(bool(candidates[0]["reader_exact_match"]))
            first_hit = next(
                (rank for rank, candidate in enumerate(candidates, 1) if candidate["answer_hit"]),
                None,
            )
            if first_hit is not None:
                for cutoff in counts[budget]:
                    if first_hit <= cutoff:
                        counts[budget][cutoff] += 1
    return {
        str(budget): {
            "candidate_k": budget,
            "questions": n,
            "hits": {str(cutoff): hits for cutoff, hits in counts[budget].items()},
            "top_k_accuracy": {
                str(cutoff): hits / n if n else 0.0
                for cutoff, hits in counts[budget].items()
            },
            "exact_match_count": exact_match_counts[budget] if exact_match_available[budget] else None,
            "exact_match_accuracy": exact_match_counts[budget] / n if n and exact_match_available[budget] else None,
        }
        for budget in budgets
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Reader per-query JSONL or JSONL.GZ")
    parser.add_argument("--budgets", type=int, nargs="+", default=[100, 200])
    parser.add_argument("--split", choices=("development", "test"), default="development",
                        help="Label the split represented in the input per-query file")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(budget < 1 for budget in args.budgets):
        parser.error("Budgets must be positive integers")
    rows = list(open_rows(args.input))
    if not rows:
        raise ValueError(f"No query rows in {args.input}")
    max_retriever_rank = max(
        int(candidate["retriever_rank"])
        for row in rows
        for candidate in row["candidates"]
    )
    if max(args.budgets) > max_retriever_rank:
        raise ValueError(
            f"Requested K={max(args.budgets)}, but input only contains retriever "
            f"ranks through {max_retriever_rank}"
        )
    report = {
        "method": "DPR Reader relevance-head reranking",
        "source_per_query_file": str(args.input),
        "selection_split": "development",
        "evaluation_split": args.split,
        "budget_selection_uses_test": False,
        "results_by_budget": summarize(rows, args.budgets),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
