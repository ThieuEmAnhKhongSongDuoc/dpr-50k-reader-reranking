"""Validate the saved version-2 feedback results; does not run a model."""

import csv
import gzip
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "v2_feedback"
EXPECTED = {
    "nq": 3610,
    "trivia": 11313,
    "webq": 2032,
    "curatedtrec": 694,
    "squad1": 10570,
}
BM25_KEYS = {
    "nq": "nq",
    "trivia": "triviaqa",
    "webq": "webquestions",
    "curatedtrec": "curatedtrec",
    "squad1": "squad",
}


def read(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def validate_metric_row(row, dataset, coverage):
    questions = EXPECTED[dataset]
    assert row["questions"] == questions, (dataset, row["questions"], questions)
    cutoffs = sorted(int(k) for k in row["hits"])
    ordered = [row["hits"][str(k)] for k in cutoffs]
    assert ordered == sorted(ordered), (dataset, ordered)
    for cutoff, count in row["hits"].items():
        assert isinstance(count, int) and 0 <= count <= coverage
        assert math.isclose(
            row["top_k_accuracy"][cutoff],
            count / questions,
            rel_tol=0,
            abs_tol=1e-12,
        )


def validate_fusion_per_query(path, expected_rows):
    seen = 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            assert row["query_id"] and row["dataset"]
            assert row["match_type"] in ("string", "regex")
            candidates = row["candidates"]
            assert len(candidates) == 100, row["query_id"]
            for rank, candidate in enumerate(candidates, start=1):
                assert candidate["rank"] == rank
                assert candidate["passage_id"]
                assert isinstance(candidate["bm25_score"], (int, float))
                assert isinstance(candidate["dpr_score"], (int, float))
                assert isinstance(candidate["fused_score"], (int, float))
            if row["dataset"] == "curatedtrec_test":
                assert row["match_type"] == "regex"
            seen += 1
    assert seen == expected_rows, (path, seen, expected_rows)
    return seen


def validate_reader_per_query(path, expected_rows, expected_match_type):
    seen = 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            assert row["query_id"] and row["match_type"] == expected_match_type
            candidates = row["candidates"]
            assert len(candidates) == 100
            for candidate in candidates:
                assert candidate.get("passage_id") is not None
                assert candidate.get("reader_relevance_score") is not None
            seen += 1
    assert seen == expected_rows, (path, seen, expected_rows)


def validate():
    multi = read("dpr_multi.json")
    single = read("dpr_single_nq.json")
    bm25 = read("lucene_bm25.json")
    fusion = read("bm25_dpr_fusion.json")
    coverage = read("coverage.json")

    assert multi["corpus_size"] == single["corpus_size"] == 50000
    assert bm25["passages"] == fusion["corpus_size"] == coverage["corpus_passages"] == 50000
    assert coverage["total_questions"] == sum(EXPECTED.values()) == 28219
    assert coverage["covered_questions"] == sum(
        row["covered_questions"] for row in coverage["per_dataset"].values()
    )
    assert math.isclose(
        coverage["overall_coverage"],
        coverage["covered_questions"] / coverage["total_questions"],
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert coverage["covered_questions"] == 18534
    assert coverage["per_dataset"]["curatedtrec"]["covered_questions"] == 609
    assert "regex_match" in coverage["matching"]

    summary_rows = []
    for dataset, questions in EXPECTED.items():
        cov = coverage["per_dataset"][dataset]
        assert cov["questions"] == questions
        assert math.isclose(cov["coverage"], cov["covered_questions"] / questions)
        systems = [
            ("Lucene BM25", bm25["test_results"][BM25_KEYS[dataset]]),
            ("DPR NQ-Single", single["test_results"][dataset + "_test"]),
            ("DPR Multi", multi["test_results"][dataset + "_test"]),
            ("BM25+DPR", fusion["test_results"][dataset + "_test"]),
        ]
        for method, row in systems:
            validate_metric_row(row, dataset, cov["covered_questions"])
            if dataset == "curatedtrec":
                assert row["match_type"] == "regex", (method, row.get("match_type"))
            summary_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "questions": questions,
                    "covered_questions": cov["covered_questions"],
                    **{
                        "hit_at_" + str(k): row["hits"].get(str(k), "")
                        for k in (1, 5, 20, 100)
                    },
                    **{
                        "accuracy_at_" + str(k): row["top_k_accuracy"].get(str(k), "")
                        for k in (1, 5, 20, 100)
                    },
                }
            )

    fusion_rows = validate_fusion_per_query(
        DATA / "bm25_dpr_fusion_per_query.jsonl.gz",
        coverage["total_questions"],
    )

    reader_files = {
        "trivia_dev": (
            "trivia_dev_reader_k100_fp16.json",
            "per_query/trivia_dev_reader_k100_fp16.jsonl.gz",
            "string",
        ),
        "curatedtrec_test": (
            "curatedtrec_reader_regex_k100_fp16.json",
            "per_query/curatedtrec_reader_regex_k100_fp16.jsonl.gz",
            "regex",
        ),
        "trivia_test": (
            "trivia_reader_k100_fp16.json",
            "per_query/trivia_reader_k100_fp16.jsonl.gz",
            "string",
        ),
    }
    missing_reader = []
    for dataset, (aggregate_name, per_query_name, match_type) in reader_files.items():
        aggregate_path = DATA / aggregate_name
        per_query_path = DATA / per_query_name
        if not aggregate_path.is_file() or not per_query_path.is_file():
            missing_reader.extend(
                str(path.relative_to(ROOT))
                for path in (aggregate_path, per_query_path)
                if not path.is_file()
            )
    reader_status = "pending"
    if not missing_reader:
        for dataset, (aggregate_name, per_query_name, match_type) in reader_files.items():
            report = read(aggregate_name)
            assert report["corpus_size"] == 50000
            assert report["faiss_type"] == "IndexFlatIP"
            row = report["test_results"][dataset]
            assert row["candidate_k"] == 100
            assert row["reader_precision"] == "fp16"
            assert row["match_type"] == match_type
            if dataset == "trivia_dev":
                assert row["questions"] == 8837
            if dataset == "trivia_test":
                assert row["questions"] == EXPECTED["trivia"]
                assert row["exact_match_accuracy"] is not None
            if dataset == "curatedtrec_test":
                assert row["questions"] == EXPECTED["curatedtrec"]
            validate_reader_per_query(
                DATA / per_query_name,
                row["questions"],
                match_type,
            )
        reader_status = "passed"

    verification = {
        "status": "passed" if reader_status == "passed" else "baseline_passed_reader_pending",
        "baseline_test_questions": coverage["total_questions"],
        "corpus_passages": coverage["corpus_passages"],
        "covered_questions": coverage["covered_questions"],
        "overall_coverage": coverage["overall_coverage"],
        "curatedtrec_coverage": coverage["per_dataset"]["curatedtrec"]["coverage"],
        "fusion_per_query_rows": fusion_rows,
        "reader_status": reader_status,
        "reader_files_missing": missing_reader,
        "scope": "Saved-result and metadata validation only; no model execution.",
    }
    return summary_rows, verification


def main():
    rows, verification = validate()
    with (DATA / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (DATA / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
