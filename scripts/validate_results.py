"""Audit saved counts and regenerate a summary. Does not execute a model."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/raw"
EXPECTED = {"nq": 3610, "trivia": 11313, "webq": 2032, "curatedtrec": 694, "squad1": 10570}
BM_KEYS = {"nq": "nq", "trivia": "triviaqa", "webq": "webquestions", "curatedtrec": "curatedtrec", "squad1": "squad"}

def read(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))

def validate():
    multi, single, bm, coverage = [read(n) for n in
        ("dpr_multi.json", "dpr_single_nq.json", "lucene_bm25.json", "coverage.json")]
    assert multi["corpus_size"] == single["corpus_size"] == bm["passages"] == coverage["corpus_passages"] == 50000
    assert coverage["total_questions"] == sum(EXPECTED.values()) == 28219
    assert coverage["covered_questions"] == sum(x["covered_questions"] for x in coverage["per_dataset"].values())
    assert math.isclose(coverage["overall_coverage"], coverage["covered_questions"] / 28219)
    rows, cells = [], 0
    def check(dataset, method, row):
        nonlocal cells
        n = EXPECTED[dataset]
        assert row["questions"] == n
        hits = row["hits"]
        ordered = [hits[str(k)] for k in (1, 5, 20, 100)]
        assert ordered == sorted(ordered)
        for k, hit in hits.items():
            assert isinstance(hit, int) and 0 <= hit <= coverage["per_dataset"][dataset]["covered_questions"] <= n
            assert math.isclose(row["top_k_accuracy"][str(k)], hit / n, rel_tol=0, abs_tol=1e-12)
            cells += 1
        rows.append({"dataset": dataset, "method": method, "questions": n,
                     **{"hit_at_" + str(k): hits[str(k)] for k in (1, 5, 20, 100)},
                     **{"accuracy_at_" + str(k): hits[str(k)] / n for k in (1, 5, 20, 100)}})
    for dataset, n in EXPECTED.items():
        cov = coverage["per_dataset"][dataset]
        assert cov["questions"] == n
        assert math.isclose(cov["coverage"], cov["covered_questions"] / n)
        check(dataset, "Lucene BM25", bm["test_results"][BM_KEYS[dataset]])
        check(dataset, "DPR NQ-Single", single["test_results"][dataset + "_test"])
        check(dataset, "DPR Multi", multi["test_results"][dataset + "_test"])
    for dataset in ("nq", "curatedtrec", "trivia"):
        for budget in ((100, 200) if dataset != "trivia" else (200,)):
            report = read(dataset + "_reader" + str(budget) + ".json")
            assert report["corpus_size"] == 50000 and report["faiss_type"] == "IndexFlatIP"
            assert report["reader_top_candidates"] == budget
            assert report["reader_batch_size"] == 16 and report["reader_gpus"] == 2
            row = report["test_results"][dataset + "_test"]
            assert row["candidate_k"] == budget
            check(dataset, "Reader K=" + str(budget), row)
    assert cells == 80
    return rows, {"status": "passed", "checked_accuracy_cells": cells,
                  "baseline_test_questions": 28219, "reranked_test_questions": 15617,
                  "corpus_passages": 50000, "covered_questions": 18142,
                  "scope": "Saved-result arithmetic and metadata only; no model execution."}

def main():
    rows, report = validate()
    with (ROOT / "results/summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "results/verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
