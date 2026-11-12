"""Build paper tables from audited summaries; never runs model inference."""

import gzip
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TABLES = ROOT / "tables"
TABLES.mkdir(exist_ok=True)


def read(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def write_table(name, lines):
    (TABLES / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def dpr_result(obj, dataset):
    return obj["test_results"][dataset + "_test"]


def bm25_result(obj, dataset):
    return obj["test_results"][dataset]


def rate(row, cutoff):
    return row["hits"][str(cutoff)] / row["questions"]


def pct(row, cutoff):
    return f"{100 * rate(row, cutoff):.2f}"


def validate_result(row):
    questions = row["questions"]
    assert questions > 0
    for cutoff, count in row["hits"].items():
        assert 0 <= count <= questions, (cutoff, count, questions)
        if cutoff in row.get("top_k_accuracy", {}):
            assert math.isclose(
                row["top_k_accuracy"][cutoff],
                count / questions,
                rel_tol=1e-10,
            ), (cutoff, row["top_k_accuracy"][cutoff], count / questions)


NAMES = ["nq", "trivia", "webq", "curatedtrec", "squad1"]
LABELS = {
    "nq": "NQ",
    "trivia": "TriviaQA",
    "webq": "WebQuestions",
    "curatedtrec": "CuratedTREC",
    "squad1": "SQuAD 1.1",
}
BM25_KEYS = {
    "nq": "nq",
    "trivia": "triviaqa",
    "webq": "webquestions",
    "curatedtrec": "curatedtrec",
    "squad1": "squad",
}

multi = read("dpr_multi.json")
single = read("dpr_single_nq.json")
bm25 = read("lucene_bm25.json")
fusion = read("bm25_dpr_fusion.json")
coverage = read("coverage.json")

for dataset in NAMES:
    rows = [
        dpr_result(multi, dataset),
        dpr_result(single, dataset),
        bm25_result(bm25, BM25_KEYS[dataset]),
        dpr_result(fusion, dataset),
    ]
    for row in rows:
        validate_result(row)
    assert len({row["questions"] for row in rows}) == 1, dataset
    cov = coverage["per_dataset"][dataset]
    assert cov["questions"] == rows[0]["questions"], dataset
    assert all(
        count <= cov["covered_questions"]
        for row in rows
        for count in row["hits"].values()
    ), dataset

trec_rows = [
    dpr_result(multi, "curatedtrec"),
    dpr_result(single, "curatedtrec"),
    bm25_result(bm25, "curatedtrec"),
    dpr_result(fusion, "curatedtrec"),
]
for row in trec_rows:
    assert row["match_type"] == "regex"

dataset_rows = []
for dataset in NAMES:
    cov = coverage["per_dataset"][dataset]
    dataset_rows.append(
        f"{LABELS[dataset]} & {cov['questions']:,} & "
        f"{cov['covered_questions']:,} & {100 * cov['coverage']:.2f} "
        r"\\"
    )
write_table(
    "datasets.tex",
    [
        r"\begin{table}[t]",
        r"\caption{Full test sizes and answer-match coverage in the 50k corpus.}",
        r"\label{tab:datasets}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Test & Questions & Covered & Coverage (\%) \\",
        r"\midrule",
        *dataset_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ],
)

baseline_rows = []
for dataset in NAMES:
    systems = [
        bm25_result(bm25, BM25_KEYS[dataset]),
        dpr_result(single, dataset),
        dpr_result(multi, dataset),
        dpr_result(fusion, dataset),
    ]
    cells = [f"{pct(row, 20)} / {pct(row, 100)}" for row in systems]
    baseline_rows.append(LABELS[dataset] + " & " + " & ".join(cells) + r"\\")
write_table(
    "baselines.tex",
    [
        r"\begin{table*}[t]",
        r"\caption{Full-test first-stage references. Cells report Hit@20 / Hit@100 (\%).}",
        r"\label{tab:baselines}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Test & Lucene BM25 & DPR NQ-Single & DPR Multi & BM25+DPR \\",
        r"\midrule",
        *baseline_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ],
)

reader_inputs = {
    "dev": DATA / "trivia_dev_reader_k100_fp16.json",
    "curatedtrec": DATA / "curatedtrec_reader_regex_k100_fp16.json",
    "trivia": DATA / "trivia_reader_k100_fp16.json",
}
per_query_inputs = {
    "dev": DATA / "per_query" / "trivia_dev_reader_k100_fp16.jsonl.gz",
    "curatedtrec": DATA / "per_query" / "curatedtrec_reader_regex_k100_fp16.jsonl.gz",
    "trivia": DATA / "per_query" / "trivia_reader_k100_fp16.jsonl.gz",
}


def validate_per_query(path, expected_questions, expected_match_type):
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
    assert seen == expected_questions, (path, seen, expected_questions)


if all(path.is_file() for path in reader_inputs.values()) and all(
    path.is_file() for path in per_query_inputs.values()
):
    dev = read(reader_inputs["dev"].name)["test_results"]["trivia_dev"]
    trec_reader = read(reader_inputs["curatedtrec"].name)["test_results"][
        "curatedtrec_test"
    ]
    trivia_reader = read(reader_inputs["trivia"].name)["test_results"]["trivia_test"]

    assert dev["candidate_k"] == 100 and dev["reader_precision"] == "fp16"
    assert trec_reader["candidate_k"] == 100
    assert trec_reader["reader_precision"] == "fp16"
    assert trec_reader["match_type"] == "regex"
    assert trivia_reader["candidate_k"] == 100
    assert trivia_reader["reader_precision"] == "fp16"
    assert trivia_reader["match_type"] == "string"
    assert trivia_reader["exact_match_accuracy"] is not None
    assert trivia_reader["questions"] == dpr_result(multi, "trivia")["questions"]
    assert trec_reader["questions"] == dpr_result(multi, "curatedtrec")["questions"]

    validate_per_query(per_query_inputs["dev"], dev["questions"], "string")
    validate_per_query(
        per_query_inputs["curatedtrec"], trec_reader["questions"], "regex"
    )
    validate_per_query(per_query_inputs["trivia"], trivia_reader["questions"], "string")

    readers = {"curatedtrec": trec_reader, "trivia": trivia_reader}
    for dataset, row in readers.items():
        validate_result(row)
        assert all(
            count <= coverage["per_dataset"][dataset]["covered_questions"]
            for count in row["hits"].values()
        )

    rerank_rows = []
    for index, dataset in enumerate(("curatedtrec", "trivia")):
        for method, row in [
            ("DPR Multi", dpr_result(multi, dataset)),
            ("Reader, $K=100$", readers[dataset]),
        ]:
            hits = " & ".join(
                f"{pct(row, cutoff)} ({row['hits'][str(cutoff)]:,})"
                for cutoff in (1, 5, 20, 100)
            )
            label = LABELS[dataset] if method == "DPR Multi" else ""
            rerank_rows.append(f"{label} & {method} & {hits} " + r"\\")
        if index == 0:
            rerank_rows.append(r"\midrule")
    write_table(
        "reranking.tex",
        [
            r"\begin{table*}[t]",
            r"\caption{Answer-containing-passage results before and after Reader reranking at $K=100$.}",
            r"\label{tab:reranking}",
            r"\centering",
            r"\small",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Test & Method & Hit@1 & Hit@5 & Hit@20 & Hit@100 \\",
            r"\midrule",
            *rerank_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ],
    )

    exact_count = trivia_reader["exact_match_count"]
    exact_questions = trivia_reader["questions"]
    write_table(
        "span_em.tex",
        [
            r"\begin{table}[t]",
            r"\caption{TriviaQA Reader span exact match at $K=100$.}",
            r"\label{tab:spanem}",
            r"\centering",
            r"\footnotesize",
            r"\begin{tabular}{lrr}",
            r"\toprule",
            r"Test & Exact Match (\%) & Count \\",
            r"\midrule",
            f"TriviaQA & {100 * trivia_reader['exact_match_accuracy']:.2f} "
            f"& {exact_count:,}/{exact_questions:,} " + r"\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ],
    )
    print("Reader tables written from complete K=100 FP16 dev/test outputs.")
else:
    # Never let a stale table survive after the run inputs are absent.
    for generated in ("reranking.tex", "span_em.tex"):
        stale_path = TABLES / generated
        if stale_path.exists():
            stale_path.unlink()
    absent = [
        str(path.relative_to(ROOT))
        for path in [*reader_inputs.values(), *per_query_inputs.values()]
        if not path.is_file()
    ]
    print(
        "Reader tables not regenerated; required Kaggle files missing: "
        + ", ".join(absent)
    )

print("Regenerated datasets.tex and baselines.tex from saved result summaries.")
