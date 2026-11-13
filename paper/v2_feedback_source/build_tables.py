"""Build paper tables from audited summaries; never runs model inference."""

import gzip
import json
import math
from itertools import zip_longest
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
cross_encoder_summary_path = DATA / "cross_encoder_summary.json"
cross_encoder_per_query_dir = (
    ROOT.parents[1] / "results" / "v2_feedback" / "cross_encoder" / "per_query"
)
cross_encoder_per_query_inputs = {
    "trivia_dev": cross_encoder_per_query_dir
    / "trivia_dev_cross_encoder_per_query.jsonl.gz",
    "curatedtrec_test": cross_encoder_per_query_dir
    / "curatedtrec_test_cross_encoder_per_query.jsonl.gz",
    "trivia_test": cross_encoder_per_query_dir
    / "trivia_test_cross_encoder_per_query.jsonl.gz",
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


def validate_shared_crossencoder(path, reader_path, expected_questions,
                                 expected_match_type, expected_summary):
    """Check that CE and Reader records contain identical candidates and labels."""
    seen = 0
    aggregate = {
        method: {str(k): 0 for k in (1, 5, 20, 100)}
        for method in ("dpr", "reader", "cross_encoder")
    }
    with gzip.open(path, "rt", encoding="utf-8") as ce_stream, gzip.open(
        reader_path, "rt", encoding="utf-8"
    ) as reader_stream:
        for ce_line, reader_line in zip_longest(ce_stream, reader_stream):
            assert ce_line is not None and reader_line is not None, path
            ce_row = json.loads(ce_line)
            reader_row = json.loads(reader_line)
            assert str(ce_row["query_id"]) == str(reader_row["query_id"])
            assert ce_row["candidate_k"] == 100
            assert ce_row["match_type"] == expected_match_type
            assert len(ce_row["candidates"]) == 100
            assert len(reader_row["candidates"]) == 100
            reader_by_id = {
                str(candidate["passage_id"]): candidate
                for candidate in reader_row["candidates"]
            }
            ce_by_id = {
                str(candidate["passage_id"]): candidate
                for candidate in ce_row["candidates"]
            }
            assert len(reader_by_id) == len(ce_by_id) == 100
            assert set(reader_by_id) == set(ce_by_id)
            ce_ranks = []
            for passage_id, candidate in ce_by_id.items():
                source = reader_by_id[passage_id]
                assert bool(candidate["answer_hit"]) == bool(source["answer_hit"])
                assert candidate["retriever_rank"] == source["retriever_rank"]
                assert candidate["rank"] == source["rank"]
                assert candidate["reader_relevance_score"] == source[
                    "reader_relevance_score"
                ]
                assert math.isfinite(float(candidate["cross_encoder_score"]))
                ce_ranks.append(candidate["cross_encoder_rank"])
            assert sorted(ce_ranks) == list(range(1, 101))
            for method, values in ce_row["ranking_metrics"].items():
                assert method in aggregate
                for cutoff, hit in values["hits"].items():
                    aggregate[method][cutoff] += int(hit)
            seen += 1
    assert seen == expected_questions, (path, seen, expected_questions)
    assert aggregate == expected_summary["hit_counts"], (path, aggregate)


if (
    all(path.is_file() for path in reader_inputs.values())
    and all(path.is_file() for path in per_query_inputs.values())
    and cross_encoder_summary_path.is_file()
    and all(path.is_file() for path in cross_encoder_per_query_inputs.values())
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

    cross_summary = read(cross_encoder_summary_path.name)
    assert cross_summary["model"] == "cross-encoder/ms-marco-MiniLM-L6-v2"
    cross_runs = {row["dataset"]: row for row in cross_summary["results"]}
    for dataset, reader_key, expected_type in [
        ("curatedtrec_test", "curatedtrec", "regex"),
        ("trivia_dev", "dev", "string"),
        ("trivia_test", "trivia", "string"),
    ]:
        run = cross_runs[dataset]
        assert run["candidate_k"] == 100
        assert run["questions"] == read(reader_inputs[reader_key].name)[
            "test_results"
        ][{"dev": "trivia_dev", "curatedtrec": "curatedtrec_test", "trivia": "trivia_test"}[reader_key]]["questions"]
        assert run["precision"] == "FP32 inference"
        assert run["matching"].startswith("saved DPR answer_hit labels")
        validate_shared_crossencoder(
            cross_encoder_per_query_inputs[dataset],
            per_query_inputs[reader_key],
            run["questions"],
            expected_type,
            run,
        )

    readers = {"curatedtrec": trec_reader, "trivia": trivia_reader}
    for dataset, row in readers.items():
        validate_result(row)
        assert all(
            count <= coverage["per_dataset"][dataset]["covered_questions"]
            for count in row["hits"].values()
        )

    run_labels = {
        "curatedtrec_test": "CuratedTREC test",
        "trivia_dev": "TriviaQA dev",
        "trivia_test": "TriviaQA test",
    }
    method_labels = {
        "dpr": "DPR order",
        "cross_encoder": "Cross-encoder",
        "reader": "DPR Reader",
    }
    rerank_rows = []
    for index, dataset in enumerate(
        ("curatedtrec_test", "trivia_dev", "trivia_test")
    ):
        run = cross_runs[dataset]
        for method in ("dpr", "cross_encoder", "reader"):
            counts = run["hit_counts"][method]
            hits = " & ".join(
                f"{100 * counts[str(cutoff)] / run['questions']:.2f} "
                f"({counts[str(cutoff)]:,})"
                for cutoff in (1, 5, 20, 100)
            )
            label = run_labels[dataset] if method == "dpr" else ""
            rerank_rows.append(
                f"{label} & {method_labels[method]} & {hits} " + r"\\"
            )
        if index < 2:
            rerank_rows.append(r"\midrule")

    paired_trivia_test = cross_runs["trivia_test"]["hit_counts"]["dpr"]
    independent_trivia_test = dpr_result(multi, "trivia")["hits"]
    discrepancy_note = ""
    if paired_trivia_test != independent_trivia_test:
        discrepancy_note = (
            r"\vspace{1mm}\par\footnotesize The TriviaQA test DPR-order values "
            r"are recomputed from the saved candidate records for a paired comparison; "
            r"the independent full-test aggregate differs slightly. All three rankings "
            r"in this table use the same 100 candidate IDs and saved answer-hit labels."
        )
    write_table(
        "reranking.tex",
        [
            r"\begin{table*}[t]",
            r"\caption{Answer-containing-passage results for the same 100 DPR candidates under DPR, cross-encoder, and Reader rankings.}",
            r"\label{tab:reranking}",
            r"\centering",
            r"\footnotesize",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Dataset/split & Ranking & Hit@1 & Hit@5 & Hit@20 & Hit@100 \\",
            r"\midrule",
            *rerank_rows,
            r"\bottomrule",
            r"\end{tabular}",
            discrepancy_note,
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
    print(
        "Reader and cross-encoder tables written after verifying identical "
        "100-candidate IDs and answer-hit labels."
    )
else:
    # Never let a stale table survive after the run inputs are absent.
    for generated in ("reranking.tex", "span_em.tex"):
        stale_path = TABLES / generated
        if stale_path.exists():
            stale_path.unlink()
    absent = [
        str(path.relative_to(ROOT))
        for path in [
            *reader_inputs.values(),
            *per_query_inputs.values(),
            cross_encoder_summary_path,
            *cross_encoder_per_query_inputs.values(),
        ]
        if not path.is_file()
    ]
    print(
        "Reader/cross-encoder tables not regenerated; required outputs missing: "
        + ", ".join(absent)
    )

print("Regenerated datasets.tex and baselines.tex from saved result summaries.")
