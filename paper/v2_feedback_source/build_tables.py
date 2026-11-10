"""Build manuscript tables from the saved, audited DPR result summaries.

This script never runs retrieval or Reader inference. Feedback Reader tables are
written only after the required Kaggle outputs have been copied into data/.
"""

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
                row["top_k_accuracy"][cutoff], count / questions, rel_tol=1e-10
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

# The fixed TREC results must explicitly identify the official regex evaluator.
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
        r"Test & Questions & Covered & Coverage (\%) " + r"\\",
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
        r"Test & Lucene BM25 & DPR NQ-Single & DPR Multi & BM25+DPR " + r"\\",
        r"\midrule",
        *baseline_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ],
)

# Feedback Reader tables are gated on the actual outputs from Kaggle.
reader_inputs = {
    "curatedtrec": DATA / "curatedtrec_reader_regex_k200_fp16.json",
    "trivia": DATA / "trivia_reader_selected_k_fp16.json",
    "dev": DATA / "trivia_dev_budget_selection.json",
}
if all(path.is_file() for path in reader_inputs.values()):
    trec_reader = read(reader_inputs["curatedtrec"].name)["test_results"]["curatedtrec_test"]
    trivia_reader = read(reader_inputs["trivia"].name)["test_results"]["trivia_test"]
    dev = read(reader_inputs["dev"].name)
    assert dev["evaluation_split"] == "development"
    assert dev["budget_selection_uses_test"] is False
    dev_rows = dev["results_by_budget"]
    selected_k = min(
        (int(k) for k in dev_rows),
        key=lambda k: (-dev_rows[str(k)]["top_k_accuracy"]["20"], k),
    )
    assert trivia_reader["candidate_k"] == selected_k
    assert trec_reader["match_type"] == "regex"
    assert trec_reader["reader_precision"] == "fp16"
    assert trivia_reader["reader_precision"] == "fp16"
    assert trivia_reader["exact_match_accuracy"] is not None

    # When development selects K=200, the required K=100 test result is
    # recovered from the same per-query scores by restricting to DPR top 100.
    trivia_k100 = trivia_reader
    if selected_k == 200:
        budget_path = DATA / "trivia_test_budget_metrics.json"
        if not budget_path.is_file():
            raise SystemExit(
                "Development selected TriviaQA K=200; add trivia_test_budget_metrics.json "
                "to provide the required full-test K=100 metrics."
            )
        test_budget_metrics = read(budget_path.name)
        assert test_budget_metrics["evaluation_split"] == "test"
        trivia_k100 = test_budget_metrics["results_by_budget"]["100"]
        assert trivia_k100["candidate_k"] == 100
        assert trivia_k100["questions"] == trivia_reader["questions"]
    else:
        assert selected_k == 100

    readers = {"curatedtrec": trec_reader, "trivia": trivia_reader}
    for dataset, row in readers.items():
        validate_result(row)
        assert row["questions"] == dpr_result(multi, dataset)["questions"]
        assert all(
            count <= coverage["per_dataset"][dataset]["covered_questions"]
            for count in row["hits"].values()
        )

    rerank_rows = []
    for index, dataset in enumerate(("curatedtrec", "trivia")):
        reader_rows = [(f"Reader, $K={readers[dataset]['candidate_k']}$", readers[dataset])]
        if dataset == "trivia" and selected_k == 200:
            reader_rows.insert(0, ("Reader, $K=100$", trivia_k100))
        for method, row in [("DPR Multi", dpr_result(multi, dataset)), *reader_rows]:
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
            r"\caption{Answer-containing-passage results before and after Reader reranking.}",
            r"\label{tab:reranking}",
            r"\centering",
            r"\small",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Test & Method & Hit@1 & Hit@5 & Hit@20 & Hit@100 " + r"\\",
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
            r"\caption{TriviaQA Reader span exact match at the development-selected budget.}",
            r"\label{tab:spanem}",
            r"\centering",
            r"\footnotesize",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Test & $K$ & Exact Match (\%) & Count " + r"\\",
            r"\midrule",
            f"TriviaQA & {selected_k} & {100 * trivia_reader['exact_match_accuracy']:.2f} "
            f"& {exact_count:,}/{exact_questions:,} " + r"\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ],
    )

    budget_rows = []
    for budget in sorted(int(k) for k in dev_rows):
        row = dev_rows[str(budget)]
        chosen = "Yes" if budget == selected_k else "No"
        budget_rows.append(
            f"{budget} & {row['questions']:,} & "
            f"{100 * row['top_k_accuracy']['20']:.2f} & {chosen} " + r"\\"
        )
    write_table(
        "budget.tex",
        [
            r"\begin{table}[t]",
            r"\caption{TriviaQA candidate budget selected on development Hit@20.}",
            r"\label{tab:budget}",
            r"\centering",
            r"\footnotesize",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"$K$ & Dev questions & Dev Hit@20 (\%) & Selected " + r"\\",
            r"\midrule",
            *budget_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ],
    )

    conditional_rows = []
    for dataset in ("curatedtrec", "trivia"):
        covered = coverage["per_dataset"][dataset]["covered_questions"]
        full_n = readers[dataset]["questions"]
        dpr = dpr_result(multi, dataset)
        reader = readers[dataset]
        conditional_rows.append(
            f"{LABELS[dataset]} & {100 * covered / full_n:.2f} & "
            f"{100 * dpr['hits']['20'] / covered:.2f} & "
            f"{100 * reader['hits']['20'] / covered:.2f} & "
            f"{100 * reader['hits']['100'] / covered:.2f} " + r"\\"
        )
    write_table(
        "conditional.tex",
        [
            r"\begin{table}[t]",
            r"\caption{Coverage-conditioned answer-hit accuracy (\%).}",
            r"\label{tab:conditional}",
            r"\centering",
            r"\footnotesize",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Test & Coverage & DPR H@20$\mid C$ & Reader H@20$\mid C$ & Reader H@100$\mid C$ " + r"\\",
            r"\midrule",
            *conditional_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ],
    )
    print(f"Reader tables written. TriviaQA K selected on development: {selected_k}.")
else:
    absent = [name for name, path in reader_inputs.items() if not path.is_file()]
    print("Reader tables not regenerated; required Kaggle files missing: " + ", ".join(absent))

print("Regenerated datasets.tex and baselines.tex from saved result summaries.")
