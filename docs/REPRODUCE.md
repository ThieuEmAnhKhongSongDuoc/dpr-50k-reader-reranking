# Reproduce and inspect the completed experiments

## Inspect metrics without a GPU

    git clone https://github.com/ThieuEmAnhKhongSongDuoc/dpr-50k-reader-reranking.git
    cd dpr-50k-reader-reranking
    python scripts/validate_results.py
    python -m pip install -r requirements-analysis.txt
    python scripts/plot_results.py

The validator uses only the standard library; plotting requires matplotlib.

## Download the exact assets

    python scripts/download_artifacts.py --group main

The main group is about 2.38 GB: Multi and Reader weights, matching Multi
passage vectors, fixed 50k corpus and five full QA test files.
The optional single group adds NQ-Single weights and matching passage vectors.
No full Wikipedia download is necessary to use the released 50k sample.

## Kaggle inference

Open notebooks/02_kaggle_reranking.ipynb, enable Internet and GPU T4 x2,
and select one dataset per run. This notebook is an unexecuted recipe.
Original completed runs are recorded under results/raw and results/logs.
With PyTorch installed for the runtime:

    python -m pip install -r requirements-rerank.txt
    python scripts/run_dpr_reader_reranking.py --reader-checkpoint artifacts/reader.cp --retriever-checkpoint artifacts/retriever.cp --passage-vectors artifacts/passage_vectors.pkl --corpus artifacts/wiki50k.tsv --qa-dir artifacts/qa --datasets nq_test --top-docs 200 --reader-batch-size 16 --query-batch-size 8 --reader-gpus 2 --output outputs/reproduced/nq200.json

Replace nq_test with curatedtrec_test or trivia_test. For K=100 use
--top-docs 100 and another output filename. Full K=100 references exist
for NQ and CuratedTREC only. Gold answers are used after ranking for evaluation.

## First-stage DPR

    python scripts/run_dpr_fulltest_eval.py --checkpoint artifacts/retriever.cp --encoded artifacts/passage_vectors.pkl --qa-config public_qa_tests --result-file outputs/reproduced/dpr_multi.json

Run from the repository root; public configs resolve artifacts/ relative
to this directory. Explicit model/vector arguments avoid legacy smoke defaults.
For NQ-Single, download the optional assets and use
run_dpr_single_nq_fulltest_eval.py with its matching checkpoint and vectors.

## Lucene BM25

The historical Windows runner and Java source are retained. It requires
a JDK and Lucene 8.11.2 core/analyzers jars, and its Windows paths need local
configuration. It is not advertised as a portable Kaggle BM25 command.
Settings: k1=0.9, b=0.4, EnglishAnalyzer, title plus passage body indexing.

## Coverage

    python scripts/measure_answer_passage_coverage.py --corpus artifacts/wiki50k.tsv --qa-dir artifacts/qa --output outputs/reproduced/coverage.json

Keep corpus, tests, checkpoint and matcher fixed. Candidate K=200 was
exploratory, not dev-selected. These runs are inference, not training.
