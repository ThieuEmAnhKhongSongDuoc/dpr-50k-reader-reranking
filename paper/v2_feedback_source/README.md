# Group 5 DPR empirical paper

This is an English IEEE conference-format manuscript and its editable LaTeX project.
It reports saved inference experiments, not new training or new retrieval runs.

Authors, in the requested order:

1. Hồ Thành Tài
2. Lê Trần Bình An
3. Phan Bá Song Toàn
4. Trần Quang Long
5. Trịnh Minh Bảo Trân

The author block uses the supplied FPT University affiliation and omits the
team label. The proposed contribution allocation covers Versions 1 and 2 and
should be confirmed by all five authors before final submission.

## Open and compile

- Main file: `main.tex`.
- Preferred Unicode compiler: XeLaTeX (or Tectonic).
- In Overleaf, upload the source ZIP, choose `main.tex`, select XeLaTeX, and
  recompile. The references use BibTeX with `IEEEtran` style.
- With a local TeX distribution: `latexmk -xelatex main.tex`.
- With Tectonic: `tectonic main.tex`.
- A pdfLaTeX compatibility branch uses T5 encoding for the Vietnamese names.
- Standard packages required: IEEEtran, fontspec/iftex, amsmath, booktabs,
  graphicx, TikZ, PGFPlots, cite and hyperref. TeX Gyre Termes supplies the
  Times-compatible Unicode text font on the XeLaTeX path. The source falls back
  to Times New Roman when TeX Gyre Termes is unavailable on Windows.

## Verify the numbers

Run `python build_tables.py` with Python 3. The script uses only the standard
library, checks stored percentages against hit counts, verifies denominators
and coverage bounds, and regenerates the baseline and coverage tables. Reader
tables are generated only after the required Kaggle JSON outputs are copied
into `data/`. The script does not download data, load a checkpoint, train, or
rerun inference.

`data/` contains compact result summaries copied from existing saved outputs.
Path-only metadata is omitted from the compact copies; experimental counts,
scores, timing and settings are retained. `provenance.json` records source file
names and SHA256 hashes, the sampled corpus digest, and resource identifiers.

## Feedback revision status

- The corpus remains the fixed 50,000-passage reservoir sample (seed 12345).
- Full-test BM25 and DPR references cover all 28,219 questions.
- Corrected CuratedTREC regex results are saved for coverage, BM25, DPR
  Single/Multi, and BM25+DPR fusion.
- The CuratedTREC evaluation path calls DPR's `regex_match` directly, with the
  same Unicode normalization and regex flags as the upstream validation code.
- The corrected five-test BM25+DPR fusion run and per-query scores are saved.
- The official Reader feedback runs are complete: CuratedTREC test (694
  questions), TriviaQA development (8,837), and TriviaQA test (11,313), all at
  K=100 with FP16 on two Kaggle T4 GPUs. TriviaQA development was run before
  test. K=100 was the fixed requested budget; no alternative-K sweep is claimed.
- Reader aggregate JSON outputs are copied into `data/`; their per-query
  JSONL.GZ files are stored under `results/v2_feedback/per_query/`.
  `python build_tables.py` regenerates the reranking and span-EM tables without
  running inference.
- A frozen `cross-encoder/ms-marco-MiniLM-L6-v2` comparison uses the same saved
  100 candidate IDs as the DPR and Reader rankings on TriviaQA dev/test and
  CuratedTREC test. It runs in FP32 with max length 512, batch size 64, and two
  Kaggle T4 GPUs. The aggregate summary is in `data/cross_encoder_summary.json`;
  complete per-query logits, ranks, candidate IDs, and answer-hit labels are in
  `../../results/v2_feedback/cross_encoder/per_query/`.
- The table builder checks every cross-encoder query against its saved Reader
  row, including identical candidate IDs, answer-hit labels, and existing DPR
  and Reader ranks/scores, then verifies the aggregate Hit@k counts before
  writing the three-way comparison table.
- `../group5-dpr-paper-v2.pdf` is the pre-cross-encoder Version 2 PDF snapshot;
  it is not synchronized with the updated LaTeX source in this branch. The
  current environment has no XeLaTeX, pdfLaTeX, latexmk, or Tectonic binary, so
  the updated source has not been compiled here. Recompile `main.tex` with
  XeLaTeX in Overleaf before submitting. The Version 1 PDF remains at
  `../group5-dpr-paper.pdf`.

The large corpus, models, API tokens, virtual environments, and Kaggle
credentials are deliberately not in the source archive.

## Corrected BM25+DPR fusion

The feedback revision normalizes bare Lucene passage IDs to the dense `wiki:`
ID format before union and lookup. Results for the corrected fusion are stored
in `data/bm25_dpr_fusion.json`; the saved run uses the top 2,000 candidates
from each retriever and the raw score sum with DPR weight 1.1.

## Implementation references in the accompanying DPR project

- `scripts/download_wiki_sample.py`
- `scripts/run_dpr_reader_reranking.py`
- `scripts/measure_answer_passage_coverage.py`
- `scripts/run_lucene_bm25_fulltest.py`
- `scripts/run_dpr_bm25_fusion_fulltest.py`
- `dpr/models/reader.py`
- `dpr/models/biencoder.py`
- `dpr/indexer/faiss_indexers.py`

The original author code and checkpoint resource registry are available at
https://github.com/facebookresearch/DPR. Third-party code and datasets retain
their original licenses. The evidence summaries are supplied for auditing the
reported arithmetic; reproducing GPU inference also requires the accompanying
DPR code, the specified model weights, and matching corpus and QA files.
