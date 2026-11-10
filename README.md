# DPR 50K: Group 5 reproduction and Reader reranking

This public repository contains the DPR project code, reproduction artifacts, and saved evaluation results for a fixed sample of 50,000 Wikipedia passages.

Authors: Hồ Thành Tài; Lê Trần Bình An; Phan Ba Song Toàn; Tran Quang Long; Trịnh Minh Bảo Trân.

Department of Artificial Intelligence, FPT University, Da Nang, Vietnam.

## Version 2 feedback status

The submitted PDF in `paper/group5-dpr-paper.pdf` is the Version 1 copy. Version 2 corrects CuratedTREC answer matching, repairs BM25+DPR passage-ID joining, and adds per-query fusion results. Reader FP16 feedback runs are still pending Kaggle execution, so the old Reader results are not presented here as corrected final results.

## Corrected full-test baselines

The table reports Hit@20 / Hit@100 on all five full DPR-formatted test sets and the 50k passage corpus. CuratedTREC uses DPR's official regex matcher.

| Dataset | Questions | Corpus coverage | Lucene BM25 | DPR Multi | BM25+DPR fusion |
|---|---:|---:|---:|---:|---:|
| NQ | 3,610 | 60.72% | 18.86 / 29.72% | 25.68 / 37.34% | 26.12 / 37.95% |
| TriviaQA | 11,313 | 72.77% | 23.52 / 35.91% | 33.60 / 48.04% | 34.62 / 48.86% |
| WebQuestions | 2,032 | 68.31% | 26.97 / 38.48% | 36.27 / 48.18% | 37.06 / 48.92% |
| CuratedTREC | 694 | 87.75% | 40.92 / 55.48% | 51.15 / 68.01% | 52.16 / 68.16% |
| SQuAD 1.1 | 10,570 | 57.82% | 16.15 / 27.59% | 21.42 / 34.37% | 22.50 / 34.91% |

Across the five tests, 18,534/28,219 questions (65.68%) have an answer match somewhere in the corpus. CuratedTREC coverage is 609/694 (87.75%); the former literal-string estimate is superseded.

The local hybrid retrieves up to 2,000 candidates per retriever, normalizes passage IDs to `wiki:<id>`, and ranks the union by `BM25 + 1.1 × DPR dot product`. It is documented as the measured local fusion setting.

## Results and reproduction files

| Requirement | Location |
|---|---|
| Original DPR architecture and entry points | `dpr/`, `conf/`, root Python files |
| Sampling, evaluation, fusion and Reader code | `scripts/` |
| Corrected TREC and five-test fusion summaries | `results/v2_feedback/` |
| Per-question BM25+DPR candidates and scores | `results/v2_feedback/bm25_dpr_fusion_per_query.jsonl.gz` |
| Kaggle Reader feedback run instructions | `docs/KAGGLE_VERSION2_RUN_GUIDE.md` |
| Version 1 paper PDF | `paper/group5-dpr-paper.pdf` |
| Editable Version 2 feedback draft (LaTeX source and verified tables) | `paper/v2_feedback_source/` |
| Model/data artifact manifest | `weights/manifest.json` |

The v2 Reader reruns will use FP16 on Kaggle T4 x2. TriviaQA's candidate budget is selected on its dev split; the full test is reserved for evaluation. The required TriviaQA Reader run also reports Exact Match from the Reader span head. Corrected CuratedTREC Reader metrics and final author contributions will be added after the Kaggle outputs and verified member roles are available.

Large datasets and checkpoints are not stored in this repository. See `docs/REPRODUCE.md` for resource identifiers and setup. Upstream DPR code and third-party resources retain their original licenses.
