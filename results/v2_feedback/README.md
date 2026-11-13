# Version 2 feedback results

These files contain the corrected full-test baselines, the validated BM25+DPR fusion run, and the completed official DPR Reader feedback runs. Reader inference used FP16, batch size 16, two Kaggle T4 GPUs, and a 100-candidate budget. TriviaQA development was run before its test split. The budget was fixed at K=100 as required; no alternative-K sweep is claimed.

## Corrected CuratedTREC

CuratedTREC answer references are evaluated with DPR's official regular-expression matcher. The 50k-corpus coverage is 609/694 questions (87.75%). Results are in `coverage.json`, `lucene_bm25.json`, `dpr_multi.json`, and `dpr_single_nq.json`.

The corrected CuratedTREC BM25, DPR Multi, and DPR NQ-Single per-query files are in `per_query/`; each contains all 694 questions and ranked candidate IDs and scores.

The corrected Reader result is 386/694 Hit@20 (55.62%) and 472/694 Hit@100 (68.01%). The per-query output in `per_query/curatedtrec_reader_regex_k100_fp16.jsonl.gz` contains 694 rows and uses `match_type: regex`. CuratedTREC span Exact Match is not reported because the references are regex patterns rather than literal answer strings.

## TriviaQA Reader

The development run has 8,837 questions, 4,279 Hit@100 (48.42%), and 1,105 Reader span exact matches (12.50%). The held-out test run has 11,313 questions, 4,678 Hit@20 (41.35%), 5,437 Hit@100 (48.06%), and 1,419 span exact matches (12.54%). The compressed per-query files in `per_query/` retain query IDs, all 100 passage IDs, reader ranks/scores, answer-hit labels, and span predictions.

## BM25+DPR fusion

The local fusion retrieves the top 2,000 results from Lucene BM25 and DPR Multi, maps both passage ID formats to `wiki:<id>`, scores the union using `BM25 + 1.1 * DPR dot product`, and stores the final top 100. `bm25_dpr_fusion.json` contains all five full-test summaries. The compressed per-query file contains 28,219 rows with 100 candidate IDs and component/fused scores per question.

The per-query artifact is 85,692,194 bytes. Its SHA256 is recorded in the root provenance file. It is included here for per-question auditing and paired analysis.

## Frozen cross-encoder comparison

The released `cross-encoder/ms-marco-MiniLM-L6-v2` checkpoint was evaluated without fine-tuning on the exact same 100 candidate passage IDs used by the saved DPR Reader reranking runs. The model scores question--(passage title + text) pairs with one relevance logit; it does not receive answers or produce answer spans. Run settings were max length 512, FP32, batch size 64, and two Kaggle Tesla T4 GPUs. TriviaQA development was processed before test; CuratedTREC uses the saved official regex answer-hit labels.

| Split | Questions | DPR Hit@20 | Cross-encoder Hit@20 | Reader Hit@20 | Cross-encoder Δ vs DPR |
|---|---:|---:|---:|---:|---:|
| TriviaQA dev | 8,837 | 34.19% | 34.53% | 41.65% | +0.34 pp |
| CuratedTREC test | 694 | 51.15% | 52.74% | 55.62% | +1.59 pp |
| TriviaQA test | 11,313 | 33.63% | 33.89% | 41.35% | +0.27 pp |

Hit@100 is unchanged within each split for all three rankings because all methods reorder the same 100 candidates. The cross-encoder improves modestly over the original DPR ordering at the top ranks but remains below the DPR Reader. These are answer-containing-passage metrics, not answer-span Exact Match.

`cross_encoder_summary.json` contains aggregate settings, counts, rates, and runtimes. `per_query/` contains the complete candidate-level score files: 8,837 TriviaQA development rows, 694 CuratedTREC test rows, and 11,313 TriviaQA test rows. Each query row stores all candidate passage IDs, DPR ranks, Reader scores, answer-hit labels, cross-encoder logits, and cross-encoder ranks. The per-query records were checked against the saved Reader records: query IDs, all 100 passage IDs, answer-hit labels, DPR ranks, and Reader ranks/scores match for every row. CuratedTREC rows all declare `match_type: regex`.

The candidate-level TriviaQA test DPR counts are reconstructed from the paired saved records, so they differ slightly from the independently aggregated DPR-only full-test counts in `dpr_multi.json` (3,804 vs. 3,801 at Hit@20; 5,437 vs. 5,435 at Hit@100). The paper reports the paired candidate-level values for the three-way comparison and retains the independent full-test aggregate in the first-stage baseline table.

The source archive was `crossencoder_results.zip` (61,077,296 bytes; SHA256 `ca2460269884b5e644f18e5f2e5a8e63a32f6de87eb3c49f80ae24d2cf5ab54e`). File sizes and SHA256 values for the committed artifacts are recorded in `manifest.json`.
