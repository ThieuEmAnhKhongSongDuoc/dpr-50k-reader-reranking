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
