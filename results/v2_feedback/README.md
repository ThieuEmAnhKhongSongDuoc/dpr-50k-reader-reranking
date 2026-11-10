# Version 2 feedback results

These files contain the corrected full-test baselines and the validated BM25+DPR fusion run. Reader feedback reruns (FP16 CuratedTREC and dev-selected TriviaQA) are pending Kaggle execution; no corrected Reader numbers are included here yet.

## Corrected CuratedTREC

CuratedTREC answer references are evaluated with DPR's official regular-expression matcher. The 50k-corpus coverage is 609/694 questions (87.75%). Results are in `coverage.json`, `lucene_bm25.json`, `dpr_multi.json`, and `dpr_single_nq.json`.

The corrected CuratedTREC BM25, DPR Multi, and DPR NQ-Single per-query files are in `per_query/`; each contains all 694 questions and ranked candidate IDs and scores.

## BM25+DPR fusion

The local fusion retrieves the top 2,000 results from Lucene BM25 and DPR Multi, maps both passage ID formats to `wiki:<id>`, scores the union using `BM25 + 1.1 * DPR dot product`, and stores the final top 100. `bm25_dpr_fusion.json` contains all five full-test summaries. The compressed per-query file contains 28,219 rows with 100 candidate IDs and component/fused scores per question.

The per-query artifact is 85,692,194 bytes. Its SHA256 is recorded in the root provenance file. It is included here for per-question auditing and paired analysis.
