# Version 2 scope

This revision addresses only the teacher's listed feedback:

- Apply the official DPR regex matcher to CuratedTREC coverage, BM25, DPR, and Reader answer-hit evaluation.
- Retain the corrected BM25+DPR passage-ID normalization and report the corrected full-test fusion results.
- Run the official Reader with FP16 at the requested K=100 setting, using TriviaQA development before the full TriviaQA test.
- Report Reader span Exact Match on TriviaQA and keep per-question candidates/scores for auditing.
- Keep the 50,000-passage corpus and existing official checkpoints fixed.

The corrected baseline and Reader output files are saved under `results/v2_feedback/`; Reader summaries and per-query outputs are also included in the paper source package where applicable. K=100 was fixed for the requested run, and no alternative-K sweep is claimed. Model training, a new corpus, and a separate reranker are outside this revision. The paper's Author Contributions section still needs verified roles from the five authors.
