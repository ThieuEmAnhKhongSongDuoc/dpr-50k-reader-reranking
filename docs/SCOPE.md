# Scope

Completed: fixed 50,000-passage reservoir sample, seed 12345; full-test
Lucene BM25, official DPR Multi and NQ-Single references on five datasets
(28,219 questions); Reader K=200 on full NQ, CuratedTREC and TriviaQA
(15,617 questions); Reader K=100 on NQ and CuratedTREC; corpus coverage.

NQ-Single on non-NQ tests is transfer, not separately trained Single models.
IndexFlatIP is used before and after Reader reranking. There is no paired
HNSW/FlatIP ablation and no claim of a new indexing or Reader architecture.

Only Reader relevance logits rank passages; answer-span outputs are ignored.
Hit@k uses NFD normalization, uncased DPR SimpleTokenizer phrase matching
in passage bodies. Coverage is an availability ceiling, not retrieval accuracy.

Old score-fusion results are excluded because BM25 numeric passage IDs were
not normalized to DPR wiki-prefixed IDs. No corrected fusion score is claimed.
Incomplete training/negative-mining runs and exploratory 3000-question samples
are outside the paper's results. Small-corpus scores are not directly
comparable to original full-Wikipedia DPR scores.
