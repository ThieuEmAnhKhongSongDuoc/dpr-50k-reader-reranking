# Upstream and group changes

Reference: facebookresearch/DPR commit a31212dc0a54dfa85d8bfa01e1669f149ac832b7.
Full upstream history and remote are retained. The old downloaded ZIP had
no Git metadata, so its exact commit cannot be proved from its directory name.
The public copy overlays the existing project source and selected group scripts.
Architecture and original DPR loss are retained. See upstream-changes.patch
for tracked-file differences to the reference.

Compatibility: Hugging Face imports, AdamW fallback, tokenizer padding keyword,
explicit legacy checkpoint loading for modern PyTorch. The last two are
publication portability changes, not new reported GPU experiments.

Group additions: reservoir sampling, Windows-safe evaluation, Lucene runner,
coverage measurement, Reader reranking, two-GPU batching, result validation,
figures, notebooks and documentation. Credentials, environments, temporary
outputs, invalid fusion and incomplete negative-mining work are excluded.
