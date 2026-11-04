# Official DPR weights

Actual checkpoint files are downloadable in the artifacts-v1 GitHub release.
They were trained by the DPR authors, not by Group 5 for the reported tests.
manifest.json lists the resource IDs, file sizes and SHA256 checksums.

Official fallback, using the authors' downloader and resource identifiers:

    python -m dpr.data.download_data --resource checkpoint.retriever.multiset.bert-base-encoder --output_dir artifacts/official
    python -m dpr.data.download_data --resource checkpoint.reader.nq-trivia-hybrid.hf-bert-base --output_dir artifacts/official
    python -m dpr.data.download_data --resource checkpoint.retriever.single.nq.bert-base-encoder --output_dir artifacts/official

The official downloader uses nested resource directories. Pass the resulting
paths explicitly when not using the release helper. Legacy files contain
pickle state. The publication copy sets weights_only=False for these trusted,
checksum-verified official files with current PyTorch.
