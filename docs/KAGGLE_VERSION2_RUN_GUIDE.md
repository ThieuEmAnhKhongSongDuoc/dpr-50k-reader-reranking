# Kaggle run guide for the teacher feedback

The requested Kaggle Reader jobs have completed. Their aggregate and per-query outputs are saved under `results/v2_feedback/`. This repository does not include the earlier unexecuted K=200 notebook recipe or the separate cross-encoder experiment.

## Re-run the Reader jobs

Use the existing DPR 50K reranking input bundle and GPU T4 x2. Set `INPUT` to the actual `/kaggle/input/...` directory containing `reader.cp`, `retriever.cp`, `passage_vectors.pkl`, `wiki50k.tsv`, and the full-test CSV files. Install the reranking requirements. The TriviaQA development CSV is downloaded separately through DPR's official resource ID and stored under the project root:

```bash
python -m pip install -r requirements-rerank.txt
python -m pip install wget==3.2
python dpr/data/download_data.py \
  --resource data.retriever.qas.trivia-dev \
  --output_dir data/raw/teacher_review_dev
```

Then, from the repository root, run each split at the fixed K=100 FP16 setting. Run development before test:

```bash
for dataset in trivia_dev curatedtrec_test trivia_test; do
  python scripts/run_dpr_reader_reranking.py \
    --reader-checkpoint "$INPUT/reader.cp" \
    --retriever-checkpoint "$INPUT/retriever.cp" \
    --passage-vectors "$INPUT/passage_vectors.pkl" \
    --corpus "$INPUT/wiki50k.tsv" \
    --qa-dir "$INPUT" \
    --datasets "$dataset" \
    --reader-gpus 2 --reader-batch-size 16 --reader-precision fp16 \
    --top-docs 100 \
    --output "/kaggle/working/${dataset}_reader_k100_fp16.json"
done
```

The `trivia_dev` split is read from `data/raw/teacher_review_dev/downloads/data/retriever/qas/trivia-dev.csv`; test CSVs are read from `$INPUT`. Replace `INPUT` with the mounted bundle path. This is a rerun recipe; the saved results in this repository were produced in Kaggle.

## Saved outputs

Aggregate JSON files and per-query `.jsonl.gz` files are committed under `results/v2_feedback/`. Paper aggregate inputs are in `paper/v2_feedback_source/data/`. Per-query Reader outputs are in `results/v2_feedback/per_query/`.
