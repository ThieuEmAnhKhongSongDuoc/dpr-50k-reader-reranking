# Kaggle run guide for DPR version 2 feedback

The laptop-side corrections are complete, but Reader inference must run on Kaggle T4 x2. The private Kaggle session is separate from this workspace; this guide does not contain or request an API token.

## 1. Prepare a Kaggle notebook

1. Open the existing private notebook **DPR 50K Reader Rerank** and choose **Copy & edit**.
2. In **Input**, attach the existing dataset **DPR 50K reranking input bundle**. It must contain reader.cp, retriever.cp, passage_vectors.pkl, wiki50k.tsv, and the official test QA CSV files.
3. In **Settings**, select **GPU T4 x2** and turn Internet on so the notebook can clone the public version-2 branch and use DPR's official small TriviaQA dev-QA resource.
4. Keep the setup cell that installs transformers==4.57.3, hydra-core==1.3.2, and jsonlines==4.0.0; also install wget for DPR's official downloader.

## 2. Replace the project-copy cell

Use this cell after the setup cell. It fetches the version-2 code from the public review branch, keeps model/data assets mounted read-only, and applies the Transformers tokenizer compatibility edit used by the original Kaggle run.

~~~python
from pathlib import Path
import shutil, subprocess, sys

PROJECT = Path('/kaggle/working/dpr-project')
if PROJECT.exists():
    shutil.rmtree(PROJECT)
subprocess.run([
    'git', 'clone', '--depth', '1', '--branch', 'version-2-feedback',
    'https://github.com/ThieuEmAnhKhongSongDuoc/dpr-50k-reader-reranking.git',
    str(PROJECT),
], check=True)

hf_path = PROJECT / 'dpr' / 'models' / 'hf_models.py'
hf_text = hf_path.read_text(encoding='utf-8')
assert hf_text.count('pad_to_max_length=False') == 2
hf_path.write_text(hf_text.replace('pad_to_max_length=False', 'padding=False'), encoding='utf-8')

subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q', 'wget==3.2'
], check=True)

dev_root = PROJECT / 'data' / 'raw' / 'teacher_review_dev'
subprocess.run([
    sys.executable, str(PROJECT / 'dpr' / 'data' / 'download_data.py'),
    '--resource', 'data.retriever.qas.trivia-dev',
    '--output_dir', str(dev_root),
], check=True)
dev_qa = dev_root / 'downloads' / 'data' / 'retriever' / 'qas' / 'trivia-dev.csv'
assert dev_qa.is_file(), dev_qa
print('Project:', PROJECT)
print('Official TriviaQA dev QA:', dev_qa, '| bytes:', dev_qa.stat().st_size)
~~~

## 3. Select candidate budget on TriviaQA dev

Run K=200 once over the full official DPR TriviaQA dev QA split. The saved per-query file contains each candidate's original DPR rank, Reader relevance score, and answer-hit label. The summarizer computes K=100 and K=200 from the same pass by restricting K=100 to the original DPR top 100 before sorting with Reader scores. Select the candidate budget using development Hit@20 only.

~~~python
import subprocess, sys
from pathlib import Path

DATA = next(Path('/kaggle/input').rglob('reader.cp')).parent
dev_output = Path('/kaggle/working/trivia_dev_reader_k200_fp16.json')
subprocess.run([
    sys.executable, str(PROJECT / 'scripts' / 'run_dpr_reader_reranking.py'),
    '--reader-checkpoint', str(DATA / 'reader.cp'),
    '--retriever-checkpoint', str(DATA / 'retriever.cp'),
    '--passage-vectors', str(DATA / 'passage_vectors.pkl'),
    '--corpus', str(DATA / 'wiki50k.tsv'),
    '--qa-dir', str(DATA),
    '--datasets', 'trivia_dev',
    '--reader-gpus', '2',
    '--reader-batch-size', '16',
    '--reader-precision', 'fp16',
    '--top-docs', '200',
    '--output', str(dev_output),
], check=True)

per_query = dev_output.with_name(dev_output.stem + '_per_query') / 'trivia_dev.jsonl.gz'
subprocess.run([
    sys.executable, str(PROJECT / 'scripts' / 'summarize_reader_candidate_budgets.py'),
    '--input', str(per_query),
    '--budgets', '100', '200',
    '--output', '/kaggle/working/trivia_dev_budget_selection.json',
], check=True)
~~~

Choose between the two candidate budgets already used in the project (K=100 and K=200) from the dev comparison using the predeclared primary cutoff (Hit@20); do not select K from test results. Keep the output JSON and per-query gzip file.

Set the test budget from the saved development comparison. Ties at Hit@20 are resolved in favor of the smaller K.

~~~python
import json
dev_summary = json.loads(Path('/kaggle/working/trivia_dev_budget_selection.json').read_text())
budget_rows = dev_summary['results_by_budget']
TRIVIA_TEST_K = min(
    (int(k) for k in budget_rows),
    key=lambda k: (-budget_rows[str(k)]['top_k_accuracy']['20'], k),
)
print('Development-selected TriviaQA test K:', TRIVIA_TEST_K)
~~~

## 4. Run corrected CuratedTREC and complete TriviaQA K=100

After recording the dev-selected K, run the requested full-test checks. These jobs can be run in separate Kaggle versions if the session approaches its time limit.

~~~python
trec_output = Path('/kaggle/working/curatedtrec_reader_regex_k200_fp16.json')
subprocess.run([
    sys.executable, str(PROJECT / 'scripts' / 'run_dpr_reader_reranking.py'),
    '--reader-checkpoint', str(DATA / 'reader.cp'),
    '--retriever-checkpoint', str(DATA / 'retriever.cp'),
    '--passage-vectors', str(DATA / 'passage_vectors.pkl'),
    '--corpus', str(DATA / 'wiki50k.tsv'),
    '--qa-dir', str(DATA),
    '--datasets', 'curatedtrec_test',
    '--reader-gpus', '2',
    '--reader-batch-size', '16',
    '--reader-precision', 'fp16',
    '--top-docs', '200',
    '--output', str(trec_output),
], check=True)
~~~

~~~python
trivia_test_output = Path(f'/kaggle/working/trivia_reader_k{TRIVIA_TEST_K}_fp16.json')
subprocess.run([
    sys.executable, str(PROJECT / 'scripts' / 'run_dpr_reader_reranking.py'),
    '--reader-checkpoint', str(DATA / 'reader.cp'),
    '--retriever-checkpoint', str(DATA / 'retriever.cp'),
    '--passage-vectors', str(DATA / 'passage_vectors.pkl'),
    '--corpus', str(DATA / 'wiki50k.tsv'),
    '--qa-dir', str(DATA),
    '--datasets', 'trivia_test',
    '--reader-gpus', '2',
    '--reader-batch-size', '16',
    '--reader-precision', 'fp16',
    '--top-docs', str(TRIVIA_TEST_K),
    '--output', str(trivia_test_output),
], check=True)
~~~

~~~python
if TRIVIA_TEST_K == 200:
    trivia_test_per_query = trivia_test_output.with_name(
        trivia_test_output.stem + '_per_query'
    ) / 'trivia_test.jsonl.gz'
    subprocess.run([
        sys.executable, str(PROJECT / 'scripts' / 'summarize_reader_candidate_budgets.py'),
        '--input', str(trivia_test_per_query),
        '--budgets', '100', '200',
        '--split', 'test',
        '--output', '/kaggle/working/trivia_test_budget_metrics.json',
    ], check=True)
~~~

## 5. Save and bring outputs back

The TriviaQA test Reader run uses the budget selected on dev, and its aggregate JSON reports answer-hit accuracy and span Exact Match at that budget. If dev selects K=200, the per-query candidate scores also derive the required K=100 answer-hit metrics without another Reader pass. CuratedTREC Exact Match is omitted because its gold answers are regex patterns.

At the end of a successful run, download these from the Kaggle notebook's Output tab:

- trivia_dev_budget_selection.json
- curatedtrec_reader_regex_k200_fp16.json and its per-query gzip file
- trivia_reader_k{selected K}_fp16.json and its per-query gzip file
- trivia_test_budget_metrics.json only if dev selected K=200
- the TriviaQA dev K=200 per-query gzip file, which audits K selection

Do not reuse the previous CuratedTREC Reader aggregate: it used string matching and must be replaced.

## Expected resource use

The previous full TriviaQA K=200 FP32 Reader run processed 2,262,600 question-passage pairs in 28,197 seconds on T4 x2. FP16 is enabled here, but actual runtime depends on Kaggle scheduling and GPU load. The planned full TriviaQA K=100 run has 1,131,300 pairs; TREC K=200 has 138,800 pairs; dev K=200 has 1,767,400 pairs. These are work-size figures, not completion-time guarantees.
