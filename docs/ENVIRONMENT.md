# Recorded execution environment

Completed Top-200 Kaggle logs report Python 3.13.15, PyTorch 2.11.0+cu128,
Transformers 4.57.3, FAISS 1.15.1 and two Tesla T4 GPUs. Hydra 1.3.2 and
jsonlines 4.0.0 are pinned in the archived notebook. Reader inference used
FP32, candidate batch 16, query batch 8, sequence length 350 and DataParallel.
Earlier Windows baselines used requirements-local.txt.

The public inference requirements target the Kaggle compatibility path.
Publishing this repository does not mean a new GPU experiment was executed.
See results/logs for actual recorded environments.
