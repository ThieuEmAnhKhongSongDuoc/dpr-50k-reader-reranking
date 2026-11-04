"""Rerank DPR+FAISS candidates with the original DPR extractive Reader.

This is a second-stage experiment: DPR's question encoder and IndexFlatIP
retrieve candidates; the DPR Reader's cross-attention relevance head rescores
each (question, passage) pair. The original retrieval/evaluation scripts and
their result files are left untouched.

The script requires an already downloaded DPR Reader checkpoint and the
passage embeddings produced with the same retriever checkpoint supplied here.
It does not download models or data.
"""

import argparse
import ast
import csv
import json
import pickle
import sys
import time
import unicodedata
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dpr.data.reader_data import _concat_pair
from dpr.indexer.faiss_indexers import DenseFlatIndexer
from dpr.models import init_biencoder_components, init_reader_components
from dpr.models.reader import _create_token_type_ids, _pad_to_len
from dpr.options import set_cfg_params_from_state, setup_cfg_gpu
from dpr.utils.model_utils import get_model_obj, load_states_from_checkpoint
from dpr.utils.tokenizers import SimpleTokenizer


RETRIEVER_CHECKPOINT = ROOT / "artifacts" / "retriever.cp"
PASSAGE_VECTORS = ROOT / "artifacts" / "passage_vectors.pkl"
CORPUS = ROOT / "artifacts" / "wiki50k.tsv"
QA_DIR = ROOT / "artifacts" / "qa"
DEFAULT_RESULT = ROOT / "outputs" / "local_retrieval" / "paper_multi_50k_fulltest_reader_reranked.json"
TEST_FILES = {
    "nq_test": "nq-test.csv",
    "trivia_test": "trivia-test.csv",
    "webq_test": "webq-test.csv",
    "curatedtrec_test": "curatedtrec-test.csv",
    "squad1_test": "squad1-test.csv",
}


def load_qa(path):
    questions, answers = [], []
    if not path.exists() and path.name.endswith("-test.csv"):
        # The official DPR downloader preserves the `.qa.csv` resource names.
        candidate = path.with_name(path.name.replace("-test.csv", "-test.qa.csv"))
        if candidate.exists():
            path = candidate
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream, delimiter="\t"):
            if len(row) < 2:
                continue
            questions.append(row[0])
            parsed = ast.literal_eval(row[1])
            answers.append(parsed if isinstance(parsed, (list, tuple)) else [parsed])
    return questions, answers


def load_passages(path):
    passages = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream, delimiter="\t"):
            if not row or row[0] == "id":
                continue
            if len(row) < 3:
                raise ValueError(f"Malformed corpus row: expected id, text, title; got {len(row)} columns")
            passages[f"wiki:{row[0]}"] = (row[2], row[1])  # title, passage text
    return passages


def encode_questions(encoder, tensorizer, questions, batch_size, device):
    vectors = []
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(questions), batch_size):
            token_ids = [tensorizer.text_to_tensor(q) for q in questions[start : start + batch_size]]
            batch = torch.nn.utils.rnn.pad_sequence(
                token_ids, batch_first=True, padding_value=tensorizer.get_pad_id()
            ).to(device)
            segments = torch.zeros_like(batch)
            attention = tensorizer.get_attn_mask(batch)
            _, encoded, _ = encoder(batch, segments, attention)
            vectors.append(encoded.detach().cpu().numpy())
    return np.concatenate(vectors, axis=0)


def make_reader_pair(question, title, text, tensorizer):
    """Match DPR Reader preprocessing: [CLS] question [SEP] title [SEP] passage."""
    question_and_title = tensorizer.text_to_tensor(text=title, title=question, add_special_tokens=True)
    passage_tokens = tensorizer.text_to_tensor(text, add_special_tokens=False)
    sequence, _ = _concat_pair(question_and_title, passage_tokens)
    sequence = _pad_to_len(sequence, tensorizer.get_pad_id(), tensorizer.max_length)
    token_types = _create_token_type_ids(sequence.unsqueeze(0), tensorizer.get_token_id("[SEP]"))[0]
    return sequence, token_types


def score_candidates(reader, tensorizer, question, candidate_ids, passages, batch_size, device):
    ranked = []
    reader.eval()
    with torch.inference_mode():
        for start in range(0, len(candidate_ids), batch_size):
            ids = candidate_ids[start : start + batch_size]
            pairs = [make_reader_pair(question, *passages[pid], tensorizer) for pid in ids]
            # Treat each question-passage pair as one item on dimension 0.
            # This lets DataParallel split a candidate batch across GPUs while
            # preserving Reader's expected [questions, passages, sequence] shape.
            input_ids = torch.stack([pair[0] for pair in pairs]).unsqueeze(1).to(device)
            token_types = torch.stack([pair[1] for pair in pairs]).unsqueeze(1).to(device)
            attention = input_ids.ne(tensorizer.get_pad_id()).long()
            _start, _end, relevance = reader(input_ids, attention, token_types)
            ranked.extend(zip(ids, relevance[:, 0].float().cpu().tolist()))
    return [pid for pid, _score in sorted(ranked, key=lambda item: item[1], reverse=True)]


def normalized_phrase(text, tokenizer):
    tokens = tokenizer.tokenize(unicodedata.normalize("NFD", str(text))).words(uncased=True)
    return " " + " ".join(tokens) + " " if tokens else ""


def normalized_passage(text, tokenizer):
    tokens = tokenizer.tokenize(unicodedata.normalize("NFD", text)).words(uncased=True)
    return " " + " ".join(tokens) + " "


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader-checkpoint", type=Path, required=True,
                        help="Official DPR Reader checkpoint; e.g. nq-trivia-hybrid hf_bert_base.cp")
    parser.add_argument("--retriever-checkpoint", type=Path, default=RETRIEVER_CHECKPOINT)
    parser.add_argument("--passage-vectors", type=Path, default=PASSAGE_VECTORS)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--qa-dir", type=Path, default=QA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--top-docs", type=int, default=100)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--reader-batch-size", type=int, default=4,
                        help="Candidate passages per cross-encoder forward pass; keep small on 4GB VRAM")
    parser.add_argument("--reader-gpus", type=int, default=1,
                        help="Number of visible GPUs for DPR Reader DataParallel inference (e.g. 2 for Kaggle T4 x2)")
    parser.add_argument("--datasets", nargs="+", choices=list(TEST_FILES), default=list(TEST_FILES),
                        help="Test splits to run; run one split per Kaggle session if needed")
    parser.add_argument("--max-questions", type=int, default=0,
                        help="Optional pipeline-smoke cap per split; do not use capped test scores for model selection")
    args = parser.parse_args()

    if args.reader_gpus < 1:
        parser.error("--reader-gpus must be at least 1")
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.reader_gpus > 1 and args.reader_gpus > available_gpus:
        parser.error(
            f"Requested --reader-gpus {args.reader_gpus}, but only {available_gpus} CUDA GPU(s) are available"
        )

    for required in (args.reader_checkpoint, args.retriever_checkpoint, args.passage_vectors, args.corpus):
        if not required.is_file():
            raise FileNotFoundError(required)

    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base="1.1"):
        retriever_cfg = compose(
            config_name="dense_retriever",
            overrides=["datasets=local_qa_tests", "ctx_sources=local_wiki50k", "hydra.job.chdir=False",
                       f"model_file={args.retriever_checkpoint.as_posix()}"],
        )
        reader_cfg = compose(
            config_name="extractive_reader_train_cfg",
            overrides=["hydra.job.chdir=False", f"model_file={args.reader_checkpoint.as_posix()}"],
        )

    retriever_cfg = setup_cfg_gpu(retriever_cfg)
    retriever_state = load_states_from_checkpoint(str(args.retriever_checkpoint))
    set_cfg_params_from_state(retriever_state.encoder_params, retriever_cfg)
    query_tensorizer, biencoder, _ = init_biencoder_components(
        retriever_cfg.encoder.encoder_model_type, retriever_cfg, inference_only=True
    )
    biencoder.load_state(retriever_state, strict=False)
    question_encoder = get_model_obj(biencoder.question_model).to(retriever_cfg.device).eval()

    reader_state = load_states_from_checkpoint(str(args.reader_checkpoint))
    set_cfg_params_from_state(reader_state.encoder_params, reader_cfg)
    reader_cfg = setup_cfg_gpu(reader_cfg)
    reader_tensorizer, reader, _ = init_reader_components(
        reader_cfg.encoder.encoder_model_type, reader_cfg, inference_only=True
    )
    reader_model = get_model_obj(reader)
    missing, unexpected = reader_model.load_state_dict(reader_state.model_dict, strict=False)
    loaded_keys = len(reader_model.state_dict()) - len(missing)
    if loaded_keys < 0.95 * len(reader_model.state_dict()):
        raise RuntimeError(
            "Reader checkpoint did not load cleanly: "
            f"loaded {loaded_keys}/{len(reader_model.state_dict())} model keys, "
            f"{len(unexpected)} unexpected keys"
        )
    if unexpected:
        print(f"Warning: ignoring {len(unexpected)} checkpoint keys not used by this Transformers version", flush=True)
    reader = reader_model.to(reader_cfg.device).eval()
    if args.reader_gpus > 1:
        reader = torch.nn.DataParallel(
            reader,
            device_ids=list(range(args.reader_gpus)),
            output_device=0,
        )
        print(f"DPR Reader DataParallel enabled on {args.reader_gpus} GPUs", flush=True)
    reader_tensorizer.set_pad_to_max(False)

    answer_matcher = SimpleTokenizer()
    passages = load_passages(args.corpus)
    normalized_passages = {pid: normalized_passage(text, answer_matcher) for pid, (_title, text) in passages.items()}
    with args.passage_vectors.open("rb") as stream:
        encoded = pickle.load(stream)
    vector_size = int(np.asarray(encoded[0][1]).shape[-1])
    indexer = DenseFlatIndexer(buffer_size=50000)
    indexer.init_index(vector_size)
    indexer.index_data(encoded)
    if indexer.index.ntotal != len(passages):
        raise ValueError(f"Passage vector count {indexer.index.ntotal} != corpus count {len(passages)}")

    results = {}
    for dataset_key in args.datasets:
        file_name = TEST_FILES[dataset_key]
        questions, answers = load_qa(args.qa_dir / file_name)
        if args.max_questions:
            questions, answers = questions[: args.max_questions], answers[: args.max_questions]
        query_vectors = encode_questions(
            question_encoder, query_tensorizer, questions, args.query_batch_size, retriever_cfg.device
        )
        candidate_k = min(args.top_docs, len(passages))
        retrieved = indexer.search_knn(query_vectors, candidate_k)
        hits = {1: 0, 5: 0, 20: 0, 100: 0}
        started = time.perf_counter()
        for q_idx, (question, answer_list, (candidate_ids, _scores)) in enumerate(
            zip(questions, answers, retrieved), start=1
        ):
            reranked_ids = score_candidates(
                reader, reader_tensorizer, question, candidate_ids, passages,
                args.reader_batch_size, reader_cfg.device,
            )
            answer_phrases = [normalized_phrase(a, answer_matcher) for a in answer_list]
            first_hit = None
            for rank, pid in enumerate(reranked_ids):
                passage_text = normalized_passages[pid]
                if any(phrase and phrase in passage_text for phrase in answer_phrases):
                    first_hit = rank
                    break
            if first_hit is not None:
                for cutoff in hits:
                    if first_hit < cutoff:
                        hits[cutoff] += 1
            if q_idx % 500 == 0:
                print(f"{dataset_key}: reranked {q_idx}/{len(questions)}", flush=True)

        elapsed = time.perf_counter() - started
        results[dataset_key] = {
            "questions": len(questions),
            "candidate_k": candidate_k,
            "hits": hits,
            "top_k_accuracy": {str(k): hits[k] / len(questions) for k in hits},
            "reranking_seconds": round(elapsed, 2),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "method": "DPR Multi + FAISS IndexFlatIP top-k + official DPR Reader relevance-head reranking",
            "retriever_checkpoint": str(args.retriever_checkpoint),
            "reader_checkpoint": str(args.reader_checkpoint),
            "passage_vectors": str(args.passage_vectors),
            "corpus": str(args.corpus),
            "corpus_size": len(passages),
            "faiss_type": "IndexFlatIP",
            "reader_top_candidates": args.top_docs,
            "reader_batch_size": args.reader_batch_size,
            "reader_gpus": args.reader_gpus,
            "test_results": results,
        }, indent=2), encoding="utf-8")
        print(f"{dataset_key}: {results[dataset_key]}", flush=True)

    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
