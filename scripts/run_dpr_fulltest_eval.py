"""Build a DPR FAISS index and evaluate the official test QA files locally.

Uses the repository's encoders, tensorizer, dense-vector generator, FAISS flat
indexer, QA reader and answer matcher. The small wrapper keeps evaluation
sequential so it works with Windows multiprocessing semantics.
"""

import argparse
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
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dpr.utils.tokenizers import SimpleTokenizer
from dpr.indexer.faiss_indexers import DenseFlatIndexer
from dpr.models import init_biencoder_components
from dpr.options import set_cfg_params_from_state, setup_cfg_gpu
from dpr.utils.model_utils import get_model_obj, load_states_from_checkpoint, setup_for_distributed_mode
from evaluation_utils import answer_match_type, passage_has_answer, write_jsonl


CHECKPOINT = ROOT / "outputs" / "local_multi_smoke" / "dpr_biencoder.0"
ENCODED = ROOT / "outputs" / "local_retrieval" / "wiki50k_ctx_0"
INDEX_PREFIX = ROOT / "outputs" / "local_retrieval" / "paper_multi_50k_fulltest_flat"
RESULT_FILE = ROOT / "outputs" / "local_retrieval" / "paper_multi_50k_fulltest_metrics.json"
TEST_KEYS = ["nq_test", "trivia_test", "webq_test", "curatedtrec_test", "squad1_test"]


def encode_questions(encoder, tensorizer, questions, batch_size, device):
    """Batch the original DPR tensorizer output safely on Windows/PyTorch."""
    vectors = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(questions), batch_size):
            token_ids = [tensorizer.text_to_tensor(q) for q in questions[start : start + batch_size]]
            batch = torch.nn.utils.rnn.pad_sequence(
                token_ids, batch_first=True, padding_value=tensorizer.get_pad_id()
            ).to(device)
            segments = torch.zeros_like(batch)
            attention = tensorizer.get_attn_mask(batch)
            _, encoded, _ = encoder(batch, segments, attention)
            vectors.append(encoded.detach().cpu().numpy())
            if (start // batch_size + 1) % 100 == 0:
                print(f"encoded {min(start + batch_size, len(questions))}/{len(questions)} queries", flush=True)
    return np.concatenate(vectors, axis=0)


def normalized_phrase(text, tokenizer):
    tokens = tokenizer.tokenize(unicodedata.normalize("NFD", text)).words(uncased=True)
    return " " + " ".join(tokens) + " " if tokens else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--encoded", type=Path, default=ENCODED)
    parser.add_argument("--result-file", type=Path, default=RESULT_FILE)
    parser.add_argument("--qa-config", default="local_qa_tests")
    parser.add_argument("--datasets", nargs="+", choices=TEST_KEYS, default=TEST_KEYS)
    parser.add_argument("--top-docs", type=int, default=100)
    parser.add_argument("--save-per-query", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file() or not args.encoded.is_file():
        raise FileNotFoundError(f"Expected checkpoint {args.checkpoint} and passage vectors {args.encoded}")

    overrides = [
        f"datasets={args.qa_config}",
        "ctx_sources=local_wiki50k",
        "hydra.job.chdir=False",
        f"model_file={args.checkpoint.as_posix()}",
    ]
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base="1.1"):
        cfg = compose(config_name="dense_retriever", overrides=overrides)

    cfg = setup_cfg_gpu(cfg)
    state = load_states_from_checkpoint(str(args.checkpoint))
    set_cfg_params_from_state(state.encoder_params, cfg)
    tensorizer, biencoder, _ = init_biencoder_components(cfg.encoder.encoder_model_type, cfg, inference_only=True)
    biencoder.load_state(state, strict=False)
    question_encoder = biencoder.question_model
    question_encoder, _ = setup_for_distributed_mode(
        question_encoder, None, cfg.device, cfg.n_gpu, cfg.local_rank, cfg.fp16
    )
    question_encoder.eval()
    vector_size = get_model_obj(question_encoder).get_out_size()

    ctx_src = hydra.utils.instantiate(cfg.ctx_sources.wiki50k)
    passages = {}
    ctx_src.load_data_to(passages)
    tokenizer = SimpleTokenizer()
    indexer = DenseFlatIndexer(buffer_size=50000)
    indexer.init_index(vector_size)
    with args.encoded.open("rb") as stream:
        encoded_passages = pickle.load(stream)
    indexer.index_data(encoded_passages)
    index_prefix = args.result_file.with_name(args.result_file.stem + "_flat")
    indexer.serialize(str(index_prefix))

    results = {}
    for dataset_key in args.datasets:
        qa_src = hydra.utils.instantiate(cfg.datasets[dataset_key])
        qa_src.load_data()
        samples = list(qa_src)
        questions = [sample.query for sample in samples]
        answers = [sample.answers for sample in samples]
        started = time.perf_counter()
        query_vectors = encode_questions(
            question_encoder,
            tensorizer,
            questions,
            args.batch_size,
            cfg.device,
        )
        top_docs = min(args.top_docs, len(passages))
        retrieved = indexer.search_knn(query_vectors, top_docs)
        hit_counts = {1: 0, 5: 0, 20: 0, 100: 0}
        per_query = []
        match_type = answer_match_type(dataset_key)
        for query_index, (sample, answer_list, (doc_ids, scores)) in enumerate(zip(samples, answers, retrieved)):
            first_hit = None
            ranked = []
            for rank, (doc_id, score) in enumerate(zip(doc_ids, scores), start=1):
                passage_text = passages[doc_id].text
                answer_hit = passage_has_answer(answer_list, passage_text, tokenizer, dataset_key)
                ranked.append({
                    "rank": rank,
                    "passage_id": str(doc_id),
                    "score": float(score),
                    "answer_hit": answer_hit,
                })
                if answer_hit and first_hit is None:
                    first_hit = rank
            if first_hit is not None:
                for cutoff in hit_counts:
                    if first_hit <= cutoff:
                        hit_counts[cutoff] += 1
            per_query.append({
                "query_id": str(sample.id) if sample.id is not None else f"{dataset_key}:{query_index}",
                "query_index": query_index,
                "dataset": dataset_key,
                "question": sample.query,
                "answers": [str(answer) for answer in answer_list],
                "match_type": match_type,
                "first_answer_rank": first_hit,
                "candidates": ranked,
            })
        per_query_path = None
        if args.save_per_query:
            per_query_path = args.result_file.with_name(args.result_file.stem + "_per_query") / f"{dataset_key}.jsonl.gz"
            write_jsonl(per_query_path, per_query)
        elapsed = time.perf_counter() - started
        results[dataset_key] = {
            "questions": len(questions),
            "candidate_k": top_docs,
            "hits": hit_counts,
            "top_k_accuracy": {str(k): hit_counts[k] / len(questions) for k in hit_counts},
            "query_and_match_seconds": round(elapsed, 2),
            "match_type": match_type,
            "per_query_results": str(per_query_path) if per_query_path else None,
        }
        print(f"{dataset_key}: {results[dataset_key]}", flush=True)
        partial_report = {
            "model_checkpoint": str(args.checkpoint),
            "passage_vectors": str(args.encoded),
            "faiss_index": str(index_prefix),
            "faiss_type": "IndexFlatIP",
            "corpus_size": indexer.index.ntotal,
            "max_sequence_length": cfg.encoder.sequence_length,
            "test_results": results,
        }
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(json.dumps(partial_report, indent=2), encoding="utf-8")

    report = {
        "model_checkpoint": str(args.checkpoint),
        "passage_vectors": str(args.encoded),
        "faiss_index": str(index_prefix),
        "faiss_type": "IndexFlatIP",
        "corpus_size": indexer.index.ntotal,
        "max_sequence_length": cfg.encoder.sequence_length,
        "test_results": results,
    }
    args.result_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {args.result_file}", flush=True)


if __name__ == "__main__":
    main()

