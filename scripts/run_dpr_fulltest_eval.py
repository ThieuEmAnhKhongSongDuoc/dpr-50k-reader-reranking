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
    passage_ids = list(passages)
    passage_id_to_idx = {pid: idx for idx, pid in enumerate(passage_ids)}
    tokenizer = SimpleTokenizer()
    passage_texts = [" " + " ".join(tokenizer.tokenize(unicodedata.normalize("NFD", passages[pid].text)).words(uncased=True)) + " " for pid in passage_ids]
    indexer = DenseFlatIndexer(buffer_size=50000)
    indexer.init_index(vector_size)
    with args.encoded.open("rb") as stream:
        encoded_passages = pickle.load(stream)
    indexer.index_data(encoded_passages)
    indexer.serialize(str(INDEX_PREFIX))

    results = {}
    for dataset_key in TEST_KEYS:
        qa_src = hydra.utils.instantiate(cfg.datasets[dataset_key])
        qa_src.load_data()
        questions = [sample.query for sample in qa_src]
        answers = [sample.answers for sample in qa_src]
        answer_phrases = [[normalized_phrase(answer, tokenizer) for answer in answer_list] for answer_list in answers]
        started = time.perf_counter()
        query_vectors = encode_questions(
            question_encoder,
            tensorizer,
            questions,
            args.batch_size,
            cfg.device,
        )
        top_docs = min(100, len(passages))
        retrieved = indexer.search_knn(query_vectors, top_docs)
        hit_counts = {1: 0, 5: 0, 20: 0, 100: 0}
        for phrases, (doc_ids, _scores) in zip(answer_phrases, retrieved):
            first_hit = None
            for rank, doc_id in enumerate(doc_ids):
                doc_phrase = passage_texts[passage_id_to_idx[doc_id]]
                if any(phrase and phrase in doc_phrase for phrase in phrases):
                    first_hit = rank
                    break
            if first_hit is not None:
                for cutoff in hit_counts:
                    if first_hit < cutoff:
                        hit_counts[cutoff] += 1
        elapsed = time.perf_counter() - started
        results[dataset_key] = {
            "questions": len(questions),
            "hits": hit_counts,
            "top_k_accuracy": {str(k): hit_counts[k] / len(questions) for k in hit_counts},
            "query_and_match_seconds": round(elapsed, 2),
        }
        print(f"{dataset_key}: {results[dataset_key]}", flush=True)
        partial_report = {
            "model_checkpoint": str(args.checkpoint),
            "passage_vectors": str(args.encoded),
            "faiss_index": str(INDEX_PREFIX),
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
        "faiss_index": str(INDEX_PREFIX),
        "faiss_type": "IndexFlatIP",
        "corpus_size": indexer.index.ntotal,
        "max_sequence_length": cfg.encoder.sequence_length,
        "test_results": results,
    }
    args.result_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {args.result_file}", flush=True)


if __name__ == "__main__":
    main()
