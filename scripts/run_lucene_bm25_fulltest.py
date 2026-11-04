#!/usr/bin/env python3
"""Run an actual Apache Lucene BM25 baseline on the DPR 50k corpus sample."""

import ast
import base64
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dpr.utils.tokenizers import SimpleTokenizer


CORPUS = Path(r"D:\DPR-data\wikipedia\psgs_w100_sample_50000.tsv")
WORK = Path(r"D:\DPR-data\lucene_bm25_fulltest")
INDEX = WORK / "lucene_index"
JAR_DIR = Path(r"D:\DPR-data\lucene-8.11.2")
CORE_JAR = JAR_DIR / "lucene-core-8.11.2.jar"
ANALYZER_JAR = JAR_DIR / "lucene-analyzers-common-8.11.2.jar"
JDK_BIN = Path(r"D:\DPR-data\temurin8\jdk8u504-b01\bin")
SOURCE = ROOT / "scripts" / "lucene_bm25" / "LuceneBm25.java"
CLASSES = WORK / "classes"
SAMPLE_DIR = ROOT / "data" / "raw" / "downloads" / "data" / "retriever" / "qas"
OUTPUT = ROOT / "outputs" / "local_retrieval" / "paper_lucene_bm25_50k_fulltest.json"
DATASETS = {
    "nq": "nq-test.csv",
    "triviaqa": "trivia-test.csv",
    "webquestions": "webq-test.csv",
    "curatedtrec": "curatedtrec-test.csv",
    "squad": "squad1-test.csv",
}


def encode(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def normalized_phrase(text: str, tokenizer: SimpleTokenizer) -> str:
    import unicodedata

    tokens = tokenizer.tokenize(unicodedata.normalize("NFD", text)).words(uncased=True)
    return " " + " ".join(tokens) + " " if tokens else ""


def main() -> None:
    for path in (CORPUS, CORE_JAR, ANALYZER_JAR, SOURCE, JDK_BIN / "javac.exe", JDK_BIN / "java.exe"):
        if not path.is_file():
            raise FileNotFoundError(path)
    WORK.mkdir(parents=True, exist_ok=True)
    CLASSES.mkdir(parents=True, exist_ok=True)

    classpath = ";".join((str(CORE_JAR), str(ANALYZER_JAR)))
    subprocess.run([str(JDK_BIN / "javac.exe"), "-encoding", "UTF-8", "-cp", classpath, "-d", str(CLASSES), str(SOURCE)], check=True)
    runtime_cp = ";".join((str(CLASSES), str(CORE_JAR), str(ANALYZER_JAR)))

    corpus_input = WORK / "passages_base64.tsv"
    passage_texts = {}
    with CORPUS.open("r", encoding="utf-8", newline="") as source, corpus_input.open("w", encoding="utf-8", newline="\n") as target:
        rows = csv.reader(source, delimiter="\t")
        header = next(rows)
        if header[:3] != ["id", "text", "title"]:
            raise ValueError(f"Unexpected corpus header: {header!r}")
        for row in rows:
            if len(row) != len(header):
                raise ValueError(f"Malformed corpus row for passage {len(passage_texts) + 1}")
            pid, text, title = row[:3]
            content = f"{title} {text}"
            passage_texts[pid] = text
            target.write(f"{pid}\t{encode(content)}\n")

    query_input = WORK / "queries_base64.tsv"
    query_data = {}
    with query_input.open("w", encoding="utf-8", newline="\n") as target:
        for dataset, filename in DATASETS.items():
            path = SAMPLE_DIR / filename
            with path.open("r", encoding="utf-8", newline="") as source:
                for index, row in enumerate(csv.reader(source, delimiter="\t")):
                    question, answers = row[:2]
                    qid = f"{dataset}:{index}"
                    answer_list = ast.literal_eval(answers)
                    if not isinstance(answer_list, list):
                        raise ValueError(f"Bad answer list for {qid}")
                    query_data[qid] = {"dataset": dataset, "answers": answer_list}
                    target.write(f"{qid}\t{encode(question)}\n")

    subprocess.run([str(JDK_BIN / "java.exe"), "-Xmx2g", "-cp", runtime_cp, "LuceneBm25", "index", str(corpus_input), str(INDEX)], check=True)
    hits_file = WORK / "hits.tsv"
    subprocess.run([str(JDK_BIN / "java.exe"), "-Xmx2g", "-cp", runtime_cp, "LuceneBm25", "search", str(query_input), str(INDEX), str(hits_file), "100"], check=True)

    tokenizer = SimpleTokenizer()
    hits = {qid: [] for qid in query_data}
    with hits_file.open("r", encoding="utf-8") as source:
        for line in source:
            qid, rank, pid = line.rstrip("\n").split("\t")
            hits[qid].append((int(rank), pid))

    results = {}
    for dataset in DATASETS:
        qids = [qid for qid, item in query_data.items() if item["dataset"] == dataset]
        cutoff_hits = {1: 0, 5: 0, 20: 0, 100: 0}
        for qid in qids:
            phrases = [normalized_phrase(answer, tokenizer) for answer in query_data[qid]["answers"]]
            first_hit = None
            for rank, pid in hits[qid]:
                text = normalized_phrase(passage_texts[pid], tokenizer)
                if any(phrase and phrase in text for phrase in phrases):
                    first_hit = rank
                    break
            if first_hit is not None:
                for cutoff in cutoff_hits:
                    if first_hit < cutoff:
                        cutoff_hits[cutoff] += 1
        results[dataset] = {
            "questions": len(qids),
            "hits": cutoff_hits,
            "top_k_accuracy": {str(k): cutoff_hits[k] / len(qids) for k in cutoff_hits},
        }
        print(dataset, results[dataset], flush=True)

    report = {
        "retriever": "Apache Lucene 8.11.2 BM25Similarity",
        "implementation": "standalone Java Lucene runner; not Pyserini/Anserini",
        "analyzer": "EnglishAnalyzer",
        "k1": 0.9,
        "b": 0.4,
        "corpus": str(CORPUS),
        "passages": len(passage_texts),
        "test_dir": str(SAMPLE_DIR),
        "test_results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
