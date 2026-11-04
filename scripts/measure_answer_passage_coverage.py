"""Measure answer-string coverage in the sampled Wikipedia corpus.

Coverage is question-level: a test question is covered when at least one of its
gold answer strings occurs in at least one corpus passage under DPR's
SimpleTokenizer normalization. This is a corpus-availability diagnostic, not a
retrieval score.
"""

import argparse
import ast
import csv
import json
import sys
import unicodedata
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dpr.utils.tokenizers import SimpleTokenizer


DEFAULT_CORPUS = Path(r"D:\DPR-data\wikipedia\psgs_w100_sample_50000.tsv")
DEFAULT_QA = ROOT / "data" / "raw" / "downloads" / "data" / "retriever" / "qas"
DATASETS = {
    "nq": "nq-test.csv",
    "trivia": "trivia-test.csv",
    "webq": "webq-test.csv",
    "curatedtrec": "curatedtrec-test.csv",
    "squad1": "squad1-test.csv",
}


def normalized_tokens(text, tokenizer):
    return tuple(tokenizer.tokenize(unicodedata.normalize("NFD", text)).words(uncased=True))


def load_questions(qa_dir, tokenizer):
    questions = []
    for dataset, filename in DATASETS.items():
        path = qa_dir / filename
        rows = []
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, delimiter="\t")
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    answers = ast.literal_eval(row[1])
                except (ValueError, SyntaxError):
                    continue
                if not isinstance(answers, (list, tuple)):
                    answers = [answers]
                rows.append({"dataset": dataset, "answers": [str(a) for a in answers if a is not None]})
        questions.extend(rows)
    return questions


class TokenAhoCorasick:
    def __init__(self):
        self.next = [{}]
        self.fail = [0]
        self.out = [[]]

    def add(self, tokens, question_id):
        state = 0
        for token in tokens:
            child = self.next[state].get(token)
            if child is None:
                child = self._new_state()
                self.next[state][token] = child
            state = child
        self.out[state].append(question_id)

    def _new_state(self):
        self.next.append({})
        self.fail.append(0)
        self.out.append([])
        return len(self.next) - 1

    def build(self):
        queue = deque()
        for child in self.next[0].values():
            queue.append(child)
        while queue:
            state = queue.popleft()
            for token, child in self.next[state].items():
                queue.append(child)
                fail = self.fail[state]
                while fail and token not in self.next[fail]:
                    fail = self.fail[fail]
                self.fail[child] = self.next[fail].get(token, 0)
                self.out[child].extend(self.out[self.fail[child]])

    def scan(self, tokens):
        state = 0
        for token in tokens:
            while state and token not in self.next[state]:
                state = self.fail[state]
            state = self.next[state].get(token, 0)
            if self.out[state]:
                yield from self.out[state]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--qa-dir", type=Path, default=DEFAULT_QA)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "local_retrieval" / "answer_passage_coverage_50k.json")
    args = parser.parse_args()

    tokenizer = SimpleTokenizer()
    questions = load_questions(args.qa_dir, tokenizer)
    machine = TokenAhoCorasick()
    no_valid_answers = 0
    for qid, question in enumerate(questions):
        has_pattern = False
        for answer in question["answers"]:
            tokens = normalized_tokens(answer, tokenizer)
            if tokens:
                machine.add(tokens, qid)
                has_pattern = True
        if not has_pattern:
            no_valid_answers += 1
    machine.build()

    corpus_count = 0
    covered = set()
    with args.corpus.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            corpus_count += 1
            tokens = normalized_tokens(row[1], tokenizer)
            covered.update(machine.scan(tokens))
            if corpus_count % 10000 == 0:
                print(f"scanned {corpus_count} passages; covered {len(covered)}/{len(questions)} questions", flush=True)

    per_dataset = {}
    for dataset in DATASETS:
        ids = [i for i, q in enumerate(questions) if q["dataset"] == dataset]
        hit_count = sum(i in covered for i in ids)
        per_dataset[dataset] = {
            "questions": len(ids),
            "covered_questions": hit_count,
            "coverage": hit_count / len(ids) if ids else None,
        }
    report = {
        "corpus_file": str(args.corpus),
        "qa_directory": str(args.qa_dir),
        "matching": "DPR SimpleTokenizer; NFD; case-insensitive token phrase match in passage text (same answer-string style as local evaluator)",
        "coverage_definition": "fraction of test questions with any answer string appearing in at least one corpus passage",
        "corpus_passages": corpus_count,
        "total_questions": len(questions),
        "questions_without_valid_answer_string": no_valid_answers,
        "covered_questions": len(covered),
        "overall_coverage": len(covered) / len(questions) if questions else None,
        "per_dataset": per_dataset,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
