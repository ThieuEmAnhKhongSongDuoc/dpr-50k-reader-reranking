"""Shared DPR answer matching and per-question result helpers."""

import gzip
import json
import unicodedata
from functools import lru_cache
from pathlib import Path

from dpr.data.qa_validation import has_answer
from dpr.utils.tokenizers import SimpleTokenizer


_STRING_MATCH_TOKENIZER = SimpleTokenizer()


def answer_match_type(dataset_key: str) -> str:
    """DPR CuratedTREC answer fields are regex patterns, unlike other datasets."""
    key = dataset_key.lower().replace("_test", "").replace("_dev", "")
    return "regex" if key in {"curatedtrec", "trec"} else "string"


def passage_has_answer(answers, passage_text: str, tokenizer, dataset_key: str) -> bool:
    """Use DPR's canonical matcher with the correct benchmark-specific mode."""
    if answer_match_type(dataset_key) == "regex":
        return has_answer(answers, passage_text, tokenizer, "regex")
    text_tokens = _cached_string_tokens(unicodedata.normalize("NFD", passage_text))
    for answer_tokens in _cached_answer_tokens(tuple(str(a) for a in answers)):
        if not answer_tokens:
            # Match the original has_answer implementation's empty-pattern behavior.
            return True
        answer_len = len(answer_tokens)
        for start in range(len(text_tokens) - answer_len + 1):
            if answer_tokens == text_tokens[start : start + answer_len]:
                return True
    return False


@lru_cache(maxsize=250000)
def _cached_string_tokens(text: str):
    return tuple(_STRING_MATCH_TOKENIZER.tokenize(text).words(uncased=True))


@lru_cache(maxsize=32768)
def _cached_answer_tokens(answers):
    return tuple(
        tuple(_STRING_MATCH_TOKENIZER.tokenize(unicodedata.normalize("NFD", answer)).words(uncased=True))
        for answer in answers
    )


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_wiki_passage_id(passage_id) -> str:
    """Normalize Lucene's raw TSV IDs to DPR's wiki:-prefixed IDs."""
    passage_id = str(passage_id)
    return passage_id if passage_id.startswith("wiki:") else "wiki:" + passage_id
