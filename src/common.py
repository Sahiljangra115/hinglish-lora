"""Data loading, prompt formats and output parsing for COMI-LINGUA MT and NER."""
import ast
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

MT_INSTR = (
    "Translate the following Hindi-English code-mixed sentence into fluent standard English. "
    "Output only the translation."
)
NER_INSTR = (
    "The sentence below is Hindi-English code-mixed text, split into numbered tokens. "
    "Tag named entities. Allowed tags: PERSON, ORGANISATION, LOCATION, GPE, DATE, TIME, HASHTAG, MENTION, EMOJI. "
    "Output only the entity tokens as index:TAG pairs separated by spaces, or the word none if there are no entities."
)
TAGS = ["PERSON", "ORGANISATION", "LOCATION", "GPE", "DATE", "TIME", "HASHTAG", "MENTION", "EMOJI"]


def load_mt(split, n=None, seed=0):
    df = pd.read_csv(DATA / f"MT_{split}.csv", usecols=["Sentences", "Annotator_1_en_translation"])
    df = df.dropna().rename(columns={"Sentences": "src", "Annotator_1_en_translation": "tgt"})
    df = df[(df.src.str.strip() != "") & (df.tgt.str.strip() != "")]
    if n:
        df = df.sample(n=min(n, len(df)), random_state=seed)
    return df.reset_index(drop=True).to_dict("records")


def parse_gold_tags(cell):
    """Annotation cell (python-literal list of {word, entity}) -> (tokens, tags)."""
    items = ast.literal_eval(cell)
    return [i["word"] for i in items], [i["entity"] for i in items]


def load_ner(split, n=None, seed=0, max_tokens=60):
    df = pd.read_csv(DATA / f"NER_{split}.csv")
    rows = []
    for a1, a2 in zip(df["Annotated by: Annotator 1"], df["Annotated by: Annotator 2"]):
        cell = a1 if isinstance(a1, str) else a2
        try:
            toks, tags = parse_gold_tags(cell)
        except Exception:
            continue
        if 0 < len(toks) <= max_tokens:
            rows.append({"tokens": toks, "tags": tags})
    random.Random(seed).shuffle(rows)
    return rows[:n] if n else rows


def ner_input(tokens):
    return " ".join(f"{i}:{t}" for i, t in enumerate(tokens))


def ner_target(tags):
    pairs = [f"{i}:{t}" for i, t in enumerate(tags) if t in TAGS]
    return " ".join(pairs) if pairs else "none"


def parse_ner_output(text, n_tokens):
    """Model output -> tag list of length n_tokens ('X' default). Ignores malformed pairs."""
    tags = ["X"] * n_tokens
    for part in text.replace(",", " ").split():
        if ":" not in part:
            continue
        i, _, t = part.partition(":")
        t = t.strip().upper()
        if i.strip().isdigit() and t in TAGS and int(i) < n_tokens:
            tags[int(i)] = t
    return tags


def mt_messages(src, shots=()):
    msgs = [{"role": "system", "content": MT_INSTR}]
    for s, t in shots:
        msgs += [{"role": "user", "content": s}, {"role": "assistant", "content": t}]
    msgs.append({"role": "user", "content": src})
    return msgs


def ner_messages(tokens, shots=()):
    msgs = [{"role": "system", "content": NER_INSTR}]
    for toks, tags in shots:
        msgs += [{"role": "user", "content": ner_input(toks)}, {"role": "assistant", "content": ner_target(tags)}]
    msgs.append({"role": "user", "content": ner_input(tokens)})
    return msgs
