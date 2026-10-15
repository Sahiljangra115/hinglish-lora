import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import pytest

from common import ner_input, ner_target, parse_gold_tags, parse_ner_output
from metrics import mt_scores, ner_scores


def test_gold_parse():
    toks, tags = parse_gold_tags("[{'word': 'Delhi', 'entity': 'GPE'}, {'word': 'hai', 'entity': 'X'}]")
    assert toks == ["Delhi", "hai"] and tags == ["GPE", "X"]


def test_ner_roundtrip():
    tags = ["X", "PERSON", "PERSON", "X", "DATE"]
    assert ner_target(tags) == "1:PERSON 2:PERSON 4:DATE"
    assert parse_ner_output(ner_target(tags), 5) == tags


def test_ner_none_and_garbage():
    assert parse_ner_output("none", 3) == ["X"] * 3
    assert parse_ner_output("9:PERSON 1:BOGUS x:GPE 0:gpe", 3) == ["GPE", "X", "X"]


def test_ner_input_numbering():
    assert ner_input(["a", "b"]) == "0:a 1:b"


def test_ner_perfect_and_wrong():
    g = [["X", "PERSON", "GPE"]]
    assert ner_scores(g, g, ["X", "PERSON", "GPE"])["macro_f1"] == 100.0
    s = ner_scores(g, [["X", "X", "X"]], ["X", "PERSON", "GPE"])
    assert s["macro_f1"] == 0.0 and s["micro_f1"] == 0.0


def test_ner_partial():
    g = [["PERSON", "PERSON", "X", "X"]]
    p = [["PERSON", "X", "X", "X"]]
    s = ner_scores(g, p, ["X", "PERSON"])
    assert s["macro_p"] == 100.0 and s["macro_r"] == 50.0 and s["micro_f1"] == pytest.approx(66.67, abs=0.01)


def test_mt_identical_is_100():
    s = mt_scores(["The cat sat on the mat today."], ["The cat sat on the mat today."])
    assert s["bleu"] == 100.0 and s["chrf++"] == 100.0


class _CharTok:
    """Stand-in tokenizer: one id per character, chat template = concatenated contents."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "".join(m["content"] for m in msgs)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_encode_masks_prompt_and_drops_overlong():
    torch = pytest.importorskip("torch")  # train.py imports torch/peft
    pytest.importorskip("peft")
    from train import encode
    msgs = [{"role": "user", "content": "abc"}]
    ids, labels = encode(_CharTok(), msgs, "xy", 100)
    assert ids[:3] == [97, 98, 99] and labels[:3] == [-100] * 3 and labels[3:] == ids[3:]
    # answer would be cut (the old truncation produced all -100 labels and a NaN loss): drop instead
    assert encode(_CharTok(), msgs, "xy", 5) is None
