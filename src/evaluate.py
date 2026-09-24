"""Evaluate base zero-shot / few-shot / LoRA on COMI-LINGUA test (MT: BLEU, chrF++; NER: token P/R/F1)."""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (MODEL_ID, ROOT, TAGS, load_mt, load_ner, mt_messages, ner_messages,
                    ner_target, parse_ner_output)

NER_MAX_NEW = 128  # 99th pct of gold answer length on test is 97 tokens; 48 cut off 21% of gold answers
from metrics import mt_scores, ner_scores


def load(adapter):
    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).cuda()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return tok, model.eval()


@torch.no_grad()
def generate(tok, model, msgs_list, max_new, bs=16):
    outs = [None] * len(msgs_list)
    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs_list]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, g in zip(idx, gen[:, enc["input_ids"].shape[1]:]):
            outs[i] = tok.decode(g, skip_special_tokens=True).strip()
    return outs


def run(mode, adapter, n_mt, n_ner, k, tag=None):
    tag = tag or mode
    tok, model = load(adapter)
    mt_test, ner_test = load_mt("test", n_mt, seed=7), load_ner("test", n_ner, seed=7)
    mt_shots = [(r["src"], r["tgt"]) for r in load_mt("train", k, seed=11)] if k else ()
    ner_shots = [(r["tokens"], r["tags"]) for r in load_ner("train", k, seed=11, max_tokens=25)] if k else ()
    t0 = time.time()
    mt_out = generate(tok, model, [mt_messages(r["src"], mt_shots) for r in mt_test], 128)
    mt_sec = time.time() - t0
    mt_out_first = [o.split("\n")[0].strip() for o in mt_out]
    t0 = time.time()
    ner_out = generate(tok, model, [ner_messages(r["tokens"], ner_shots) for r in ner_test], NER_MAX_NEW)
    ner_sec = time.time() - t0
    preds = [parse_ner_output(o, len(r["tokens"])) for o, r in zip(ner_out, ner_test)]
    res = {
        "mode": mode, "n_mt": len(mt_test), "n_ner": len(ner_test), "shots": k,
        "mt": mt_scores(mt_out_first, [r["tgt"] for r in mt_test]),
        "ner": ner_scores([r["tags"] for r in ner_test], preds, TAGS + ["X"]),
        "mt_sec_per_sentence": round(mt_sec / len(mt_test), 3),
        "ner_sec_per_sentence": round(ner_sec / len(ner_test), 3),
        # output that is not "none" yet yields no valid index:TAG pair
        "ner_unparseable_rate": round(sum(o.strip().lower() != "none" and set(p) == {"X"} for o, p in zip(ner_out, preds)) / len(ner_out), 3),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(res, open(ROOT / "results" / f"eval_{tag}.json", "w"), indent=1)
    json.dump({"mt": [{"src": r["src"], "ref": r["tgt"], "hyp": h} for r, h in list(zip(mt_test, mt_out))[:25]],
               "ner": [{"tokens": r["tokens"], "gold": ner_target(r["tags"]), "raw": o} for r, o in list(zip(ner_test, ner_out))[:25]]},
              open(ROOT / "results" / f"samples_{tag}.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n_mt", type=int, default=200)
    ap.add_argument("--n_ner", type=int, default=200)
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--tag", default=None, help="output file suffix, defaults to --mode")
    a = ap.parse_args()
    run(a.mode, a.adapter, a.n_mt, a.n_ner, a.shots, a.tag)
