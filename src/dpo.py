"""SFT then DPO arm. Manual DPO (trl is not installed in the ml venv). GPU needed.

Stage 1 (--stage pairs): sample SFT outputs on train prompts, pair against gold.
  chosen = gold reference, rejected = SFT output that scores clearly worse than gold
  (MT: sentence chrF below --mt_thresh; NER: output tags differ from gold).
  No hosted judge is used. To use judge-labelled pairs instead, write a JSONL with
  {"messages": [...], "chosen": str, "rejected": str} and pass --pairs_file.
Stage 2 (--stage train): DPO on those pairs, starting from the SFT adapter. The reference
  log-probs are computed once up front from the starting (SFT) policy, so only one model is in memory.
Then evaluate with:  python evaluate.py --mode lora_sft_dpo --adapter ../adapters/qwen05b-comi-dpo
"""
import argparse
import json
import random
import time

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (MODEL_ID, ROOT, load_mt, load_ner, mt_messages, ner_messages, ner_target,
                    parse_ner_output)


def build_pairs(a):
    from sacrebleu.metrics import CHRF
    from evaluate import NER_MAX_NEW, generate, load
    tok, model = load(a.sft_adapter)
    chrf = CHRF(word_order=2)
    mt = load_mt("train", a.n_mt, seed=21)       # seeds differ from SFT subset (seed=1)
    ner = load_ner("train", a.n_ner, seed=21)
    mt_out = [o.split("\n")[0].strip() for o in generate(tok, model, [mt_messages(r["src"]) for r in mt], 128)]
    ner_out = generate(tok, model, [ner_messages(r["tokens"]) for r in ner], NER_MAX_NEW)
    pairs = []
    for r, o in zip(mt, mt_out):
        if o and chrf.sentence_score(o, [r["tgt"]]).score < a.mt_thresh:
            pairs.append({"messages": mt_messages(r["src"]), "chosen": r["tgt"], "rejected": o})
    for r, o in zip(ner, ner_out):
        if parse_ner_output(o, len(r["tokens"])) != r["tags"]:
            pairs.append({"messages": ner_messages(r["tokens"]), "chosen": ner_target(r["tags"]), "rejected": o.strip() or "none"})
    out = ROOT / "results" / "dpo_pairs.jsonl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("pairs", len(pairs), "of", len(mt) + len(ner), "->", out)


def enc(tok, msgs, answer, max_len):
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    r = tok(answer + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return p + r, len(p)


def seq_logp(model, ids, plen):
    x = torch.tensor([ids]).cuda()
    logits = model(input_ids=x).logits[0, :-1].float()
    lp = torch.log_softmax(logits, -1).gather(-1, x[0, 1:, None])[:, 0]
    return lp[plen - 1:].sum()


def train(a):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).cuda()
    model = PeftModel.from_pretrained(base, a.sft_adapter, is_trainable=True)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    path = a.pairs_file or str(ROOT / "results" / "dpo_pairs.jsonl")
    rows = [json.loads(l) for l in open(path)]
    random.Random(0).shuffle(rows)
    data = []
    for r in rows:
        c, pl = enc(tok, r["messages"], r["chosen"], a.max_len)
        j, _ = enc(tok, r["messages"], r["rejected"], a.max_len)
        if max(len(c), len(j)) <= a.max_len:  # drop, never truncate
            data.append((c, j, pl))
    print("usable pairs", len(data))

    model.eval()
    with torch.no_grad():  # reference = starting SFT policy
        ref = [(seq_logp(model, c, pl).item(), seq_logp(model, j, pl).item()) for c, j, pl in data]
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    log, t0, run, acc = [], time.time(), 0.0, 0
    seen = 0
    for i, ((c, j, pl), (rc, rj)) in enumerate(zip(data, ref)):
        if time.time() - t0 > 60 * a.max_minutes:
            print("time cap hit at pair", i, flush=True)
            break
        seen = i + 1
        margin = a.beta * ((seq_logp(model, c, pl) - rc) - (seq_logp(model, j, pl) - rj))
        loss = -F.logsigmoid(margin)
        (loss / a.accum).backward()
        run += loss.item(); acc += int(margin.item() > 0)
        if (i + 1) % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad()
        if (i + 1) % 50 == 0:
            rec = {"pair": i + 1, "of": len(data), "loss": round(run / 50, 4), "pref_acc": round(acc / 50, 3), "sec": round(time.time() - t0)}
            print(rec, flush=True); log.append(rec); run, acc = 0.0, 0
    if seen % a.accum:  # flush the last partial accumulation
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad()
    model.save_pretrained(a.out)
    json.dump({"args": vars(a), "n_pairs": len(data), "pairs_trained": seen, "log": log, "train_seconds": round(time.time() - t0),
               "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)},
              open(ROOT / "results" / "dpo_log.json", "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pairs", "train"], required=True)
    ap.add_argument("--sft_adapter", default=str(ROOT / "adapters" / "qwen05b-comi"))
    ap.add_argument("--out", default=str(ROOT / "adapters" / "qwen05b-comi-dpo"))
    ap.add_argument("--pairs_file", default=None)
    ap.add_argument("--n_mt", type=int, default=1000)
    ap.add_argument("--n_ner", type=int, default=1000)
    ap.add_argument("--mt_thresh", type=float, default=40.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--max_minutes", type=float, default=15, help="wall-clock cap on the DPO update loop")
    a = ap.parse_args()
    build_pairs(a) if a.stage == "pairs" else train(a)
