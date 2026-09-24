"""LoRA SFT of Qwen2.5-0.5B-Instruct on COMI-LINGUA MT + NER (one adapter, both tasks)."""
import argparse
import json
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import MODEL_ID, ROOT, load_mt, load_ner, mt_messages, ner_messages, ner_target


def encode(tok, msgs, answer, max_len):
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(answer + "<|im_end|>", add_special_tokens=False)["input_ids"]
    if len(p) + len(a) > max_len:  # drop, never truncate: a cut answer teaches a wrong stop, an empty one gives NaN loss
        return None
    return p + a, [-100] * len(p) + a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_mt", type=int, default=3000)
    ap.add_argument("--n_ner", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--log_name", default="train_log.json")
    ap.add_argument("--grad_ckpt", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "adapters" / "qwen05b-comi"))
    a = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).cuda()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    for p in model.parameters():  # keep trainable LoRA weights in fp32 for stable updates
        if p.requires_grad:
            p.data = p.data.float()

    ex = [encode(tok, mt_messages(r["src"]), r["tgt"], a.max_len) for r in load_mt("train", a.n_mt, seed=1)]
    ex += [encode(tok, ner_messages(r["tokens"]), ner_target(r["tags"]), a.max_len) for r in load_ner("train", a.n_ner, seed=1)]
    dropped = sum(e is None for e in ex)
    ex = [e for e in ex if e is not None]
    print("examples", len(ex), "dropped over max_len", dropped, "mean len", sum(len(i) for i, _ in ex) / len(ex))

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps_total = a.epochs * (len(ex) // (a.bs * a.accum))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / max(1, steps_total)))
    model.train()
    log, step, t0 = [], 0, time.time()
    for ep in range(a.epochs):
        random.Random(ep).shuffle(ex)
        # bucket by length inside chunks of 64 to cut padding
        chunks = [sorted(ex[i:i + 64], key=lambda x: len(x[0])) for i in range(0, len(ex), 64)]
        batches = [c[i:i + a.bs] for c in chunks for i in range(0, len(c), a.bs)]
        random.Random(100 + ep).shuffle(batches)
        run = 0.0
        for bi, b in enumerate(batches):
            L = max(len(i) for i, _ in b)
            ids = torch.tensor([i + [tok.pad_token_id] * (L - len(i)) for i, _ in b]).cuda()
            lab = torch.tensor([l + [-100] * (L - len(l)) for _, l in b]).cuda()
            att = torch.tensor([[1] * len(i) + [0] * (L - len(i)) for i, _ in b]).cuda()
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            (loss / a.accum).backward()
            run += loss.item()
            if (bi + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 10 == 0:
                    rec = {"step": step, "of": steps_total, "loss": round(run / (10 * a.accum), 4), "sec": round(time.time() - t0)}
                    print(rec, flush=True); log.append(rec); run = 0.0
    model.save_pretrained(a.out)
    json.dump({"args": vars(a), "n_examples": len(ex), "dropped_over_max_len": dropped, "log": log, "train_seconds": round(time.time() - t0),
               "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)},
              open(ROOT / "results" / a.log_name, "w"), indent=1)
    print("saved", a.out, "peak GB", torch.cuda.max_memory_allocated() / 1e9)


if __name__ == "__main__":
    main()
