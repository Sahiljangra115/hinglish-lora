"""HTTP endpoint for base + LoRA adapter (or merged export, or mini-inference-engine).

  POST /generate  {"task": "mt"|"ner", "text": "...", "max_new_tokens": 128}  (ner: text is whitespace-tokenised)
  GET  /health

Backends (--backend):
  hf      base model + PeftModel adapter, transformers generate. Uses the GPU (cuda), so do not run while GPU is busy.
  engine  mini-inference-engine LLM (offline engine; no LoRA support, so --model must be a merged export from
          merge_export.py). Needs --engine_path (or MINI_ENGINE_DIR) pointing at a mini-inference-engine checkout.
Requests are serialised with a lock (simple, correct, not batched).
"""
import argparse
import os
import sys
import threading
import time

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from common import MODEL_ID, ROOT, mt_messages, ner_messages

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["hf", "engine"], default="hf")
ap.add_argument("--adapter", default=str(ROOT / "adapters" / "qwen05b-comi"))
ap.add_argument("--model", default=str(ROOT / "export" / "qwen05b-comi-merged"), help="engine backend: merged checkpoint dir")
ap.add_argument("--engine_path", default=os.environ.get("MINI_ENGINE_DIR", str(ROOT.parent.parent / "Projects" / "mini-inference-engine")))
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8010)
args, _ = ap.parse_known_args()

app, lock = FastAPI(), threading.Lock()


class Req(BaseModel):
    task: str = "mt"
    text: str
    max_new_tokens: int = 128


def build():
    if args.backend == "hf":
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
        m = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).cuda()
        m = PeftModel.from_pretrained(m, args.adapter).merge_and_unload().eval()

        @torch.no_grad()
        def gen(msgs, n):
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            e = tok(p, return_tensors="pt", add_special_tokens=False).to("cuda")
            o = m.generate(**e, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
            ids = o[0, e["input_ids"].shape[1]:]
            return tok.decode(ids, skip_special_tokens=True).strip(), int(ids.shape[0]), int(e["input_ids"].shape[1])
        return gen
    sys.path.insert(0, args.engine_path)
    from engine.config import SamplingParams
    from engine.llm import LLM
    llm = LLM(args.model)

    def gen(msgs, n):
        p = llm.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        out = llm.generate([p], SamplingParams(max_tokens=n))[0]
        return out.text.strip(), len(out.token_ids), len(out.prompt_token_ids)
    return gen


GEN = build()


@app.get("/health")
def health():
    return {"ok": True, "backend": args.backend, "pid": os.getpid()}


@app.post("/generate")
def generate(r: Req):
    msgs = mt_messages(r.text) if r.task == "mt" else ner_messages(r.text.split())
    t0 = time.perf_counter()
    with lock:
        text, n, pn = GEN(msgs, r.max_new_tokens)
    return {"output": text, "completion_tokens": n, "prompt_tokens": pn, "latency_s": round(time.perf_counter() - t0, 4)}


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
