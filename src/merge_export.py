"""Merge a LoRA adapter into the base weights and save a plain HF checkpoint (safetensors).

CPU only by default, no GPU needed. The output directory loads with AutoModelForCausalLM,
vLLM, llama.cpp conversion, and mini-inference-engine (which has no LoRA support, so it needs merged weights).
Optional --gguf_script points at llama.cpp's convert_hf_to_gguf.py to also emit GGUF.
"""
import argparse
import subprocess
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import MODEL_ID, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default=str(ROOT / "adapters" / "qwen05b-comi"))
ap.add_argument("--out", default=str(ROOT / "export" / "qwen05b-comi-merged"))
ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
ap.add_argument("--gguf_script", default=None, help="path to llama.cpp convert_hf_to_gguf.py (optional)")
a = ap.parse_args()

dt = getattr(torch, a.dtype)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dt, device_map="cpu")
model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
model.save_pretrained(a.out, safe_serialization=True)
AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(a.out)
print("merged model saved to", a.out)
if a.gguf_script:
    subprocess.run([sys.executable, a.gguf_script, a.out, "--outfile", a.out + ".gguf"], check=True)
