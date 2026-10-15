"""Cost and latency comparison, local LoRA model vs a hosted API model. Pure arithmetic, no network, no API key.

Inputs: configs/pricing.yaml (hosted list prices with source URL and date) and results/bench_serve_<tag>.json
(measured locally, including mean GPU board power). Hosted latency is not measured here (no API key).
Prompt tokens: the measured Qwen count from the bench (hosted tokenizers differ), else chars/--chars_per_token.
Local cost is energy only: measured GPU watts x time. USD needs usd_per_kwh in the config; it stays null otherwise.
"""
import argparse
import json

import yaml

from common import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(ROOT / "configs" / "pricing.yaml"))
ap.add_argument("--bench", default=str(ROOT / "results" / "bench_serve_hf_lora.json"))
ap.add_argument("--chars_per_token", type=float, default=3.0)
ap.add_argument("--prompt_overhead_tokens", type=int, default=60, help="system prompt + chat template tokens")
a = ap.parse_args()

cfg, b = yaml.safe_load(open(a.config)), json.load(open(a.bench))
h, loc = cfg["hosted"], cfg["local"]
# measured Qwen prompt tokens (incl. system prompt) when available; hosted tokenizers differ, so this is approximate
p_tok = b.get("mean_prompt_tokens") or b["mean_prompt_chars"] / a.chars_per_token + a.prompt_overhead_tokens
c_tok = b["mean_completion_tokens"]
sec_per_req = b["wall_s"] / b["n"]
hosted_per_1k_req = 1000 * (p_tok * h["input_per_1m"] + c_tok * h["output_per_1m"]) / 1e6
# Power: nvidia-smi gives whole-board power only, never a per-process share. We use board power during the run
# minus the idle board power measured just before it (model loaded, no requests). This counts only the extra energy
# that inference caused. It is exact only when no other process used the GPU (checked via other_gpu_pids).
kwh = lambda w: 1000 * sec_per_req * w / 3.6e6 if w else None
inc_w, board_w = b.get("incremental_gpu_power_w"), b.get("mean_gpu_power_w")
kwh_per_1k_req = kwh(inc_w if inc_w is not None else board_w)
others = b.get("other_gpu_pids")
tok_per_req = p_tok + c_tok  # prompt + completion tokens per request
usd_kwh = loc.get("usd_per_kwh")
local_usd = round(kwh_per_1k_req * usd_kwh, 6) if kwh_per_1k_req is not None and usd_kwh else None
out = {
    "assumed_tokens_per_request": {"prompt": round(p_tok, 1), "completion": c_tok},
    "local": {"p50_latency_s": b["p50_latency_s"], "p95_latency_s": b["p95_latency_s"],
              "requests_per_s": round(1 / sec_per_req, 3), "mean_board_power_w": board_w,
              "idle_board_power_w": b.get("idle_gpu_power_w"), "incremental_power_w": inc_w,
              "power_basis": "incremental (board minus idle)" if inc_w is not None else "whole board (no idle baseline)",
              "other_gpu_pids_during_run": others,
              "gpu_exclusive": (others == []) if others is not None else None,
              "kwh_per_1k_requests_board_upper_bound": round(kwh(board_w), 5) if board_w else None,
              "kwh_per_1k_requests": round(kwh_per_1k_req, 5) if kwh_per_1k_req is not None else None,
              "kwh_per_1k_tokens": round(kwh_per_1k_req / tok_per_req, 8) if kwh_per_1k_req is not None else None,
              "usd_per_1k_requests": local_usd,
              "usd_note": None if local_usd is not None else "pending: no cited electricity tariff in configs/pricing.yaml"},
    "hosted": {"model": h["name"], "price_source": h.get("source_url"), "price_date": h.get("retrieved"),
               "p50_latency_s": h.get("p50_latency_s"), "usd_per_1k_requests": round(hosted_per_1k_req, 6),
               "usd_per_1k_tokens_blended": round(hosted_per_1k_req / tok_per_req, 8)},
    "note": ("Local cost is GPU inference energy only (no hardware, idle time, CPU or host power). "
             "Power is whole-board, not per-process; the idle baseline is subtracted. "
             + ("No other GPU process was seen during the run. " if others == [] else
                "Other GPU processes were seen, so the energy figure is contaminated. " if others else
                "GPU exclusivity was not recorded. ")
             + "Hosted latency not measured."),
}
(ROOT / "results").mkdir(exist_ok=True)
json.dump(out, open(ROOT / "results" / "cost_compare.json", "w"), indent=1)
print(json.dumps(out, indent=1))
