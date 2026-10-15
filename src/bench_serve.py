"""Latency/throughput benchmark against a running serve.py endpoint. Uses the COMI-LINGUA MT test split as prompts.

Writes results/bench_serve_<tag>.json: p50/p95 latency, tokens/s per request, aggregate tokens/s, and the
mean prompt/completion tokens (used by cost_compare.py).
"""
import argparse
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from common import ROOT, load_mt

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8010")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--concurrency", type=int, default=1)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--tag", default="hf_lora")
ap.add_argument("--idle_s", type=float, default=15, help="seconds of idle power sampling before the timed run")
a = ap.parse_args()

rows = load_mt("test", a.n + a.warmup, seed=5)
call = lambda r: requests.post(f"{a.url}/generate", json={"task": "mt", "text": r["src"]}, timeout=300).json()
for r in rows[:a.warmup]:
    call(r)
server_pid = requests.get(f"{a.url}/health", timeout=10).json().get("pid")


def smi(query, kind="gpu"):
    q = subprocess.run(["nvidia-smi", f"--query-{kind}={query}", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return [l.strip() for l in q.stdout.splitlines() if l.strip()]


def sample_power(watts, pids, stop):
    # nvidia-smi reports whole-board power only (no per-process power or SM share on this GPU),
    # so we also record every compute PID on the GPU to show whether the board was ours alone.
    while not stop.is_set():
        try:
            watts.append(float(smi("power.draw")[0]))
        except (ValueError, IndexError):
            pass
        for l in smi("pid", "compute-apps"):
            if l.isdigit():
                pids.add(int(l))
        stop.wait(0.5)


def sampled(seconds=None, fn=None):
    watts, pids, stop = [], set(), threading.Event()
    th = threading.Thread(target=sample_power, args=(watts, pids, stop), daemon=True)
    th.start()
    t0 = time.perf_counter()
    r = fn() if fn else stop.wait(seconds)
    wall = time.perf_counter() - t0
    stop.set(); th.join()
    return r, wall, watts, pids


# idle baseline: model loaded on the GPU, no requests in flight
_, _, idle_w, idle_pids = sampled(seconds=a.idle_s)
with ThreadPoolExecutor(a.concurrency) as ex:
    res, wall, watts, pids = sampled(fn=lambda: list(ex.map(call, rows[a.warmup:])))
lat = sorted(x["latency_s"] for x in res)
toks = sum(x["completion_tokens"] for x in res)
pct = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))]
out = {"tag": a.tag, "n": len(res), "concurrency": a.concurrency, "wall_s": round(wall, 2),
       "p50_latency_s": pct(0.5), "p95_latency_s": pct(0.95),
       "aggregate_tokens_per_s": round(toks / wall, 2), "mean_completion_tokens": round(toks / len(res), 1),
       "mean_gpu_power_w": round(sum(watts) / len(watts), 1) if watts else None, "power_samples": len(watts),
       "idle_gpu_power_w": round(sum(idle_w) / len(idle_w), 1) if idle_w else None, "idle_power_samples": len(idle_w),
       "incremental_gpu_power_w": round(sum(watts) / len(watts) - sum(idle_w) / len(idle_w), 1) if watts and idle_w else None,
       "server_pid": server_pid, "gpu_pids_seen": sorted(pids | idle_pids),
       "other_gpu_pids": sorted((pids | idle_pids) - {server_pid}),
       "date": time.strftime("%Y-%m-%d"),
       "mean_prompt_tokens": round(sum(x["prompt_tokens"] for x in res) / len(res), 1),
       "mean_prompt_chars": round(sum(len(r["src"]) for r in rows[a.warmup:]) / len(res), 1)}
(ROOT / "results").mkdir(exist_ok=True)
json.dump(out, open(ROOT / "results" / f"bench_serve_{a.tag}.json", "w"), indent=1)
print(json.dumps(out, indent=1))
