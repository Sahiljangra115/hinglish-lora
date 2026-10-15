#!/usr/bin/env bash
# Full pipeline. Each GPU stage takes the shared GPU lock for its own duration only.
# Env: ML_VENV (python env), GPU_LOCK (flock file, default $LOCKS/gpu.lock or /tmp/gpu.lock).
# Optional stages: RUN_DPO=1 RUN_EXPORT=1 RUN_SERVE_BENCH=1 RUN_ENGINE_BENCH=1 ./run_all.sh
set -eu
[ -n "${ML_VENV:-}" ] && source "$ML_VENV/bin/activate"
cd "$(dirname "$0")"
mkdir -p results logs
cd src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 TRANSFORMERS_VERBOSITY=error
LOCK="${GPU_LOCK:-${LOCKS:-/tmp}/gpu.lock}"
# Inside the lock, also wait for VRAM to be free (a process outside the lock once caused an OOM).
wait_vram() { for i in $(seq 1 120); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 600 ] && return; sleep 5; done; }
export -f wait_vram
gpu() { local name=$1; shift; flock "$LOCK" bash -c 'wait_vram; exec "$@"' _ "$@" > "../logs/$name.log" 2>&1; }

[ "${SKIP_TRAIN:-0}" = 1 ] || gpu train python train.py --n_mt 2500 --n_ner 2500 --bs 2 --accum 8 --max_len 512
gpu eval_base_zeroshot python evaluate.py --mode base_zeroshot --n_mt 300 --n_ner 300
gpu eval_base_3shot python evaluate.py --mode base_3shot --shots 3 --n_mt 300 --n_ner 300
gpu eval_lora_sft python evaluate.py --mode lora_sft --adapter ../adapters/qwen05b-comi --n_mt 300 --n_ner 300
gpu shrota python shrota_eval.py

if [ "${RUN_DPO:-0}" = 1 ]; then
  gpu dpo_pairs python dpo.py --stage pairs
  gpu dpo_train python dpo.py --stage train
  gpu eval_lora_sft_dpo python evaluate.py --mode lora_sft_dpo --adapter ../adapters/qwen05b-comi-dpo --n_mt 300 --n_ner 300
fi
if [ "${RUN_EXPORT:-0}" = 1 ]; then
  CUDA_VISIBLE_DEVICES= python merge_export.py > ../logs/merge_export.log 2>&1   # CPU only
fi
if [ "${RUN_SERVE_BENCH:-0}" = 1 ]; then
  # Server, bench and shutdown all run inside one lock hold.
  flock "$LOCK" bash -c 'wait_vram
    python serve.py --backend hf --port 8010 > ../logs/serve.log 2>&1 & SPID=$!
    trap "kill $SPID" EXIT
    until curl -sf http://127.0.0.1:8010/health > /dev/null; do kill -0 $SPID || exit 1; sleep 2; done
    python bench_serve.py --tag hf_lora --n 30 --concurrency 1' > ../logs/bench_serve.log 2>&1
  python cost_compare.py --bench ../results/bench_serve_hf_lora.json > ../logs/cost_compare.log 2>&1
fi
if [ "${RUN_ENGINE_BENCH:-0}" = 1 ]; then
  # mini-inference-engine on the merged export (needs RUN_EXPORT first). Same bench, same lock pattern.
  flock "$LOCK" bash -c 'wait_vram
    python serve.py --backend engine --port 8011 > ../logs/serve_engine.log 2>&1 & SPID=$!
    trap "kill $SPID" EXIT
    until curl -sf http://127.0.0.1:8011/health > /dev/null; do kill -0 $SPID || exit 1; sleep 2; done
    python bench_serve.py --url http://127.0.0.1:8011 --tag engine_merged --n 30 --concurrency 1' > ../logs/bench_engine.log 2>&1
fi
echo done
