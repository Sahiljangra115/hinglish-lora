# Hinglish LoRA on COMI-LINGUA

LoRA fine-tune of a small base model (Qwen2.5-0.5B-Instruct) for two COMI-LINGUA tasks: Hinglish to English machine translation (MT) and named entity recognition (NER). The goal is a number comparable to the COMI-LINGUA paper (Sheth, Beniwal, Singh; Findings of EMNLP 2025, arXiv 2503.21670), plus a transfer check on real Shrota transcripts.

## Status

Code, tests and scripts are written. **No training or evaluation has been run yet** (the GPU was needed for other jobs). Every results table below is `pending, run later`. Nothing here is a measured model number. The unit tests in `tests/` were written but not executed in their final form either (a 7-test run passed once earlier on metrics and parsing only, before the last edits).

## Data (downloaded, measured)

Source: `LingoIITGN/COMI-LINGUA` on Hugging Face (CC BY 4.0). Files in `data/`: `MT_train.csv` (24,692 rows), `MT_test.csv` (5,000), `NER_train.csv` (19,913), `NER_test.csv` (5,000), plus the dataset README.

- MT target: `Annotator_1_en_translation`.
- NER gold: Annotator 1 tags, falling back to Annotator 2 where Annotator 1 is empty (28 of 5,000 test rows). Tag set: PERSON, ORGANISATION, LOCATION, GPE, DATE, TIME, HASHTAG, MENTION, EMOJI, X (non-entity).

## Metrics (what the paper reports)

Read from the paper PDF:

- MT: corpus BLEU and chrF++ computed separately for English, Romanized Hindi and Devanagari Hindi. This project does English only.
- NER: token-level precision, recall and F1 at the macro level.
- Paper fine-tuned LLaMA-3.1-8B-Instruct, MT English: BLEU 56.1, chrF++ 78.7 (Table 7).
- Paper abstract: up to 95.25 F1 on NER (model not named in the abstract; I did not extract the per-model NER table).

Differences from the paper, stated plainly: the NER output format here is my own (numbered tokens in, `index:TAG` pairs out); the macro average here is over entity labels excluding X (`macro_f1`), with `macro_all_f1` (X included) and `micro_f1` also reported; I could not confirm the paper's exact per-sentence averaging. The model is 0.5B parameters against the paper's 7-8B, and training is a reduced run (see below), so the paper numbers are a reference bar, not a like-for-like comparison.

## Setup

- Base: `Qwen/Qwen2.5-0.5B-Instruct`, bf16. The GPU is a 4 GB RTX 2050, so plain LoRA (no 4-bit QLoRA; `bitsandbytes` is not installed).
- LoRA r=16, alpha=32, dropout 0.05, all attention and MLP projections (8.8M trainable params of 502.8M).
- Reduced training: 3,000 MT + 3,000 NER examples, 1 epoch, effective batch 16, lr 2e-4, max length 320 tokens.
- Evaluation: seeded random 300-sentence subsets of each test split (not the full 5,000).
- Comparison arms: base zero-shot, base 3-shot, LoRA SFT. SFT then DPO (fourth arm) has a script, `src/dpo.py`, written but never run. Hosted-API cost comparison is a config-driven script, never run.

## How to run

```
source $ML_VENV/bin/activate   # ml venv; peft installed there
./run_all.sh          # waits for free GPU, trains, evaluates 3 arms, runs Shrota transfer
python -m pytest tests
```

Outputs go to `results/` (`eval_*.json`, `samples_*.json`, `train_log.json`, `shrota_transfer.json`) and the adapter to `adapters/qwen05b-comi`. `run_all.sh` sets `expandable_segments` and retries on out-of-memory because the GPU is shared; it needs roughly 3.5 GB free. Edit the sizes in `run_all.sh` and the `--n_mt/--n_ner` eval flags for a bigger run.

## Optional stages (code written, never run)

Enable from `run_all.sh` with env vars: `RUN_DPO=1 RUN_EXPORT=1 RUN_SERVE_BENCH=1 BACKEND=hf ./run_all.sh`. Or run by hand from `src/`:

```
# DPO arm (manual DPO, trl is not installed). Needs GPU and a trained SFT adapter.
python dpo.py --stage pairs     # sample SFT outputs, pair gold (chosen) vs worse SFT output (rejected), no hosted judge
python dpo.py --stage train     # DPO from the SFT adapter -> adapters/qwen05b-comi-dpo
python evaluate.py --mode lora_sft_dpo --adapter ../adapters/qwen05b-comi-dpo

# Merge adapter and export a plain HF safetensors checkpoint (CPU only)
python merge_export.py          # -> export/qwen05b-comi-merged ; add --gguf_script <llama.cpp convert_hf_to_gguf.py> for GGUF

# Serve and benchmark
python serve.py --backend hf --port 8000          # base + adapter, POST /generate, GET /health (uses the GPU)
python bench_serve.py --tag hf_lora --n 100 --concurrency 1   # p50/p95 latency, tokens/s -> results/bench_serve_*.json

# Cost and latency vs a hosted API (arithmetic only, no network, no key)
# edit configs/pricing.yaml first (prices default to 0.0 on purpose)
python cost_compare.py --bench ../results/bench_serve_hf_lora.json
```

DPO pairs: the plan wants preferences from a large model's judgments. That needs a hosted API and is not done; `dpo.py` instead uses gold references as chosen and the SFT model's clearly worse outputs as rejected. You can supply judge-labelled pairs with `--pairs_file` (JSONL: `messages`, `chosen`, `rejected`). The learning rate, beta and thresholds are untuned guesses.

mini-inference-engine (separate local project, `Projects/mini-inference-engine`): it is an offline engine (`engine.llm.LLM`, no HTTP layer) and has **no LoRA support**, so the adapter must be merged first. Run `merge_export.py`, then `python serve.py --backend engine --model ../export/qwen05b-comi-merged`; `serve.py` wraps `LLM.generate` in the HTTP endpoint. Its README lists the core model, scheduler and kernels as still unwritten stubs, so this path is untested and may not run until that project is finished. Native adapter-swapping in the engine would be future work there.

## Results on COMI-LINGUA test (300-sample subsets)

| Arm | MT BLEU | MT chrF++ | NER macro F1 | NER micro F1 |
|---|---|---|---|---|
| Base zero-shot | pending, run later | pending, run later | pending, run later | pending, run later |
| Base 3-shot | pending, run later | pending, run later | pending, run later | pending, run later |
| LoRA SFT | pending, run later | pending, run later | pending, run later | pending, run later |
| Paper, fine-tuned LLaMA-3.1-8B | 56.1 | 78.7 | see paper (up to 95.25 F1 in abstract) | n/a |

Training time, peak VRAM, seconds per sentence: pending, run later.

## Shrota transfer

`src/shrota_eval.py` collects transcript segments from the Shrota project in the separate local project `Projects/lets` (`data/transcripts/*.json` and `eval/cache/*.transcript.json`). Measured: only **78 unique segments with 3 or more words** exist there, not the 200 the plan assumed, and most are not Hinglish (46 English, 11 Italian, 8 Spanish, 11 Hindi, 2 hi-en by the pipeline's language tag). So a 100-segment hand-labelled transfer test is not possible from the current data; more Hinglish audio would have to be transcribed first.

The script has no gold labels, so it writes base and SFT outputs plus reference-free sanity stats (length ratio, Devanagari leakage, copy rate, empty-entity rate). A benchmark-to-real gap needs hand labels. Results: pending, run later.

## Layout

- `src/common.py` data loading, prompt formats, NER output parsing
- `src/metrics.py` BLEU, chrF++, token-level NER P/R/F1
- `src/train.py` LoRA SFT loop (assistant-only loss)
- `src/evaluate.py` generation and scoring for any arm
- `src/shrota_eval.py` Shrota transfer run
- `tests/test_core.py` unit tests for parsing and metrics
- `src/dpo.py` DPO arm (pairs + train)
- `src/merge_export.py` merge adapter, export HF checkpoint
- `src/serve.py`, `src/bench_serve.py` HTTP endpoint and latency/throughput benchmark
- `src/cost_compare.py`, `configs/pricing.yaml` cost and latency comparison vs hosted API
- `run_all.sh` full pipeline plus optional stages

## Known gaps

- No measured results yet.
- DPO, serving, benchmark, export and cost scripts are unrun and untested (compile-checked only). Hosted-API latency is not measured by anything here, only entered in the config. The mini-inference-engine backend depends on that project being finished.
- Paper NER per-model table and exact NER averaging not verified.
- Small base model and reduced data; expect numbers well below the paper's 7-8B fine-tunes.
