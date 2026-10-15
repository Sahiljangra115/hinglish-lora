"""Transfer run: apply base and SFT model to real Shrota transcript segments.

No gold labels exist for these segments, so this script produces model outputs and
reference-free sanity statistics only. Hand labels are required for a real transfer gap.
"""
import argparse
import glob
import json
import os
import re

from common import ROOT, mt_messages, ner_messages
from evaluate import NER_MAX_NEW, generate, load

# Shrota project checkout (read-only). Default assumes ~/Developer/{Portfolio/hinglish-lora,Projects/lets}.
SHROTA = os.environ.get("SHROTA_DIR", str(ROOT.parent.parent / "Projects" / "lets"))
DEVA = re.compile(r"[ऀ-ॿ]")


def collect():
    seen, rows = set(), []
    files = sorted(glob.glob(f"{SHROTA}/data/transcripts/*.json") + glob.glob(f"{SHROTA}/eval/cache/*.transcript.json"))
    for f in files:
        d = json.load(open(f))
        for s in d.get("segments", []):
            t = s["text"].strip()
            if len(t.split()) >= 3 and t not in seen:
                seen.add(t)
                rows.append({"file": f.split("/")[-1], "lang": s.get("language"), "text": t})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(ROOT / "adapters" / "qwen05b-comi"))
    a = ap.parse_args()
    rows = collect()
    if not rows:
        raise SystemExit(f"no transcript segments found under {SHROTA}; set SHROTA_DIR to the Shrota checkout")
    out = {"n_segments": len(rows), "by_file": {}, "segments": []}
    for r in rows:
        out["by_file"][r["file"]] = out["by_file"].get(r["file"], 0) + 1
    for name, ad in [("base", None), ("sft", a.adapter)]:
        tok, model = load(ad)
        mt = generate(tok, model, [mt_messages(r["text"]) for r in rows], 128)
        ner = generate(tok, model, [ner_messages(r["text"].split()) for r in rows], NER_MAX_NEW)
        for r, m, n in zip(rows, mt, ner):
            r.setdefault("out", {})[name] = {"mt": m.split("\n")[0], "ner_raw": n}
        out[f"{name}_stats"] = {
            "mean_len_ratio_out_over_in_words": round(sum(len(m.split()) / len(r["text"].split()) for r, m in zip(rows, mt)) / len(rows), 3),
            "pct_outputs_containing_devanagari": round(100 * sum(bool(DEVA.search(m)) for m in mt) / len(rows), 1),
            "pct_outputs_identical_to_input": round(100 * sum(m.strip() == r["text"] for r, m in zip(rows, mt)) / len(rows), 1),
            "pct_ner_none": round(100 * sum(n.strip().lower() == "none" for n in ner) / len(rows), 1),
        }
        del model
        import torch; torch.cuda.empty_cache()
    out["segments"] = rows
    json.dump(out, open(ROOT / "results" / "shrota_transfer.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "segments"}, indent=1))


if __name__ == "__main__":
    main()
