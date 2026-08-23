"""Metrics matching the COMI-LINGUA paper: BLEU + chrF++ for MT, token-level P/R/F1 for NER."""
from sacrebleu.metrics import BLEU, CHRF


def mt_scores(hyps, refs):
    bleu = BLEU().corpus_score(hyps, [refs]).score
    chrfpp = CHRF(word_order=2).corpus_score(hyps, [refs]).score
    return {"bleu": round(bleu, 2), "chrf++": round(chrfpp, 2)}


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def ner_scores(gold_seqs, pred_seqs, labels):
    """Token-level scores pooled over the whole test set.

    macro_*: unweighted mean over the entity labels (without X) that occur in gold or pred.
    macro_all_*: same but X included (closest reading of the paper's 'macro over tag set').
    micro_*: micro-average over entity labels only (X excluded).
    """
    counts = {l: [0, 0, 0] for l in labels}  # tp, fp, fn
    for g, p in zip(gold_seqs, pred_seqs):
        assert len(g) == len(p)
        for gt, pt in zip(g, p):
            if gt == pt:
                counts.setdefault(gt, [0, 0, 0])[0] += 1
            else:
                counts.setdefault(pt, [0, 0, 0])[1] += 1
                counts.setdefault(gt, [0, 0, 0])[2] += 1
    per = {l: _prf(*c) for l, c in counts.items()}
    active = [l for l, c in counts.items() if l != "X" and (c[0] + c[2] > 0 or c[1] > 0)]
    allact = [l for l, c in counts.items() if c[0] + c[1] + c[2] > 0]

    def mean(ls, k):
        return sum(per[l][k] for l in ls) / len(ls) if ls else 0.0

    tp = sum(counts[l][0] for l in active)
    fp = sum(counts[l][1] for l in active)
    fn = sum(counts[l][2] for l in active)
    mp, mr, mf = _prf(tp, fp, fn)
    return {
        "macro_p": round(100 * mean(active, 0), 2),
        "macro_r": round(100 * mean(active, 1), 2),
        "macro_f1": round(100 * mean(active, 2), 2),
        "macro_all_f1": round(100 * mean(allact, 2), 2),
        "micro_f1": round(100 * mf, 2),
        "per_label_f1": {l: round(100 * per[l][2], 1) for l in sorted(per)},
    }
