"""Band sweep, after the fact: does the training objective prefer the anchor runs end up with?

    python3 scripts/stage4_loss_anchor.py

Within each rho, compares twin A's training report loss (train_report_ce_ema) between
cue-anchored runs (gap > 0) and self-anchored ones, against the loss of reading the cue
alone. Not pre-registered: written after the band result.
"""
import glob
import json
import math
import statistics

RHOS = [0.6, 0.65, 0.7, 0.75, 0.8]
BINS = 5


def cue_only_ce(rho):
    # Best loss from the cue alone: the target equals the cue with p = rho + (1 - rho)/5.
    p = rho + (1 - rho) / BINS
    return -(p * math.log(p) + (1 - p) * math.log((1 - p) / (BINS - 1)))


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    for k, i in enumerate(order):
        r[i] = float(k)
    return r


def main():
    rows = []
    for p in glob.glob("results/stage4_band/self_twins_s*_r*_a_t*_*.json") + \
            glob.glob("results/stage4_sweep/self_twins_s*_r*_a_*.json"):
        with open(p) as f:
            d = json.load(f)
        for run in d["stage4"]["runs"]:
            rho = round(run["rho"], 2)
            if d["config"]["seed"] in (7, 8, 9) and rho in RHOS:
                a = run["twins"]["A (own channel)"]
                rows.append((rho, a["slope_cue"] - a["slope_sigma_bin"], a["train_report_ce_ema"]))
    out = {}
    print(f"{'rho':>5} {'cue-only CE':>12} {'cue-anchored CE':>18} {'self-anchored CE':>18} {'spearman':>9}")
    for rho in RHOS:
        cell = [r for r in rows if r[0] == rho]
        cue = [ce for _, g, ce in cell if g > 0]
        own = [ce for _, g, ce in cell if g <= 0]
        rs = statistics.correlation(ranks([g for _, g, _ in cell]), ranks([ce for _, _, ce in cell]))
        out[rho] = {"cue_only_ce": cue_only_ce(rho), "cue_anchored": cue, "self_anchored": own,
                    "spearman_gap_ce": rs}
        print(f"{rho:5.2f} {cue_only_ce(rho):12.3f} {statistics.mean(cue):11.3f} (n {len(cue):2}) "
              f"{statistics.mean(own):11.3f} (n {len(own):2}) {rs:+9.2f}")
    with open("results/stage4_loss_anchor_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/stage4_loss_anchor_summary.json")


if __name__ == "__main__":
    main()
