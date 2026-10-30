"""Score the stage-4 band sweep against its pre-registration (P5-P7 in self_report.rs).

    python3 scripts/stage4_band.py BAND_JSON_DIR

Reads the band runs (self_twins_s*_r*_a_t*_*.json) from BAND_JSON_DIR and the crossover
sweep's runs (results/stage4_sweep, training seed = base seed) as the sixth run per cell.
"""
import glob
import json
import os
import statistics
import sys

RHOS = [0.6, 0.65, 0.7, 0.75, 0.8]
BASES = [7, 8, 9]


def runs_in(pattern):
    for p in glob.glob(pattern):
        with open(p) as f:
            d = json.load(f)
        c = d["config"]
        for run in d["stage4"]["runs"]:
            a = run["twins"]["A (own channel)"]
            yield (c["seed"], round(run["rho"], 2), c.get("train_seed") or c["seed"],
                   a["slope_cue"] - a["slope_sigma_bin"], a)


def main():
    cells = {}
    for base, rho, ts, gap, a in list(runs_in(os.path.join(sys.argv[1], "self_twins_s*_r*_a_t*_*.json"))) + \
            list(runs_in("results/stage4_sweep/self_twins_s*_r*_a_*.json")):
        if base in BASES and rho in RHOS:
            cells.setdefault((base, rho), {})[ts] = (gap, a)

    print(f"gap = slope on cue - slope on own noise bin (> 0: cue-anchored); runs per cell")
    print(f"{'rho':>5} " + " ".join(f"{'base ' + str(b):>34}" for b in BASES))
    sds, mixed_rho, frac = [], [], {}
    for rho in RHOS:
        row, all_mixed, cue, n = [], True, 0, 0
        for b in BASES:
            gaps = [g for _, (g, _) in sorted(cells.get((b, rho), {}).items())]
            row.append(" ".join(f"{g:+.2f}" for g in gaps).rjust(34))
            if len(gaps) > 1:
                sds.append(statistics.stdev(gaps))
            all_mixed = all_mixed and any(g > 0 for g in gaps) and any(g <= 0 for g in gaps)
            cue += sum(g > 0 for g in gaps)
            n += len(gaps)
        frac[rho] = cue / n if n else float("nan")
        if all_mixed:
            mixed_rho.append(rho)
        print(f"{rho:5.2f} " + " ".join(row) + f"   cue-anchored {cue}/{n}")
    counts = {k: len(v) for k, v in cells.items()}
    print(f"\nruns per cell: min {min(counts.values())}, max {max(counts.values())} "
          f"(expected 6), cells {len(counts)} (expected 15)")
    half = statistics.mean((a["slope_cue_ci95"][1] - a["slope_cue_ci95"][0]) / 2
                           for c in cells.values() for _, a in c.values())
    p5 = bool(mixed_rho)
    p6 = statistics.mean(sds) >= 0.10
    p7 = frac[0.8] - frac[0.6] >= 0.30
    print(f"P5 {'holds' if p5 else 'fails'}: rho with every base mixed: {mixed_rho}")
    print(f"P6 {'holds' if p6 else 'fails'}: mean within-cell SD of gap {statistics.mean(sds):.3f} "
          f"(threshold 0.10; mean bootstrap half-width of the cue slope {half:.3f})")
    print(f"P7 {'holds' if p7 else 'fails'}: cue-anchored fraction {frac[0.6]:.2f} at 0.60 -> "
          f"{frac[0.8]:.2f} at 0.80 (needs +0.30)")
    out = {"cells": {f"{b}_{r}": {str(t): g for t, (g, _) in v.items()} for (b, r), v in cells.items()},
           "cue_fraction": frac, "mean_within_cell_sd": statistics.mean(sds),
           "mean_cue_slope_halfwidth": half, "mixed_rho": mixed_rho,
           "P5": p5, "P6": p6, "P7": p7}
    with open("results/stage4_band_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/stage4_band_summary.json")


if __name__ == "__main__":
    main()
