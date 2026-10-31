"""Score the stage-4 long runs against their pre-registration (check, P8-P10 in self_report.rs).

    python3 scripts/stage4_long.py [LONG_JSON_DIR]

Reads the long runs (default results/stage4_long) and, for the check, the band runs they
replay: results/stage4_band (training seeds 101-105) and results/stage4_sweep (training
seed = base seed).
"""
import glob
import json
import os
import statistics
import sys

RHOS = [0.65, 0.7]
BASES = [7, 8, 9]


def gap(row):
    return row["slope_cue"] - row["slope_sigma_bin"]


def band_rows():
    rows = {}
    for p in glob.glob("results/stage4_band/self_twins_s*_r*_a_t*_*.json") + \
            glob.glob("results/stage4_sweep/self_twins_s*_r*_a_*.json"):
        with open(p) as f:
            d = json.load(f)
        c = d["config"]
        for run in d["stage4"]["runs"]:
            rows[(c["seed"], round(run["rho"], 2), c.get("train_seed") or c["seed"])] = \
                run["twins"]["A (own channel)"]
    return rows


def main():
    long_dir = sys.argv[1] if len(sys.argv) > 1 else "results/stage4_long"
    band = band_rows()
    runs = {}
    for p in glob.glob(os.path.join(long_dir, "self_twins_s*_x30000_*.json")):
        with open(p) as f:
            d = json.load(f)
        c = d["config"]
        for run in d["stage4"]["runs"]:
            runs[(c["seed"], round(run["rho"], 2), c["train_seed"])] = run["twins"]["A (own channel)"]
    print(f"{len(runs)} long runs (expected 36)")

    mismatch = [k for k, a in runs.items() if a["at_report_steps"] != band.get(k)]
    print(f"Check {'holds' if not mismatch else 'fails'}: test at 10000 equals the band run "
          f"in {len(runs) - len(mismatch)} of {len(runs)}" + (f"; differs: {sorted(mismatch)}" if mismatch else ""))

    at = {k: {t["step"]: t for t in a["trace"]} for k, a in runs.items()}
    g10 = {k: gap(a["at_report_steps"]) for k, a in runs.items()}
    g40 = {k: gap(a) for k, a in runs.items()}
    cue = sorted(k for k in runs if g10[k] >= 0.10)
    own = sorted(k for k in runs if g10[k] <= -0.10)
    print(f"at 10000: {len(cue)} cue-anchored (expected 14), {len(own)} self-anchored (expected 17)")

    print(f"\n{'run':>16} {'gap 1000':>9} {'gap 10000':>10} {'gap 40000':>10} {'val CE 10000':>13} "
          f"{'val CE 40000':>13}  first trace step with gap <= 0")
    for k in sorted(runs):
        tr = at[k]
        first = next((s for s in sorted(tr) if gap(tr[s]) <= 0), None)
        cls = "cue " if k in cue else "self" if k in own else "    "
        print(f"{cls} s{k[0]} r{k[1]:<4} t{k[2]:<4} {gap(tr[1000]):+9.2f} {g10[k]:+10.2f} {g40[k]:+10.2f} "
              f"{tr[10000]['val_report_ce']:13.3f} {tr[40000]['val_report_ce']:13.3f}  {first}")

    first_pos = sum(gap(at[k][1000]) > 0 for k in runs)
    own_kept = sum(g40[k] <= 0 for k in own)
    cue_down = sum(g40[k] < g10[k] for k in cue)
    cue_switched = sum(g40[k] <= 0 for k in cue)
    p8 = first_pos >= 30
    p9 = own_kept >= 15 and cue_down >= 11
    p10 = cue_switched >= 7
    print(f"\nP8 {'holds' if p8 else 'fails'}: {first_pos} of {len(runs)} runs have gap > 0 at step 1000 (needs 30)")
    print(f"P9 {'holds' if p9 else 'fails'}: {own_kept} of {len(own)} self-anchored runs still have gap <= 0 "
          f"at 40000 (needs 15); {cue_down} of {len(cue)} cue-anchored runs have a lower gap (needs 11)")
    print(f"P10 {'holds' if p10 else 'fails'}: {cue_switched} of {len(cue)} cue-anchored runs have gap <= 0 "
          f"at 40000 (needs 7)")

    # Not pre-registered: does run-to-run spread shrink, and what does the objective say?
    sd = {}
    for rho in RHOS:
        for b in BASES:
            ks = [k for k in runs if k[0] == b and k[1] == rho]
            if len(ks) > 1:
                sd[f"{b}_{rho}"] = (statistics.stdev(g10[k] for k in ks), statistics.stdev(g40[k] for k in ks))
    if sd:
        print(f"\nmean within-cell SD of gap: {statistics.mean(v[0] for v in sd.values()):.3f} at 10000, "
              f"{statistics.mean(v[1] for v in sd.values()):.3f} at 40000")
    for name, ks in (("cue-anchored", cue), ("self-anchored", own)):
        if ks:
            print(f"{name}: mean val report CE {statistics.mean(at[k][10000]['val_report_ce'] for k in ks):.3f} "
                  f"at 10000 -> {statistics.mean(at[k][40000]['val_report_ce'] for k in ks):.3f} at 40000")

    out = {"runs": {f"{k[0]}_{k[1]}_{k[2]}": {"gap_1000": gap(at[k][1000]), "gap_10000": g10[k],
                                               "gap_40000": g40[k],
                                               "trace": [{"step": s, "gap": gap(t), "val_report_ce": t["val_report_ce"]}
                                                         for s, t in sorted(at[k].items())]}
                    for k in sorted(runs)},
           "check_mismatches": [list(k) for k in sorted(mismatch)],
           "cue_anchored_10000": [list(k) for k in cue], "self_anchored_10000": [list(k) for k in own],
           "within_cell_sd": sd, "P8": p8, "P9": p9, "P10": p10}
    with open("results/stage4_long_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/stage4_long_summary.json")


if __name__ == "__main__":
    main()
