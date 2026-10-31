"""Early replay check for the stage-4 long runs, from their stdout logs.

    python3 scripts/stage4_long_early.py LONG_LOGS_DIR

Each run prints its full test at step 10000 ("rho R A @10000 ...") about a quarter of the
way in. The printed values (3 decimals, LM loss 4) must match the band run it replays;
the exact check on the JSON comes later, in scripts/stage4_long.py.
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage4_long import band_rows  # noqa: E402


def main():
    band = band_rows()
    counts = {"match": 0, "differ": 0, "waiting": 0}
    for p in sorted(glob.glob(os.path.join(sys.argv[1], "s*_r*_t*.out"))):
        m = re.search(r"s(\d+)_r([\d.]+)_t(\d+)\.out$", p)
        k = (int(m.group(1)), round(float(m.group(2)), 2), int(m.group(3)))
        with open(p) as f:
            line = next((l for l in f if "@10000" in l), None)
        if line is None:
            counts["waiting"] += 1
            print(f"{os.path.basename(p)}: not at step 10000 yet")
            continue
        got = [float(x) for x in re.sub(r"[\[\],]", " ", line.split("@10000", 1)[1]).split()]
        b = band[k]
        want = [b["slope_sigma_bin"], *b["slope_sigma_ci95"], b["slope_cue"], *b["slope_cue_ci95"],
                b["acc_vs_sigma_bin"], b["acc_vs_cue"], b["probe_lm_loss_noisy"]]
        tol = [0.00051] * 8 + [0.000051]
        ok = len(got) == len(want) and all(abs(g - w) <= t for g, w, t in zip(got, want, tol))
        counts["match" if ok else "differ"] += 1
        gap = b["slope_cue"] - b["slope_sigma_bin"]
        print(f"{os.path.basename(p)}: {'matches' if ok else 'DIFFERS from'} the band run (band gap {gap:+.2f})"
              + ("" if ok else f"\n    got  {got}\n    band {[round(w, 4) for w in want]}"))
    print(f"\n{counts['match']} match, {counts['differ']} differ, {counts['waiting']} not yet at step 10000")


if __name__ == "__main__":
    main()
