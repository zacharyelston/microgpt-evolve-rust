"""Score the stage-4 crossover sweep against its pre-registration (fd6eb99).

    python3 scripts/stage4_crossover.py SWEEP_JSON_DIR

Reads results/self_twins_s{7,8,9}_*.json (the stage-4 runs, for the cross-machine
check) and the sweep's self_twins_s*_r*_a_*.json files from SWEEP_JSON_DIR.
rho* is where A's slope on the cue minus its slope on sigma's bin changes sign,
by linear interpolation between adjacent grid points.
"""
import glob
import json
import os
import re
import sys

PRED = {7: 0.585, 8: 0.599, 9: 0.601}
TOL = 0.075


def load(path):
    with open(path) as f:
        return json.load(f)


def twin_a(run):
    return run["twins"]["A (own channel)"]


def main():
    sweep_dir = sys.argv[1]
    sweep = {}
    for p in glob.glob(os.path.join(sweep_dir, "self_twins_s*_r*_a_*.json")):
        d = load(p)
        seed = d["config"]["seed"]
        for run in d["stage4"]["runs"]:
            sweep[(seed, round(run["rho"], 2))] = twin_a(run)
    stage4 = {}
    for p in glob.glob("results/self_twins_s*_2026*.json"):
        if re.search(r"_r[0-9.-]+_", p):
            continue
        d = load(p)
        for run in d["stage4"]["runs"]:
            stage4[(d["config"]["seed"], round(run["rho"], 2))] = twin_a(run)

    out = {"prediction": PRED, "tolerance": TOL, "seeds": {}}
    p4 = True
    for seed in sorted(PRED):
        rhos = sorted(r for s, r in sweep if s == seed)
        print(f"\nseed {seed}: twin A, slope on sigma bin (s) and on cue (c)")
        print(f"{'rho':>5} {'s':>7} {'c':>7} {'c - s':>7} {'acc s':>6} {'acc c':>6}")
        gap = []
        for r in rhos:
            a = sweep[(seed, r)]
            g = a["slope_cue"] - a["slope_sigma_bin"]
            gap.append(g)
            print(f"{r:5.2f} {a['slope_sigma_bin']:7.3f} {a['slope_cue']:7.3f} {g:+7.3f} "
                  f"{a['acc_vs_sigma_bin']:6.3f} {a['acc_vs_cue']:6.3f}")
        cross = None
        for i in range(len(rhos) - 1):
            if gap[i] < 0 <= gap[i + 1]:
                cross = rhos[i] + (rhos[i + 1] - rhos[i]) * (-gap[i]) / (gap[i + 1] - gap[i])
                break
        ok = cross is not None and abs(cross - PRED[seed]) <= TOL
        p4 = p4 and ok
        print(f"rho* = {cross if cross is None else round(cross, 3)}  predicted {PRED[seed]}  "
              f"{'within' if ok else 'NOT within'} {TOL}")
        checks = {}
        for r in (0.5, 0.9):
            a, b = sweep[(seed, r)], stage4.get((seed, r))
            if b is None:
                continue
            inside = all(lo <= a[k] <= hi for k, (lo, hi) in
                         [("slope_sigma_bin", b["slope_sigma_ci95"]), ("slope_cue", b["slope_cue_ci95"])])
            same = all(abs(a[k] - b[k]) < 1e-9 for k in ("slope_sigma_bin", "slope_cue"))
            checks[str(r)] = {"inside_stage4_ci": inside, "identical": same}
            print(f"cross-machine rho {r}: x86 s {a['slope_sigma_bin']:.3f} c {a['slope_cue']:.3f} | "
                  f"arm64 s {b['slope_sigma_bin']:.3f} c {b['slope_cue']:.3f} | "
                  f"{'identical' if same else ('inside CI' if inside else 'OUTSIDE CI')}")
        out["seeds"][seed] = {"rho": rhos, "gap_c_minus_s": gap, "rho_star": cross,
                              "within_tolerance": ok, "cross_machine": checks}
    out["P4_holds"] = p4
    print(f"\nP4 {'holds' if p4 else 'fails'} (every seed within {TOL} of its prediction)")
    with open(os.path.join("results", "stage4_crossover_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/stage4_crossover_summary.json")


if __name__ == "__main__":
    main()
