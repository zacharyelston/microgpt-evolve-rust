# Handover: stage-4 long runs (P8–P10)

You are picking up one pre-registered experiment and running it to the end. In order:

1. Launch 36 training runs across three lab hosts.
2. Check early that they replay the earlier runs exactly.
3. Collect the results.
4. Score them against predictions that are already committed.
5. Record the result in the source and report back.

Everything needed is in this repo. The previous agent worked from a laptop that is now off the lab network, so it can't reach the hosts; don't try to reach it.

## State right now

- **Code.** Branch `exp/self-report` of `github.com/zacharyelston/microgpt-evolve-rust`. The pre-registration commit is `d4ef7e2`; this handover is the commit after it.
- **Nothing has run.** No long run has started on any host, and no `long/` or `long_logs/` directory should exist there yet. The laptop-side launchers were stopped, so nothing else will start these jobs.
- **Getting the branch:**

  ```bash
  git clone -b exp/self-report git@github.com:zacharyelston/microgpt-evolve-rust.git && cd microgpt-evolve-rust
  ```

- **The hosts note.** You were also given `stage4-long-hosts.md`, which is kept out of this public repo. It has each host's login, OS and cautions. Before running the commands below, set `CYBER2`, `ML` and `ML2` from it, each as `user@address`. Don't commit that note or copy its contents into the repo, an issue or a PR.

## Background (read the source header for the full record)

`src/bin/self_report.rs`, in the comment block that starts at "Stage 4: the model reports", is the canonical log: each design, its predictions (fixed before the run), and each result. Read it before doing anything.

**The setup.**
- A small character-level GPT is pretrained on names (one "base" per seed: 7, 8 and 9).
- A copy of the base, twin A, then learns to emit a report token R0..R4 after a name. Its target is the bin of a noise level σ that is injected inside the network.
- The input also carries a public cue token. In training, the cue equals σ's bin with probability ρ.
- At test, σ and the cue are drawn independently. The expected report is regressed on each.
- **gap** = slope on cue − slope on own noise. A run is **cue-anchored** if gap ≥ +0.10 and **self-anchored** if gap ≤ −0.10.

**Results so far.**
- In the band sweep (ρ 0.60–0.80, 3 bases × 6 training seeds), which anchor a run ends up with is partly chance (P5–P7 held).
- Run-to-run spread is about 9× the bootstrap intervals, which only cover test-name sampling.
- Self-anchored runs end with lower training report loss than cue-anchored ones at every ρ. So the objective prefers the own-state anchor, yet most runs at ρ ≥ 0.70 take the cue.

**What this run tests.**
- The 36 band runs at ρ 0.65 and 0.70 are replayed exactly for 10,000 steps (phase 1).
- Then each continues 30,000 more steps with the learning rate restarted (phase 2).
- Slopes and held-out report loss are measured along the way.

**Predictions, verbatim in the header.**
- **Check:** at step 10,000 the full test equals the band run exactly, in all 36 runs.
- **P8, cue first:** at step 1,000, at least 30 of 36 runs have gap > 0.
- **P9, one-way pull:** at least 15 of the 17 self-anchored runs still have gap ≤ 0 at step 40,000. At least 11 of the 14 cue-anchored runs have a lower gap at 40,000 than at 10,000.
- **P10, the fork wears off:** at least 7 of the 14 cue-anchored runs have gap ≤ 0 at step 40,000.

## Hosts

Three hosts, called cyber2, ml and ml2 here. Logins, OS details and cautions are in the hosts note.

- **Hardware.** All three have the same 16-core / 32-thread CPU. The work is CPU only.
- **Jobs files.** `scripts/stage4_long/jobs_cyber2.txt`, `jobs_ml.txt` and `jobs_ml2.txt`.
- **Cargo.** `deploy.sh` builds with `~/.cargo/bin` put first on PATH, which covers hosts where cargo isn't on PATH.
- **Access.** Key-based SSH only. Your machine's key has to be in each account's `authorized_keys`; if it isn't, ask the user to add it. Never type or enter a password.
- **Layout.** Each host has `~/microgpt-stage4` with:
  - `src/`, `Cargo.toml`, `Cargo.lock` and `input.txt`;
  - `bases/base7.json`, `bases/base8.json` and `bases/base9.json`, the pretrained bases every band run used (`run_host.sh` checks their md5);
  - `results/`.
- **cyber2 also holds** the band sweep's own outputs: `band/`, `band_logs/`, `results_pretrain/` and `logs/`. Leave them alone.
- **Other services.** Leave alone anything else running on these hosts; the hosts note says what. Jobs run at `nice -n 10`, 12 per host.

## Steps

### 1. Check access (from your machine, at the repo root)

```bash
for T in "$CYBER2" "$ML" "$ML2"; do ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$T" 'hostname; uptime; ls ~/microgpt-stage4; pgrep -fa self_report || echo idle'; done
```

Every host should answer, list `bases`, and be idle. If a host already has `long/` or `long_logs/`, stop and ask: someone has already started this.

### 2. Deploy and launch, one host at a time

```bash
scripts/stage4_long/deploy.sh "$CYBER2" scripts/stage4_long/jobs_cyber2.txt
```

```bash
scripts/stage4_long/deploy.sh "$ML" scripts/stage4_long/jobs_ml.txt
```

```bash
scripts/stage4_long/deploy.sh "$ML2" scripts/stage4_long/jobs_ml2.txt
```

`deploy.sh` does the following:
1. Copies `src/` to the host. It never copies `Cargo.toml`, because each host's copy has the macOS-only `metal` feature removed.
2. Copies `scripts/stage4_long/run_host.sh` to the host as `long4.sh`, and the jobs file as `long_jobs.txt`.
3. Rebuilds on the host.
4. Starts `long4.sh` detached.
5. After 60 s, prints the number of running jobs (expect 12) and `long_logs/launch.log`.

`run_host.sh` refuses to start if long runs already exist there, if stage-4 jobs are already running, if a base's md5 differs, or if the binary is older than its source.

Each job runs:

```
self_report --stage 4 --steps 10000 --report-steps 10000 --restart-steps 30000 --trace 500 --probe 2000 --twin a --seed B --rho R --train-seed T --base-file bases/baseB.json
```

- B is the base (7, 8 or 9) and R is 0.65 or 0.7.
- T is 101–105, or the base seed itself. `--train-seed` equal to the base seed reproduces the crossover sweep's run, which had no `--train-seed`, so that run is the sixth per cell.
- Logs go to `long_logs/sB_rR_tT.out` (stdout) and `.err` (stderr).

### 3. Early check, about a quarter of the way in

- **When.** Each run prints its full test at step 10,000 to stdout, as a line starting `rho R A @10000`. I estimate that comes 30–45 minutes after launch.
- **Progress.** Trace lines in the `.err` files look like `trace  5000  slope s … gap …`, and loss lines look like `step  5000 …`.
- **Fetch the logs and check them:**

```bash
mkdir -p ../stage4_long_logs/cyber2 ../stage4_long_logs/ml ../stage4_long_logs/ml2
```

```bash
rsync -a "$CYBER2:microgpt-stage4/long_logs/" ../stage4_long_logs/cyber2/ && rsync -a "$ML:microgpt-stage4/long_logs/" ../stage4_long_logs/ml/ && rsync -a "$ML2:microgpt-stage4/long_logs/" ../stage4_long_logs/ml2/
```

```bash
for h in cyber2 ml ml2; do python3 scripts/stage4_long_early.py ../stage4_long_logs/$h; done
```

- **Expected result:** every run that has reached step 10,000 matches its band run.
- **If any run differs, stop and report.** Don't patch the code and rerun silently, because a mismatch means phase 1 isn't the band run. Any change to the plan, including a rerun, gets written into the header as a deviation before results are read.

### 4. Wait for completion

- **Estimate.** About 2–2.5 hours per host after launch. This is an estimate, not measured. Each run does 4× the band run's training plus about 25% for measurements, at 12 jobs per 16-core host. A band run took about 1,670 s per 10,000 steps at 28 jobs on cyber2.
- **Done marker.** A host is done when `~/microgpt-stage4/long_logs/DONE` exists. Its 12 JSONs are then in `~/microgpt-stage4/long/`.
- **No repeated polling.** Check every 20–30 minutes at most.

### 5. Collect and score (repo root)

```bash
mkdir -p results/stage4_long && for T in "$CYBER2" "$ML" "$ML2"; do rsync -a "$T:microgpt-stage4/long/" results/stage4_long/; done && ls results/stage4_long | wc -l
```

Expect 36 files, named `self_twins_sB_rR_a_tT_x30000_<timestamp>.json`. Fetch the logs one last time, as in step 3, and keep them outside the repo; band logs weren't committed either.

```bash
python3 scripts/stage4_long.py
```

The scorer prints the following and writes `results/stage4_long_summary.json`:
- the check, comparing the exact JSON at step 10,000 with `results/stage4_band` and `results/stage4_sweep`;
- the class counts (expect 14 cue-anchored and 17 self-anchored);
- a per-run table of gap at steps 1,000, 10,000 and 40,000, held-out report loss, and the first traced step with gap ≤ 0;
- P8, P9 and P10 as holds or fails;
- two figures that were not pre-registered: within-cell SD of gap, and mean held-out loss by class.

### 6. Record and commit

1. **Record.** In `src/bin/self_report.rs`, directly under the "Long runs" pre-registration paragraph, add a result paragraph in the same style as the earlier ones ("Band result (…): P5–P7 held. …"). Start it with "Long result (results/stage4_long, scripts/stage4_long.py):". Give the check outcome, each of P8–P10 as held or failed with its numbers, and anything else marked "not pre-registered".
2. **Keep the predictions as written.** Never edit their text or thresholds.
3. **Commit** `results/stage4_long/`, `results/stage4_long_summary.json` and the header change on `exp/self-report`. Match the earlier subjects, for example: `self_report --stage 4 long runs: <one-line finding>`.
4. **Push only if the user says so.**

### 7. Report back to the user

Write a short plain-language summary covering:
- whether the replay check passed;
- which predictions held, with their numbers;
- what happened to the cue-anchored runs (switched, drifted, or stayed);
- whether the result is clear-cut.

How the outcomes bear on the question behind stage 4, whether a trained self-report's objective fixes what it reports on:

| Outcome | Reading |
|---|---|
| P9 and P10 hold | The cue anchor is a finite-training effect: given more training, the objective eventually decides. |
| P9 holds, P10 fails | The objective pulls toward the own state, but the cue anchor survives 4× the training. The training path fixes the anchor. |
| P9 fails | Phase 2 doesn't move runs consistently toward the lower-loss anchor. Look at held-out loss and the switches in both directions before saying more. |

## Rules

- **Pre-registration is fixed.** Predictions, thresholds and the run plan in `d4ef7e2` don't change. Report failures as failures.
- **Shell hygiene, as the user requires:**
  - never `rm -rf` a path held in a variable;
  - put multi-step commands in script files, not long quote-heavy one-liners or inline heredocs;
  - in zsh, `$VAR` holding several flags is not word-split, so it silently passes one argument. This has already sent a run to the wrong stage once.
- **Keep every result.** Never delete result files or logs on the hosts or locally. Don't commit or touch the untracked `results/self_report_20261028_234307.json`.
- **No secrets** in files, logs, URLs or commit messages.
- **Ask before** pushing, opening PRs, touching any cluster or service on the hosts, or anything outside this experiment.
- **Keep host details out of the repo.** It is public: no logins, addresses or descriptions of the hosts' other roles in commits, results or docs.

## Pitfalls already hit

- **Machine determinism.** Runs are bit-identical across the three x86 hosts: cyber2's results were reproduced exactly on ml and ml2, 4 runs in full. An Apple-silicon Mac diverges over 20k steps. Don't compare a Mac run with a host run, and don't run these jobs on a Mac.
- **Build on each host.** The hosts' glibc versions differ, so a binary built on cyber2 won't run on ml or ml2.
- **ssh hangs** unless the remote job is detached. `deploy.sh` uses `ssh -n` with `(setsid nohup … &)`.
- **The binary doesn't contain flag names as strings.** The compiler turns those comparisons into integer compares, so `grep` on the binary won't show whether `--restart-steps` is compiled in. Freshness is checked by the binary being newer than the source, which `deploy.sh` touches before building.
- **ρ is spelled `0.7`, not `0.70`,** in filenames and logs, matching the band files.
- **Bootstrap intervals only cover test-name sampling.** Variation between training runs is about 9× larger, so don't read much into a single run.

## Files

| File | What it is |
|---|---|
| `src/bin/self_report.rs` | The model, every stage, and the header log of designs, predictions and results |
| `scripts/stage4_long/deploy.sh` | Sync, build and launch on one host |
| `scripts/stage4_long/run_host.sh` | Runs on the host: guards, then 12 jobs in parallel, then `long/` and `DONE` |
| `scripts/stage4_long/jobs_*.txt` | The 36 jobs as `base rho train_seed`, 12 per host, mixed across bases and ρ |
| `scripts/stage4_long_early.py` | Early replay check from stdout logs |
| `scripts/stage4_long.py` | Final scorer: exact check plus P8–P10 |
| `scripts/stage4_band.py`, `scripts/stage4_loss_anchor.py` | Earlier analyses (band sweep P5–P7; training loss against anchor) |
| `results/stage4_band/`, `results/stage4_sweep/` | The runs being replayed: band training seeds 101–105, and the sweep's run with training seed = base seed |
