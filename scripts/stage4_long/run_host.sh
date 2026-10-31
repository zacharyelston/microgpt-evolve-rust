#!/bin/bash
# Stage 4 long runs (pre-registered in d4ef7e2), run on one host from ~/microgpt-stage4:
# every job in long_jobs.txt ("base rho train_seed" per line) at once, then the results
# move to long/ and long_logs/DONE is written. deploy.sh copies this file there as long4.sh.
cd ~/microgpt-stage4 || exit 1
if [ -e long_logs/DONE ] || [ -n "$(ls long 2>/dev/null)" ]; then
  echo "long runs already here, stopping"
  exit 1
fi
if pgrep -f 'self_report --stage 4' > /dev/null; then
  echo "stage-4 jobs already running here, stopping"
  exit 1
fi
# the bases every band run used (md5 of the copies the band sweep and the ml hosts used)
while read -r want f; do
  [ "$(md5sum "$f" 2>/dev/null | cut -d' ' -f1)" = "$want" ] || { echo "$f missing or changed"; exit 1; }
done <<'SUMS'
3a1a078b49438cb59e497a29bd7a7d7d bases/base7.json
5de3684543381e385742df17b296df52 bases/base8.json
f19bc99f5668558c0d3475a65ec3c119 bases/base9.json
SUMS
[ target/release/self_report -nt src/bin/self_report.rs ] || { echo "binary older than its source"; exit 1; }
mkdir -p long long_logs results
BIN=./target/release/self_report
COMMON="--stage 4 --steps 10000 --report-steps 10000 --restart-steps 30000 --trace 500 --probe 2000 --twin a"
N=$(wc -l < long_jobs.txt)
echo "starting $N jobs $(date)"
xargs -P "$N" -L 1 bash -c 'nice -n 10 '"$BIN $COMMON"' --seed $0 --rho $1 --train-seed $2 \
    --base-file bases/base$0.json > long_logs/s$0_r$1_t$2.out 2> long_logs/s$0_r$1_t$2.err' \
    < long_jobs.txt
mv results/self_twins_s*_x30000_*.json long/
echo "finished $(date)"
touch long_logs/DONE
