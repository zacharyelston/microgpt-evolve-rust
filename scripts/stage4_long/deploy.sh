#!/bin/bash
# Copy this checkout's src/ and the long-run scripts to one host, build there, and start
# that host's jobs detached. Run from the repo root (exp/self-report at or after d4ef7e2):
#
#     scripts/stage4_long/deploy.sh user@cyber2-address scripts/stage4_long/jobs_cyber2.txt
#
# Only src/ is copied: each host's Cargo.toml has the macOS-only metal feature removed.
T=$1
J=$2
SSH=(ssh -n -o BatchMode=yes -o ConnectTimeout=10)
[ -f "$J" ] || { echo "no job file $J"; exit 1; }
[ -f src/bin/self_report.rs ] || { echo "run from the repo root"; exit 1; }
"${SSH[@]}" "$T" 'test -d ~/microgpt-stage4/bases' || { echo "$T: no ~/microgpt-stage4/bases"; exit 1; }
rsync -a src/ "$T:microgpt-stage4/src/" || exit 1
rsync -a scripts/stage4_long/run_host.sh "$T:microgpt-stage4/long4.sh" || exit 1
rsync -a "$J" "$T:microgpt-stage4/long_jobs.txt" || exit 1
# touch first so the binary is newer than its source (run_host.sh checks this)
"${SSH[@]}" "$T" 'cd ~/microgpt-stage4 && touch src/bin/self_report.rs && sleep 1 \
  && PATH=$HOME/.cargo/bin:$PATH cargo build --release --bin self_report 2>&1 | tail -n 1' || exit 1
"${SSH[@]}" "$T" 'cd ~/microgpt-stage4 && mkdir -p long_logs && (setsid nohup bash long4.sh > long_logs/launch.log 2>&1 &)' || exit 1
sleep 60
echo "$T after 60 s: $("${SSH[@]}" "$T" 'pgrep -c -x self_report') jobs running"
"${SSH[@]}" "$T" 'cat ~/microgpt-stage4/long_logs/launch.log'
