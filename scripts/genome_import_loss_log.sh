#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  $0 --log <experiments/evolve_*.log> [--author <name>] [--notes <text>] [--all-candidates]

Imports candidate lines from evolve_loss logs into genomes/loss and appends entries to genomes/registry.json.
By default it keeps top-1 per generation; use --all-candidates to import every ranked candidate.
USAGE
}

log_file=""
author="unknown"
notes=""
all_candidates="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log) log_file="$2"; shift 2 ;;
    --author) author="$2"; shift 2 ;;
    --notes) notes="$2"; shift 2 ;;
    --all-candidates) all_candidates="1"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$log_file" || ! -f "$log_file" ]]; then
  echo "Valid --log is required" >&2
  usage
  exit 1
fi

python3 - "$log_file" "$author" "$notes" "$all_candidates" <<'PY'
import datetime, hashlib, json, os, re, sys

log_file, author, notes, all_candidates = sys.argv[1:5]
all_candidates = all_candidates == "1"
repo = os.getcwd()
registry_path = os.path.join(repo, "genomes", "registry.json")
out_dir = os.path.join(repo, "genomes", "runs")
os.makedirs(out_dir, exist_ok=True)

with open(log_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Example line:
# > #1: Emb:16  Head:4 Lay:3 Ctx:24 FF:3 LR:0.0051 Steps:500  | Loss: 1.5278 [random]
pat = re.compile(
    r"^\s*>?\s*#(?P<rank>\d+):\s+Emb:(?P<n_emb>\d+)\s+Head:(?P<n_head>\d+)\s+Lay:(?P<n_layer>\d+)\s+Ctx:(?P<n_ctx>\d+)\s+FF:(?P<n_ff_exp>\d+)\s+LR:(?P<lr>[0-9.]+)\s+Steps:(?P<steps>\d+)\s+\|\s+Loss:\s+(?P<loss>[0-9.]+)\s+\[(?P<origin>[^\]]+)\]"
)
gen_pat = re.compile(r"^--- Generation\s+(\d+)/(\d+)\s+---")

entries = []
current_gen = None
for line in lines:
    g = gen_pat.match(line.strip())
    if g:
        current_gen = int(g.group(1))
        continue
    m = pat.match(line.rstrip("\n"))
    if not m:
        continue
    d = m.groupdict()
    if current_gen is None:
        continue
    entries.append({
        "generation": current_gen,
        "rank": int(d["rank"]),
        "n_emb": int(d["n_emb"]),
        "n_head": int(d["n_head"]),
        "n_layer": int(d["n_layer"]),
        "n_ctx": int(d["n_ctx"]),
        "n_ff_exp": int(d["n_ff_exp"]),
        "steps": int(d["steps"]),
        "lr": float(d["lr"]),
        "loss": float(d["loss"]),
        "origin": d["origin"],
    })

if not entries:
    raise SystemExit("No candidate lines parsed from log")

created = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
log_base = os.path.basename(log_file).replace(".log", "")
run_id = f"{created[:10]}_{log_base}"

# write full run snapshot
run_path = os.path.join(out_dir, f"{run_id}.json")
run_payload = {
    "run_id": run_id,
    "created_at": created,
    "author": author,
    "notes": notes,
    "objective": "loss",
    "source_log": os.path.relpath(log_file, repo),
    "candidates": entries,
}
with open(run_path, "w", encoding="utf-8") as f:
    json.dump(run_payload, f, indent=2)
    f.write("\n")

loss_dir = os.path.join(repo, "genomes", "loss")
os.makedirs(loss_dir, exist_ok=True)
if all_candidates:
    selected = sorted(entries, key=lambda x: (x["generation"], x["rank"]))
else:
    by_gen = {}
    for e in entries:
        if e["generation"] not in by_gen or e["loss"] < by_gen[e["generation"]]["loss"]:
            by_gen[e["generation"]] = e
    selected = [by_gen[g] for g in sorted(by_gen)]

if os.path.exists(registry_path):
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)
else:
    registry = {"version": 1, "entries": []}
registry.setdefault("version", 1)
registry.setdefault("entries", [])

added = 0
for e in selected:
    gen = e["generation"]
    sig = json.dumps(e, sort_keys=True).encode("utf-8")
    short_hash = hashlib.sha1(sig).hexdigest()[:10]
    gid = f"{created[:10]}_loss_g{gen:02d}_{short_hash}"
    fname = f"{gid}.json"
    path = os.path.join(loss_dir, fname)
    payload = {
        "id": gid,
        "objective": "loss",
        "created_at": created,
        "author": author,
        "notes": notes,
        "source": os.path.relpath(log_file, repo),
        **e,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    registry["entries"].append({
        "id": gid,
        "path": f"genomes/loss/{fname}",
        "objective": "loss",
        "created_at": created,
        "author": author,
        "loss": e["loss"],
        "generation": gen,
        "notes": (f"{notes} ({('rank ' + str(e['rank'])) if all_candidates else 'top-1'} in generation {gen} from {os.path.basename(log_file)})").strip(),
        "run_log": os.path.relpath(log_file, repo),
    })
    added += 1

with open(registry_path, "w", encoding="utf-8") as f:
    json.dump(registry, f, indent=2)
    f.write("\n")

print(f"Imported run snapshot: {os.path.relpath(run_path, repo)}")
mode = "all candidates" if all_candidates else "top-per-generation"
print(f"Added {added} {mode} genomes to genomes/loss and registry")
PY
