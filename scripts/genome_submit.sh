#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  $0 --objective <loss|aesthetic> [--source <path>] [--author <name>] [--notes <text>] [--run-log <path>]

Defaults:
  --source genome.json
USAGE
}

objective=""
source_file="genome.json"
author="unknown"
notes=""
run_log=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --objective) objective="$2"; shift 2 ;;
    --source) source_file="$2"; shift 2 ;;
    --author) author="$2"; shift 2 ;;
    --notes) notes="$2"; shift 2 ;;
    --run-log) run_log="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$objective" ]]; then
  echo "--objective is required" >&2
  usage
  exit 1
fi

if [[ "$objective" != "loss" && "$objective" != "aesthetic" ]]; then
  echo "objective must be one of: loss, aesthetic" >&2
  exit 1
fi

if [[ ! -f "$source_file" ]]; then
  echo "source genome not found: $source_file" >&2
  exit 1
fi

python3 - "$objective" "$source_file" "$author" "$notes" "$run_log" <<'PY'
import json, os, sys, datetime, hashlib

objective, source_file, author, notes, run_log = sys.argv[1:6]
repo = os.getcwd()
registry_path = os.path.join(repo, "genomes", "registry.json")
out_dir = os.path.join(repo, "genomes", objective)
os.makedirs(out_dir, exist_ok=True)

with open(source_file, "r", encoding="utf-8") as f:
    data = json.load(f)

required = ["n_emb", "n_ctx", "n_layer", "n_head", "n_ff_exp", "steps", "lr"]
missing = [k for k in required if k not in data]
if missing:
    raise SystemExit(f"source genome missing fields: {missing}")

loss = data.get("loss")
generation = data.get("generation")
created = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
base_blob = json.dumps(data, sort_keys=True).encode("utf-8")
short_hash = hashlib.sha1(base_blob).hexdigest()[:10]
slug = f"{created[:10]}_{objective}_{short_hash}"
out_name = f"{slug}.json"
out_path = os.path.join(out_dir, out_name)

entry = {
    "id": slug,
    "path": f"genomes/{objective}/{out_name}",
    "objective": objective,
    "created_at": created,
    "author": author,
    "loss": loss,
    "generation": generation,
    "notes": notes,
    "run_log": run_log or None,
}

payload = {
    **{k: data[k] for k in required},
    "loss": loss,
    "generation": generation,
    "objective": objective,
    "author": author,
    "notes": notes,
    "created_at": created,
    "source": os.path.relpath(source_file, repo),
    "run_log": run_log or None,
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")

if os.path.exists(registry_path):
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)
else:
    registry = {"version": 1, "entries": []}

registry.setdefault("version", 1)
registry.setdefault("entries", []).append(entry)

with open(registry_path, "w", encoding="utf-8") as f:
    json.dump(registry, f, indent=2)
    f.write("\n")

print(f"Submitted genome: {entry['id']}")
print(f"  file: {entry['path']}")
print(f"  registry: genomes/registry.json")
PY
