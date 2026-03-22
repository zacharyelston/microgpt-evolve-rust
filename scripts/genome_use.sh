#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <registry-genome-json-path>" >&2
  exit 1
fi

src="$1"
if [[ ! -f "$src" ]]; then
  echo "Genome file not found: $src" >&2
  exit 1
fi

cp "$src" genome.json
echo "Active genome set from: $src"
