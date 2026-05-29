#!/usr/bin/env python3
"""Workflow runner for Argo: writes config, runs evolution binary, captures output.

Usage:
    python3 workflow_run_evolution.py \
        --generations 20 \
        --population-size 12 \
        --target-loss 1.2 \
        --output-dir /results
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def run(args: argparse.Namespace) -> None:
    config_src = Path("/app/evolution_config.json")
    cfg = json.loads(config_src.read_text())
    cfg["evolution_settings"]["generations"] = args.generations
    cfg["evolution_settings"]["population_size"] = args.population_size
    cfg["evolution_settings"]["target_loss"] = args.target_loss

    run_config = Path("evolution_config.json")
    run_config.write_text(json.dumps(cfg, indent=2))
    print(f"[run] Config written: generations={args.generations}, "
          f"population={args.population_size}, target_loss={args.target_loss}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "evolve.log"

    binary = Path("/app/evolve_genesis_config_gpu")
    if not binary.exists():
        binary = Path("/app/evolve_genesis_config_cpu")

    print(f"[run] Starting binary: {binary}")
    result = subprocess.run(
        [str(binary)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = result.stdout.decode()
    print(output)
    log_path.write_text(output)

    m = re.search(r"Best loss: ([\d.]+)", output)
    best_loss = m.group(1) if m else "N/A"
    Path("/tmp/best-loss.txt").write_text(best_loss + "\n")
    print(f"[run] Best loss: {best_loss}")

    if result.returncode != 0:
        print(f"[run] Binary exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=12)
    parser.add_argument("--target-loss", type=float, default=1.2)
    parser.add_argument("--output-dir", default="/results")
    run(parser.parse_args())
