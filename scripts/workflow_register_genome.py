#!/usr/bin/env python3
"""Workflow step: read genome-best.json, update next run's genesis_config.

Reads the best genome written by the Rust binary, updates the evolution
config so the next run seeds from it rather than the hardcoded genesis.

Usage:
    python3 workflow_register_genome.py \
        --genome /results/genome-best.json \
        --config /app/evolution_config.json \
        --run-id microgpt-evolve-abc123 \
        --output-dir /tmp
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def register(args: argparse.Namespace) -> None:
    genome_path = Path(args.genome)
    config_path = Path(args.config)
    out = Path(args.output_dir)
    node_name = os.environ.get("EVOLVE_NODE_NAME", "unknown")
    out.mkdir(parents=True, exist_ok=True)

    if not genome_path.exists():
        print(f"[register] No genome file at {genome_path}, using existing genesis")
        (out / "next-genesis.json").write_text(
            config_path.read_text() if config_path.exists() else "{}"
        )
        (out / "genome-loss.txt").write_text("N/A\n")
        (out / "genome-config.txt").write_text("no genome\n")
        return

    genome = json.loads(genome_path.read_text())
    loss = genome.get("loss", 99.0)
    generation = genome.get("generation", "?")

    try:
        loss = float(loss)
    except (TypeError, ValueError):
        loss = 99.0

    if loss >= 90.0:
        print(f"[register] Fallback genome (loss={loss}) — no valid genome to register, using existing genesis")
        (out / "genome-loss.txt").write_text(str(loss) + "\n")
        (out / "genome-config.txt").write_text("fallback — no valid genome\n")
        cfg = json.loads(config_path.read_text()) if config_path.exists() else {}
        (out / "next-genesis.json").write_text(json.dumps(cfg, indent=2) + "\n")
        return

    print(f"[register] Best genome — loss={loss:.6f}, generation={generation}")
    n_emb = genome.get("n_emb") or 1
    n_ctx = genome.get("n_ctx") or 1
    n_layer = genome.get("n_layer") or 1
    n_head = genome.get("n_head") or 1
    n_ff_exp = genome.get("n_ff_exp") or 1
    lr = genome.get("lr") or 0.01
    steps = genome.get("steps") or 1000
    print(f"[register] Config: n_emb={n_emb} n_ctx={n_ctx} n_layer={n_layer} "
          f"n_head={n_head} n_ff_exp={n_ff_exp} lr={lr:.6f} steps={steps}")

    cfg = json.loads(config_path.read_text())

    cfg["genesis_config"] = {
        "n_emb": n_emb,
        "n_ctx": n_ctx,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_ff_exp": n_ff_exp,
        "lr": lr,
        "steps": steps,
    }

    if loss < 1.3:
        print("[register] Good loss — widening search space for next run")
        cfg["parameter_ranges"]["n_emb"]["max"] = min(8, cfg["parameter_ranges"]["n_emb"]["max"] + 1)
        cfg["parameter_ranges"]["n_layer"]["max"] = min(6, cfg["parameter_ranges"]["n_layer"]["max"] + 1)

    next_genesis_path = out / "next-genesis.json"
    next_genesis_path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"[register] Next genesis config written to {next_genesis_path}")

    registry_entry = {
        "id": f"{args.run_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "run_id": args.run_id,
        "node": node_name,
        "loss": loss,
        "generation": generation,
        "genome": genome,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    print(f"[register] Node: {node_name}")
    (out / "registry-entry.json").write_text(json.dumps(registry_entry, indent=2) + "\n")

    (out / "genome-loss.txt").write_text(str(loss) + "\n")
    desc = (f"Emb:{genome.get('n_emb')} Head:{genome.get('n_head')} "
            f"Lay:{genome.get('n_layer')} Ctx:{genome.get('n_ctx')} "
            f"FF:{genome.get('n_ff_exp')} LR:{genome.get('lr', 0):.4f} Steps:{genome.get('steps')}")
    (out / "genome-config.txt").write_text(desc + "\n")
    print(f"[register] {desc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome", default="/results/genome-best.json")
    parser.add_argument("--config", default="/app/evolution_config.json")
    parser.add_argument("--run-id", default="unknown")
    parser.add_argument("--output-dir", default="/tmp")
    register(parser.parse_args())
