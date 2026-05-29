#!/usr/bin/env python3
"""Workflow step: merge results from N island evolution runs.

Receives best-loss and genome-json from each island, picks the
winner (minimum loss), writes outputs for the register step.

Usage:
    python3 workflow_merge_islands.py \
        --losses "1.44,1.38,1.51,1.42" \
        --genomes '/path/island-0.json,/path/island-1.json,...' \
        --nodes "hh-b01,hh-b02,hh-b03,hh-b05" \
        --output-dir /tmp
"""
import argparse
import json
from pathlib import Path


def merge(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    losses_raw = [s.strip() for s in args.losses.split(",") if s.strip()]
    genome_files = [s.strip() for s in args.genomes.split(",") if s.strip()]
    nodes = [s.strip() for s in args.nodes.split(",") if s.strip()]

    results = []
    for i, (loss_str, genome_file, node) in enumerate(zip(losses_raw, genome_files, nodes)):
        try:
            loss = float(loss_str)
        except ValueError:
            loss = float("inf")
            print(f"[merge] Island {i} ({node}): loss parse failed ('{loss_str}'), skipping")

        genome = None
        gpath = Path(genome_file)
        if gpath.exists():
            try:
                genome = json.loads(gpath.read_text())
            except Exception as e:
                print(f"[merge] Island {i} ({node}): genome parse failed: {e}")

        results.append({"island": i, "node": node, "loss": loss, "genome": genome})
        print(f"[merge] Island {i} ({node}): loss={loss:.6f} genome={'yes' if genome else 'missing'}")

    valid = [r for r in results if r["genome"] is not None and r["loss"] < float("inf")]
    if not valid:
        print("[merge] No valid islands — writing fallback outputs")
        (out / "best-loss.txt").write_text("N/A\n")
        (out / "best-node.txt").write_text("none\n")
        (out / "island-summary.txt").write_text("No valid island results\n")
        (out / "best-genome.json").write_text("{}\n")
        return

    winner = min(valid, key=lambda r: r["loss"])
    print(f"[merge] Winner: island {winner['island']} on {winner['node']} — loss={winner['loss']:.6f}")

    summary_lines = ["=== Island Evolution Merge ==="]
    for r in results:
        marker = " <-- WINNER" if r is winner else ""
        loss_str = f"{r['loss']:.6f}" if r["loss"] < float("inf") else "FAILED"
        summary_lines.append(f"  [{r['island']}] {r['node']}: loss={loss_str}{marker}")
    summary_lines.append(f"\nBest loss: {winner['loss']:.6f} on {winner['node']}")
    summary = "\n".join(summary_lines)
    print(summary)

    (out / "best-loss.txt").write_text(str(winner["loss"]) + "\n")
    (out / "best-node.txt").write_text(winner["node"] + "\n")
    (out / "island-summary.txt").write_text(summary + "\n")
    (out / "best-genome.json").write_text(json.dumps(winner["genome"], indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--losses", required=True, help="Comma-separated loss values per island")
    parser.add_argument("--genomes", required=True, help="Comma-separated paths to genome JSON files")
    parser.add_argument("--nodes", required=True, help="Comma-separated node names per island")
    parser.add_argument("--output-dir", default="/tmp")
    merge(parser.parse_args())
