#!/usr/bin/env python3
"""Workflow analysis script for Argo Workflows.

Parses evolution log output and writes Argo output parameter files.
No matplotlib dependency - text output only.

Usage:
    python3 workflow_analyze.py --log /tmp/evolve.log
"""
import argparse
import json
import re
import sys
from pathlib import Path


def parse_log(log_file: str) -> dict:
    content = Path(log_file).read_text()

    generations = re.findall(r'--- Generation (\d+)/\d+ ---', content)
    best_losses = re.findall(r'Best: (\d+\.\d+)', content)
    best_loss_final = re.search(r'Best loss: (\d+\.\d+)', content)
    total_time = re.search(r'Total time: (\d+\.\d+)s', content)
    best_config = re.search(r'Best config: (.+)', content)
    total_evals = re.search(r'Total evaluations: (\d+)', content)

    result = {
        "total_generations": len(generations),
        "best_losses": [float(x) for x in best_losses],
        "final_best_loss": float(best_loss_final.group(1)) if best_loss_final else (float(best_losses[-1]) if best_losses else None),
        "initial_best_loss": float(best_losses[0]) if best_losses else None,
        "total_time_seconds": float(total_time.group(1)) if total_time else None,
        "total_evaluations": int(total_evals.group(1)) if total_evals else None,
        "best_config": best_config.group(1).strip() if best_config else None,
    }

    if result["final_best_loss"] and result["initial_best_loss"]:
        result["total_improvement"] = result["initial_best_loss"] - result["final_best_loss"]

    return result


def build_summary(result: dict) -> str:
    lines = ["=== MicroGPT Evolution Results ==="]
    lines.append(f"Generations:       {result['total_generations']}")
    if result.get("initial_best_loss"):
        lines.append(f"Initial loss:      {result['initial_best_loss']:.4f}")
    if result.get("final_best_loss"):
        lines.append(f"Final best loss:   {result['final_best_loss']:.4f}")
    if result.get("total_improvement"):
        lines.append(f"Total improvement: {result['total_improvement']:.4f}")
    if result.get("total_time_seconds"):
        mins = result["total_time_seconds"] / 60
        lines.append(f"Total time:        {mins:.1f} min")
        if result["total_generations"] > 0:
            lines.append(f"Time/generation:   {mins/result['total_generations']:.2f} min")
    if result.get("total_evaluations"):
        lines.append(f"Total evaluations: {result['total_evaluations']}")
    if result.get("best_config"):
        lines.append(f"Best config:       {result['best_config']}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--output-dir", default="/tmp")
    args = parser.parse_args()

    if not Path(args.log).exists():
        print(f"[analyze] Log file not found: {args.log}", file=sys.stderr)
        Path(args.output_dir, "summary.txt").write_text("No log file found\n")
        Path(args.output_dir, "best-loss.txt").write_text("N/A\n")
        Path(args.output_dir, "best-genome.json").write_text("{}\n")
        sys.exit(0)

    result = parse_log(args.log)
    summary = build_summary(result)
    print(summary)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.txt").write_text(summary + "\n")
    (out / "best-loss.txt").write_text(str(result.get("final_best_loss", "N/A")) + "\n")
    (out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
