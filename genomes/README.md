# Genome Registry

This folder is a Git-backed registry of contributed genomes.

## Layout

- `registry.json`: index of submitted genomes and metadata.
- `loss/`: genomes optimized for loss.
- `aesthetic/`: genomes optimized for aesthetic score.
- `runs/`: optional imported run snapshots (many candidates from one experiment log).

## Contribution flow

1. Run an evolution command (example):
   - `cargo run --release --bin evolve_loss`
2. Submit the winning `genome.json` to the registry:
   - `./scripts/genome_submit.sh --objective loss --author <name>`
3. (Optional) Import all candidates from a run log for a fuller picture:
   - `./scripts/genome_import_loss_log.sh --log experiments/evolve_YYYYMMDD_HHMMSS.log --author <name>`
4. Commit the generated `genomes/*` files and `genomes/registry.json`.

## Selection flow

- Set the active genome for `cargo run --release`:
  - `./scripts/genome_use.sh genomes/loss/<file>.json`
