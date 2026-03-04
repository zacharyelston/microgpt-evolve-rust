# Experiment: Expanded Parameter Ranges (Robust yet Efficient)

## 1. Hypothesis
By increasing the training steps significantly (from 10 to 40) while reducing the population size (from 8 to 4) to maintain efficiency, we expect:
1.  **Better Convergence**: The larger models (`n_layer` > 8) will actually have enough gradient updates to learn basic phonetics.
2.  **Cleaner Separation**: The fitness scores between "random luck" and "actual learning" should become distinct.
3.  **Efficiency**: Total runtime estimated at ~35-40 minutes (2x the pilot, but much more valuable data).

## 2. Experimental Setup
- **Binary**: `hydra_report`
- **Generations**: 5
- **Steps per Gen**: 40
- **Population**: 4 per head (16 total organisms per gen)
- **Parameter Ranges** (Unchanged):
  - `n_emb`: 8 - 64
  - `n_layer`: 1 - 12
  - `n_ctx`: 8 - 32

## 3. Execution
Command:
```bash
cargo run --release --bin hydra_report -- --gens 5 --steps 40 --pop 4
```

## 4. Results
*(To be filled after analysis)*

### 4.1 Fitness Trajectories
- **Weaver**: [Observation]
- **Mirror**: [Observation]
- **Spark**: [Observation]
- **Origin**: [Observation]

### 4.2 Model Complexity
- [Analysis]

## 5. Conclusion
[Pending]
