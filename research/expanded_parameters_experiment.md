# Experiment: Expanded Parameter Ranges

## 1. Hypothesis
By significantly increasing the allowable range for model hyperparameters, we expect to see:
1.  **Higher Ceiling**: The "Spark" (Creativity) and "Origin" (Loss) heads should achieve better scores as they can leverage deeper networks (`n_layer` up to 12) and richer embeddings (`n_emb` up to 64).
2.  **Slower Convergence**: Larger models require more training steps to converge, so we might see initial instability or slower fitness growth compared to the "insect-scale" models (1-8 parameters).
3.  **Emergent Complexity**: We hope to see "lizard-scale" behavior—more coherent names and potentially better structural patterns in the "Mirror" head.

## 2. Experimental Setup
- **Binary**: `hydra_report`
- **Generations**: 20
- **Steps per Gen**: 50
- **Population**: 8 per head
- **Parameter Ranges**:
  - `n_emb`: 8 - 64 (previously 1-8)
  - `n_layer`: 1 - 12 (previously 1-8)
  - `n_ctx`: 8 - 32 (previously 1-8)
  - `n_head`: 1 - 8 (constrained by `n_emb % n_head == 0`)

## 3. Execution
Command:
```bash
cargo run --release --bin hydra_report -- --gens 20 --steps 50
```

## 4. Results

### 4.1 Pilot Run (20260303_130517)
**Config:** 5 Gens, 10 Steps/Gen, Pop 8.
- **Outcome**: 10 steps was insufficient. Models were undertrained (Weaver score < 0).

### 4.2 Robust yet Efficient Run (20260303_133023)
**Config:** 5 Gens, 40 Steps/Gen, Pop 4.
**Total Time:** ~9.3 mins.

#### Fitness Trajectories
- **Weaver (Flow)**: Still struggled (Score 0.0). Phonetics require significant training to master.
- **Mirror (Symmetry)**: Oscillated (0.2 -> 0.9). The small population (4) likely contributed to this instability (genetic drift).
- **Spark (Creativity)**: Consistently high (~2.5). Novelty is the easiest objective to satisfy even with random weights.
- **Origin (Loss)**: Loss ~2.8 (Score ~0.35). Improved slightly over the 10-step run, but still far from the <1.0 loss seen in smaller models.

#### Model Complexity (The "Lizard" Brains)
Distinct architectural preferences emerged:
- **Spark (Deep & Narrow)**: Consistently favored `n_layer=11`, `n_emb=10`. It seems depth helps generate novel/weird patterns.
- **Origin (Wide & Shallow)**: Favored `n_emb=44`, `n_layer=3`. Wider embeddings likely help memorize character relationships better for pure prediction with limited training steps.

### 4.3 Images
![Score History](../experiments/run_20260303_133023/score_history.png)
![Complexity History](../experiments/run_20260303_133023/complexity_history.png)

## 5. Conclusion
1.  **Complexity Emerges**: Evolution successfully explores the new "lizard-scale" parameter space (`n_layer` > 8, `n_emb` > 40).
2.  **Trade-off**: These larger brains are significantly harder to train. 40 steps is barely a warm-up.
3.  **Architecture Specialization**: "Creative" heads seem to prefer depth (Layers), while "Analytical" (Loss/Mirror) heads prefer width (Embeddings).

**Recommendation**: To get high-quality results from these larger models, we likely need **100+ training steps** per generation. Efficiency optimization (parallelism) is key to making this viable.
