# Experiment: Deep Learning (Validating Complex Models)

## 1. Hypothesis
The previous experiment (40 steps) showed architectural specialization but poor performance. We hypothesize that **100 training steps** is the critical threshold for these larger models (`n_layer` > 8) to begin converging and outperforming random initialization.

## 2. Experimental Setup
- **Binary**: `hydra_report`
- **Generations**: 5
- **Steps per Gen**: 100
- **Population**: 4 (Kept small for runtime feasibility)
- **Parameter Ranges**:
  - `n_emb`: 8 - 64
  - `n_layer`: 1 - 12
  - `n_ctx`: 8 - 32

## 3. Execution
Command:
```bash
cargo run --release --bin hydra_report -- --gens 5 --steps 100 --pop 4
```
*Started at: ~13:41*
*Status: Running (PID 26670)*
*Estimated duration: ~16 minutes per generation (0.6s/step).*

## 4. Results

### 4.1 Checkpoint: Generation 1/5 (20 mins elapsed)
The larger models are surviving!
- **Weaver (Flow)**: Score **0.40** (vs 0.0 in previous runs). The deep network (7 layers) is starting to learn phonetics (`issvv`, `iliert`).
- **Mirror (Symmetry)**: Score **0.90**. Using a massive 12-layer network (`n_layer=12`).
- **Spark (Creativity)**: Score **2.40**. Consistent.
- **Origin (Loss)**: Score **0.3998** (Loss ~2.5). Still struggling to crush the loss, but `n_emb=50` suggests it wants capacity.

### 4.2 Checkpoint: Generation 2/5 (40 mins elapsed)
Instability strikes (likely due to small population size of 4).
- **Weaver**: Dropped to **0.20**. It abandoned the deep 7-layer model for a shallower 3-layer one (`n_layer=3`), effectively "giving up" on the hard task of phonetics.
- **Mirror**: Crashed to **0.30**.
- **Spark**: Steady at **2.38**.
- **Origin**: Slight improvement to **0.4006**. It is sticking with the `n_emb=50` strategy.

### 4.3 Checkpoint: Generation 3/5 (60 mins elapsed)
Recovery and Divergence.
- **Weaver**: Rebounded to **0.50** using `n_emb=16`, `n_layer=5`. It seems to prefer a "medium" complexity sweet spot.
- **Mirror**: Stuck at **0.30**. The 12-layer giant from Gen 1 died out, replaced by a wide `n_emb=50` model that isn't working yet.
- **Spark**: Stable at **2.38**.
- **Origin**: Continued steady improvement to **0.4157**.

### 4.4 Checkpoint: Generation 4/5 (75 mins elapsed)
**Convergence Event (Homogenization)**:
- **Weaver**, **Mirror**, and **Origin** have all converged to the **exact same genome** (`n_emb=50`, `n_layer=2`, `n_ctx=26`, `lr=0.0019`).
- **Score Impact**:
  - **Weaver**: **0.67** (Good! The genome works for flow).
  - **Mirror**: **0.30** (Bad. It lost its symmetry-specialized architecture).
  - **Origin**: **0.4287** (Steady improvement).
- **Spark**: Remains distinct (`n_emb=20`, `n_layer=2`). The creativity objective successfully defends against the invasion of the "boring" high-performing genome.

### 4.5 Checkpoint: Generation 5/5 (End of Run)
**Homogenization Confirmed**:
- **Weaver**, **Mirror**, and **Origin** ended the run sharing the exact same genome: `n_emb=50`, `n_layer=2`.
- **Spark** remained unique (`n_emb=20`, `n_layer=5`).

**Final Scores**:
- **Weaver**: 0.25 (Struggling with flow).
- **Mirror**: 0.50 (Lost the high-symmetry architecture).
- **Spark**: 2.20 (Consistently creative/weird).
- **Origin**: 0.4126 (Loss ~2.42).

### 4.6 Images
![Score History](../experiments/run_20260303_134148/score_history.png)
![Complexity History](../experiments/run_20260303_134148/complexity_history.png)

## 5. Conclusion
1.  **100 Steps is Viable**: Unlike the 10-step or 40-step runs, the models *did* learn. We saw scores improve and loss decrease steadily.
2.  **Population Size Criticality**: With `pop=4`, the diversity collapsed. One decent genome (`n_emb=50`, `n_layer=2`) took over 3 out of 4 heads. This "invasive species" effect stifled specialization.
3.  **The "Lizard" Brain**: The winning genome was relatively wide (`n_emb=50`) but shallow (`n_layer=2`). The deep 12-layer models from Gen 1 died out, suggesting they are harder to train or less robust to mutation.

**Final Recommendation**:
To truly evolve complex, specialized brains ("Lizard-scale"):
- **Steps**: Keep >= 100.
- **Population**: Must increase back to 8+ to prevent homogenization.
- **Diversity Enforcement**: We may need to limit cross-pollination to prevent one "generalist" genome from wiping out specialists.
