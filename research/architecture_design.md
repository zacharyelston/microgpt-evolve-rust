# MicroHydraGPT Architecture & Design Philosophy

## 1. The Organism: Not a Neuron, But a Brain
A common misconception is that a MicroGPT instance is a single processing unit (a "neuron"). In reality, each individual organism in the population is a **Deep Neural Network** (Transformer) composed of thousands of simulated neurons.

### 1.1 The Atom: `Val` (The Synapse)
At the lowest level (`src/lib.rs`), the `Val` struct represents a single scalar value with automatic differentiation capabilities.
- It is not a neuron; it is akin to a **synapse** (connection strength) or a **signal** (voltage).
- It holds:
  - `data`: The value.
  - `grad`: The gradient (how much this value affects the error).
  - `prev`: The history (computation graph) for backpropagation.

### 1.2 The Layer: `Mat2` (Cortical Tissue)
A layer in the network (e.g., `fc1` or `wq`) is a matrix of these `Val` atoms.
- **Example:** A Feed-Forward layer with `n_emb=64` and `n_ff_exp=4` creates **256 distinct neurons**.
- Each neuron performs a weighted sum of inputs and applies a non-linear activation function (`ReLU`).
- **Code Reference:**
  ```rust
  // 256 neurons firing in parallel
  let h = linear(&xn, &self.fc1[i]).iter().map(|v| v.relu()).collect::<Vec1>();
  ```

### 1.3 The Body: `GPT` (The Nervous System)
The `GPT` struct represents the entire nervous system of one organism. With the upgraded parameter ranges (`n_layer=12`, `n_emb=64`), a single organism possesses:
- **Sensory Input:** Token and Positional Embeddings.
- **Processing Power:** ~3,000+ neurons in the MLP layers alone, plus Attention mechanisms.
- **Routing:** Multiple "Heads" (Attention Heads) that decide which parts of the context to focus on.

## 2. The Hydra: The Species / Ecosystem
The "Hydra" architecture is not a single brain, but a **multi-objective evolutionary system** managing distinct sub-populations.

### 2.1 The Heads (Tribes)
Each "Head" of the Hydra (Weaver, Mirror, Spark, Origin) is a **separate tribe** of organisms.
- **Weaver:** Optimized for linguistic flow and pronounceability.
- **Mirror:** Optimized for symmetry and structural patterns.
- **Spark:** Optimized for novelty and creativity (deviation from training data).
- **Origin:** Optimized for raw predictive accuracy (lowest loss).

### 2.2 The Body (The Exchange)
The central "Body" of the Hydra acts as an exchange hub.
- **Independent Evolution:** Each tribe evolves in isolation for several generations, specializing in their specific objective.
- **The Gathering:** Periodically, tribes submit their "champions" (best genomes) to the Body.
- **Cross-Pollination:** The Body redistributes these champions as "immigrants" to other tribes.
  - *Example:* A high-scoring "Spark" genome (creative) might be sent to the "Weaver" tribe. The Weaver tribe then tries to optimize this creative genome for flow, potentially creating a "creative flow" hybrid.

## 3. Evolutionary Scale
We are not limited by the design, but by the **scale** of the simulation.
- **Micro Scale:** We use small numbers (e.g., 64 dimensions) to allow evolution to run on a standard CPU.
- **Macro Capability:** The architecture is mathematically identical to large language models (LLMs).
  - Increasing `n_emb` = Richer concept representation.
  - Increasing `n_layer` = Deeper logical reasoning.
  - Increasing `n_ctx` = Longer short-term memory.

By expanding the parameter ranges (Embedding: 8-64, Layers: 1-12), we have effectively upgraded the organisms from "insects" to "lizards" in terms of cognitive capacity.
