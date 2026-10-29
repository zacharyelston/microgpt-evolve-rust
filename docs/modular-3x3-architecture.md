# Modular 3x3 Vision Block Architecture

## Overview

This document describes the modular 3x3 vision block system designed for reusable vision processing that can be assembled into larger grids.

## Architecture Components

### 1. VisionBlock3x3 (`src/blocks/block_3x3.rs`)

The fundamental building block that processes 3x3 pixel grids.

**Key Features:**
- Processes 3x3 grids with one-hot encoding (90 features: 3×3×10)
- Linear layer with configurable embedding dimension
- Learnable synaptic weight for block-level modulation
- GPU-accelerated forward pass using Candle framework
- Genome extraction for weight persistence

**Usage:**
```rust
let block = VisionBlock3x3::new("block_0", 24, &device)?;
let output = block.forward(&grid_pixels)?;
let genome = block.extract_genome()?;
```

### 2. BlockAssembly (`src/blocks/assembly.rs`)

Combines multiple 3x3 blocks into larger vision systems.

**Layout Configurations:**
- `Single`: 1×1 blocks (3×3 pixels)
- `Grid2x2`: 2×2 blocks (6×6 pixels)
- `Grid3x3`: 3×3 blocks (9×9 pixels)
- `Grid4x4`: 4×4 blocks (12×12 pixels)

**Key Features:**
- Configurable block layouts
- Shared evaluator for combining block outputs
- Individual block replacement capability
- Genome-based assembly construction

**Usage:**
```rust
let assembly = BlockAssembly::new("assembly_1", (3, 3), 24, 48, &device)?;
let logits = assembly.forward(&grid_data)?;
```

### 3. Genome Management (`src/blocks/genome.rs`)

Handles serialization and management of trained block weights.

**Components:**
- `BlockGenome`: Serializable weights for a single block
- `GenomeCollection`: Multiple genomes with performance metrics
- `GenomeManager`: File-based collection management

**Key Features:**
- JSON serialization for easy storage
- Performance tracking (accuracy, loss, training time)
- Collection statistics and best-genome selection
- Export/import for assembly construction

## Training Pipeline

### 1. Train Individual Blocks

```bash
cargo run --release --bin train_modular_3x3 [collection_id]
```

**Process:**
1. Load 3x3 dataset with 9 directional labels
2. Create single VisionBlock3x3
3. Train with AdamW optimizer on GPU
4. Extract and save genome to collection
5. Track performance metrics

### 2. Test Assemblies

```bash
cargo run --release --bin test_assembly [collection_id] [layout]
```

**Process:**
1. Load genome collection
2. Create assembly with specified layout
3. Test with sample data
4. Report accuracy and performance

## Performance Characteristics

### Single Block Performance
- **Training Speed**: ~573 steps/sec (GPU Metal)
- **Inference Speed**: ~0.002 sec per forward pass
- **Memory Usage**: ~10MB per block
- **Accuracy**: 9.5% (single block, 9 directions)

### Assembly Scaling
- **Linear scaling** with block count
- **GPU acceleration** maintained
- **Modular replacement** possible
- **Persistent genomes** enable reuse

## File Structure

```
src/blocks/
├── mod.rs              # Module exports
├── block_3x3.rs       # Individual 3x3 block
├── assembly.rs         # Multi-block assembly
└── genome.rs           # Genome management

genomes/
├── collection_1.json   # Trained block genomes
└── collection_2.json   # Different training runs

bin/
├── train_modular_3x3.rs  # Block training
├── test_assembly.rs       # Assembly testing
└── generate_eyespot_3x3.rs # Dataset generation
```

## Usage Examples

### Training a Specialized Block

```rust
// Train block for specific task
let collection_id = "edge_detector_v1";
let config = TrainingConfig {
    learning_rate: 0.02,
    embedding_dim: 32,
    training_steps: 10000,
    // ...
};
```

### Building Custom Assembly

```rust
// Create 2x2 assembly from trained genomes
let genomes = manager.get_collection("edge_detector_v1")?;
let assembly = BlockAssembly::from_genomes("custom_2x2", &genomes.genomes[0..4], 64, &device)?;
```

### Replacing Underperforming Blocks

```rust
// Replace block with better performing variant
let better_block = VisionBlock3x3::from_genome(&best_genome, &device)?;
assembly.replace_block(1, 0, better_block)?;
```

## Design Principles

### 1. **Modularity**
- Each 3x3 block is independently trainable
- Blocks can be mixed and matched
- Clear separation of concerns

### 2. **Reusability**
- Genomes can be saved and loaded
- Performance metrics guide selection
- Assembly patterns are reusable

### 3. **Scalability**
- Linear scaling with block count
- GPU acceleration maintained
- Memory usage predictable

### 4. **Extensibility**
- New block types can be added
- Custom layouts supported
- Different evaluators possible

## Future Enhancements

### 1. **Specialized Blocks**
- Edge detection blocks
- Color processing blocks
- Motion detection blocks

### 2. **Advanced Assembly**
- Hierarchical assemblies
- Dynamic block selection
- Adaptive routing

### 3. **Performance Optimization**
- Block quantization
- Model pruning
- Batch processing

### 4. **Training Improvements**
- Transfer learning between blocks
- Multi-task learning
- Curriculum learning

## Conclusion

The modular 3x3 vision block system provides a solid foundation for building scalable vision processing systems. By separating concerns into trainable blocks, manageable assemblies, and persistent genomes, we enable rapid experimentation and reuse of proven components.

The system is designed to scale from single blocks (3×3 pixels) to large assemblies (12×12+ pixels) while maintaining GPU acceleration and modular flexibility.
