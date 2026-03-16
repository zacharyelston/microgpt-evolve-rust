# Large-Scale Evolution Setup for MicroGPT

This document describes the setup for running large-scale evolution tests (100 and 1000 generations) on the MicroGPT project.

## 🎯 Objective

Test the evolutionary capabilities of MicroGPT with extended runs:
- **100 generations**: Medium-scale test (~2-4 hours)
- **1000 generations**: Large-scale test (~20-40 hours)

## 📁 Files Created

### Core Evolution Binary
- `src/bin/evolve_large.rs` - Configurable evolution engine
- Supports any number of generations via command line
- Auto-configures population size and targets based on generation count

### Configuration Files
- `evolution_config_100.json` - 100-generation configuration
- `evolution_config_1000.json` - 1000-generation configuration

### Scripts and Tools
- `run_evolution_100.sh` - Linux/Mac script for 100 generations
- `run_evolution_1000.sh` - Linux/Mac script for 1000 generations
- `test_small_run.ps1` - Windows PowerShell test script
- `analyze_results.py` - Python script for result analysis
- `checkpoint_1000.sh` - Progress monitoring for long runs

## 🚀 Usage

### Quick Start
```bash
# Test with 5 generations
cargo run --release --bin evolve_large -- 5

# Run 100 generations
cargo run --release --bin evolve_large -- 100

# Run 1000 generations
cargo run --release --bin evolve_large -- 1000
```

### Configuration Details

#### 100 Generations
- Population size: 12
- Target loss: 1.0
- Expected duration: 2-4 hours
- Memory usage: ~2-4GB

#### 1000 Generations
- Population size: 16
- Target loss: 0.8
- Expected duration: 20-40 hours
- Memory usage: ~4-8GB

#### Custom Generation Counts
The system auto-configures:
- Population size: `max(8, min(20, generations / 10))`
- Target loss: `0.8` for ≥500 generations, `1.0` otherwise

## 📊 Current Status

### ✅ Completed
1. **Analysis** - Understood existing evolution system
2. **Configuration** - Created configs for 100/1000 generations
3. **Monitoring** - Set up logging and progress tracking
4. **Testing** - Verified system with 5-generation test
5. **Execution** - 100-generation evolution currently running

### 🔄 In Progress
- **100-generation test** - Running in background (started 14:02)

### ⏳ Pending
- **1000-generation test** - Ready to start after 100-gen completes

## 📈 Expected Results

### Performance Metrics
- **Loss improvement**: From ~1.4 to <1.0 (100-gen) or <0.8 (1000-gen)
- **Architecture discovery**: Optimal layer/emb/head configurations
- **Learning rate tuning**: Best LR for the specific problem
- **Training efficiency**: Optimal steps vs quality trade-off

### Hardware Utilization
- **CPU cores**: Full utilization during evaluation phases
- **Memory**: Proportional to population size
- **Disk**: ~100MB logs per 100 generations

## 🔧 Monitoring

### Progress Indicators
The system reports progress every 10 generations:
```
Progress: 10/100 generations (10.0%) - ETA: 1800s - Best: 1.2345
```

### Log Files
- Location: `experiments/evolve_YYYYMMDD_HHMMSS.log`
- Contains: Generation results, best configs, species diversity

### Checkpoint Commands
```bash
# For 1000-gen runs
./checkpoint_1000.sh

# Manual monitoring
tail -f experiments/evolve_*.log
ps aux | grep evolve_large
```

## 📊 Analysis Tools

### Python Analysis Script
```bash
python analyze_results.py experiments/evolve_YYYYMMDD_HHMMSS.log
```

Outputs:
- `*_evolution_plots.png` - Progress visualizations
- `*_analysis.json` - Detailed statistics
- Convergence analysis
- Parameter importance ranking

## 🏆 Best Results

### Current Best (from previous runs)
```
Generation: 10
Loss: 1.4676
Config: Emb:24 Head:2 Lay:1 Ctx:32 FF:3 LR:0.0119 Steps:500
```

### Expected Improvements
- **100 generations**: 10-20% loss improvement
- **1000 generations**: 30-50% loss improvement
- **Architecture discovery**: Novel optimal configurations

## 🛠️ Technical Details

### Evolution Mechanics
- **Selection**: Tournament selection (k=3)
- **Crossover**: Uniform parameter crossover
- **Mutation**: Random parameter perturbation
- **Elitism**: Best individual preserved each generation

### Search Space
- **Embeddings**: 8-64 dimensions
- **Heads**: 1-8 attention heads
- **Layers**: 1-6 transformer layers
- **Context**: 8-32 token context
- **Learning rate**: 0.001-0.1 (log scale)
- **Steps**: 200-3000 training steps

### Parallelization
- **Evaluation**: Parallel across population (rayon)
- **CPU usage**: All available cores
- **Memory**: Linear with population size

## ⚠️ Considerations

### Resource Requirements
- **Power**: Stable for long runs (especially 1000-gen)
- **Heat**: Ensure adequate cooling for extended CPU usage
- **Storage**: Several GB for logs and checkpoints

### Interruption Handling
- **Checkpointing**: Genome saved after each generation
- **Recovery**: Can restart from saved genome
- **Logs**: Continuous logging for progress tracking

### Performance Tuning
- **Population size**: Larger = better exploration but slower
- **Generation count**: Diminishing returns after ~500
- **Target loss**: Aggressive targets may require more generations

## 📚 Next Steps

1. **Monitor 100-gen run** - Check progress every hour
2. **Analyze results** - Use Python script when complete
3. **Start 1000-gen run** - If 100-gen shows promise
4. **Compare architectures** - Study evolved configurations
5. **Document findings** - Record optimal parameters

## 🎯 Success Criteria

### 100 Generations
- [ ] Loss < 1.0 achieved
- [ ] Stable convergence pattern
- [ ] Reasonable time (<6 hours)

### 1000 Generations  
- [ ] Loss < 0.8 achieved
- [ ] Novel architecture discovered
- [ ] Completion within 48 hours

### Overall
- [ ] System stability for long runs
- [ ] Reproducible results
- [ ] Clear performance improvement over baseline

---

**Last Updated**: 2026-03-12 14:05
**Current Status**: 100-generation evolution running
**Next Check**: 2026-03-12 15:00
