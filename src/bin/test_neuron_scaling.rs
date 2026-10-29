//! Test Neuron Scaling Limits
//! Explore how increasing neuron count affects perspective and grouping capabilities

use candle_core::{Device, Tensor, Result, DType};
use candle_nn::{VarBuilder, Module, Linear};
use rand::seq::SliceRandom;
use std::time::Instant;

#[derive(Debug, Clone)]
struct PerspectiveSample {
    grid: [[u8; 3]; 3], // 3x3 grid with 11 grayscale levels
    grouping: usize,    // 0-4: isolated, pair, line, cluster, perspective
    complexity: usize, // 1-5: simple to complex patterns
}

impl PerspectiveSample {
    fn parse_line(line: &str) -> Option<Self> {
        let parts: Vec<&str> = line.trim().split_whitespace().collect();
        if parts.len() != 12 {
            return None;
        }
        
        let mut grid = [[0u8; 3]; 3];
        for i in 0..9 {
            grid[i / 3][i % 3] = parts[i].parse().ok()?;
        }
        
        let grouping = parts[9].parse().ok()?;
        let complexity = parts[10].parse().ok()?;
        // let _confidence = parts[11].parse().ok()?; // unused for now
        
        Some(PerspectiveSample { grid, grouping, complexity })
    }
}

struct ScalingModel {
    layers: Vec<Linear>,
    device: Device,
}

impl ScalingModel {
    fn new(input_dim: usize, hidden_dims: &[usize], output_dim: usize, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        let mut prev_dim = input_dim;
        
        for &hidden_dim in hidden_dims {
            let layer = candle_nn::linear(prev_dim, hidden_dim, vb.pp(&format!("layer_{}", layers.len())))?;
            layers.push(layer);
            prev_dim = hidden_dim;
        }
        
        let output_layer = candle_nn::linear(prev_dim, output_dim, vb.pp("output"))?;
        layers.push(output_layer);
        
        Ok(Self {
            layers,
            device: vb.device().clone(),
        })
    }
    
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            if i < self.layers.len() - 1 {
                x = x.relu()?;
            }
        }
        
        Ok(x)
    }
}

struct ScalingTest {
    device: Device,
}

impl ScalingTest {
    fn new(device: Device) -> Self {
        Self { device }
    }
    
    fn test_scaling_levels(&self) -> Result<Vec<f32>> {
        println!("=== Testing Neuron Scaling Limits ===");
        
        let mut results = Vec::new();
        
        // Test different neuron counts
        let neuron_configs = vec![
            (vec![32], "32 neurons"),
            (vec![64], "64 neurons"),
            (vec![128], "128 neurons"),
            (vec![32, 16], "32+16 neurons"),
            (vec![64, 32], "64+32 neurons"),
            (vec![128, 64], "128+64 neurons"),
            (vec![256, 128], "256+128 neurons"),
            (vec![64, 32, 16], "64+32+16 neurons"),
            (vec![128, 64, 32], "128+64+32 neurons"),
            (vec![256, 128, 64], "256+128+64 neurons"),
        ];
        
        for (hidden_dims, description) in neuron_configs {
            println!("\n🧪 Testing: {}", description);
            
            // Generate perspective dataset
            let samples = self.generate_perspective_dataset(2000)?;
            
            // Train model
            let accuracy = self.train_perspective_model(&samples, &hidden_dims, 1000)?;
            results.push(accuracy);
            
            println!("📊 {} -> {:.1}% accuracy", description, accuracy * 100.0);
            
            // Analyze what the model learned
            if accuracy > 0.7 {
                self.analyze_capabilities(&samples, &hidden_dims)?;
            }
        }
        
        Ok(results)
    }
    
    fn generate_perspective_dataset(&self, size: usize) -> Result<Vec<PerspectiveSample>> {
        let mut samples = Vec::new();
        
        for grouping in 0..5 {
            for complexity in 1..=5 {
                let samples_per_type = size / 20; // 5 groupings * 5 complexities
                
                for _ in 0..samples_per_type {
                    let grid = generate_perspective_pattern(grouping, complexity);
                    samples.push(PerspectiveSample {
                        grid,
                        grouping,
                        complexity,
                    });
                }
            }
        }
        
        // Shuffle samples
        let mut rng = rand::thread_rng();
        samples.shuffle(&mut rng);
        
        println!("Generated {} perspective samples", samples.len());
        Ok(samples)
    }
    
    fn train_perspective_model(&self, samples: &[PerspectiveSample], hidden_dims: &[usize], steps: usize) -> Result<f32> {
        use candle_nn::{VarMap, AdamW, ParamsAdamW, Optimizer};
        
        // Create model with VarMap pattern
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &self.device);
        let model = ScalingModel::new(9 * 11, hidden_dims, 5, vb)?; // 5 grouping types
        
        let mut optimizer = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: 0.001,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
            },
        )?;
        
        let mut best_accuracy = 0.0f32;
        
        for step in 0..steps {
            // Create batch
            let batch_size = 32.min(samples.len());
            let mut indices: Vec<usize> = (0..samples.len()).collect();
            indices.shuffle(&mut rand::thread_rng());
            let batch_indices: Vec<usize> = indices.into_iter().take(batch_size).collect();
            
            let samples_refs: Vec<&PerspectiveSample> = batch_indices.iter().map(|&i| &samples[i]).collect();
            let (input, target) = self.create_perspective_batch(&samples_refs, &batch_indices)?;
            
            // Forward pass
            let logits = model.forward(&input)?;
            let loss = candle_nn::loss::cross_entropy(&logits, &target)?;
            
            // Backward pass
            optimizer.backward_step(&loss)?;
            
            // Calculate accuracy
            if step % 100 == 0 {
                let accuracy = self.calculate_perspective_accuracy(&model, samples, 500)?;
                if accuracy > best_accuracy {
                    best_accuracy = accuracy;
                }
                
                if step % 200 == 0 {
                    println!("  Step {}: {:.1}% accuracy", step, accuracy * 100.0);
                }
                
                if accuracy >= 0.8 {
                    break;
                }
            }
        }
        
        Ok(best_accuracy)
    }
    
    fn create_perspective_batch(&self, samples: &[&PerspectiveSample], indices: &[usize]) -> Result<(Tensor, Tensor)> {
        let batch_size = indices.len();
        let mut input_data = Vec::new();
        let mut target_data = Vec::new();
        
        for &idx in indices {
            let sample = &samples[idx];
            
            // Input features - one-hot encode grayscale
            for row in &sample.grid {
                for &pixel in row {
                    for i in 0..11 {
                        input_data.push(if pixel == i as u8 { 1.0 } else { 0.0 });
                    }
                }
            }
            
            // Target - grouping type
            target_data.push(sample.grouping as u32);
        }
        
        let input = Tensor::from_vec(input_data, (batch_size, 9 * 11), &self.device)?;
        let target = Tensor::from_vec(target_data, batch_size, &self.device)?;
        
        Ok((input, target))
    }
    
    fn calculate_perspective_accuracy(&self, model: &ScalingModel, samples: &[PerspectiveSample], test_size: usize) -> Result<f32> {
        let mut correct = 0;
        let test_indices: Vec<usize> = (0..samples.len()).take(test_size).collect();
        
        for chunk in test_indices.chunks(32) {
            let (input, _target) = self.create_perspective_batch(samples, chunk)?;
            let logits = model.forward(&input)?;
            let predictions = logits.argmax(1)?.to_vec1::<u32>()?;
            
            for (i, &idx) in chunk.iter().enumerate() {
                if predictions[i] as usize == samples[idx].grouping {
                    correct += 1;
                }
            }
        }
        
        Ok(correct as f32 / test_size as f32)
    }
    
    fn analyze_capabilities(&self, samples: &[PerspectiveSample], hidden_dims: &[usize]) -> Result<()> {
        println!("  🔍 Analyzing capabilities...");
        
        // Test different complexity levels
        for complexity in 1..=5 {
            let complexity_samples: Vec<_> = samples.iter()
                .filter(|s| s.complexity == complexity)
                .take(100)
                .collect();
            
            if !complexity_samples.is_empty() {
                // Create model and test on this complexity
                use candle_nn::{VarMap, AdamW, ParamsAdamW, Optimizer};
                
                let varmap = VarMap::new();
                let vb = VarBuilder::from_varmap(&varmap, DType::F32, &self.device);
                let model = ScalingModel::new(9 * 11, hidden_dims, 5, vb)?;
                
                let indices: Vec<usize> = (0..complexity_samples.len()).collect();
                let samples_refs: Vec<&PerspectiveSample> = complexity_samples.iter().collect();
                let (input, target) = self.create_perspective_batch(&samples_refs, &indices)?;
                let logits = model.forward(&input)?;
                let predictions = logits.argmax(1)?.to_vec1::<u32>()?;
                
                let mut correct = 0;
                for (i, sample) in complexity_samples.iter().enumerate() {
                    if predictions[i] as usize == sample.grouping {
                        correct += 1;
                    }
                }
                
                let accuracy = correct as f32 / complexity_samples.len() as f32;
                println!("    Complexity {}: {:.1}% accuracy", complexity, accuracy * 100.0);
            }
        }
        
        Ok(())
    }
}

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Device: {:?}", device);
    
    let test = ScalingTest::new(device);
    let results = test.test_scaling_levels()?;
    
    println!("\n📈 Scaling Results Summary:");
    println!("Best accuracy: {:.1}%", results.iter().fold(0.0f32, |a, &b| a.max(b)) * 100.0);
    println!("Average accuracy: {:.1}%", results.iter().sum::<f32>() / results.len() as f32 * 100.0);
    
    // Find optimal neuron count
    let max_idx = results.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    
    let configs = vec![
        (vec![32], "32 neurons"),
        (vec![64], "64 neurons"),
        (vec![128], "128 neurons"),
        (vec![32, 16], "32+16 neurons"),
        (vec![64, 32], "64+32 neurons"),
        (vec![128, 64], "128+64 neurons"),
        (vec![256, 128], "256+128 neurons"),
        (vec![64, 32, 16], "64+32+16 neurons"),
        (vec![128, 64, 32], "128+64+32 neurons"),
        (vec![256, 128, 64], "256+128+64 neurons"),
    ];
    
    if let Some((best_config, best_name)) = configs.get(max_idx) {
        println!("🏆 Best configuration: {} with {:.1}% accuracy", 
                best_name, results[max_idx] * 100.0);
        println!("   Architecture: {:?}", best_config);
    }
    
    Ok(())
}

fn generate_perspective_pattern(grouping: usize, complexity: usize) -> [[u8; 3]; 3] {
    let mut grid = [[5u8; 3]; 3]; // Start with middle gray
    
    match grouping {
        0 => { // Isolated - single bright pixel
            let pos = (complexity - 1) % 9;
            grid[pos / 3][pos % 3] = 10;
        }
        1 => { // Pair - two connected pixels
            let pairs = [(0, 1), (3, 4), (6, 7), (1, 2), (4, 5), (7, 8)];
            if let Some((a, b)) = pairs.get((complexity - 1) % pairs.len()) {
                grid[a / 3][a % 3] = 10;
                grid[b / 3][b % 3] = 10;
            }
        }
        2 => { // Line - three pixels in line
            match complexity {
                1 => { grid[0][0] = 10; grid[0][1] = 10; grid[0][2] = 10; } // Top row
                2 => { grid[1][0] = 10; grid[1][1] = 10; grid[1][2] = 10; } // Middle row
                3 => { grid[2][0] = 10; grid[2][1] = 10; grid[2][2] = 10; } // Bottom row
                4 => { grid[0][0] = 10; grid[1][0] = 10; grid[2][0] = 10; } // Left column
                5 => { grid[0][1] = 10; grid[1][1] = 10; grid[2][1] = 10; } // Middle column
                _ => { grid[0][2] = 10; grid[1][2] = 10; grid[2][2] = 10; } // Right column
            }
        }
        3 => { // Cluster - 4 pixels in square
            match complexity {
                1 => { grid[0][0] = 10; grid[0][1] = 10; grid[1][0] = 10; grid[1][1] = 10; } // Top-left
                2 => { grid[0][1] = 10; grid[0][2] = 10; grid[1][1] = 10; grid[1][2] = 10; } // Top-right
                3 => { grid[1][0] = 10; grid[1][1] = 10; grid[2][0] = 10; grid[2][1] = 10; } // Bottom-left
                4 => { grid[1][1] = 10; grid[1][2] = 10; grid[2][1] = 10; grid[2][2] = 10; } // Bottom-right
                5 => { grid[0][0] = 8; grid[0][1] = 9; grid[1][0] = 9; grid[1][1] = 10; } // Gradient cluster
                _ => { grid[0][0] = 10; grid[0][2] = 10; grid[2][0] = 10; grid[2][2] = 10; } // Corners
            }
        }
        4 => { // Perspective - patterns suggesting depth
            match complexity {
                1 => { // Small to large
                    grid[0][1] = 7;
                    grid[1][1] = 8;
                    grid[2][1] = 10;
                }
                2 => { // Diagonal perspective
                    grid[0][0] = 6;
                    grid[1][1] = 8;
                    grid[2][2] = 10;
                }
                3 => { // Converging lines
                    grid[0][0] = 10; grid[0][2] = 10;
                    grid[1][1] = 8;
                    grid[2][1] = 6;
                }
                4 => { // Complex perspective
                    grid[0][0] = 10; grid[0][2] = 10;
                    grid[1][0] = 8;  grid[1][2] = 8;
                    grid[2][1] = 6;
                }
                5 => { // Full perspective scene
                    grid[0][0] = 9; grid[0][1] = 7; grid[0][2] = 9;
                    grid[1][0] = 8; grid[1][1] = 10; grid[1][2] = 8;
                    grid[2][1] = 6;
                }
                _ => { // Advanced perspective
                    grid[0][0] = 10; grid[0][1] = 8; grid[0][2] = 10;
                    grid[1][0] = 8;  grid[1][1] = 6;  grid[1][2] = 8;
                    grid[2][0] = 6;  grid[2][1] = 4;  grid[2][2] = 6;
                }
            }
        }
        _ => {}
    }
    
    // Add noise based on complexity
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let noise_level = (complexity - 1) as i8;
    
    for row in &mut grid {
        for pixel in row {
            if *pixel != 10 && *pixel != 0 { // Don't modify brightest or darkest
                *pixel = (*pixel as i8 + rng.gen_range(-noise_level..=noise_level)).max(0).min(10) as u8;
            }
        }
    }
    
    grid
}
