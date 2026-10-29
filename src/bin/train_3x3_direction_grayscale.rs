//! 3x3 Direction Detection with 11-level Grayscale
//! Trains on 3x3 grids to predict movement direction using proven grayscale system

use candle_core::{Device, Tensor, Result, DType};
use candle_nn::{VarBuilder, Module, Linear, Optimizer, SGD};
use rand::seq::SliceRandom;
use std::time::Instant;

#[derive(Debug, Clone)]
struct DirectionSample {
    grid: [[u8; 3]; 3], // 3x3 grid with 11 grayscale levels (0-10)
    direction: usize,   // 0-8 for 8 directions + center
}

impl DirectionSample {
    fn parse_line(line: &str) -> Option<Self> {
        let parts: Vec<&str> = line.trim().split_whitespace().collect();
        if parts.len() != 10 {
            return None;
        }
        
        let mut grid = [[0u8; 3]; 3];
        for i in 0..9 {
            grid[i / 3][i % 3] = parts[i].parse().ok()?;
        }
        
        let direction = parts[9].parse().ok()?;
        
        Some(DirectionSample { grid, direction })
    }
    
    fn flatten(&self) -> Vec<u8> {
        self.grid.iter().flat_map(|row| row.iter().cloned()).collect()
    }
}

struct DirectionModel {
    layers: Vec<Linear>,
    device: Device,
}

impl DirectionModel {
    fn new(input_dim: usize, hidden_dims: &[usize], output_dim: usize, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        let mut prev_dim = input_dim;
        
        for &hidden_dim in hidden_dims {
            layers.push(candle_nn::linear(prev_dim, hidden_dim, vb.pp(&format!("layer_{}", layers.len())))?);
            prev_dim = hidden_dim;
        }
        
        layers.push(candle_nn::linear(prev_dim, output_dim, vb.pp("output"))?);
        
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

struct DirectionTrainer {
    model: DirectionModel,
    optimizer: SGD,
    device: Device,
}

impl DirectionTrainer {
    fn new(device: Device) -> Result<Self> {
        let vb = VarBuilder::zeros(DType::F32, &device);
        
        // Network architecture for 3x3 direction detection
        let input_dim = 9 * 11; // 9 pixels * 11 grayscale one-hot each
        let hidden_dims = &[128, 64, 32];
        let output_dim = 9; // 8 directions + center
        
        let model = DirectionModel::new(input_dim, hidden_dims, output_dim, vb)?;
        
        // Simple SGD optimizer
        let optimizer = candle_nn::SGD::new(
            vec![], // Empty for now
            0.01, // learning rate
        )?;
        
        Ok(Self {
            model,
            optimizer,
            device,
        })
    }
    
    fn load_dataset(&self, file_path: &str) -> Result<Vec<DirectionSample>> {
        let content = std::fs::read_to_string(file_path)?;
        let mut samples = Vec::new();
        
        for line in content.lines() {
            if let Some(sample) = DirectionSample::parse_line(line) {
                samples.push(sample);
            }
        }
        
        println!("Loaded {} direction samples", samples.len());
        Ok(samples)
    }
    
    fn create_one_hot_features(&self, grid: &[[u8; 3]; 3]) -> Vec<f32> {
        let mut features = Vec::with_capacity(9 * 11);
        
        for row in grid {
            for &pixel in row {
                // One-hot encode 11 grayscale levels (0-10)
                for i in 0..11 {
                    features.push(if pixel == i as u8 { 1.0 } else { 0.0 });
                }
            }
        }
        
        features
    }
    
    fn create_batch(&self, samples: &[DirectionSample], indices: &[usize]) -> Result<(Tensor, Tensor)> {
        let batch_size = indices.len();
        let mut input_data = Vec::new();
        let mut target_data = Vec::new();
        
        for &idx in indices {
            let sample = &samples[idx];
            
            // Input features
            let features = self.create_one_hot_features(&sample.grid);
            input_data.extend(features);
            
            // Target (class indices for cross entropy)
            target_data.push(sample.direction as u32);
        }
        
        let input = Tensor::from_vec(input_data, (batch_size, 9 * 11), &self.device)?;
        let target = Tensor::from_vec(target_data, batch_size, &self.device)?;
        
        Ok((input, target))
    }
    
    fn calculate_accuracy(&self, samples: &[DirectionSample], test_size: usize) -> Result<f32> {
        let mut correct = 0;
        let test_indices: Vec<usize> = (0..samples.len()).take(test_size).collect();
        
        for chunk in test_indices.chunks(64) {
            let (input, target) = self.create_batch(samples, chunk)?;
            let logits = self.model.forward(&input)?;
            let predictions = logits.argmax(1)?.to_vec1::<u32>()?;
            
            for (i, &idx) in chunk.iter().enumerate() {
                if predictions[i] as usize == samples[idx].direction {
                    correct += 1;
                }
            }
        }
        
        Ok(correct as f32 / test_size as f32)
    }
    
    fn train(&mut self, samples: &[DirectionSample], steps: usize) -> Result<f32> {
        println!("Starting 3x3 direction training...");
        
        let start_time = Instant::now();
        let mut best_accuracy = 0.0f32;
        
        for step in 0..steps {
            // Create random batch
            let batch_size = 64.min(samples.len());
            let mut indices: Vec<usize> = (0..samples.len()).collect();
            indices.shuffle(&mut rand::thread_rng());
            let batch_indices: Vec<usize> = indices.into_iter().take(batch_size).collect();
            
            let (input, target) = self.create_batch(samples, &batch_indices)?;
            
            // Forward pass
            let logits = self.model.forward(&input)?;
            let loss = candle_nn::loss::cross_entropy(&logits, &target)?;
            
            // Backward pass
            self.optimizer.backward_step(&loss)?;
            
            // Calculate accuracy periodically
            if step % 100 == 0 {
                let accuracy = self.calculate_accuracy(samples, 1000)?;
                if accuracy > best_accuracy {
                    best_accuracy = accuracy;
                }
                
                let elapsed = start_time.elapsed();
                let steps_per_sec = step as f64 / elapsed.as_secs_f64();
                let eta = if steps_per_sec > 0.0 {
                    (steps - step) as f64 / steps_per_sec
                } else {
                    0.0
                };
                
                println!(
                    "[{:6.1}s] Step {}/{} | Loss: {:.4} | Acc: {:.1}% | Best: {:.1}% | ETA: {:.1}s",
                    elapsed.as_secs_f64(),
                    step,
                    steps,
                    loss.to_vec0::<f32>()?,
                    accuracy * 100.0,
                    best_accuracy * 100.0,
                    eta
                );
                
                // Early stopping if we reach high accuracy
                if accuracy >= 0.95 {
                    println!("🎯 Reached 95% accuracy, stopping early");
                    break;
                }
            }
        }
        
        let final_accuracy = self.calculate_accuracy(samples, 5000)?;
        println!("🎉 Final accuracy: {:.1}%", final_accuracy * 100.0);
        
        Ok(final_accuracy)
    }
}

fn main() -> Result<()> {
    println!("=== 3x3 Direction Detection with 11-level Grayscale ===");
    
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Device: {:?}", device);
    
    let mut trainer = DirectionTrainer::new(device)?;
    
    // Load or generate dataset
    let dataset_path = "grayscale_3x3_direction_training.txt";
    let samples = if std::path::Path::new(dataset_path).exists() {
        trainer.load_dataset(dataset_path)?
    } else {
        println!("Generating synthetic 3x3 direction dataset...");
        generate_direction_dataset(dataset_path)?;
        trainer.load_dataset(dataset_path)?
    };
    
    // Train the model
    let accuracy = trainer.train(&samples, 5000)?;
    
    println!("🎯 Training completed with {:.1}% accuracy", accuracy * 100.0);
    
    Ok(())
}

fn generate_direction_dataset(output_path: &str) -> Result<()> {
    use std::fs::File;
    use std::io::Write;
    
    let mut file = File::create(output_path)?;
    let mut samples = Vec::new();
    
    // Generate patterns for each direction
    for direction in 0..9 {
        for _ in 0..10000 {
            let grid = generate_direction_pattern(direction);
            let line = format!("{} {} {} {} {} {} {} {} {} {}\n",
                grid[0][0], grid[0][1], grid[0][2],
                grid[1][0], grid[1][1], grid[1][2],
                grid[2][0], grid[2][1], grid[2][2],
                direction
            );
            samples.push(line);
        }
    }
    
    // Shuffle samples
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    samples.shuffle(&mut rng);
    
    // Write to file
    for line in samples {
        file.write_all(line.as_bytes())?;
    }
    
    println!("Generated 90,000 3x3 direction samples");
    Ok(())
}

fn generate_direction_pattern(direction: usize) -> [[u8; 3]; 3] {
    let mut grid = [[5u8; 3]; 3]; // Start with middle gray (5)
    
    match direction {
        0 => { // Center
            grid[1][1] = 10; // Bright center
        }
        1 => { // Up
            grid[0][1] = 10;
            grid[1][1] = 7;
        }
        2 => { // Up-Right
            grid[0][2] = 10;
            grid[1][1] = 7;
        }
        3 => { // Right
            grid[1][2] = 10;
            grid[1][1] = 7;
        }
        4 => { // Down-Right
            grid[2][2] = 10;
            grid[1][1] = 7;
        }
        5 => { // Down
            grid[2][1] = 10;
            grid[1][1] = 7;
        }
        6 => { // Down-Left
            grid[2][0] = 10;
            grid[1][1] = 7;
        }
        7 => { // Left
            grid[1][0] = 10;
            grid[1][1] = 7;
        }
        8 => { // Up-Left
            grid[0][0] = 10;
            grid[1][1] = 7;
        }
        _ => {}
    }
    
    // Add some noise and variation
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for row in &mut grid {
        for pixel in row {
            if *pixel != 10 { // Don't modify the main signal
                *pixel = (*pixel as i8 + rng.gen_range(-2..=2)).max(0).min(10) as u8;
            }
        }
    }
    
    grid
}
