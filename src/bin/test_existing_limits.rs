//! Test Limits of Existing Systems
//! Push our proven 1x1 and 3x3 systems to their breaking points

use candle_core::{Device, Tensor, Result, DType};
use candle_nn::{VarBuilder, Module, Linear, Optimizer};
use rand::seq::SliceRandom;
use rand::Rng;

#[derive(Debug, Clone)]
struct LimitTestSample {
    input: Vec<f32>,
    target: u32,
    difficulty: usize, // 1-10: easy to impossible
    description: String,
}

struct LimitTester {
    device: Device,
}

impl LimitTester {
    fn new(device: Device) -> Self {
        Self { device }
    }
    
    fn test_1x1_limits(&self) -> Result<Vec<f32>> {
        println!("=== Testing 1x1 Neuron Limits ===");
        
        let mut results = Vec::new();
        
        // Test 1: Basic grayscale (our baseline)
        println!("🧪 Test 1: Basic 11-level grayscale");
        let accuracy = self.test_1x1_basic_grayscale()?;
        results.push(accuracy);
        println!("  Result: {:.1}% accuracy", accuracy * 100.0);
        
        // Test 2: Extended grayscale (more levels)
        println!("🧪 Test 2: Extended grayscale (20 levels)");
        let accuracy = self.test_1x1_extended_grayscale(20)?;
        results.push(accuracy);
        println!("  Result: {:.1}% accuracy", accuracy * 100.0);
        
        // Test 3: Many grayscale levels (50 levels)
        println!("🧪 Test 3: Many grayscale levels (50 levels)");
        let accuracy = self.test_1x1_extended_grayscale(50)?;
        results.push(accuracy);
        println!("  Result: {:.1}% accuracy", accuracy * 100.0);
        
        // Test 4: Extreme grayscale (100 levels)
        println!("🧪 Test 4: Extreme grayscale (100 levels)");
        let accuracy = self.test_1x1_extended_grayscale(100)?;
        results.push(accuracy);
        println!("  Result: {:.1}% accuracy", accuracy * 100.0);
        
        // Test 5: Noisy grayscale
        println!("🧪 Test 5: Noisy grayscale (11 levels + noise)");
        let accuracy = self.test_1x1_noisy_grayscale()?;
        results.push(accuracy);
        println!("  Result: {:.1}% accuracy", accuracy * 100.0);
        
        // Test 6: Pattern recognition
        println!("🧪 Test 6: Pattern recognition (mathematical patterns)");
        let accuracy = self.test_1x1_patterns()?;
        results.push(accuracy);
        println!("  Result: {:.1}% accuracy", accuracy * 100.0);
        
        // Test 7: Temporal sequences
        println!("🧪 Test 7: Temporal sequences (predict next in series)");
        let accuracy = self.test_1x1_sequences()?;
        results.push(accuracy);
        println!("  Result: {:.1}% accuracy", accuracy * 100.0);
        
        Ok(results)
    }
    
    fn test_3x3_limits(&self) -> Result<Vec<f32>> {
        println!("\n=== Testing 3x3 Direction Limits ===");
        
        let mut results = Vec::new();
        
        // Test 1: Basic directions (our baseline)
        println!("🧪 Test 1: Basic 8 directions + center");
        let accuracy = self.test_3x3_basic_directions()?;
        results.push(accuracy);
        println!("  Result: {:.1}% accuracy", accuracy * 100.0);
        
        // Test 2: More directions (16 directions)
        println!("🧪 Test 2: 16 directions (finer granularity)");
        let accuracy = self.test_3x3_extended_directions(16)?;
        results.push(accuracy);
        println!("  Result: {:.1}% accuracy", accuracy * 100.0);
        
        // Test 3: Many directions (32 directions)
        println!("🧪 Test 3: 32 directions (very fine granularity)");
        let accuracy = self.test_3x3_extended_directions(32)?;
        results.push(accuracy);
        println!("  Result: {:.1}% accuracy", accuracy * 100.0);
        
        // Test 4: Multiple objects
        println!("🧪 Test 4: Multiple objects in grid");
        let accuracy = self.test_3x3_multiple_objects()?;
        results.push(accuracy);
        println!("  Result: {:.1}% accuracy", accuracy * 100.0);
        
        // Test 5: Intensity variations
        println!("🧪 Test 5: Intensity variations (same pattern, different brightness)");
        let accuracy = self.test_3x3_intensity_variations()?;
        results.push(accuracy);
        println!("  Result: {:.1}% accuracy", accuracy * 100.0);
        
        // Test 6: Partial patterns
        println!("🧪 Test 6: Partial patterns (missing information)");
        let accuracy = self.test_3x3_partial_patterns()?;
        results.push(accuracy);
        println!("  Result: {:.1}% accuracy", accuracy * 100.0);
        
        // Test 7: Complex scenes
        println!("🧪 Test 7: Complex scenes (multiple features)");
        let accuracy = self.test_3x3_complex_scenes()?;
        results.push(accuracy);
        println!("  Result: {:.1}% accuracy", accuracy * 100.0);
        
        Ok(results)
    }
    
    fn test_1x1_basic_grayscale(&self) -> Result<f32> {
        // Use our exact working neuron trainer setup
        let samples = self.generate_basic_grayscale_samples(1000);
        self.train_1x1_model(&samples, 1000)
    }
    
    fn test_1x1_extended_grayscale(&self, levels: usize) -> Result<f32> {
        let samples = self.generate_extended_grayscale_samples(1000, levels);
        self.train_1x1_model(&samples, 2000)
    }
    
    fn test_1x1_noisy_grayscale(&self) -> Result<f32> {
        let samples = self.generate_noisy_grayscale_samples(1000);
        self.train_1x1_model(&samples, 2000)
    }
    
    fn test_1x1_patterns(&self) -> Result<f32> {
        let samples = self.generate_pattern_samples(1000);
        self.train_1x1_model(&samples, 3000)
    }
    
    fn test_1x1_sequences(&self) -> Result<f32> {
        let samples = self.generate_sequence_samples(1000);
        self.train_1x1_model(&samples, 3000)
    }
    
    fn test_3x3_basic_directions(&self) -> Result<f32> {
        // Use our exact working 3x3 setup
        let samples = self.generate_basic_direction_samples(2000);
        self.train_3x3_model(&samples, 2000)
    }
    
    fn test_3x3_extended_directions(&self, directions: usize) -> Result<f32> {
        let samples = self.generate_extended_direction_samples(2000, directions);
        self.train_3x3_model(&samples, 3000)
    }
    
    fn test_3x3_multiple_objects(&self) -> Result<f32> {
        let samples = self.generate_multiple_object_samples(2000);
        self.train_3x3_model(&samples, 3000)
    }
    
    fn test_3x3_intensity_variations(&self) -> Result<f32> {
        let samples = self.generate_intensity_variation_samples(2000);
        self.train_3x3_model(&samples, 3000)
    }
    
    fn test_3x3_partial_patterns(&self) -> Result<f32> {
        let samples = self.generate_partial_pattern_samples(2000);
        self.train_3x3_model(&samples, 3000)
    }
    
    fn test_3x3_complex_scenes(&self) -> Result<f32> {
        let samples = self.generate_complex_scene_samples(2000);
        self.train_3x3_model(&samples, 4000)
    }
    
    // Training methods using our proven patterns
    fn train_1x1_model(&self, samples: &[LimitTestSample], steps: usize) -> Result<f32> {
        use candle_nn::{VarMap, AdamW, ParamsAdamW};
        
        let input_dim = samples[0].input.len();
        let output_dim = 10; // 10 classes
        
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &self.device);
        let model = SimpleModel::new(input_dim, &[64, 32], output_dim, vb)?;
        
        let mut optimizer = AdamW::new_lr(varmap.all_vars(), 0.001)?;
        
        let mut best_accuracy = 0.0f32;
        
        for step in 0..steps {
            let batch_size = 32.min(samples.len());
            let mut indices: Vec<usize> = (0..samples.len()).collect();
            indices.shuffle(&mut rand::thread_rng());
            let batch_indices: Vec<usize> = indices.into_iter().take(batch_size).collect();
            
            let (input, target) = self.create_batch(samples, &batch_indices)?;
            
            let logits = model.forward(&input)?;
            let loss = candle_nn::loss::cross_entropy(&logits, &target)?;
            
            optimizer.backward_step(&loss)?;
            
            if step % 100 == 0 {
                let accuracy = self.calculate_accuracy(&model, samples, 500)?;
                if accuracy > best_accuracy {
                    best_accuracy = accuracy;
                }
                
                if accuracy >= 0.95 {
                    break;
                }
            }
        }
        
        Ok(best_accuracy)
    }
    
    fn train_3x3_model(&self, samples: &[LimitTestSample], steps: usize) -> Result<f32> {
        use candle_nn::{VarMap, AdamW, ParamsAdamW};
        
        let input_dim = samples[0].input.len();
        let output_dim = samples.iter().map(|s| s.target as usize).max().unwrap() + 1;
        
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &self.device);
        let model = SimpleModel::new(input_dim, &[128, 64], output_dim, vb)?;
        
        let mut optimizer = AdamW::new_lr(varmap.all_vars(), 0.001)?;
        
        let mut best_accuracy = 0.0f32;
        
        for step in 0..steps {
            let batch_size = 32.min(samples.len());
            let mut indices: Vec<usize> = (0..samples.len()).collect();
            indices.shuffle(&mut rand::thread_rng());
            let batch_indices: Vec<usize> = indices.into_iter().take(batch_size).collect();
            
            let (input, target) = self.create_batch(samples, &batch_indices)?;
            
            let logits = model.forward(&input)?;
            let loss = candle_nn::loss::cross_entropy(&logits, &target)?;
            
            optimizer.backward_step(&loss)?;
            
            if step % 100 == 0 {
                let accuracy = self.calculate_accuracy(&model, samples, 500)?;
                if accuracy > best_accuracy {
                    best_accuracy = accuracy;
                }
                
                if accuracy >= 0.90 {
                    break;
                }
            }
        }
        
        Ok(best_accuracy)
    }
    
    // Helper methods
    fn create_batch(&self, samples: &[LimitTestSample], indices: &[usize]) -> Result<(Tensor, Tensor)> {
        let batch_size = indices.len();
        let input_dim = samples[0].input.len();
        
        let mut input_data: Vec<f32> = Vec::new();
        let mut target_data = Vec::new();
        
        for &idx in indices {
            let sample = &samples[idx];
            input_data.extend(&sample.input);
            target_data.push(sample.target);
        }
        
        let input = Tensor::from_vec(input_data, (batch_size, input_dim), &self.device)?;
        let target = Tensor::from_vec(target_data, batch_size, &self.device)?;
        
        Ok((input, target))
    }
    
    fn calculate_accuracy(&self, model: &SimpleModel, samples: &[LimitTestSample], test_size: usize) -> Result<f32> {
        let mut correct = 0;
        let test_indices: Vec<usize> = (0..samples.len()).take(test_size).collect();
        
        for chunk in test_indices.chunks(32) {
            let (input, _target) = self.create_batch(samples, chunk)?;
            let logits = model.forward(&input)?;
            let predictions = logits.argmax(1)?.to_vec1::<u32>()?;
            
            for (i, &idx) in chunk.iter().enumerate() {
                if predictions[i] == samples[idx].target {
                    correct += 1;
                }
            }
        }
        
        Ok(correct as f32 / test_size as f32)
    }
    
    // Data generation methods
    fn generate_basic_grayscale_samples(&self, count: usize) -> Vec<LimitTestSample> {
        let mut samples = Vec::new();
        
        for i in 0..count {
            let value = (i % 10) as u8; // Changed from 11 to 10 to stay within range
            let mut input = vec![0.0f32; 11];
            input[value as usize] = 1.0;
            
            samples.push(LimitTestSample {
                input,
                target: value as u32,
                difficulty: 1,
                description: format!("grayscale_{}", value),
            });
        }
        
        samples
    }
    
    fn generate_extended_grayscale_samples(&self, count: usize, levels: usize) -> Vec<LimitTestSample> {
        let mut samples = Vec::new();
        
        for i in 0..count {
            let value = (i % levels) as u8;
            let mut input = vec![0.0f32; levels];
            input[value as usize] = 1.0;
            
            samples.push(LimitTestSample {
                input,
                target: (value as f32 / levels as f32 * 10.0) as u32, // Map to 0-10 range
                difficulty: levels / 10,
                description: format!("extended_grayscale_{}_{}", levels, value),
            });
        }
        
        samples
    }
    
    fn generate_noisy_grayscale_samples(&self, count: usize) -> Vec<LimitTestSample> {
        let mut samples = Vec::new();
        let mut rng = rand::thread_rng();
        
        for i in 0..count {
            let value = (i % 10) as u8; // Stay within 0-9
            let mut input = vec![0.0f32; 10]; // Match target size
            input[value as usize] = 1.0;
            
            // Add noise
            for j in 0..10 {
                if j != value as usize {
                    input[j] = rng.gen_range(0.0..0.2); // Small noise
                }
            }
            
            samples.push(LimitTestSample {
                input,
                target: value as u32,
                difficulty: 3,
                description: format!("noisy_grayscale_{}", value),
            });
        }
        
        samples
    }
    
    fn generate_pattern_samples(&self, count: usize) -> Vec<LimitTestSample> {
        let mut samples = Vec::new();
        
        for i in 0..count {
            let pattern_type = i % 5;
            let value = match pattern_type {
                0 => i % 3, // Repeating pattern
                1 => (i * 2) % 11, // Multiplication pattern
                2 => (i * i) % 11, // Square pattern
                3 => (i + (i / 3)) % 11, // Complex pattern
                _ => i % 11, // Simple pattern
            };
            
            let mut input = vec![0.0f32; 11];
            input[value as usize] = 1.0;
            
            samples.push(LimitTestSample {
                input,
                target: pattern_type as u32,
                difficulty: 5,
                description: format!("pattern_type_{}_value_{}", pattern_type, value),
            });
        }
        
        samples
    }
    
    fn generate_sequence_samples(&self, count: usize) -> Vec<LimitTestSample> {
        let mut samples = Vec::new();
        
        for i in 0..count {
            // Generate sequences like: 1,2,3,4,5 -> predict 6
            let sequence_pos = i % 5;
            let sequence_type = i % 3;
            
            let value = match sequence_type {
                0 => (sequence_pos + 1) % 11, // Increment sequence
                1 => (sequence_pos * 2) % 11, // Double sequence
                _ => (sequence_pos * sequence_pos + 1) % 11, // Square+1 sequence
            };
            
            let mut input = vec![0.0f32; 11];
            input[value as usize] = 1.0;
            
            samples.push(LimitTestSample {
                input,
                target: ((sequence_pos + 1) % 5) as u32, // Predict next position
                difficulty: 7,
                description: format!("sequence_type_{}_pos_{}", sequence_type, sequence_pos),
            });
        }
        
        samples
    }
    
    fn generate_basic_direction_samples(&self, count: usize) -> Vec<LimitTestSample> {
        let mut samples = Vec::new();
        
        for i in 0..count {
            let direction = i % 9; // 8 directions + center
            let grid = generate_direction_grid(direction);
            let input = grid_to_features(&grid);
            
            samples.push(LimitTestSample {
                input,
                target: direction as u32,
                difficulty: 2,
                description: format!("direction_{}", direction),
            });
        }
        
        samples
    }
    
    fn generate_extended_direction_samples(&self, count: usize, directions: usize) -> Vec<LimitTestSample> {
        let mut samples = Vec::new();
        
        for i in 0..count {
            let direction = i % directions;
            let grid = generate_extended_direction_grid(direction, directions);
            let input = grid_to_features(&grid);
            
            samples.push(LimitTestSample {
                input,
                target: direction as u32,
                difficulty: directions / 8,
                description: format!("extended_direction_{}_{}", directions, direction),
            });
        }
        
        samples
    }
    
    fn generate_multiple_object_samples(&self, count: usize) -> Vec<LimitTestSample> {
        let mut samples = Vec::new();
        
        for i in 0..count {
            let object_count = (i % 3) + 1; // 1-3 objects
            let grid = generate_multiple_object_grid(object_count);
            let input = grid_to_features(&grid);
            
            samples.push(LimitTestSample {
                input,
                target: (object_count - 1) as u32,
                difficulty: 4,
                description: format!("multiple_objects_{}", object_count),
            });
        }
        
        samples
    }
    
    fn generate_intensity_variation_samples(&self, count: usize) -> Vec<LimitTestSample> {
        let mut samples = Vec::new();
        
        for i in 0..count {
            let direction = i % 9;
            let intensity = (i % 5) + 5; // 5-9 intensity levels
            let grid = generate_intensity_grid(direction, intensity as u8);
            let input = grid_to_features(&grid);
            
            samples.push(LimitTestSample {
                input,
                target: direction as u32,
                difficulty: 3,
                description: format!("intensity_{}_direction_{}", intensity, direction),
            });
        }
        
        samples
    }
    
    fn generate_partial_pattern_samples(&self, count: usize) -> Vec<LimitTestSample> {
        let mut samples = Vec::new();
        
        for i in 0..count {
            let direction = i % 9;
            let missing_pixels = (i % 3) + 1; // 1-3 missing pixels
            let grid = generate_partial_grid(direction, missing_pixels);
            let input = grid_to_features(&grid);
            
            samples.push(LimitTestSample {
                input,
                target: direction as u32,
                difficulty: 6,
                description: format!("partial_{}_direction_{}", missing_pixels, direction),
            });
        }
        
        samples
    }
    
    fn generate_complex_scene_samples(&self, count: usize) -> Vec<LimitTestSample> {
        let mut samples = Vec::new();
        
        for i in 0..count {
            let scene_type = i % 5;
            let grid = generate_complex_scene(scene_type);
            let input = grid_to_features(&grid);
            
            samples.push(LimitTestSample {
                input,
                target: scene_type as u32,
                difficulty: 8,
                description: format!("complex_scene_{}", scene_type),
            });
        }
        
        samples
    }
}

// Simple model for testing
struct SimpleModel {
    layers: Vec<Linear>,
    device: Device,
}

impl SimpleModel {
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

// Helper functions for grid generation
fn grid_to_features(grid: &[[u8; 3]; 3]) -> Vec<f32> {
    let mut features = Vec::with_capacity(9 * 11);
    
    for row in grid {
        for &pixel in row {
            for i in 0..11 {
                features.push(if pixel == i as u8 { 1.0 } else { 0.0 });
            }
        }
    }
    
    features
}

fn generate_direction_grid(direction: usize) -> [[u8; 3]; 3] {
    let mut grid = [[5u8; 3]; 3];
    
    match direction {
        0 => { grid[1][1] = 10; } // Center
        1 => { grid[0][1] = 10; grid[1][1] = 7; } // Up
        2 => { grid[0][2] = 10; grid[1][1] = 7; } // Up-Right
        3 => { grid[1][2] = 10; grid[1][1] = 7; } // Right
        4 => { grid[2][2] = 10; grid[1][1] = 7; } // Down-Right
        5 => { grid[2][1] = 10; grid[1][1] = 7; } // Down
        6 => { grid[2][0] = 10; grid[1][1] = 7; } // Down-Left
        7 => { grid[1][0] = 10; grid[1][1] = 7; } // Left
        8 => { grid[0][0] = 10; grid[1][1] = 7; } // Up-Left
        _ => {}
    }
    
    grid
}

fn generate_extended_direction_grid(direction: usize, total_directions: usize) -> [[u8; 3]; 3] {
    let mut grid = [[5u8; 3]; 3];
    
    // Map direction to position in 3x3 grid
    let angle = (direction as f32 / total_directions as f32) * 2.0 * std::f32::consts::PI;
    let x = (angle.cos() * 1.5 + 1.5) as usize;
    let y = (angle.sin() * 1.5 + 1.5) as usize;
    
    if x < 3 && y < 3 {
        grid[y][x] = 10;
        grid[1][1] = 7;
    }
    
    grid
}

fn generate_multiple_object_grid(object_count: usize) -> [[u8; 3]; 3] {
    let mut grid = [[5u8; 3]; 3];
    let positions = [(0, 0), (0, 2), (2, 2), (2, 0), (1, 1)];
    
    for i in 0..object_count.min(positions.len()) {
        let (x, y) = positions[i];
        grid[y][x] = 10;
    }
    
    grid
}

fn generate_intensity_grid(direction: usize, intensity: u8) -> [[u8; 3]; 3] {
    let mut grid = generate_direction_grid(direction);
    
    // Adjust intensity
    for row in &mut grid {
        for pixel in row {
            if *pixel == 10 {
                *pixel = intensity;
            }
        }
    }
    
    grid
}

fn generate_partial_grid(direction: usize, missing_pixels: usize) -> [[u8; 3]; 3] {
    let mut grid = generate_direction_grid(direction);
    
    // Remove some pixels
    let mut removed = 0;
    for row in &mut grid {
        for pixel in row {
            if *pixel > 5 && removed < missing_pixels {
                *pixel = 5;
                removed += 1;
            }
        }
    }
    
    grid
}

fn generate_complex_scene(scene_type: usize) -> [[u8; 3]; 3] {
    let mut grid = [[5u8; 3]; 3];
    
    match scene_type {
        0 => { // Corner pattern
            grid[0][0] = 10; grid[0][2] = 10; grid[2][0] = 10; grid[2][2] = 10;
        }
        1 => { // Cross pattern
            grid[0][1] = 10; grid[1][0] = 10; grid[1][2] = 10; grid[2][1] = 10;
        }
        2 => { // Diamond pattern
            grid[1][0] = 8; grid[0][1] = 9; grid[1][2] = 8; grid[2][1] = 9;
        }
        3 => { // Gradient pattern
            grid[0][0] = 6; grid[0][1] = 7; grid[0][2] = 8;
            grid[1][0] = 7; grid[1][1] = 9; grid[1][2] = 7;
            grid[2][0] = 8; grid[2][1] = 7; grid[2][2] = 6;
        }
        4 => { // Random pattern
            use rand::Rng;
            let mut rng = rand::thread_rng();
            for row in &mut grid {
                for pixel in row {
                    *pixel = rng.gen_range(0..11);
                }
            }
        }
        _ => {}
    }
    
    grid
}

fn main() -> Result<()> {
    println!("=== Testing Limits of Existing Systems ===");
    
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Device: {:?}", device);
    
    let tester = LimitTester::new(device);
    
    // Test 1x1 limits
    let results_1x1 = tester.test_1x1_limits()?;
    
    // Test 3x3 limits
    let results_3x3 = tester.test_3x3_limits()?;
    
    // Analysis
    println!("\n📊 LIMIT ANALYSIS");
    println!("=== 1x1 Neuron Limits ===");
    println!("Best performance: {:.1}%", results_1x1.iter().fold(0.0f32, |a, &b| a.max(b)) * 100.0);
    println!("Average performance: {:.1}%", results_1x1.iter().sum::<f32>() / results_1x1.len() as f32 * 100.0);
    
    println!("\n=== 3x3 Direction Limits ===");
    println!("Best performance: {:.1}%", results_3x3.iter().fold(0.0f32, |a, &b| a.max(b)) * 100.0);
    println!("Average performance: {:.1}%", results_3x3.iter().sum::<f32>() / results_3x3.len() as f32 * 100.0);
    
    // Find breaking points
    println!("\n🔍 BREAKING POINTS");
    
    let breaking_1x1 = results_1x1.iter().enumerate()
        .find(|(_, &acc)| acc < 0.8)
        .map(|(i, _)| i);
    
    let breaking_3x3 = results_3x3.iter().enumerate()
        .find(|(_, &acc)| acc < 0.8)
        .map(|(i, _)| i);
    
    if let Some(test) = breaking_1x1 {
        println!("1x1 breaks at test {}", test + 1);
    } else {
        println!("1x1 handles all tests well");
    }
    
    if let Some(test) = breaking_3x3 {
        println!("3x3 breaks at test {}", test + 1);
    } else {
        println!("3x3 handles all tests well");
    }
    
    Ok(())
}
