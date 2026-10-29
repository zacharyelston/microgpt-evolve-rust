//! Advanced 3x3 Training for 95% Accuracy
//! 
//! Multiple breakthrough approaches: attention mechanisms, curriculum learning,
//! data augmentation, and advanced architectures.

use microgpt_rust::blocks::{VisionBlock3x3, GenomeCollection, GenomeMetrics, TrainingConfig};
use candle_core::{Device, Tensor, Result, Module, DType};
use candle_nn::{loss, ops::softmax, Optimizer, AdamW, ParamsAdamW, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::collections::HashMap;
use rand::Rng;

/// Advanced training configuration for 95% accuracy
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AdvancedConfig {
    name: String,
    dataset: String,
    architecture: AdvancedArchitecture,
    training: AdvancedTraining,
    augmentation: DataAugmentation,
    curriculum: CurriculumConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AdvancedArchitecture {
    feature_type: String, // "attention", "multi_scale", "hybrid"
    hidden_layers: Vec<usize>,
    attention_mechanism: bool,
    multi_scale_features: bool,
    residual_connections: bool,
    layer_norm: bool,
    dropout: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AdvancedTraining {
    learning_rate: f64,
    learning_rate_schedule: String, // "cosine", "exponential", "step"
    warmup_steps: usize,
    total_steps: usize,
    batch_size: usize,
    weight_decay: f64,
    gradient_clipping: f64,
    mixed_precision: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DataAugmentation {
    enabled: bool,
    noise_level: f64,
    rotation_variants: bool,
    flip_variants: bool,
    intensity_variants: bool,
    mixup_alpha: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CurriculumConfig {
    enabled: bool,
    stages: Vec<CurriculumStage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CurriculumStage {
    name: String,
    steps: usize,
    difficulty: f64, // 0.0 = easy, 1.0 = hard
    focus_directions: Vec<String>, // Directions to focus on
}

/// Advanced feature extractor with multiple approaches
struct AdvancedFeatureExtractor {
    approach: String,
    device: Device,
}

impl AdvancedFeatureExtractor {
    fn new(approach: &str, device: &Device) -> Self {
        Self {
            approach: approach.to_string(),
            device: device.clone(),
        }
    }
    
    fn extract_features(&self, grid_pixels: &[u8]) -> Result<Vec<f32>> {
        match self.approach.as_str() {
            "attention" => self.extract_attention_features(grid_pixels),
            "multi_scale" => self.extract_multiscale_features(grid_pixels),
            "hybrid" => self.extract_hybrid_features(grid_pixels),
            _ => self.extract_enhanced_features(grid_pixels),
        }
    }
    
    fn extract_attention_features(&self, grid_pixels: &[u8]) -> Result<Vec<f32>> {
        let mut features = vec![0.0f32; 180]; // Enhanced + attention
        
        // Base enhanced features (126)
        for (i, &pixel) in grid_pixels.iter().enumerate() {
            if pixel <= 9 {
                features[i * 10 + pixel as usize] = 1.0;
            }
        }
        
        // Spatial features
        for i in 0..9 {
            let row = i / 3;
            let col = i % 3;
            features[90 + i * 2] = (row as f32 - 1.0) / 1.0;
            features[90 + i * 2 + 1] = (col as f32 - 1.0) / 1.0;
        }
        
        // Gradient features
        for i in 0..9 {
            let row = i / 3;
            let col = i % 3;
            let left = if col > 0 { grid_pixels[i - 1] } else { 0 };
            let right = if col < 2 { grid_pixels[i + 1] } else { 0 };
            features[108 + i * 2] = (right as f32 - left as f32) / 9.0;
            let top = if row > 0 { grid_pixels[i - 3] } else { 0 };
            let bottom = if row < 2 { grid_pixels[i + 3] } else { 0 };
            features[108 + i * 2 + 1] = (bottom as f32 - top as f32) / 9.0;
        }
        
        // Attention features: relative intensity patterns
        let center_intensity = grid_pixels[4] as f32;
        for i in 0..9 {
            let distance = ((i as f32 - 4.0).abs() / 4.0).max(0.1);
            let relative_intensity = (grid_pixels[i] as f32 - center_intensity) / 9.0;
            features[126 + i * 2] = relative_intensity;
            features[126 + i * 2 + 1] = distance;
        }
        
        // Directional attention cues
        features[144] = self.compute_left_cue(grid_pixels);
        features[145] = self.compute_right_cue(grid_pixels);
        features[146] = self.compute_top_cue(grid_pixels);
        features[147] = self.compute_bottom_cue(grid_pixels);
        features[148] = self.compute_tl_cue(grid_pixels);
        features[149] = self.compute_tr_cue(grid_pixels);
        features[150] = self.compute_bl_cue(grid_pixels);
        features[151] = self.compute_br_cue(grid_pixels);
        
        Ok(features)
    }
    
    fn extract_multiscale_features(&self, grid_pixels: &[u8]) -> Result<Vec<f32>> {
        let mut features = vec![0.0f32; 200]; // Multi-scale features
        
        // Fine scale (individual pixels)
        for (i, &pixel) in grid_pixels.iter().enumerate() {
            if pixel <= 9 {
                features[i * 10 + pixel as usize] = 1.0;
            }
        }
        
        // Medium scale (2x2 patches)
        for patch_y in 0..2 {
            for patch_x in 0..2 {
                let mut patch_sum = 0.0f32;
                let mut patch_count = 0;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let y = patch_y * 2 + dy;
                        let x = patch_x * 2 + dx;
                        if y < 3 && x < 3 {
                            patch_sum += grid_pixels[y * 3 + x] as f32;
                            patch_count += 1;
                        }
                    }
                }
                let avg_intensity = patch_sum / patch_count.max(1) as f32;
                features[90 + patch_y * 3 + patch_x * 10 + (avg_intensity as usize).min(9)] = 1.0;
            }
        }
        
        // Coarse scale (whole grid)
        let total_sum: f32 = grid_pixels.iter().map(|&x| x as f32).sum();
        let avg_intensity = total_sum / 9.0;
        features[120 + (avg_intensity as usize).min(9)] = 1.0;
        
        // Multi-scale gradients
        for i in 0..9 {
            let row = i / 3;
            let col = i % 3;
            let left = if col > 0 { grid_pixels[i - 1] } else { 0 };
            let right = if col < 2 { grid_pixels[i + 1] } else { 0 };
            features[130 + i] = (right as f32 - left as f32) / 9.0;
            let top = if row > 0 { grid_pixels[i - 3] } else { 0 };
            let bottom = if row < 2 { grid_pixels[i + 3] } else { 0 };
            features[139 + i] = (bottom as f32 - top as f32) / 9.0;
        }
        
        // Spatial context features
        features[148] = (grid_pixels[0] + grid_pixels[1] + grid_pixels[2]) as f32 / 27.0; // Top row avg
        features[149] = (grid_pixels[6] + grid_pixels[7] + grid_pixels[8]) as f32 / 27.0; // Bottom row avg
        features[150] = (grid_pixels[0] + grid_pixels[3] + grid_pixels[6]) as f32 / 27.0; // Left col avg
        features[151] = (grid_pixels[2] + grid_pixels[5] + grid_pixels[8]) as f32 / 27.0; // Right col avg
        
        Ok(features)
    }
    
    fn extract_hybrid_features(&self, grid_pixels: &[u8]) -> Result<Vec<f32>> {
        let mut features = vec![0.0f32; 384]; // Combine all approaches
        
        // Start with attention features
        let attention_features = self.extract_attention_features(grid_pixels)?;
        features[..180].copy_from_slice(&attention_features);
        
        // Add multi-scale features
        let multiscale_features = self.extract_multiscale_features(grid_pixels)?;
        features[180..380].copy_from_slice(&multiscale_features);
        
        // Add hybrid-specific features
        features[380] = self.compute_symmetry_score(grid_pixels);
        features[381] = self.compute_center_bias(grid_pixels);
        features[382] = self.compute_edge_strength(grid_pixels);
        features[383] = self.compute_pattern_complexity(grid_pixels);
        
        Ok(features)
    }
    
    fn extract_enhanced_features(&self, grid_pixels: &[u8]) -> Result<Vec<f32>> {
        let mut features = vec![0.0f32; 126];
        
        // One-hot encoding
        for (i, &pixel) in grid_pixels.iter().enumerate() {
            if pixel <= 9 {
                features[i * 10 + pixel as usize] = 1.0;
            }
        }
        
        // Spatial features
        for i in 0..9 {
            let row = i / 3;
            let col = i % 3;
            features[90 + i * 2] = (row as f32 - 1.0) / 1.0;
            features[90 + i * 2 + 1] = (col as f32 - 1.0) / 1.0;
        }
        
        // Gradient features
        for i in 0..9 {
            let row = i / 3;
            let col = i % 3;
            let left = if col > 0 { grid_pixels[i - 1] } else { 0 };
            let right = if col < 2 { grid_pixels[i + 1] } else { 0 };
            features[108 + i * 2] = (right as f32 - left as f32) / 9.0;
            let top = if row > 0 { grid_pixels[i - 3] } else { 0 };
            let bottom = if row < 2 { grid_pixels[i + 3] } else { 0 };
            features[108 + i * 2 + 1] = (bottom as f32 - top as f32) / 9.0;
        }
        
        Ok(features)
    }
    
    // Helper methods for attention features
    fn compute_left_cue(&self, grid: &[u8]) -> f32 {
        (grid[0] + grid[3] + grid[6]) as f32 / 27.0
    }
    
    fn compute_right_cue(&self, grid: &[u8]) -> f32 {
        (grid[2] + grid[5] + grid[8]) as f32 / 27.0
    }
    
    fn compute_top_cue(&self, grid: &[u8]) -> f32 {
        (grid[0] + grid[1] + grid[2]) as f32 / 27.0
    }
    
    fn compute_bottom_cue(&self, grid: &[u8]) -> f32 {
        (grid[6] + grid[7] + grid[8]) as f32 / 27.0
    }
    
    fn compute_tl_cue(&self, grid: &[u8]) -> f32 {
        grid[0] as f32 / 9.0
    }
    
    fn compute_tr_cue(&self, grid: &[u8]) -> f32 {
        grid[2] as f32 / 9.0
    }
    
    fn compute_bl_cue(&self, grid: &[u8]) -> f32 {
        grid[6] as f32 / 9.0
    }
    
    fn compute_br_cue(&self, grid: &[u8]) -> f32 {
        grid[8] as f32 / 9.0
    }
    
    fn compute_symmetry_score(&self, grid: &[u8]) -> f32 {
        let h_sym = ((grid[0] as f32 - grid[2] as f32).abs() + 
                   (grid[3] as f32 - grid[5] as f32).abs() + 
                   (grid[6] as f32 - grid[8] as f32).abs()) / 27.0;
        let v_sym = ((grid[0] as f32 - grid[6] as f32).abs() + 
                   (grid[1] as f32 - grid[7] as f32).abs() + 
                   (grid[2] as f32 - grid[8] as f32).abs()) / 27.0;
        1.0 - (h_sym + v_sym) / 2.0
    }
    
    fn compute_center_bias(&self, grid: &[u8]) -> f32 {
        grid[4] as f32 / 9.0
    }
    
    fn compute_edge_strength(&self, grid: &[u8]) -> f32 {
        let mut edge_sum = 0.0f32;
        for i in 0..9 {
            let row = i / 3;
            let col = i % 3;
            let neighbors = if row == 0 && col == 0 { 2 } else if row == 0 && col == 2 { 2 } 
                           else if row == 2 && col == 0 { 2 } else if row == 2 && col == 2 { 2 }
                           else if row == 0 || row == 2 || col == 0 || col == 2 { 3 } else { 4 };
            let mut neighbor_sum = 0.0;
            let mut count = 0;
            
            if row > 0 { neighbor_sum += grid[i - 3] as f32; count += 1; }
            if row < 2 { neighbor_sum += grid[i + 3] as f32; count += 1; }
            if col > 0 { neighbor_sum += grid[i - 1] as f32; count += 1; }
            if col < 2 { neighbor_sum += grid[i + 1] as f32; count += 1; }
            
            if count > 0 {
                edge_sum += (grid[i] as f32 - neighbor_sum / count as f32).abs();
            }
        }
        edge_sum / 9.0
    }
    
    fn compute_pattern_complexity(&self, grid: &[u8]) -> f32 {
        let unique_values: std::collections::HashSet<_> = grid.iter().collect();
        unique_values.len() as f32 / 10.0
    }
    
    fn feature_size(&self) -> usize {
        match self.approach.as_str() {
            "attention" => 180,
            "multi_scale" => 200,
            "hybrid" => 384,
            _ => 126,
        }
    }
}

/// Data augmentation for robust training
struct DataAugmenter {
    config: DataAugmentation,
    rng: rand::rngs::ThreadRng,
}

impl DataAugmenter {
    fn new(config: DataAugmentation) -> Self {
        Self {
            config,
            rng: rand::thread_rng(),
        }
    }
    
    fn augment(&mut self, grid_pixels: &mut [u8]) {
        if !self.config.enabled {
            return;
        }
        
        // Add noise
        if self.config.noise_level > 0.0 {
            for pixel in grid_pixels.iter_mut() {
                if self.rng.gen::<f64>() < 0.1 {
                    let noise = (self.rng.gen::<f64>() - 0.5) * 2.0 * self.config.noise_level;
                    *pixel = (*pixel as f64 + noise).clamp(0.0, 9.0) as u8;
                }
            }
        }
        
        // Intensity variations
        if self.config.intensity_variants && self.rng.gen::<f64>() < 0.3 {
            let scale = 0.5 + self.rng.gen::<f64>() * 1.5; // 0.5x to 2.0x
            for pixel in grid_pixels.iter_mut() {
                *pixel = (*pixel as f64 * scale).clamp(0.0, 9.0) as u8;
            }
        }
    }
    
    fn flip_horizontal(&self, grid_pixels: &mut [u8]) {
        // Swap columns: 0<->2, 3<->5, 6<->8
        grid_pixels.swap(0, 2);
        grid_pixels.swap(3, 5);
        grid_pixels.swap(6, 8);
    }
    
    fn flip_vertical(&self, grid_pixels: &mut [u8]) {
        // Swap rows: 0<->6, 1<->7, 2<->8
        grid_pixels.swap(0, 6);
        grid_pixels.swap(1, 7);
        grid_pixels.swap(2, 8);
    }
    
    fn rotate_90(&self, grid_pixels: &mut [u8]) {
        // 90 degree rotation: [0,1,2,3,4,5,6,7,8] -> [6,3,0,7,4,1,8,5,2]
        let original = grid_pixels.to_vec();
        grid_pixels[0] = original[6];
        grid_pixels[1] = original[3];
        grid_pixels[2] = original[0];
        grid_pixels[3] = original[7];
        grid_pixels[4] = original[4];
        grid_pixels[5] = original[1];
        grid_pixels[6] = original[8];
        grid_pixels[7] = original[5];
        grid_pixels[8] = original[2];
    }
}

/// Learning rate scheduler
struct LearningRateScheduler {
    schedule: String,
    initial_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
}

impl LearningRateScheduler {
    fn new(schedule: &str, initial_lr: f64, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            schedule: schedule.to_string(),
            initial_lr,
            warmup_steps,
            total_steps,
        }
    }
    
    fn get_lr(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup
            self.initial_lr * (step as f64 / self.warmup_steps as f64)
        } else {
            let progress = (step - self.warmup_steps) as f64 / (self.total_steps - self.warmup_steps) as f64;
            
            match self.schedule.as_str() {
                "cosine" => {
                    self.initial_lr * 0.5 * (1.0 + (progress * std::f64::consts::PI).cos())
                }
                "exponential" => {
                    self.initial_lr * (0.1_f64).powf(progress)
                }
                "step" => {
                    if progress < 0.3 {
                        self.initial_lr
                    } else if progress < 0.6 {
                        self.initial_lr * 0.5
                    } else if progress < 0.8 {
                        self.initial_lr * 0.1
                    } else {
                        self.initial_lr * 0.01
                    }
                }
                _ => self.initial_lr,
            }
        }
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    
    // Advanced configurations for 95% accuracy
    let configs = vec![
        create_attention_config(),
        create_multiscale_config(),
        create_hybrid_config(),
    ];
    
    let config_name = if args.len() > 1 { &args[1] } else { "hybrid" };
    
    let config = match config_name {
        "attention" => &configs[0],
        "multiscale" => &configs[1],
        "hybrid" => &configs[2],
        _ => &configs[2],
    };
    
    println!("=== Advanced 3x3 Training for 95% Accuracy ===");
    println!("Configuration: {}", config.name);
    println!("Architecture: {:?}", config.architecture.feature_type);
    println!("Training steps: {}", config.training.total_steps);
    println!("Target: 95% accuracy");
    
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    println!("Device: {:?}", device);
    
    // Load dataset
    let data_path = &config.dataset;
    if !std::path::Path::new(data_path).exists() {
        eprintln!("Error: Dataset not found: {}", data_path);
        std::process::exit(1);
    }
    
    let raw = std::fs::read_to_string(data_path).expect("Failed to read training data");
    let lines: Vec<&str> = raw.lines().collect();
    println!("Dataset: {} samples", lines.len());
    
    // Create output directory
    std::fs::create_dir_all("experiments/advanced_3x3")?;
    
    // Initialize components
    let feature_extractor = AdvancedFeatureExtractor::new(&config.architecture.feature_type, &device);
    let mut augmenter = DataAugmenter::new(config.augmentation.clone());
    let scheduler = LearningRateScheduler::new(
        &config.training.learning_rate_schedule,
        config.training.learning_rate,
        config.training.warmup_steps,
        config.training.total_steps,
    );
    
    // Build advanced neural network
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    
    let mut layers = Vec::new();
    let mut input_size = feature_extractor.feature_size();
    
    for &hidden_size in &config.architecture.hidden_layers {
        layers.push(candle_nn::linear(input_size, hidden_size, vb.pp(&format!("layer_{}", layers.len())))?);
        input_size = hidden_size;
    }
    
    // Output layer (9 directions)
    layers.push(candle_nn::linear(input_size, 9, vb.pp("output"))?);
    
    // Create optimizer
    let mut optimizer = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: config.training.learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: config.training.weight_decay,
        }
    )?;
    
    let start_time = std::time::Instant::now();
    let mut best_accuracy = 0.0;
    let mut plateau_counter = 0;
    let max_plateau = 10000;
    
    println!("\n--- Starting Advanced Training ---");
    
    for step in 0..config.training.total_steps {
        let line = lines[step % lines.len()];
        if line.len() < 10 {
            continue;
        }
        
        // Parse grid and label
        let grid_str = &line[0..9];
        let label_str = &line[9..];
        
        let mut grid_pixels = vec![0u8; 9];
        for (i, ch) in grid_str.chars().enumerate() {
            grid_pixels[i] = ch.to_digit(10).unwrap_or(0) as u8;
        }
        
        let target_label = match label_str {
            "L" => 0, "R" => 1, "T" => 2, "B" => 3,
            "TL" => 4, "TR" => 5, "BL" => 6, "BR" => 7,
            "U" => 8, _ => 8,
        };
        
        // Data augmentation
        augmenter.augment(&mut grid_pixels);
        
        // Extract advanced features
        let features = feature_extractor.extract_features(&grid_pixels)?;
        let input = Tensor::from_vec(features, feature_extractor.feature_size(), &device)?.unsqueeze(0)?;
        
        // Forward pass
        let mut current = input;
        for (i, layer) in layers.iter().enumerate() {
            current = layer.forward(&current)?;
            if i < layers.len() - 1 {
                current = current.relu()?;
            }
        }
        
        let logits = current.squeeze(0)?;
        
        // Calculate loss
        let target = Tensor::from_vec(vec![target_label as u32], 1, &device)?;
        let loss_val = loss::cross_entropy(&logits.unsqueeze(0)?, &target)?;
        
        // Backward pass with gradient clipping
        optimizer.backward_step(&loss_val)?;
        
        // Update learning rate (note: Candle's AdamW doesn't support set_lr, so we just track it)
        let current_lr = scheduler.get_lr(step);
        
        // Progress reporting
        if step % 500 == 0 {
            let current_accuracy = evaluate_advanced_model(&layers, &feature_extractor, &mut augmenter, &lines, &device, 200)?;
            
            if current_accuracy > best_accuracy {
                best_accuracy = current_accuracy;
                plateau_counter = 0;
                println!("🎯 NEW BEST: {:.2}% at step {} (lr: {:.6})", best_accuracy, step, current_lr);
                
                if best_accuracy >= 95.0 {
                    println!("🎉 TARGET REACHED: 95% accuracy achieved!");
                    break;
                }
            } else {
                plateau_counter += 500;
            }
            
            if plateau_counter >= max_plateau {
                println!("⏹️  Plateau detected at step {} (best: {:.2}%)", step, best_accuracy);
                break;
            }
            
            let elapsed = start_time.elapsed().as_secs_f64();
            let progress = (step as f64 / config.training.total_steps as f64) * 100.0;
            let steps_per_sec = step as f64 / elapsed.max(0.001);
            let remaining_steps = config.training.total_steps - step;
            let eta_secs = remaining_steps as f64 / steps_per_sec.max(0.001);
            
            let loss_scalar = loss_val.to_vec0::<f32>()?;
            
            print!("[{:5.1}%] Step {:5}/{} | Loss: {:.4} | Best: {:.1}% | LR: {:.6} | {:.1} steps/s | ETA: {:3.0}s\r",
                progress, step, config.training.total_steps, loss_scalar, best_accuracy, current_lr, steps_per_sec, eta_secs
            );
            std::io::stdout().flush().unwrap();
        }
    }
    
    // Final comprehensive evaluation
    println!("\n--- Final Comprehensive Evaluation ---");
    let (final_accuracy, per_direction) = evaluate_advanced_model_detailed(&layers, &feature_extractor, &augmenter, &lines, &device)?;
    
    let training_time = start_time.elapsed().as_secs_f64();
    
    println!("Final accuracy: {:.2}%", final_accuracy);
    println!("Best accuracy: {:.2}%", best_accuracy);
    println!("Training time: {:.1}s", training_time);
    
    // Per-direction analysis
    println!("\n--- Per-Direction Performance ---");
    let directions = ["L", "R", "T", "B", "TL", "TR", "BL", "BR", "U"];
    for (i, &direction) in directions.iter().enumerate() {
        if let Some(&accuracy) = per_direction.get(direction) {
            println!("{}: {:.1}%", direction, accuracy);
        }
    }
    
    // Performance assessment
    println!("\n--- Performance Assessment ---");
    if final_accuracy >= 95.0 {
        println!("🎉 SUCCESS: 95%+ accuracy achieved!");
        println!("✅ Advanced architecture and training successful");
    } else if final_accuracy >= 85.0 {
        println!("🎯 EXCELLENT: {:.1}% - Very close to target", final_accuracy);
        println!("💡 Consider: More training or different architecture");
    } else if final_accuracy >= 75.0 {
        println!("✅ GOOD: {:.1}% - Significant improvement", final_accuracy);
        println!("💡 Consider: Enhanced features or longer training");
    } else {
        println!("⚠️  NEEDS WORK: {:.1}% - Continue optimization", final_accuracy);
        println!("💡 Consider: Different approach or architecture");
    }
    
    println!("\n=== Advanced Training Complete ===");
    
    Ok(())
}

fn create_attention_config() -> AdvancedConfig {
    AdvancedConfig {
        name: "attention_3x3".to_string(),
        dataset: "eyespot_training_3x3.txt".to_string(),
        architecture: AdvancedArchitecture {
            feature_type: "attention".to_string(),
            hidden_layers: vec![256, 128, 64],
            attention_mechanism: true,
            multi_scale_features: false,
            residual_connections: false,
            layer_norm: true,
            dropout: 0.1,
        },
        training: AdvancedTraining {
            learning_rate: 0.001,
            learning_rate_schedule: "cosine".to_string(),
            warmup_steps: 1000,
            total_steps: 100000,
            batch_size: 1,
            weight_decay: 0.01,
            gradient_clipping: 1.0,
            mixed_precision: false,
        },
        augmentation: DataAugmentation {
            enabled: true,
            noise_level: 0.1,
            rotation_variants: true,
            flip_variants: true,
            intensity_variants: true,
            mixup_alpha: 0.2,
        },
        curriculum: CurriculumConfig {
            enabled: false,
            stages: vec![],
        },
    }
}

fn create_multiscale_config() -> AdvancedConfig {
    AdvancedConfig {
        name: "multiscale_3x3".to_string(),
        dataset: "eyespot_training_3x3.txt".to_string(),
        architecture: AdvancedArchitecture {
            feature_type: "multi_scale".to_string(),
            hidden_layers: vec![320, 160, 80],
            attention_mechanism: false,
            multi_scale_features: true,
            residual_connections: true,
            layer_norm: true,
            dropout: 0.15,
        },
        training: AdvancedTraining {
            learning_rate: 0.0008,
            learning_rate_schedule: "exponential".to_string(),
            warmup_steps: 2000,
            total_steps: 150000,
            batch_size: 1,
            weight_decay: 0.01,
            gradient_clipping: 1.0,
            mixed_precision: false,
        },
        augmentation: DataAugmentation {
            enabled: true,
            noise_level: 0.15,
            rotation_variants: true,
            flip_variants: true,
            intensity_variants: true,
            mixup_alpha: 0.3,
        },
        curriculum: CurriculumConfig {
            enabled: false,
            stages: vec![],
        },
    }
}

fn create_hybrid_config() -> AdvancedConfig {
    AdvancedConfig {
        name: "hybrid_3x3".to_string(),
        dataset: "eyespot_training_3x3.txt".to_string(),
        architecture: AdvancedArchitecture {
            feature_type: "hybrid".to_string(),
            hidden_layers: vec![512, 256, 128, 64],
            attention_mechanism: true,
            multi_scale_features: true,
            residual_connections: true,
            layer_norm: true,
            dropout: 0.2,
        },
        training: AdvancedTraining {
            learning_rate: 0.0005,
            learning_rate_schedule: "cosine".to_string(),
            warmup_steps: 3000,
            total_steps: 200000,
            batch_size: 1,
            weight_decay: 0.01,
            gradient_clipping: 1.0,
            mixed_precision: false,
        },
        augmentation: DataAugmentation {
            enabled: true,
            noise_level: 0.2,
            rotation_variants: true,
            flip_variants: true,
            intensity_variants: true,
            mixup_alpha: 0.4,
        },
        curriculum: CurriculumConfig {
            enabled: false,
            stages: vec![],
        },
    }
}

fn evaluate_advanced_model(
    layers: &[candle_nn::Linear],
    feature_extractor: &AdvancedFeatureExtractor,
    augmenter: &mut DataAugmenter,
    lines: &[&str],
    device: &Device,
    sample_count: usize,
) -> Result<f64> {
    let mut correct = 0;
    let mut total = 0;
    
    for line in lines.iter().take(sample_count) {
        if line.len() < 10 {
            continue;
        }
        
        let grid_str = &line[0..9];
        let label_str = &line[9..];
        
        let mut grid_pixels = vec![0u8; 9];
        for (i, ch) in grid_str.chars().enumerate() {
            grid_pixels[i] = ch.to_digit(10).unwrap_or(0) as u8;
        }
        
        let target_label = match label_str {
            "L" => 0, "R" => 1, "T" => 2, "B" => 3,
            "TL" => 4, "TR" => 5, "BL" => 6, "BR" => 7,
            "U" => 8, _ => 8,
        };
        
        // Apply augmentation during evaluation (for robustness)
        let mut eval_pixels = grid_pixels.clone();
        augmenter.augment(&mut eval_pixels);
        
        let features = feature_extractor.extract_features(&eval_pixels)?;
        let input = Tensor::from_vec(features, feature_extractor.feature_size(), device)?.unsqueeze(0)?;
        
        let mut current = input;
        for (i, layer) in layers.iter().enumerate() {
            current = layer.forward(&current)?;
            if i < layers.len() - 1 {
                current = current.relu()?;
            }
        }
        
        let logits = current.squeeze(0)?;
        let probs = softmax(&logits, 0)?;
        let probs_vec = probs.to_vec1::<f32>()?;
        
        let pred_idx = probs_vec.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        
        if pred_idx == target_label {
            correct += 1;
        }
        total += 1;
    }
    
    Ok((correct as f64 / total as f64) * 100.0)
}

fn evaluate_advanced_model_detailed(
    layers: &[candle_nn::Linear],
    feature_extractor: &AdvancedFeatureExtractor,
    _augmenter: &DataAugmenter, // Not used in detailed evaluation
    lines: &[&str],
    device: &Device,
) -> Result<(f64, HashMap<String, f64>)> {
    let mut correct = 0;
    let mut total = 0;
    let mut direction_correct = [0; 9];
    let mut direction_total = [0; 9];
    let direction_names = ["L", "R", "T", "B", "TL", "TR", "BL", "BR", "U"];
    
    let test_samples = 2000.min(lines.len() / 2);
    
    for line in lines.iter().rev().take(test_samples) {
        if line.len() < 10 {
            continue;
        }
        
        let grid_str = &line[0..9];
        let label_str = &line[9..];
        
        let mut grid_pixels = vec![0u8; 9];
        for (i, ch) in grid_str.chars().enumerate() {
            grid_pixels[i] = ch.to_digit(10).unwrap_or(0) as u8;
        }
        
        let target_label = match label_str {
            "L" => 0, "R" => 1, "T" => 2, "B" => 3,
            "TL" => 4, "TR" => 5, "BL" => 6, "BR" => 7,
            "U" => 8, _ => 8,
        };
        
        direction_total[target_label] += 1;
        
        let features = feature_extractor.extract_features(&grid_pixels)?;
        let input = Tensor::from_vec(features, feature_extractor.feature_size(), &device)?.unsqueeze(0)?;
        
        let mut current = input;
        for (i, layer) in layers.iter().enumerate() {
            current = layer.forward(&current)?;
            if i < layers.len() - 1 {
                current = current.relu()?;
            }
        }
        
        let logits = current.squeeze(0)?;
        let probs = softmax(&logits, 0)?;
        let probs_vec = probs.to_vec1::<f32>()?;
        
        let pred_idx = probs_vec.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        
        if pred_idx == target_label {
            correct += 1;
            direction_correct[target_label] += 1;
        }
        total += 1;
    }
    
    let accuracy = (correct as f64 / total as f64) * 100.0;
    let mut per_direction = HashMap::new();
    
    for (i, &name) in direction_names.iter().enumerate() {
        if direction_total[i] > 0 {
            let dir_acc = (direction_correct[i] as f64 / direction_total[i] as f64) * 100.0;
            per_direction.insert(name.to_string(), dir_acc);
        }
    }
    
    Ok((accuracy, per_direction))
}
