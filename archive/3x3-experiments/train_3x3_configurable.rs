//! Configurable 3x3 Block Training Tool
//! 
//! Flexible training system for testing different datasets, architectures, and configurations.

use microgpt_rust::blocks::{VisionBlock3x3, GenomeCollection, GenomeMetrics, TrainingConfig};
use candle_core::{Device, Tensor, Result, Module, DType};
use candle_nn::{loss, ops::softmax, Optimizer, AdamW, ParamsAdamW, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::collections::HashMap;

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExperimentConfig {
    /// Experiment name
    name: String,
    /// Dataset file path
    dataset: String,
    /// Feature extraction type
    feature_type: FeatureType,
    /// Architecture configuration
    architecture: ArchitectureConfig,
    /// Training parameters
    training: TrainingParams,
    /// Output settings
    output: OutputConfig,
}

/// Feature extraction types
#[derive(Debug, Clone, Serialize, Deserialize)]
enum FeatureType {
    /// Basic one-hot encoding (90 features)
    Basic,
    /// One-hot + spatial coordinates (108 features)
    Spatial,
    /// One-hot + spatial + intensity gradients (126 features)
    Enhanced,
    /// Custom feature configuration
    Custom { size: usize, description: String },
}

/// Architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ArchitectureConfig {
    /// Hidden layer sizes
    hidden_layers: Vec<usize>,
    /// Activation function
    activation: String,
    /// Use dropout
    dropout: Option<f32>,
    /// Use layer normalization
    layer_norm: bool,
}

/// Training parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrainingParams {
    /// Learning rate
    learning_rate: f64,
    /// Number of training steps
    steps: usize,
    /// Batch size
    batch_size: usize,
    /// Weight decay
    weight_decay: f64,
    /// Early stopping patience
    patience: usize,
    /// Evaluation frequency
    eval_frequency: usize,
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OutputConfig {
    /// Save directory
    save_dir: String,
    /// Save frequency
    save_frequency: usize,
    /// Enable detailed logging
    verbose: bool,
    /// Generate plots
    plots: bool,
}

/// Default configurations for common experiments
impl ExperimentConfig {
    fn basic_3x3() -> Self {
        Self {
            name: "basic_3x3".to_string(),
            dataset: "eyespot_training_3x3.txt".to_string(),
            feature_type: FeatureType::Basic,
            architecture: ArchitectureConfig {
                hidden_layers: vec![64],
                activation: "relu".to_string(),
                dropout: None,
                layer_norm: false,
            },
            training: TrainingParams {
                learning_rate: 0.01,
                steps: 5000,
                batch_size: 1,
                weight_decay: 0.01,
                patience: 1000,
                eval_frequency: 100,
            },
            output: OutputConfig {
                save_dir: "experiments/basic_3x3".to_string(),
                save_frequency: 1000,
                verbose: true,
                plots: false,
            },
        }
    }
    
    fn spatial_3x3() -> Self {
        Self {
            name: "spatial_3x3".to_string(),
            dataset: "eyespot_training_3x3.txt".to_string(),
            feature_type: FeatureType::Spatial,
            architecture: ArchitectureConfig {
                hidden_layers: vec![128, 64],
                activation: "relu".to_string(),
                dropout: None,
                layer_norm: false,
            },
            training: TrainingParams {
                learning_rate: 0.005,
                steps: 20000,
                batch_size: 1,
                weight_decay: 0.01,
                patience: 2000,
                eval_frequency: 500,
            },
            output: OutputConfig {
                save_dir: "experiments/spatial_3x3".to_string(),
                save_frequency: 2000,
                verbose: true,
                plots: false,
            },
        }
    }
    
    fn enhanced_3x3() -> Self {
        Self {
            name: "enhanced_3x3".to_string(),
            dataset: "eyespot_training_3x3.txt".to_string(),
            feature_type: FeatureType::Enhanced,
            architecture: ArchitectureConfig {
                hidden_layers: vec![256, 128, 64],
                activation: "relu".to_string(),
                dropout: Some(0.1),
                layer_norm: true,
            },
            training: TrainingParams {
                learning_rate: 0.003,
                steps: 50000,
                batch_size: 1,
                weight_decay: 0.01,
                patience: 5000,
                eval_frequency: 1000,
            },
            output: OutputConfig {
                save_dir: "experiments/enhanced_3x3".to_string(),
                save_frequency: 5000,
                verbose: true,
                plots: true,
            },
        }
    }
}

/// Feature extractor for different feature types
struct FeatureExtractor {
    feature_type: FeatureType,
    device: Device,
}

impl FeatureExtractor {
    fn new(feature_type: FeatureType, device: &Device) -> Self {
        Self {
            feature_type,
            device: device.clone(),
        }
    }
    
    fn extract_features(&self, grid_pixels: &[u8]) -> Result<Vec<f32>> {
        match &self.feature_type {
            FeatureType::Basic => {
                let mut features = vec![0.0f32; 90]; // 9 * 10
                for (i, &pixel) in grid_pixels.iter().enumerate() {
                    if pixel <= 9 {
                        features[i * 10 + pixel as usize] = 1.0;
                    }
                }
                Ok(features)
            }
            FeatureType::Spatial => {
                let mut features = vec![0.0f32; 108]; // 9 * 10 + 9 * 2
                for (i, &pixel) in grid_pixels.iter().enumerate() {
                    if pixel <= 9 {
                        features[i * 10 + pixel as usize] = 1.0;
                    }
                }
                // Add spatial features
                for i in 0..9 {
                    let row = i / 3;
                    let col = i % 3;
                    features[90 + i * 2] = (row as f32 - 1.0) / 1.0; // -1, 0, 1
                    features[90 + i * 2 + 1] = (col as f32 - 1.0) / 1.0; // -1, 0, 1
                }
                Ok(features)
            }
            FeatureType::Enhanced => {
                let mut features = vec![0.0f32; 126]; // 9 * 10 + 9 * 2 + 9 * 2
                for (i, &pixel) in grid_pixels.iter().enumerate() {
                    if pixel <= 9 {
                        features[i * 10 + pixel as usize] = 1.0;
                    }
                }
                // Add spatial features
                for i in 0..9 {
                    let row = i / 3;
                    let col = i % 3;
                    features[90 + i * 2] = (row as f32 - 1.0) / 1.0;
                    features[90 + i * 2 + 1] = (col as f32 - 1.0) / 1.0;
                }
                // Add intensity gradient features
                for i in 0..9 {
                    let row = i / 3;
                    let col = i % 3;
                    // Horizontal gradient
                    let left = if col > 0 { grid_pixels[i - 1] } else { 0 };
                    let right = if col < 2 { grid_pixels[i + 1] } else { 0 };
                    features[108 + i * 2] = (right as f32 - left as f32) / 9.0;
                    // Vertical gradient
                    let top = if row > 0 { grid_pixels[i - 3] } else { 0 };
                    let bottom = if row < 2 { grid_pixels[i + 3] } else { 0 };
                    features[108 + i * 2 + 1] = (bottom as f32 - top as f32) / 9.0;
                }
                Ok(features)
            }
            FeatureType::Custom { size, .. } => {
                Ok(vec![0.0f32; *size])
            }
        }
    }
    
    fn feature_size(&self) -> usize {
        match &self.feature_type {
            FeatureType::Basic => 90,
            FeatureType::Spatial => 108,
            FeatureType::Enhanced => 126,
            FeatureType::Custom { size, .. } => *size,
        }
    }
}

/// Training results
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrainingResults {
    experiment_name: String,
    config: ExperimentConfig,
    final_accuracy: f64,
    best_accuracy: f64,
    training_steps: usize,
    training_time: f64,
    per_direction_accuracy: HashMap<String, f64>,
    plateau_detected: bool,
    convergence_step: Option<usize>,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    
    // Parse command line arguments
    let config = if args.len() > 1 {
        match args[1].as_str() {
            "basic" => ExperimentConfig::basic_3x3(),
            "spatial" => ExperimentConfig::spatial_3x3(),
            "enhanced" => ExperimentConfig::enhanced_3x3(),
            "config" => {
                if args.len() < 3 {
                    eprintln!("Usage: {} config <config_file>", args[0]);
                    std::process::exit(1);
                }
                load_config_from_file(&args[2])?
            }
            _ => {
                eprintln!("Usage: {} [basic|spatial|enhanced|config <file>]", args[0]);
                std::process::exit(1);
            }
        }
    } else {
        ExperimentConfig::spatial_3x3() // Default to spatial
    };
    
    println!("=== Configurable 3x3 Training ===");
    println!("Experiment: {}", config.name);
    println!("Dataset: {}", config.dataset);
    println!("Feature type: {:?}", config.feature_type);
    println!("Architecture: {:?}", config.architecture.hidden_layers);
    println!("Training steps: {}", config.training.steps);
    
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    println!("Device: {:?}", device);
    
    // Load dataset
    if !std::path::Path::new(&config.dataset).exists() {
        eprintln!("Error: Dataset not found: {}", config.dataset);
        std::process::exit(1);
    }
    
    let raw = std::fs::read_to_string(&config.dataset).expect("Failed to read training data");
    let lines: Vec<&str> = raw.lines().collect();
    println!("Dataset: {} samples", lines.len());
    
    // Create output directory
    std::fs::create_dir_all(&config.output.save_dir)?;
    
    // Save configuration
    let config_path = format!("{}/config.json", config.output.save_dir);
    std::fs::write(&config_path, serde_json::to_string_pretty(&config)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?)?;
    println!("Configuration saved to: {}", config_path);
    
    // Run training
    let results = run_training(&config, &lines, &device)?;
    
    // Save results
    let results_path = format!("{}/results.json", config.output.save_dir);
    std::fs::write(&results_path, serde_json::to_string_pretty(&results)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?)?;
    println!("Results saved to: {}", results_path);
    
    // Print summary
    print_experiment_summary(&results);
    
    Ok(())
}

fn load_config_from_file(path: &str) -> Result<ExperimentConfig> {
    let json = std::fs::read_to_string(path)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let config: ExperimentConfig = serde_json::from_str(&json)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    Ok(config)
}

fn run_training(config: &ExperimentConfig, lines: &[&str], device: &Device) -> Result<TrainingResults> {
    let start_time = std::time::Instant::now();
    
    // Create feature extractor
    let feature_extractor = FeatureExtractor::new(config.feature_type.clone(), device);
    let feature_size = feature_extractor.feature_size();
    
    // Build neural network
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    
    let mut layers = Vec::new();
    let mut input_size = feature_size;
    
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
    
    let mut best_accuracy = 0.0;
    let mut plateau_counter = 0;
    let mut convergence_step = None;
    let mut accuracy_history = Vec::new();
    
    println!("\n--- Starting Training ---");
    
    for step in 0..config.training.steps {
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
        
        // Extract features
        let features = feature_extractor.extract_features(&grid_pixels)?;
        let input = Tensor::from_vec(features, feature_size, device)?.unsqueeze(0)?;
        
        // Forward pass
        let mut current = input;
        for (i, layer) in layers.iter().enumerate() {
            current = layer.forward(&current)?;
            if i < layers.len() - 1 {
                // Apply activation to hidden layers only
                current = current.relu()?;
            }
        }
        
        let logits = current.squeeze(0)?;
        
        // Calculate loss
        let target = Tensor::from_vec(vec![target_label as u32], 1, device)?;
        let loss_val = loss::cross_entropy(&logits.unsqueeze(0)?, &target)?;
        
        // Backward pass
        optimizer.backward_step(&loss_val)?;
        
        // Progress reporting
        if step % config.training.eval_frequency == 0 {
            let current_accuracy = evaluate_model(&layers, &feature_extractor, lines, device, 200)?;
            accuracy_history.push((step, current_accuracy));
            
            if current_accuracy > best_accuracy {
                best_accuracy = current_accuracy;
                plateau_counter = 0;
                if convergence_step.is_none() {
                    convergence_step = Some(step);
                }
                if config.output.verbose {
                    println!("🎯 New best: {:.2}% at step {}", best_accuracy, step);
                }
            } else {
                plateau_counter += config.training.eval_frequency;
            }
            
            if plateau_counter >= config.training.patience {
                if config.output.verbose {
                    println!("⏹️  Plateau detected at step {}", step);
                }
                break;
            }
            
            if config.output.verbose {
                let elapsed = start_time.elapsed().as_secs_f64();
                let progress = (step as f64 / config.training.steps as f64) * 100.0;
                let steps_per_sec = step as f64 / elapsed.max(0.001);
                let remaining_steps = config.training.steps - step;
                let eta_secs = remaining_steps as f64 / steps_per_sec.max(0.001);
                
                let loss_scalar = loss_val.to_vec0::<f32>()?;
                
                print!("[{:5.1}%] Step {:5}/{} | Loss: {:.4} | Best: {:.1}% | {:.1} steps/s | ETA: {:3.0}s\r",
                    progress, step, config.training.steps, loss_scalar, best_accuracy, steps_per_sec, eta_secs
                );
                std::io::stdout().flush().unwrap();
            }
        }
    }
    
    // Final evaluation
    println!("\n--- Final Evaluation ---");
    let (final_accuracy, per_direction) = evaluate_model_detailed(&layers, &feature_extractor, lines, device)?;
    
    let training_time = start_time.elapsed().as_secs_f64();
    let plateau_detected = plateau_counter >= config.training.patience;
    
    Ok(TrainingResults {
        experiment_name: config.name.clone(),
        config: config.clone(),
        final_accuracy,
        best_accuracy,
        training_steps: config.training.steps.min(accuracy_history.last().map(|(s, _)| *s).unwrap_or(0)),
        training_time,
        per_direction_accuracy: per_direction,
        plateau_detected,
        convergence_step,
    })
}

fn evaluate_model(
    layers: &[candle_nn::Linear],
    feature_extractor: &FeatureExtractor,
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
        
        let features = feature_extractor.extract_features(&grid_pixels)?;
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

fn evaluate_model_detailed(
    layers: &[candle_nn::Linear],
    feature_extractor: &FeatureExtractor,
    lines: &[&str],
    device: &Device,
) -> Result<(f64, HashMap<String, f64>)> {
    let mut correct = 0;
    let mut total = 0;
    let mut direction_correct = [0; 9];
    let mut direction_total = [0; 9];
    let direction_names = ["L", "R", "T", "B", "TL", "TR", "BL", "BR", "U"];
    
    let test_samples = 1000.min(lines.len() / 2);
    
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

fn print_experiment_summary(results: &TrainingResults) {
    println!("\n=== Experiment Summary ===");
    println!("Name: {}", results.experiment_name);
    println!("Final accuracy: {:.2}%", results.final_accuracy);
    println!("Best accuracy: {:.2}%", results.best_accuracy);
    println!("Training steps: {}", results.training_steps);
    println!("Training time: {:.1}s", results.training_time);
    println!("Plateau detected: {}", results.plateau_detected);
    
    if let Some(step) = results.convergence_step {
        println!("Convergence step: {}", step);
    }
    
    println!("\n--- Per-Direction Performance ---");
    for (direction, accuracy) in &results.per_direction_accuracy {
        println!("{}: {:.1}%", direction, accuracy);
    }
    
    println!("\n--- Configuration ---");
    println!("Feature type: {:?}", results.config.feature_type);
    println!("Architecture: {:?}", results.config.architecture.hidden_layers);
    println!("Learning rate: {:.4}", results.config.training.learning_rate);
    println!("Weight decay: {:.4}", results.config.training.weight_decay);
}
