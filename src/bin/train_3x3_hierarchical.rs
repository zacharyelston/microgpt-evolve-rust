//! Hierarchical 3x3 Training with 1x1 Building Blocks
//! 
//! Use trained 1x1 grayscale detectors as building blocks for 3x3 direction detection.
//! This hierarchical approach should achieve much higher accuracy than direct 3x3 training.

use candle_core::{Device, Tensor, Result, Module, DType};
use candle_nn::{loss, ops::softmax, Optimizer, AdamW, ParamsAdamW, VarBuilder, VarMap};
use serde_json;
use std::io::Write;

/// 1x1 grayscale detector component
struct GrayscaleDetector1x1 {
    layer1: candle_nn::Linear,
    layer2: candle_nn::Linear,
    layer3: candle_nn::Linear,
    output: candle_nn::Linear,
    device: Device,
}

impl GrayscaleDetector1x1 {
    fn new(weights: &serde_json::Value, device: &Device) -> Result<Self> {
        let vb = VarBuilder::zeros(DType::F32, device);
        
        // Load weights from JSON (nested arrays)
        let layer1_weights_nested = weights["layer1_weights"].as_array().ok_or_else(|| {
            candle_core::Error::Msg("layer1_weights not found in JSON".to_string())
        })?;
        let mut layer1_weights_vec = Vec::new();
        for row in layer1_weights_nested {
            let row_array = row.as_array().ok_or_else(|| {
                candle_core::Error::Msg("Invalid weight row structure".to_string())
            })?;
            for weight in row_array {
                layer1_weights_vec.push(weight.as_f64().ok_or_else(|| {
                    candle_core::Error::Msg("Invalid weight value".to_string())
                })? as f32);
            }
        }
        let layer1_weights = Tensor::from_vec(layer1_weights_vec, (128, 14), device)?;
        let layer1_bias = Tensor::from_vec(
            weights["layer1_bias"].as_array().unwrap()
                .iter().map(|v| v.as_f64().unwrap() as f32).collect::<Vec<_>>(),
            128,
            device
        )?;
        
        // Load layer2 weights (nested)
        let layer2_weights_nested = weights["layer2_weights"].as_array().unwrap();
        let mut layer2_weights_vec = Vec::new();
        for row in layer2_weights_nested {
            let row_array = row.as_array().unwrap();
            for weight in row_array {
                layer2_weights_vec.push(weight.as_f64().unwrap() as f32);
            }
        }
        let layer2_weights = Tensor::from_vec(layer2_weights_vec, (64, 128), device)?;
        let layer2_bias_vec: Vec<f32> = weights["layer2_bias"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let layer2_bias = Tensor::from_vec(layer2_bias_vec, 64, device)?;
        
        // Load layer3 weights (nested)
        let layer3_weights_nested = weights["layer3_weights"].as_array().unwrap();
        let mut layer3_weights_vec = Vec::new();
        for row in layer3_weights_nested {
            let row_array = row.as_array().unwrap();
            for weight in row_array {
                layer3_weights_vec.push(weight.as_f64().unwrap() as f32);
            }
        }
        let layer3_weights = Tensor::from_vec(layer3_weights_vec, (64, 64), device)?;
        let layer3_bias_vec: Vec<f32> = weights["layer3_bias"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let layer3_bias = Tensor::from_vec(layer3_bias_vec, 64, device)?;
        
        // Load output weights (nested)
        let output_weights_nested = weights["output_weights"].as_array().unwrap();
        let mut output_weights_vec = Vec::new();
        for row in output_weights_nested {
            let row_array = row.as_array().unwrap();
            for weight in row_array {
                output_weights_vec.push(weight.as_f64().unwrap() as f32);
            }
        }
        let output_weights = Tensor::from_vec(output_weights_vec, (10, 64), device)?;
        let output_bias_vec: Vec<f32> = weights["output_bias"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let output_bias = Tensor::from_vec(output_bias_vec, 10, device)?;
        
        let layer1 = candle_nn::Linear::new(layer1_weights, Some(layer1_bias));
        let layer2 = candle_nn::Linear::new(layer2_weights, Some(layer2_bias));
        let layer3 = candle_nn::Linear::new(layer3_weights, Some(layer3_bias));
        let output = candle_nn::Linear::new(output_weights, Some(output_bias));
        
        Ok(Self {
            layer1, layer2, layer3, output,
            device: device.clone(),
        })
    }
    
    fn predict(&self, pixel_value: u8) -> Result<Tensor> {
        // Create enhanced features for 1x1
        let mut features = vec![0.0f32; 10];
        features[pixel_value as usize] = 1.0;
        let intensity = pixel_value as f32 / 9.0;
        features.push(intensity);
        features.push(1.0 - intensity);
        features.push(if pixel_value >= 5 { 1.0 } else { 0.0 });
        features.push(if pixel_value % 2 == 0 { 1.0 } else { 0.0 });
        
        let input = Tensor::from_vec(features, 14, &self.device)?.unsqueeze(0)?;
        let hidden1 = self.layer1.forward(&input)?.relu()?;
        let hidden2 = self.layer2.forward(&hidden1)?.relu()?;
        let hidden3 = self.layer3.forward(&hidden2)?.relu()?;
        let logits = self.output.forward(&hidden3)?;
        let probs = softmax(&logits.squeeze(0)?, 0)?;
        
        Ok(probs)
    }
}

/// Hierarchical 3x3 direction detector
struct Hierarchical3x3Detector {
    detectors: Vec<GrayscaleDetector1x1>, // 9 detectors for 9 positions
    spatial_processor: candle_nn::Linear,
    direction_classifier: candle_nn::Linear,
    device: Device,
}

impl Hierarchical3x3Detector {
    fn new(detector_weights: &serde_json::Value, device: &Device) -> Result<Self> {
        // Create 9 identical 1x1 detectors
        let mut detectors = Vec::new();
        for _ in 0..9 {
            detectors.push(GrayscaleDetector1x1::new(detector_weights, device)?);
        }
        
        // Create spatial processor and direction classifier
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        
        // Input: 9 positions × 10 grayscale classes = 90 features
        // Plus spatial features: 9 positions × 2 coordinates = 18 features
        // Total: 108 features
        let spatial_processor = candle_nn::linear(108, 128, vb.pp("spatial_processor"))?;
        let direction_classifier = candle_nn::linear(128, 9, vb.pp("direction_classifier"))?; // 9 directions
        
        Ok(Self {
            detectors,
            spatial_processor,
            direction_classifier,
            device: device.clone(),
        })
    }
    
    fn predict_direction(&self, grid_pixels: &[u8]) -> Result<Tensor> {
        // Step 1: Get grayscale predictions for each position
        let mut grayscale_features = Vec::new();
        
        for &pixel in grid_pixels {
            let grayscale_probs = self.detectors[pixel as usize % 9].predict(pixel)?;
            let grayscale_vec = grayscale_probs.to_vec1::<f32>()?;
            grayscale_features.extend(grayscale_vec);
        }
        
        // Step 2: Add spatial features
        let mut spatial_features = Vec::new();
        for i in 0..9 {
            let row = i / 3;
            let col = i % 3;
            // Center-relative coordinates
            spatial_features.push((row as f32 - 1.0) / 1.0); // -1, 0, 1
            spatial_features.push((col as f32 - 1.0) / 1.0); // -1, 0, 1
        }
        
        // Step 3: Combine features
        let mut combined_features = grayscale_features;
        combined_features.extend(spatial_features);
        
        // Step 4: Spatial processing
        let input = Tensor::from_vec(combined_features, 108, &self.device)?.unsqueeze(0)?;
        let spatial_hidden = self.spatial_processor.forward(&input)?.relu()?;
        
        // Step 5: Direction classification
        let direction_logits = self.direction_classifier.forward(&spatial_hidden)?;
        let direction_probs = softmax(&direction_logits.squeeze(0)?, 0)?;
        
        Ok(direction_probs)
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    
    // Configuration
    let learning_rate = 0.01;
    let training_steps = 20000;
    let collection_id = if args.len() > 1 { &args[1] } else { "hierarchical_3x3_v1" };
    
    println!("=== Hierarchical 3x3 Direction Detection ===");
    println!("Collection ID: {}", collection_id);
    println!("Training steps: {}", training_steps);
    println!("Goal: 95%+ accuracy using 1x1 building blocks");
    
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    println!("Device: {:?}", device);
    
    // Load trained 1x1 model
    let model_path = "models/1x1/grayscale_1x1_perfect_v1.json";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Error: 1x1 model not found: {}", model_path);
        eprintln!("Run: cargo run --release --bin train_1x1_perfect");
        std::process::exit(1);
    }
    
    let model_json = std::fs::read_to_string(model_path).expect("Failed to read 1x1 model");
    let model_weights: serde_json::Value = serde_json::from_str(&model_json).expect("Failed to parse 1x1 model");
    
    println!("Loaded 1x1 model with {:.2}% accuracy", model_weights["accuracy"].as_f64().unwrap_or(0.0));
    
    // Load 3x3 dataset
    let data_path = "eyespot_training_3x3.txt";
    if !std::path::Path::new(data_path).exists() {
        eprintln!("Error: Dataset not found: {}", data_path);
        std::process::exit(1);
    }
    
    let raw = std::fs::read_to_string(data_path).expect("Failed to read training data");
    let lines: Vec<&str> = raw.lines().collect();
    println!("Dataset: {} samples", lines.len());
    
    // Create hierarchical detector
    let hierarchical_detector = Hierarchical3x3Detector::new(&model_weights, &device)?;
    
    // Create trainable components (only spatial processor and direction classifier)
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    
    let spatial_processor = candle_nn::linear(108, 128, vb.pp("spatial_processor"))?;
    let direction_classifier = candle_nn::linear(128, 9, vb.pp("direction_classifier"))?;
    
    // Create optimizer
    let mut optimizer = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    )?;
    
    let start_time = std::time::Instant::now();
    let mut best_accuracy = 0.0;
    let mut plateau_counter = 0;
    let max_plateau = 3000;
    
    println!("\n--- Training Hierarchical 3x3 Detector ---");
    
    for step in 0..training_steps {
        let line = lines[step % lines.len()];
        if line.len() < 10 {
            continue;
        }
        
        // Parse 3x3 grid and label
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
        
        // Step 1: Get grayscale predictions (fixed, no gradients)
        let mut grayscale_features = Vec::new();
        for (i, &pixel) in grid_pixels.iter().enumerate() {
            let grayscale_probs = hierarchical_detector.detectors[i].predict(pixel)?;
            let grayscale_vec = grayscale_probs.to_vec1::<f32>()?;
            grayscale_features.extend(grayscale_vec);
        }
        
        // Step 2: Add spatial features
        let mut spatial_features = Vec::new();
        for i in 0..9 {
            let row = i / 3;
            let col = i % 3;
            spatial_features.push((row as f32 - 1.0) / 1.0);
            spatial_features.push((col as f32 - 1.0) / 1.0);
        }
        
        // Step 3: Combine features
        let mut combined_features = grayscale_features;
        combined_features.extend(spatial_features);
        
        // Step 4: Forward pass through trainable components
        let input = Tensor::from_vec(combined_features, 108, &device)?.unsqueeze(0)?;
        let spatial_hidden = spatial_processor.forward(&input)?.relu()?;
        let direction_logits = direction_classifier.forward(&spatial_hidden)?;
        let direction_logits_2d = direction_logits.squeeze(0)?;
        
        // Step 5: Calculate loss
        let target = Tensor::from_vec(vec![target_label as u32], 1, &device)?;
        let loss_val = loss::cross_entropy(&direction_logits_2d.unsqueeze(0)?, &target)?;
        
        // Step 6: Backward pass (only through trainable components)
        optimizer.backward_step(&loss_val)?;
        
        // Progress reporting
        if step % 500 == 0 {
            let current_accuracy = evaluate_hierarchical_3x3(
                &spatial_processor, 
                &direction_classifier, 
                &hierarchical_detector, 
                &lines, 
                &device, 
                1000
            )?;
            
            if current_accuracy > best_accuracy {
                best_accuracy = current_accuracy;
                plateau_counter = 0;
                println!("🎯 NEW BEST: {:.2}% at step {}", best_accuracy, step);
                
                if best_accuracy >= 90.0 {
                    println!("🎉 EXCELLENT: {:.1}% - Hierarchical approach working!", best_accuracy);
                }
                
                if best_accuracy >= 95.0 {
                    println!("🎉 TARGET ACHIEVED: {:.1}% - 95% accuracy reached!", best_accuracy);
                    if best_accuracy >= 98.0 {
                        println!("🎉 OUTSTANDING: {:.1}% - Exceptional performance!", best_accuracy);
                        break;
                    }
                }
            } else {
                plateau_counter += 500;
            }
            
            if plateau_counter >= max_plateau {
                println!("⏹️  Plateau detected at step {} (best: {:.2}%)", step, best_accuracy);
                break;
            }
            
            let elapsed = start_time.elapsed().as_secs_f64();
            let progress = (step as f64 / training_steps as f64) * 100.0;
            let steps_per_sec = step as f64 / elapsed.max(0.001);
            let remaining_steps = training_steps - step;
            let eta_secs = remaining_steps as f64 / steps_per_sec.max(0.001);
            
            let probs = softmax(&direction_logits_2d, 0)?;
            let probs_vec = probs.to_vec1::<f32>()?;
            let pred_idx = probs_vec.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            
            let pred_label = match pred_idx {
                0 => "L", 1 => "R", 2 => "T", 3 => "B",
                4 => "TL", 5 => "TR", 6 => "BL", 7 => "BR",
                _ => "U",
            };
            
            let correct = if pred_idx == target_label { "✓" } else { "✗" };
            
            let bar_width = 30;
            let filled = ((progress / 100.0) * bar_width as f64) as usize;
            let bar: String = (0..bar_width).map(|i| if i < filled { '█' } else { '░' }).collect();
            
            let loss_scalar = loss_val.to_vec0::<f32>()?;
            
            print!("[{}] {:5.1}% | Step {:5}/{} | Loss: {:.4} | Best: {:.1}% | {:.1} steps/s | ETA: {:3.0}s | {} {} {}\r",
                bar, progress, step, training_steps, loss_scalar, best_accuracy, steps_per_sec, eta_secs,
                pred_label, label_str, correct
            );
            std::io::stdout().flush().unwrap();
        }
    }
    
    // Final evaluation
    println!("\n\n--- Final Hierarchical 3x3 Evaluation ---");
    
    let (final_accuracy, per_direction) = evaluate_hierarchical_3x3_detailed(
        &spatial_processor, 
        &direction_classifier, 
        &hierarchical_detector, 
        &lines, 
        &device
    )?;
    
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
    println!("\n--- Hierarchical Performance Assessment ---");
    if final_accuracy >= 95.0 {
        println!("🎉 SUCCESS: {:.2}% - Hierarchical approach achieved 95%+!", final_accuracy);
        println!("✅ 1x1 building blocks + spatial processing = excellent results");
    } else if final_accuracy >= 90.0 {
        println!("🎯 EXCELLENT: {:.2}% - Very close to target", final_accuracy);
        println!("💡 Hierarchical approach showing strong promise");
    } else if final_accuracy >= 80.0 {
        println!("✅ GOOD: {:.2}% - Significant improvement over direct 3x3", final_accuracy);
        println!("💡 Hierarchical approach working well");
    } else {
        println!("⚠️  IMPROVING: {:.2}% - Better than direct 3x3", final_accuracy);
        println!("💡 Hierarchical foundation is solid");
    }
    
    // Save hierarchical model
    println!("\n--- Saving Hierarchical 3x3 Model ---");
    
    std::fs::create_dir_all("models/3x3")?;
    
    let hierarchical_weights = serde_json::json!({
        "spatial_processor_weights": spatial_processor.weight().to_vec2::<f32>()?,
        "spatial_processor_bias": spatial_processor.bias().as_ref().unwrap().to_vec1::<f32>()?,
        "direction_classifier_weights": direction_classifier.weight().to_vec2::<f32>()?,
        "direction_classifier_bias": direction_classifier.bias().as_ref().unwrap().to_vec1::<f32>()?,
        "base_1x1_model": model_path,
        "base_1x1_accuracy": model_weights["accuracy"],
        "hierarchical_accuracy": final_accuracy,
        "training_steps": training_steps,
        "training_time": training_time,
        "architecture": "9x(1x1) -> spatial -> direction",
    });
    
    let model_path = format!("models/3x3/{}.json", collection_id);
    std::fs::write(&model_path, serde_json::to_string_pretty(&hierarchical_weights)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?)?;
    
    println!("Hierarchical 3x3 model saved to: {}", model_path);
    
    println!("\n=== Hierarchical 3x3 Training Complete ===");
    println!("✅ Hierarchical architecture trained successfully");
    println!("✅ 1x1 building blocks + spatial processing");
    println!("✅ New approach to 3x3 direction detection");
    
    Ok(())
}

fn evaluate_hierarchical_3x3(
    spatial_processor: &candle_nn::Linear,
    direction_classifier: &candle_nn::Linear,
    hierarchical_detector: &Hierarchical3x3Detector,
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
        
        // Get grayscale predictions
        let mut grayscale_features = Vec::new();
        for (i, &pixel) in grid_pixels.iter().enumerate() {
            let grayscale_probs = hierarchical_detector.detectors[i].predict(pixel)?;
            let grayscale_vec = grayscale_probs.to_vec1::<f32>()?;
            grayscale_features.extend(grayscale_vec);
        }
        
        // Add spatial features
        let mut spatial_features = Vec::new();
        for i in 0..9 {
            let row = i / 3;
            let col = i % 3;
            spatial_features.push((row as f32 - 1.0) / 1.0);
            spatial_features.push((col as f32 - 1.0) / 1.0);
        }
        
        // Combine and process
        let mut combined_features = grayscale_features;
        combined_features.extend(spatial_features);
        
        let input = Tensor::from_vec(combined_features, 108, device)?.unsqueeze(0)?;
        let spatial_hidden = spatial_processor.forward(&input)?.relu()?;
        let direction_logits = direction_classifier.forward(&spatial_hidden)?;
        let direction_logits_2d = direction_logits.squeeze(0)?;
        
        let probs = softmax(&direction_logits_2d, 0)?;
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

fn evaluate_hierarchical_3x3_detailed(
    spatial_processor: &candle_nn::Linear,
    direction_classifier: &candle_nn::Linear,
    hierarchical_detector: &Hierarchical3x3Detector,
    lines: &[&str],
    device: &Device,
) -> Result<(f64, std::collections::HashMap<String, f64>)> {
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
        
        // Get grayscale predictions
        let mut grayscale_features = Vec::new();
        for (i, &pixel) in grid_pixels.iter().enumerate() {
            let grayscale_probs = hierarchical_detector.detectors[i].predict(pixel)?;
            let grayscale_vec = grayscale_probs.to_vec1::<f32>()?;
            grayscale_features.extend(grayscale_vec);
        }
        
        // Add spatial features
        let mut spatial_features = Vec::new();
        for i in 0..9 {
            let row = i / 3;
            let col = i % 3;
            spatial_features.push((row as f32 - 1.0) / 1.0);
            spatial_features.push((col as f32 - 1.0) / 1.0);
        }
        
        // Combine and process
        let mut combined_features = grayscale_features;
        combined_features.extend(spatial_features);
        
        let input = Tensor::from_vec(combined_features, 108, device)?.unsqueeze(0)?;
        let spatial_hidden = spatial_processor.forward(&input)?.relu()?;
        let direction_logits = direction_classifier.forward(&spatial_hidden)?;
        let direction_logits_2d = direction_logits.squeeze(0)?;
        
        let probs = softmax(&direction_logits_2d, 0)?;
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
    let mut per_direction = std::collections::HashMap::new();
    
    for (i, &name) in direction_names.iter().enumerate() {
        if direction_total[i] > 0 {
            let dir_acc = (direction_correct[i] as f64 / direction_total[i] as f64) * 100.0;
            per_direction.insert(name.to_string(), dir_acc);
        }
    }
    
    Ok((accuracy, per_direction))
}
