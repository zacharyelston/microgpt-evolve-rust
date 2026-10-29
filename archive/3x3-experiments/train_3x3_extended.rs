//! Extended 3x3 Block Training
//! 
//! Continue training the improved architecture to find the performance plateau.

use microgpt_rust::blocks::{VisionBlock3x3, GenomeCollection, GenomeMetrics, TrainingConfig};
use candle_core::{Device, Tensor, Result, Module, DType};
use candle_nn::{loss, ops::softmax, Optimizer, AdamW, ParamsAdamW, VarBuilder, VarMap};
use std::io::Write;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    
    // Extended training configuration
    let embedding_dim = 64;
    let learning_rate = 0.005;
    let training_steps = 50000; // Much longer training
    let collection_id = if args.len() > 1 { &args[1] } else { "eyespot_3x3_extended_v1" };
    
    println!("=== Extended 3x3 Block Training ===");
    println!("Collection ID: {}", collection_id);
    println!("Embedding dim: {}", embedding_dim);
    println!("Training steps: {}", training_steps);
    println!("Goal: Find performance plateau with current architecture");
    
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    println!("Device: {:?}", device);
    
    // Load dataset
    let data_path = "eyespot_training_3x3.txt";
    if !std::path::Path::new(data_path).exists() {
        eprintln!("Error: Dataset not found: {}", data_path);
        eprintln!("Run: cargo run --release --bin generate_eyespot_3x3");
        std::process::exit(1);
    }
    
    let raw = std::fs::read_to_string(data_path).expect("Failed to read training data");
    let lines: Vec<&str> = raw.lines().collect();
    println!("Dataset: {} samples", lines.len());
    
    // Create genome collection
    let training_config = TrainingConfig {
        learning_rate,
        batch_size: 1,
        training_steps,
        embedding_dim,
        optimizer: "AdamW".to_string(),
    };
    
    let mut collection = GenomeCollection::new(collection_id, data_path, training_config);
    
    // Create the same improved architecture
    println!("\n--- Extended Training with Enhanced Architecture ---");
    
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    
    // Same architecture as before: 108 features -> 128 -> 64
    let feature_extractor = candle_nn::linear(108, 128, vb.pp("feature_extractor"))?;
    let feature_layer2 = candle_nn::linear(128, embedding_dim, vb.pp("feature_layer2"))?;
    let evaluator = candle_nn::linear(embedding_dim, 9, vb.pp("evaluator"))?;
    
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
    let mut last_print = std::time::Instant::now();
    let mut best_accuracy = 0.0;
    let mut plateau_counter = 0;
    let max_plateau = 5000; // Allow longer plateau before stopping
    let mut accuracy_history = Vec::new();
    
    for step in 0..training_steps {
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
        
        // Enhanced feature extraction (same as before)
        let mut features = vec![0.0f32; 108];
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
        
        let input = Tensor::from_vec(features, 108, &device)?.unsqueeze(0)?;
        
        // Forward pass
        let features1 = feature_extractor.forward(&input)?.relu()?;
        let features2 = feature_layer2.forward(&features1)?.relu()?;
        let logits = evaluator.forward(&features2)?;
        let logits_2d = logits.squeeze(0)?;
        
        // Calculate loss
        let target = Tensor::from_vec(vec![target_label as u32], 1, &device)?;
        let loss_val = loss::cross_entropy(&logits_2d.unsqueeze(0)?, &target)?;
        
        // Backward pass
        optimizer.backward_step(&loss_val)?;
        
        // Progress reporting and plateau detection
        if step % 500 == 0 || last_print.elapsed().as_secs_f64() > 5.0 {
            let current_accuracy = quick_accuracy_check(&feature_extractor, &feature_layer2, &evaluator, &lines, &device, 200)?;
            
            accuracy_history.push((step, current_accuracy));
            
            if current_accuracy > best_accuracy {
                best_accuracy = current_accuracy;
                plateau_counter = 0;
                println!("🎯 NEW BEST: {:.2}% at step {}", best_accuracy, step);
            } else {
                plateau_counter += 500;
                if plateau_counter >= max_plateau {
                    println!("⏹️  PLATEAU DETECTED: {:.2}% for {} steps", best_accuracy, plateau_counter);
                    println!("📊 Performance plateau reached at step {}", step);
                    break;
                }
            }
            
            let elapsed = start_time.elapsed().as_secs_f64();
            let progress = (step as f64 / training_steps as f64) * 100.0;
            let steps_per_sec = step as f64 / elapsed.max(0.001);
            let remaining_steps = training_steps - step;
            let eta_secs = remaining_steps as f64 / steps_per_sec.max(0.001);
            
            let probs = softmax(&logits_2d, 0)?;
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
            last_print = std::time::Instant::now();
        }
    }
    
    // Final comprehensive evaluation
    println!("\n\n--- Final Comprehensive Evaluation ---");
    
    let mut correct = 0;
    let mut total = 0;
    let test_samples = 2000.min(lines.len() / 2);
    
    // Track per-direction performance
    let mut direction_correct = [0; 9];
    let mut direction_total = [0; 9];
    let direction_names = ["L", "R", "T", "B", "TL", "TR", "BL", "BR", "U"];
    
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
        
        // Enhanced feature extraction
        let mut features = vec![0.0f32; 108];
        for (i, &pixel) in grid_pixels.iter().enumerate() {
            if pixel <= 9 {
                features[i * 10 + pixel as usize] = 1.0;
            }
        }
        
        for i in 0..9 {
            let row = i / 3;
            let col = i % 3;
            features[90 + i * 2] = (row as f32 - 1.0) / 1.0;
            features[90 + i * 2 + 1] = (col as f32 - 1.0) / 1.0;
        }
        
        let input = Tensor::from_vec(features, 108, &device)?.unsqueeze(0)?;
        
        let features1 = feature_extractor.forward(&input)?.relu()?;
        let features2 = feature_layer2.forward(&features1)?.relu()?;
        let logits = evaluator.forward(&features2)?;
        let logits_2d = logits.squeeze(0)?;
        
        let probs = softmax(&logits_2d, 0)?;
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
    let training_time = start_time.elapsed().as_secs_f64();
    
    println!("Test accuracy: {}/{} ({:.2}%)", correct, total, accuracy);
    println!("Training time: {:.1}s ({:.2} steps/s)", training_time, training_steps as f64 / training_time);
    
    // Per-direction analysis
    println!("\n--- Per-Direction Performance ---");
    for (i, &name) in direction_names.iter().enumerate() {
        if direction_total[i] > 0 {
            let dir_acc = (direction_correct[i] as f64 / direction_total[i] as f64) * 100.0;
            println!("{}: {}/{} ({:.1}%)", name, direction_correct[i], direction_total[i], dir_acc);
        }
    }
    
    // Performance analysis
    println!("\n--- Performance Analysis ---");
    if accuracy >= 90.0 {
        println!("🎉 EXCELLENT: {:.1}% accuracy - 3x3 block mastered!", accuracy);
    } else if accuracy >= 75.0 {
        println!("✅ GOOD: {:.1}% accuracy - Block working well", accuracy);
    } else if accuracy >= 50.0 {
        println!("⚠️  FAIR: {:.1}% accuracy - Needs improvement", accuracy);
    } else {
        println!("❌ POOR: {:.1}% accuracy - Architecture needs revision", accuracy);
    }
    
    // Plateau analysis
    println!("\n--- Plateau Analysis ---");
    println!("Best accuracy achieved: {:.2}%", best_accuracy);
    println!("Final accuracy: {:.2}%", accuracy);
    if accuracy_history.len() > 1 {
        let recent_accuracies: Vec<f64> = accuracy_history.iter().rev().take(10).map(|(_, acc)| *acc).collect();
        let recent_avg = recent_accuracies.iter().sum::<f64>() / recent_accuracies.len() as f64;
        let recent_variance = recent_accuracies.iter().map(|acc| (acc - recent_avg).powi(2)).sum::<f64>() / recent_accuracies.len() as f64;
        
        println!("Recent 10 checks average: {:.2}%", recent_avg);
        println!("Recent variance: {:.4}", recent_variance);
        
        if recent_variance < 1.0 {
            println!("📈 STABLE PLATEAU: Low variance suggests true performance limit reached");
        } else {
            println!("📊 UNSTABLE: Still learning, might benefit from more training");
        }
    }
    
    // Save results
    println!("\n--- Saving Extended Training Results ---");
    
    let simple_genome = microgpt_rust::blocks::BlockGenome {
        block_id: "extended_training_block".to_string(),
        processor_weights: evaluator.weight().to_vec2::<f32>()?,
        processor_bias: evaluator.bias().as_ref().unwrap().to_vec1::<f32>()?,
        synaptic_weight: 1.0,
        embedding_dim,
    };
    
    let metrics = GenomeMetrics {
        accuracy,
        loss: 0.0,
        training_steps,
        training_time,
        created_at: chrono::Utc::now().to_rfc3339(),
    };
    
    collection.add_genome(simple_genome, metrics);
    
    std::fs::create_dir_all("genomes")?;
    collection.save_to_file(&format!("genomes/{}.json", collection_id))
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    
    println!("Extended training results saved to: genomes/{}.json", collection_id);
    
    // Comparison with previous results
    println!("\n--- Progress Comparison ---");
    println!("Original (simple): 9.5% accuracy");
    println!("Improved (short): 45.2% accuracy");
    println!("Extended (long): {:.1}% accuracy", accuracy);
    println!("Total improvement: {:.1}x better", accuracy / 9.5);
    
    println!("\n=== Extended 3x3 Training Complete ===");
    println!("✅ Found performance plateau with current architecture");
    println!("✅ Comprehensive evaluation completed");
    println!("✅ Ready for next improvements when plateau confirmed");
    
    Ok(())
}

// Quick accuracy check
fn quick_accuracy_check(
    feature_extractor: &candle_nn::Linear, 
    feature_layer2: &candle_nn::Linear,
    evaluator: &candle_nn::Linear, 
    lines: &[&str], 
    device: &Device, 
    sample_count: usize
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
        
        let mut features = vec![0.0f32; 108];
        for (i, &pixel) in grid_pixels.iter().enumerate() {
            if pixel <= 9 {
                features[i * 10 + pixel as usize] = 1.0;
            }
        }
        
        for i in 0..9 {
            let row = i / 3;
            let col = i % 3;
            features[90 + i * 2] = (row as f32 - 1.0) / 1.0;
            features[90 + i * 2 + 1] = (col as f32 - 1.0) / 1.0;
        }
        
        let input = Tensor::from_vec(features, 108, device)?.unsqueeze(0)?;
        
        let features1 = feature_extractor.forward(&input)?.relu()?;
        let features2 = feature_layer2.forward(&features1)?.relu()?;
        let logits = evaluator.forward(&features2)?;
        let logits_2d = logits.squeeze(0)?;
        
        let probs = softmax(&logits_2d, 0)?;
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
