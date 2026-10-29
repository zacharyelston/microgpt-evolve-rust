//! 1x1 Grayscale Detection Training
//! 
//! Train a simple but highly accurate 1x1 grayscale detector
//! that can be composed into 3x3 assemblies.

use candle_core::{Device, Tensor, Result, Module, DType};
use candle_nn::{loss, ops::softmax, Optimizer, AdamW, ParamsAdamW, VarBuilder, VarMap};
use std::io::Write;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    
    // Configuration
    let embedding_dim = 32;
    let learning_rate = 0.01;
    let training_steps = 10000;
    let collection_id = if args.len() > 1 { &args[1] } else { "grayscale_1x1_v1" };
    
    println!("=== 1x1 Grayscale Detection Training ===");
    println!("Collection ID: {}", collection_id);
    println!("Embedding dim: {}", embedding_dim);
    println!("Training steps: {}", training_steps);
    println!("Goal: Perfect grayscale detection (10 classes)");
    
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    println!("Device: {:?}", device);
    
    // Generate 1x1 grayscale dataset
    let data_path = "grayscale_training_1x1.txt";
    generate_1x1_dataset(data_path)?;
    
    // Load dataset
    let raw = std::fs::read_to_string(data_path).expect("Failed to read training data");
    let lines: Vec<&str> = raw.lines().collect();
    println!("Dataset: {} samples", lines.len());
    
    // Create neural network for 1x1 grayscale detection
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    
    // Simple but effective architecture: 10 (one-hot) -> 64 -> 32 -> 10
    let layer1 = candle_nn::linear(10, 64, vb.pp("layer1"))?;
    let layer2 = candle_nn::linear(64, embedding_dim, vb.pp("layer2"))?;
    let output = candle_nn::linear(embedding_dim, 10, vb.pp("output"))?;
    
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
    let max_plateau = 2000;
    
    println!("\n--- Training 1x1 Grayscale Detector ---");
    
    for step in 0..training_steps {
        let line = lines[step % lines.len()];
        if line.len() < 2 {
            continue;
        }
        
        // Parse 1x1 grid and label
        let pixel_char = line.chars().next().unwrap();
        let label_char = line.chars().nth(1).unwrap();
        
        let pixel_value = pixel_char.to_digit(10).unwrap_or(0) as u8;
        let target_label = label_char.to_digit(10).unwrap_or(0) as usize;
        
        // Create one-hot encoding for pixel (10 values: 0-9)
        let mut features = vec![0.0f32; 10];
        features[pixel_value as usize] = 1.0;
        
        // Forward pass
        let input = Tensor::from_vec(features, 10, &device)?.unsqueeze(0)?;
        let hidden1 = layer1.forward(&input)?.relu()?;
        let hidden2 = layer2.forward(&hidden1)?.relu()?;
        let logits = output.forward(&hidden2)?;
        let logits_2d = logits.squeeze(0)?;
        
        // Calculate loss
        let target = Tensor::from_vec(vec![target_label as u32], 1, &device)?;
        let loss_val = loss::cross_entropy(&logits_2d.unsqueeze(0)?, &target)?;
        
        // Backward pass
        optimizer.backward_step(&loss_val)?;
        
        // Progress reporting
        if step % 100 == 0 {
            let current_accuracy = evaluate_1x1_model(&layer1, &layer2, &output, &lines, &device, 500)?;
            
            if current_accuracy > best_accuracy {
                best_accuracy = current_accuracy;
                plateau_counter = 0;
                println!("🎯 NEW BEST: {:.2}% at step {}", best_accuracy, step);
                
                if best_accuracy >= 98.0 {
                    println!("🎉 EXCELLENT: {:.1}% accuracy achieved!", best_accuracy);
                }
                
                if best_accuracy >= 99.5 {
                    println!("🎉 NEAR PERFECT: {:.1}% accuracy!", best_accuracy);
                    if best_accuracy >= 99.9 {
                        println!("🎉 PERFECT DETECTION: {:.1}%!", best_accuracy);
                        break;
                    }
                }
            } else {
                plateau_counter += 100;
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
            
            let probs = softmax(&logits_2d, 0)?;
            let probs_vec = probs.to_vec1::<f32>()?;
            let pred_idx = probs_vec.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            
            let correct = if pred_idx == target_label { "✓" } else { "✗" };
            
            let bar_width = 30;
            let filled = ((progress / 100.0) * bar_width as f64) as usize;
            let bar: String = (0..bar_width).map(|i| if i < filled { '█' } else { '░' }).collect();
            
            let loss_scalar = loss_val.to_vec0::<f32>()?;
            
            print!("[{}] {:5.1}% | Step {:4}/{} | Loss: {:.4} | Best: {:.1}% | {:.1} steps/s | ETA: {:3.0}s | {} {} {}\r",
                bar, progress, step, training_steps, loss_scalar, best_accuracy, steps_per_sec, eta_secs,
                pred_idx, target_label, correct
            );
            std::io::stdout().flush().unwrap();
        }
    }
    
    // Final evaluation
    println!("\n\n--- Final 1x1 Evaluation ---");
    
    let mut correct = 0;
    let mut total = 0;
    let mut per_digit_correct = [0; 10];
    let mut per_digit_total = [0; 10];
    
    let test_samples = 2000.min(lines.len() / 2);
    
    for line in lines.iter().rev().take(test_samples) {
        if line.len() < 2 {
            continue;
        }
        
        let pixel_char = line.chars().next().unwrap();
        let label_char = line.chars().nth(1).unwrap();
        
        let pixel_value = pixel_char.to_digit(10).unwrap_or(0) as u8;
        let target_label = label_char.to_digit(10).unwrap_or(0) as usize;
        
        per_digit_total[target_label] += 1;
        
        // Create one-hot encoding
        let mut features = vec![0.0f32; 10];
        features[pixel_value as usize] = 1.0;
        
        let input = Tensor::from_vec(features, 10, &device)?.unsqueeze(0)?;
        let hidden1 = layer1.forward(&input)?.relu()?;
        let hidden2 = layer2.forward(&hidden1)?.relu()?;
        let logits = output.forward(&hidden2)?;
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
            per_digit_correct[target_label] += 1;
        }
        total += 1;
    }
    
    let accuracy = (correct as f64 / total as f64) * 100.0;
    let training_time = start_time.elapsed().as_secs_f64();
    
    println!("Test accuracy: {}/{} ({:.2}%)", correct, total, accuracy);
    println!("Training time: {:.1}s ({:.2} steps/s)", training_time, training_steps as f64 / training_time);
    
    // Per-digit analysis
    println!("\n--- Per-Digit Performance ---");
    for digit in 0..10 {
        if per_digit_total[digit] > 0 {
            let digit_acc = (per_digit_correct[digit] as f64 / per_digit_total[digit] as f64) * 100.0;
            println!("Digit {}: {}/{} ({:.1}%)", digit, per_digit_correct[digit], per_digit_total[digit], digit_acc);
        }
    }
    
    // Performance assessment
    println!("\n--- Performance Assessment ---");
    if accuracy >= 99.9 {
        println!("🎉 PERFECT: {:.2}% - 1x1 grayscale detection mastered!", accuracy);
        println!("✅ Ready for 3x3 assembly");
    } else if accuracy >= 99.0 {
        println!("🎯 EXCELLENT: {:.2}% - Near-perfect detection", accuracy);
        println!("✅ Excellent foundation for 3x3 assembly");
    } else if accuracy >= 95.0 {
        println!("✅ VERY GOOD: {:.2}% - Strong performance", accuracy);
        println!("💡 Good foundation, but can improve before assembly");
    } else {
        println!("⚠️  NEEDS WORK: {:.2}% - Improve before assembly", accuracy);
        println!("💡 Focus on 1x1 accuracy before moving to 3x3");
    }
    
    // Save the 1x1 model weights
    println!("\n--- Saving 1x1 Model ---");
    
    // Create output directory
    std::fs::create_dir_all("models/1x1")?;
    
    // Save weights to JSON
    let weights_1x1 = serde_json::json!({
        "layer1_weights": layer1.weight().to_vec2::<f32>()?,
        "layer1_bias": layer1.bias().as_ref().unwrap().to_vec1::<f32>()?,
        "layer2_weights": layer2.weight().to_vec2::<f32>()?,
        "layer2_bias": layer2.bias().as_ref().unwrap().to_vec1::<f32>()?,
        "output_weights": output.weight().to_vec2::<f32>()?,
        "output_bias": output.bias().as_ref().unwrap().to_vec1::<f32>()?,
        "embedding_dim": embedding_dim,
        "accuracy": accuracy,
        "training_steps": training_steps,
        "training_time": training_time,
    });
    
    let model_path = format!("models/1x1/{}.json", collection_id);
    std::fs::write(&model_path, serde_json::to_string_pretty(&weights_1x1)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?)?;
    
    println!("1x1 model saved to: {}", model_path);
    
    println!("\n=== 1x1 Grayscale Training Complete ===");
    println!("✅ High-accuracy 1x1 detector trained");
    println!("✅ Model weights saved for 3x3 assembly");
    println!("✅ Ready for hierarchical 3x3 construction");
    
    Ok(())
}

fn generate_1x1_dataset(path: &str) -> Result<()> {
    println!("Generating 1x1 grayscale dataset...");
    
    let mut samples = Vec::new();
    
    // Generate balanced dataset with noise and variations
    for digit in 0..10 {
        // Clean samples
        for _ in 0..1000 {
            samples.push(format!("{}{}", digit, digit));
        }
        
        // Noisy samples (wrong label)
        for _ in 0..200 {
            let wrong_digit = (digit + rand::random::<u8>() % 9 + 1) % 10;
            samples.push(format!("{}{}", digit, wrong_digit));
        }
    }
    
    // Shuffle the dataset
    use rand::seq::SliceRandom;
    samples.shuffle(&mut rand::thread_rng());
    
    // Write to file
    let content = samples.join("\n");
    std::fs::write(path, content)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    
    println!("Generated {} samples to {}", samples.len(), path);
    Ok(())
}

fn evaluate_1x1_model(
    layer1: &candle_nn::Linear,
    layer2: &candle_nn::Linear,
    output: &candle_nn::Linear,
    lines: &[&str],
    device: &Device,
    sample_count: usize,
) -> Result<f64> {
    let mut correct = 0;
    let mut total = 0;
    
    for line in lines.iter().take(sample_count) {
        if line.len() < 2 {
            continue;
        }
        
        let pixel_char = line.chars().next().unwrap();
        let label_char = line.chars().nth(1).unwrap();
        
        let pixel_value = pixel_char.to_digit(10).unwrap_or(0) as u8;
        let target_label = label_char.to_digit(10).unwrap_or(0) as usize;
        
        // Create one-hot encoding
        let mut features = vec![0.0f32; 10];
        features[pixel_value as usize] = 1.0;
        
        let input = Tensor::from_vec(features, 10, device)?.unsqueeze(0)?;
        let hidden1 = layer1.forward(&input)?.relu()?;
        let hidden2 = layer2.forward(&hidden1)?.relu()?;
        let logits = output.forward(&hidden2)?;
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
