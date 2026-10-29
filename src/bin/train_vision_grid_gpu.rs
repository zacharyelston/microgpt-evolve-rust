use microgpt_rust::bundle::grid_config::*;
use microgpt_rust::candle_gpt::*;
use candle_core::{Device, Tensor, Result};
use candle_nn::{loss, ops::softmax, Optimizer, AdamW, ParamsAdamW};
use std::io::Write;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    
    let config = if args.len() > 1 {
        match args[1].as_str() {
            "8x8" => ExperimentConfig::eyespot_8x8(),
            "12x12" => ExperimentConfig::eyespot_12x12(),
            "16x16" => ExperimentConfig::eyespot_16x16(),
            "32x32" => ExperimentConfig::eyespot_32x32_center_focus(),
            _ => {
                eprintln!("Usage: train_vision_grid_gpu [8x8|12x12|16x16|32x32]");
                std::process::exit(1);
            }
        }
    } else {
        ExperimentConfig::eyespot_12x12()
    };
    
    println!("=== Vision Grid Training (GPU-ACCELERATED): {} ===\n", config.name);
    
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    println!("Device: {:?}", device);
    
    let data_path = format!("eyespot_training_{}x{}.txt", 
        config.grid_config.grid_size, config.grid_config.grid_size);
    
    if !std::path::Path::new(&data_path).exists() {
        eprintln!("Error: Dataset not found: {}", data_path);
        std::process::exit(1);
    }
    
    let raw = std::fs::read_to_string(&data_path).expect("Failed to read training data");
    let lines: Vec<&str> = raw.lines().collect();
    
    println!("\nGrid Configuration:");
    println!("  Grid size: {}x{}", config.grid_config.grid_size, config.grid_config.grid_size);
    println!("  Total quadrants: {}", config.grid_config.total_quadrants);
    println!("  Dataset: {} samples", lines.len());
    
    let pixel_vocab = 10;
    let label_vocab = 5;
    
    let mut upstream_configs = Vec::new();
    for quadrant_idx in 0..config.grid_config.total_quadrants {
        let (n_emb, n_ctx, n_layer, n_head, n_ff_exp) = 
            config.bundle_pattern.get_quadrant_config(quadrant_idx, &config.grid_config);
        
        upstream_configs.push(CandleGPTConfig {
            vocab_size: pixel_vocab,
            n_emb,
            n_ctx,
            n_layer,
            n_head,
            n_ff_exp,
        });
    }
    
    let evaluator_config = CandleGPTConfig {
        vocab_size: label_vocab,
        n_emb: config.evaluator_n_emb,
        n_ctx: config.grid_config.total_quadrants,
        n_layer: config.evaluator_n_layer,
        n_head: config.evaluator_n_head,
        n_ff_exp: 4,
    };
    
    println!("\nBuilding GPU assembly...");
    let mut assembly = CandleAssembly::new(upstream_configs, evaluator_config, device.clone())?;
    println!("  Upstream nodes: {}", assembly.upstream_nodes.len());
    println!("  Device: {:?}", assembly.device);
    
    let mut optimizer = AdamW::new(
        assembly.varmap.all_vars(),
        ParamsAdamW {
            lr: config.learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    )?;
    
    println!("\nTraining configuration:");
    println!("  Steps: {}", config.training_steps);
    println!("  Learning rate: {}", config.learning_rate);
    
    println!("\n--- Training (GPU) ---");
    
    let start_time = std::time::Instant::now();
    let mut last_print = std::time::Instant::now();
    
    for step in 0..config.training_steps {
        let line = lines[step % lines.len()];
        let expected_len = config.grid_config.total_pixels() + 1;
        if line.len() < expected_len {
            continue;
        }
        
        let grid_str = &line[0..config.grid_config.total_pixels()];
        let label_char = line.chars().nth(config.grid_config.total_pixels()).unwrap();
        let target_label = match label_char {
            'L' => 0,
            'R' => 1,
            'T' => 2,
            'B' => 3,
            _ => 4,
        };
        
        let mut quadrant_inputs = Vec::new();
        for quadrant_idx in 0..config.grid_config.total_quadrants {
            let slice = config.grid_config.get_quadrant_slice(quadrant_idx);
            let x_start = slice[0];
            let x_end = slice[1];
            let y_start = slice[2];
            let y_end = slice[3];
            
            let mut tokens = Vec::new();
            for y in y_start..y_end {
                for x in x_start..x_end {
                    let idx = y * config.grid_config.grid_size + x;
                    if idx < grid_str.len() {
                        let ch = grid_str.chars().nth(idx).unwrap();
                        let token = ch.to_digit(10).unwrap_or(0) as u32;
                        tokens.push(token);
                    }
                }
            }
            
            let len = tokens.len();
            let input_tensor = Tensor::from_vec(tokens, len, &device)?;
            quadrant_inputs.push(input_tensor);
        }
        
        let logits = assembly.forward(&quadrant_inputs)?;
        let logits_2d = logits.reshape((1, label_vocab))?;
        
        let target = Tensor::from_vec(vec![target_label as u32], 1, &device)?;
        let loss_val = loss::cross_entropy(&logits_2d, &target)?;
        
        optimizer.backward_step(&loss_val)?;
        
        if step % 10 == 0 || last_print.elapsed().as_secs_f64() > 0.5 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let progress = (step as f64 / config.training_steps as f64) * 100.0;
            let steps_per_sec = step as f64 / elapsed.max(0.001);
            let remaining_steps = config.training_steps - step;
            let eta_secs = remaining_steps as f64 / steps_per_sec.max(0.001);
            
            let probs = softmax(&logits, 0)?;
            let probs_vec = probs.to_vec1::<f32>()?;
            let pred_idx = probs_vec.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            
            let pred_label = match pred_idx {
                0 => 'L',
                1 => 'R',
                2 => 'T',
                3 => 'B',
                _ => 'U',
            };
            
            let correct = if pred_idx == target_label { "✓" } else { "✗" };
            
            let bar_width = 30;
            let filled = ((progress / 100.0) * bar_width as f64) as usize;
            let bar: String = (0..bar_width).map(|i| if i < filled { '█' } else { '░' }).collect();
            
            let loss_scalar = loss_val.to_vec0::<f32>()?;
            
            print!("[{}] {:5.1}% | Step {:4}/{} | Loss: {:.4} | {:.1} steps/s | ETA: {:3.0}s | {} {} {}\r",
                bar, progress, step, config.training_steps, loss_scalar, steps_per_sec, eta_secs,
                pred_label, label_char, correct
            );
            std::io::stdout().flush().unwrap();
            last_print = std::time::Instant::now();
        }
    }
    
    println!("\n\n--- Final Evaluation ---");
    
    let mut correct = 0;
    let mut total = 0;
    let test_samples = 200.min(lines.len() / 10);
    
    for line in lines.iter().rev().take(test_samples) {
        let expected_len = config.grid_config.total_pixels() + 1;
        if line.len() < expected_len {
            continue;
        }
        
        let grid_str = &line[0..config.grid_config.total_pixels()];
        let label_char = line.chars().nth(config.grid_config.total_pixels()).unwrap();
        let target_label = match label_char {
            'L' => 0,
            'R' => 1,
            'T' => 2,
            'B' => 3,
            _ => 4,
        };
        
        let mut quadrant_inputs = Vec::new();
        for quadrant_idx in 0..config.grid_config.total_quadrants {
            let slice = config.grid_config.get_quadrant_slice(quadrant_idx);
            let x_start = slice[0];
            let x_end = slice[1];
            let y_start = slice[2];
            let y_end = slice[3];
            
            let mut tokens = Vec::new();
            for y in y_start..y_end {
                for x in x_start..x_end {
                    let idx = y * config.grid_config.grid_size + x;
                    if idx < grid_str.len() {
                        let ch = grid_str.chars().nth(idx).unwrap();
                        let token = ch.to_digit(10).unwrap_or(0) as u32;
                        tokens.push(token);
                    }
                }
            }
            
            let len = tokens.len();
            let input_tensor = Tensor::from_vec(tokens, len, &device)?;
            quadrant_inputs.push(input_tensor);
        }
        
        let logits = assembly.forward(&quadrant_inputs)?;
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
    
    let accuracy = (correct as f64 / total as f64) * 100.0;
    let training_time = start_time.elapsed().as_secs_f64();
    
    println!("Test accuracy: {}/{} ({:.2}%)", correct, total, accuracy);
    println!("Training time: {:.1}s ({:.2} steps/s)", training_time, config.training_steps as f64 / training_time);
    println!("GPU Speedup vs CPU baseline (3.3 steps/s): {:.1}x", 
        (config.training_steps as f64 / training_time) / 3.3);
    
    Ok(())
}
