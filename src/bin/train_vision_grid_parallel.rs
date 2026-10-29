use microgpt_rust::bundle::{assembly::*, grid_config::*};
use microgpt_rust::{softmax, Val};
use std::collections::HashMap;
use std::io::Write;
use rayon::prelude::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    let config = if args.len() > 1 {
        match args[1].as_str() {
            "8x8" => ExperimentConfig::eyespot_8x8(),
            "12x12" => ExperimentConfig::eyespot_12x12(),
            "16x16" => ExperimentConfig::eyespot_16x16(),
            "32x32" => ExperimentConfig::eyespot_32x32_center_focus(),
            _ => {
                eprintln!("Usage: train_vision_grid_parallel [8x8|12x12|16x16|32x32]");
                std::process::exit(1);
            }
        }
    } else {
        ExperimentConfig::eyespot_8x8()
    };
    
    let batch_size = std::env::var("BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    
    println!("=== Vision Grid Training (PARALLEL): {} ===\n", config.name);
    println!("Batch size: {} (parallel processing)", batch_size);
    println!("CPU threads: {}", rayon::current_num_threads());
    
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
    println!("  Quadrant size: {}x{}", config.grid_config.quadrant_size, config.grid_config.quadrant_size);
    println!("  Total quadrants: {}", config.grid_config.total_quadrants);
    
    println!("\nDataset: {} samples", lines.len());
    
    let chars = build_vocab_for_visual(9);
    let pixel_vocab = chars.len();
    let label_vocab = 5;
    
    let init_scale = 0.1;
    let rms_eps = 1e-5;
    
    let mut upstream_config = Vec::new();
    for quadrant_idx in 0..config.grid_config.total_quadrants {
        let (n_emb, n_ctx, n_layer, n_head, n_ff_exp) = 
            config.bundle_pattern.get_quadrant_config(quadrant_idx, &config.grid_config);
        
        upstream_config.push((
            format!("q{}", quadrant_idx),
            pixel_vocab,
            n_ctx,
            n_emb,
            n_layer,
            n_head,
            n_ff_exp,
        ));
    }
    
    println!("\nBuilding assembly:");
    println!("  Upstream nodes: {}", upstream_config.len());
    println!("  Evaluator: n_emb={}, n_layer={}", config.evaluator_n_emb, config.evaluator_n_layer);
    
    let mut assembly = Assembly::new(
        upstream_config,
        label_vocab,
        config.grid_config.total_quadrants,
        config.evaluator_n_emb,
        config.evaluator_n_layer,
        config.evaluator_n_head,
        4,
        init_scale,
        rms_eps,
        config.l1_penalty,
        config.prune_threshold,
        config.max_synapses,
    );
    
    let params = assembly.params();
    let num_params = params.len();
    println!("  Total parameters: {}", num_params);
    
    let (mut m, mut v) = (vec![0.; params.len()], vec![0.; params.len()]);
    
    println!("\nTraining configuration:");
    println!("  Steps: {}", config.training_steps);
    println!("  Learning rate: {}", config.learning_rate);
    
    println!("\n--- Training (Parallel Batches) ---");
    
    let start_time = std::time::Instant::now();
    let mut last_print = std::time::Instant::now();
    let adam_beta1 = 0.9;
    let adam_beta2 = 0.999;
    let adam_eps = 1e-8;
    
    let num_batches = (config.training_steps + batch_size - 1) / batch_size;
    
    for batch_idx in 0..num_batches {
        let batch_start = batch_idx * batch_size;
        let batch_end = (batch_start + batch_size).min(config.training_steps);
        let actual_batch_size = batch_end - batch_start;
        
        let batch_gradients: Vec<Vec<f64>> = (batch_start..batch_end)
            .into_par_iter()
            .map(|step| {
                let line = lines[step % lines.len()];
                let expected_len = config.grid_config.total_pixels() + 1;
                if line.len() < expected_len {
                    return vec![0.0; params.len()];
                }
                
                let grid_str = &line[0..config.grid_config.total_pixels()];
                let label_char = line.chars().nth(config.grid_config.total_pixels()).unwrap();
                let target_label = tokenize_label(label_char);
                
                let grid_tokens = tokenize_visual_grid(grid_str, &chars);
                
                let mut inputs = HashMap::new();
                for quadrant_idx in 0..config.grid_config.total_quadrants {
                    let quadrant_tokens = config.grid_config.extract_quadrant(&grid_tokens, quadrant_idx);
                    inputs.insert(format!("q{}", quadrant_idx), quadrant_tokens);
                }
                
                let mut k_cache = HashMap::new();
                let mut v_cache = HashMap::new();
                let mut eval_k = vec![vec![]; config.evaluator_n_layer];
                let mut eval_v = vec![vec![]; config.evaluator_n_layer];
                
                let logits = assembly.forward(&inputs, &mut k_cache, &mut v_cache, &mut eval_k, &mut eval_v);
                let probs = softmax(&logits);
                
                let mut loss = -probs[target_label].log();
                
                if step > (config.training_steps * 2 / 3) && step % 10 == 0 {
                    let l1_reg = assembly.l1_regularization();
                    loss = &loss + &(&l1_reg * &Val::new(0.001));
                }
                
                for p in &params {
                    p.zero();
                }
                loss.backward();
                
                params.iter().map(|p| p.grad()).collect()
            })
            .collect();
        
        let avg_gradients: Vec<f64> = (0..params.len())
            .map(|i| {
                batch_gradients.iter().map(|g| g[i]).sum::<f64>() / actual_batch_size as f64
            })
            .collect();
        
        let step = batch_end - 1;
        let lr_t = config.learning_rate * (1. - step as f64 / config.training_steps as f64);
        
        for (i, p) in params.iter().enumerate() {
            let g = avg_gradients[i];
            m[i] = adam_beta1 * m[i] + (1. - adam_beta1) * g;
            v[i] = adam_beta2 * v[i] + (1. - adam_beta2) * g * g;
            let m_hat = m[i] / (1. - adam_beta1.powi(step as i32 + 1));
            let v_hat = v[i] / (1. - adam_beta2.powi(step as i32 + 1));
            p.0.borrow_mut().data -= lr_t * m_hat / (v_hat.sqrt() + adam_eps);
        }
        
        if step > (config.training_steps * 4 / 5) && step % 200 == 0 {
            let pruned = assembly.prune_synapses();
            if pruned > 0 {
                println!("\n  Step {}: Pruned {} synapses", step, pruned);
            }
        }
        
        if batch_idx % 5 == 0 || last_print.elapsed().as_secs_f64() > 1.0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let progress = (step as f64 / config.training_steps as f64) * 100.0;
            let steps_per_sec = step as f64 / elapsed.max(0.001);
            let remaining_steps = config.training_steps - step;
            let eta_secs = remaining_steps as f64 / steps_per_sec.max(0.001);
            
            let bar_width = 30;
            let filled = ((progress / 100.0) * bar_width as f64) as usize;
            let bar: String = (0..bar_width).map(|i| if i < filled { '█' } else { '░' }).collect();
            
            print!("[{}] {:5.1}% | Step {:4}/{} | {:.1} steps/s | ETA: {:3.0}s | Syn: {}/{}\r",
                bar, progress, step, config.training_steps, steps_per_sec, eta_secs,
                assembly.active_synapses(), assembly.synaptic_weights.len()
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
        let target_label = tokenize_label(label_char);
        
        let grid_tokens = tokenize_visual_grid(grid_str, &chars);
        
        let mut inputs = HashMap::new();
        for quadrant_idx in 0..config.grid_config.total_quadrants {
            let quadrant_tokens = config.grid_config.extract_quadrant(&grid_tokens, quadrant_idx);
            inputs.insert(format!("q{}", quadrant_idx), quadrant_tokens);
        }
        
        let mut k_cache = HashMap::new();
        let mut v_cache = HashMap::new();
        let mut eval_k = vec![vec![]; config.evaluator_n_layer];
        let mut eval_v = vec![vec![]; config.evaluator_n_layer];
        
        let logits = assembly.forward(&inputs, &mut k_cache, &mut v_cache, &mut eval_k, &mut eval_v);
        let probs = softmax(&logits);
        
        let pred_idx = probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.data().partial_cmp(&b.data()).unwrap())
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
    println!("Speedup vs single-thread: {:.1}x (estimated)", 
        (config.training_steps as f64 / training_time) / 3.3);
}
