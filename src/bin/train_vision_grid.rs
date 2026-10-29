use microgpt_rust::bundle::{assembly::*, grid_config::*};
use microgpt_rust::{softmax, Val};
use std::collections::HashMap;
use std::io::Write;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    let config = if args.len() > 1 {
        match args[1].as_str() {
            "8x8" => ExperimentConfig::eyespot_8x8(),
            "12x12" => ExperimentConfig::eyespot_12x12(),
            "16x16" => ExperimentConfig::eyespot_16x16(),
            "32x32" => ExperimentConfig::eyespot_32x32_center_focus(),
            _ => {
                eprintln!("Usage: train_vision_grid [8x8|12x12|16x16|32x32]");
                std::process::exit(1);
            }
        }
    } else {
        ExperimentConfig::eyespot_8x8()
    };
    
    println!("=== Vision Grid Training: {} ===\n", config.name);
    
    let data_path = format!("eyespot_training_{}x{}.txt", 
        config.grid_config.grid_size, config.grid_config.grid_size);
    
    if !std::path::Path::new(&data_path).exists() {
        eprintln!("Error: Dataset not found: {}", data_path);
        eprintln!("Generate it with: cargo run --release --bin generate_eyespot_{}x{}", 
            config.grid_config.grid_size, config.grid_config.grid_size);
        std::process::exit(1);
    }
    
    let raw = std::fs::read_to_string(&data_path)
        .expect("Failed to read training data");
    let lines: Vec<&str> = raw.lines().collect();
    
    println!("Grid Configuration:");
    println!("  Grid size: {}x{}", config.grid_config.grid_size, config.grid_config.grid_size);
    println!("  Quadrant size: {}x{}", config.grid_config.quadrant_size, config.grid_config.quadrant_size);
    println!("  Total quadrants: {}", config.grid_config.total_quadrants);
    println!("  Pixels per quadrant: {}", config.grid_config.pixels_per_quadrant());
    
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
    
    println!("\nBundle Pattern:");
    match &config.bundle_pattern {
        BundlePattern::Uniform { n_emb, n_layer, .. } => {
            println!("  Type: Uniform");
            println!("  All quadrants: n_emb={}, n_layer={}", n_emb, n_layer);
        }
        BundlePattern::CenterFocus { center_n_emb, center_n_layer, peripheral_n_emb, peripheral_n_layer, .. } => {
            println!("  Type: Center Focus");
            println!("  Center quadrants: n_emb={}, n_layer={}", center_n_emb, center_n_layer);
            println!("  Peripheral quadrants: n_emb={}, n_layer={}", peripheral_n_emb, peripheral_n_layer);
        }
        BundlePattern::Custom { .. } => {
            println!("  Type: Custom");
        }
    }
    
    println!("\nBuilding assembly:");
    println!("  Upstream nodes: {}", upstream_config.len());
    println!("  Evaluator: n_emb={}, n_layer={}, n_head={}", 
        config.evaluator_n_emb, config.evaluator_n_layer, config.evaluator_n_head);
    
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
    println!("  Initial synapses: {}", assembly.synaptic_weights.len());
    
    let (mut m, mut v) = (vec![0.; params.len()], vec![0.; params.len()]);
    
    println!("\nTraining configuration:");
    println!("  Steps: {}", config.training_steps);
    println!("  Learning rate: {}", config.learning_rate);
    println!("  L1 penalty: {}", config.l1_penalty);
    println!("  Prune threshold: {}", config.prune_threshold);
    println!("  Max synapses: {}", config.max_synapses);
    
    println!("\n--- Training ---");
    
    let start_time = std::time::Instant::now();
    let mut last_print = std::time::Instant::now();
    let adam_beta1 = 0.9;
    let adam_beta2 = 0.999;
    let adam_eps = 1e-8;
    
    for step in 0..config.training_steps {
        let line = lines[step % lines.len()];
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
        
        let mut loss = -probs[target_label].log();
        
        if step > (config.training_steps * 2 / 3) && step % 10 == 0 {
            let l1_reg = assembly.l1_regularization();
            loss = &loss + &(&l1_reg * &Val::new(0.001));
        }
        
        for p in &params {
            p.zero();
        }
        loss.backward();
        
        let lr_t = config.learning_rate * (1. - step as f64 / config.training_steps as f64);
        for (i, p) in params.iter().enumerate() {
            let g = p.grad();
            m[i] = adam_beta1 * m[i] + (1. - adam_beta1) * g;
            v[i] = adam_beta2 * v[i] + (1. - adam_beta2) * g * g;
            let m_hat = m[i] / (1. - adam_beta1.powi(step as i32 + 1));
            let v_hat = v[i] / (1. - adam_beta2.powi(step as i32 + 1));
            p.0.borrow_mut().data -= lr_t * m_hat / (v_hat.sqrt() + adam_eps);
        }
        
        if step > (config.training_steps * 4 / 5) && step % 200 == 0 {
            let pruned = assembly.prune_synapses();
            if pruned > 0 {
                println!("\n  Step {}: Pruned {} synapses, {} active", step, pruned, assembly.active_synapses());
            }
        }
        
        if step % 10 == 0 || last_print.elapsed().as_secs_f64() > 0.5 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let progress = (step as f64 / config.training_steps as f64) * 100.0;
            let steps_per_sec = step as f64 / elapsed.max(0.001);
            let remaining_steps = config.training_steps - step;
            let eta_secs = remaining_steps as f64 / steps_per_sec.max(0.001);
            
            let pred_idx = probs.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.data().partial_cmp(&b.data()).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            
            let pred_label = match pred_idx {
                0 => 'L',
                1 => 'R',
                2 => 'T',
                3 => 'B',
                _ => 'U',
            };
            
            let correct = if pred_label == label_char { "✓" } else { "✗" };
            
            let bar_width = 30;
            let filled = ((progress / 100.0) * bar_width as f64) as usize;
            let bar: String = (0..bar_width).map(|i| if i < filled { '█' } else { '░' }).collect();
            
            print!("[{}] {:5.1}% | Step {:4}/{} | Loss: {:.4} | {:.1} steps/s | ETA: {:3.0}s | {} {} {} | Syn: {}/{}\r",
                bar, progress, step, config.training_steps, loss.data(), steps_per_sec, eta_secs,
                pred_label, label_char, correct,
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
    println!("Final active synapses: {}/{}", assembly.active_synapses(), assembly.synaptic_weights.len());
    println!("Target synapses: {}", config.max_synapses);
    
    println!("\n--- Top Active Synapses ---");
    let mut active_synapses: Vec<_> = assembly.synaptic_weights.iter()
        .filter(|conn| conn.weight.data().abs() >= config.prune_threshold)
        .collect();
    
    active_synapses.sort_by(|a, b| b.weight.data().abs().partial_cmp(&a.weight.data().abs()).unwrap());
    
    for (i, conn) in active_synapses.iter().take(config.max_synapses.min(16)).enumerate() {
        println!("  {:2}. {} -> weight: {:.4}", 
            i + 1, conn.source_node_id, conn.weight.data());
    }
}
