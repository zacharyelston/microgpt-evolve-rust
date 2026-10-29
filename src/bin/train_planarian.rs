use microgpt_rust::bundle::{Bundle, assembly::*};
use microgpt_rust::{softmax, Val};
use std::collections::HashMap;
use std::io::Write;

fn main() {
    println!("=== Planarian Eyespot Training ===\n");
    
    let bundle_path = "research/bundles/bundle-1-planarian-eye/planarian_eyespot-prd.yaml";
    let data_path = "eyespot_training.txt";
    
    let bundle = Bundle::from_yaml_file(bundle_path)
        .expect("Failed to load bundle");
    bundle.validate().expect("Bundle validation failed");
    
    println!("Bundle: {}", bundle.metadata.name);
    println!("Description: {}", bundle.metadata.description.as_ref().unwrap());
    
    let raw = std::fs::read_to_string(data_path)
        .expect("Failed to read training data");
    let lines: Vec<&str> = raw.lines().collect();
    
    println!("\nDataset: {} samples", lines.len());
    
    let chars = build_vocab_for_visual(9);
    let pixel_vocab = chars.len();
    let label_vocab = 5;
    
    println!("Pixel vocabulary: {} (0-9)", pixel_vocab);
    println!("Label vocabulary: {} (L/R/T/B/U)", label_vocab);
    
    let arch = &bundle.architecture;
    let init_scale = 0.1;
    let rms_eps = 1e-5;
    
    let mut upstream_config = Vec::new();
    if let microgpt_rust::bundle::UpstreamNodes::StaticAllocation { nodes } = &bundle.upstream_nodes {
        for (node_id, node_spec) in nodes {
            upstream_config.push((
                node_id.clone(),
                pixel_vocab,
                node_spec.n_ctx,
                arch.evaluator_core.n_emb,
                1,
                arch.evaluator_core.n_head,
                arch.evaluator_core.n_ff_exp,
            ));
        }
    }
    
    println!("\nBuilding assembly:");
    println!("  Upstream nodes: {}", upstream_config.len());
    println!("  Evaluator: n_emb={}, n_ctx={}, n_layer={}, n_head={}", 
        arch.evaluator_core.n_emb,
        arch.evaluator_core.n_ctx,
        arch.evaluator_core.n_layer,
        arch.evaluator_core.n_head
    );
    
    let mut assembly = Assembly::new(
        upstream_config,
        label_vocab,
        arch.evaluator_core.n_ctx,
        arch.evaluator_core.n_emb,
        arch.evaluator_core.n_layer,
        arch.evaluator_core.n_head,
        arch.evaluator_core.n_ff_exp,
        init_scale,
        rms_eps,
        arch.plasticity.l1_prune_penalty,
        arch.plasticity.prune_threshold,
        arch.plasticity.max_surviving_synapses,
    );
    
    let params = assembly.params();
    let num_params = params.len();
    println!("  Total parameters: {}", num_params);
    println!("  Initial synapses: {}", assembly.synaptic_weights.len());
    
    let steps = 3000;
    let lr = 0.01;
    let adam_beta1 = 0.9;
    let adam_beta2 = 0.999;
    let adam_eps = 1e-8;
    
    println!("\nTraining configuration:");
    println!("  Steps: {}", steps);
    println!("  Learning rate: {}", lr);
    println!("  L1 penalty: {}", arch.plasticity.l1_prune_penalty);
    println!("  Prune threshold: {}", arch.plasticity.prune_threshold);
    
    let (mut m, mut v) = (vec![0.; params.len()], vec![0.; params.len()]);
    
    println!("\n--- Training ---");
    
    let start_time = std::time::Instant::now();
    let mut last_print = std::time::Instant::now();
    
    for step in 0..steps {
        let line = lines[step % lines.len()];
        if line.len() < 65 {
            continue;
        }
        
        let grid_str = &line[0..64];
        let label_char = line.chars().nth(64).unwrap();
        let target_label = tokenize_label(label_char);
        
        let grid_tokens = tokenize_visual_grid(grid_str, &chars);
        
        let mut inputs = HashMap::new();
        if let microgpt_rust::bundle::UpstreamNodes::StaticAllocation { nodes } = &bundle.upstream_nodes {
            for (node_id, node_spec) in nodes {
                let quadrant_tokens = extract_quadrant(&grid_tokens, &node_spec.slice);
                inputs.insert(node_id.clone(), quadrant_tokens);
            }
        }
        
        let mut k_cache = HashMap::new();
        let mut v_cache = HashMap::new();
        let mut eval_k = vec![vec![]; arch.evaluator_core.n_layer];
        let mut eval_v = vec![vec![]; arch.evaluator_core.n_layer];
        
        let logits = assembly.forward(&inputs, &mut k_cache, &mut v_cache, &mut eval_k, &mut eval_v);
        let probs = softmax(&logits);
        
        let mut loss = -probs[target_label].log();
        
        if step > 1500 && step % 10 == 0 {
            let l1_reg = assembly.l1_regularization();
            loss = &loss + &(&l1_reg * &Val::new(0.001));
        }
        
        for p in &params {
            p.zero();
        }
        loss.backward();
        
        let lr_t = lr * (1. - step as f64 / steps as f64);
        for (i, p) in params.iter().enumerate() {
            let g = p.grad();
            m[i] = adam_beta1 * m[i] + (1. - adam_beta1) * g;
            v[i] = adam_beta2 * v[i] + (1. - adam_beta2) * g * g;
            let m_hat = m[i] / (1. - adam_beta1.powi(step as i32 + 1));
            let v_hat = v[i] / (1. - adam_beta2.powi(step as i32 + 1));
            p.0.borrow_mut().data -= lr_t * m_hat / (v_hat.sqrt() + adam_eps);
        }
        
        if step > 1800 && step % 200 == 0 {
            let pruned = assembly.prune_synapses();
            if pruned > 0 {
                println!("\n  Step {}: Pruned {} synapses, {} active", step, pruned, assembly.active_synapses());
            }
        }
        
        if step % 10 == 0 || last_print.elapsed().as_secs_f64() > 0.5 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let progress = (step as f64 / steps as f64) * 100.0;
            let steps_per_sec = step as f64 / elapsed;
            let remaining_steps = steps - step;
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
                bar, progress, step, steps, loss.data(), steps_per_sec, eta_secs,
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
    let test_samples = 100;
    
    for line in lines.iter().rev().take(test_samples) {
        if line.len() < 65 {
            continue;
        }
        
        let grid_str = &line[0..64];
        let label_char = line.chars().nth(64).unwrap();
        let target_label = tokenize_label(label_char);
        
        let grid_tokens = tokenize_visual_grid(grid_str, &chars);
        
        let mut inputs = HashMap::new();
        if let microgpt_rust::bundle::UpstreamNodes::StaticAllocation { nodes } = &bundle.upstream_nodes {
            for (node_id, node_spec) in nodes {
                let quadrant_tokens = extract_quadrant(&grid_tokens, &node_spec.slice);
                inputs.insert(node_id.clone(), quadrant_tokens);
            }
        }
        
        let mut k_cache = HashMap::new();
        let mut v_cache = HashMap::new();
        let mut eval_k = vec![vec![]; arch.evaluator_core.n_layer];
        let mut eval_v = vec![vec![]; arch.evaluator_core.n_layer];
        
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
    
    println!("Test accuracy: {}/{} ({:.2}%)", correct, total, accuracy);
    println!("Final active synapses: {}/{}", assembly.active_synapses(), assembly.synaptic_weights.len());
    println!("Target synapses: {}", arch.plasticity.max_surviving_synapses);
    
    println!("\n--- Synaptic Weights ---");
    for (i, conn) in assembly.synaptic_weights.iter().enumerate() {
        if conn.weight.data().abs() >= arch.plasticity.prune_threshold {
            println!("  Synapse {}: {} -> weight: {:.4}", 
                i, conn.source_node_id, conn.weight.data());
        }
    }
}
