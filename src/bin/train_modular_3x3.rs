//! Modular 3x3 Block Training
//! 
//! Trains individual 3x3 vision blocks and saves their genomes for reuse
//! in larger assemblies.

use microgpt_rust::blocks::{VisionBlock3x3, GenomeCollection, GenomeMetrics, TrainingConfig};
use candle_core::{Device, Tensor, Result, Module, DType};
use candle_nn::{loss, ops::softmax, Optimizer, AdamW, ParamsAdamW, VarBuilder, VarMap};
use std::io::Write;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    
    // Training configuration
    let embedding_dim = 24;
    let learning_rate = 0.02;
    let training_steps = 5000;
    let collection_id = if args.len() > 1 { &args[1] } else { "eyespot_3x3_v1" };
    
    println!("=== Modular 3x3 Block Training ===");
    println!("Collection ID: {}", collection_id);
    println!("Embedding dim: {}", embedding_dim);
    println!("Training steps: {}", training_steps);
    
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
    
    // Train single 3x3 block
    println!("\n--- Training 3x3 Vision Block ---");
    
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    
    // Create the block
    let block = VisionBlock3x3::new("primary_block", embedding_dim, &device)?;
    
    // Create evaluator (9 directions: L,R,T,B,TL,TR,BL,BR,U)
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
    
    for step in 0..training_steps {
        let line = lines[step % lines.len()];
        if line.len() < 10 { // 9 pixels + 1-2 char label
            continue;
        }
        
        // Parse grid and label
        let grid_str = &line[0..9];
        let label_str = &line[9..];
        
        // Convert grid string to pixel values
        let mut grid_pixels = vec![0u8; 9];
        for (i, ch) in grid_str.chars().enumerate() {
            grid_pixels[i] = ch.to_digit(10).unwrap_or(0) as u8;
        }
        
        // Map label to index
        let target_label = match label_str {
            "L" => 0,
            "R" => 1,
            "T" => 2,
            "B" => 3,
            "TL" => 4,
            "TR" => 5,
            "BL" => 6,
            "BR" => 7,
            "U" => 8,
            _ => 8,
        };
        
        // Forward pass
        let block_output = block.forward(&grid_pixels)?;
        let logits = evaluator.forward(&block_output.unsqueeze(0)?)?;
        let logits_2d = logits.squeeze(0)?;
        
        // Calculate loss
        let target = Tensor::from_vec(vec![target_label as u32], 1, &device)?;
        let loss_val = loss::cross_entropy(&logits_2d.unsqueeze(0)?, &target)?;
        
        // Backward pass
        optimizer.backward_step(&loss_val)?;
        
        // Progress reporting
        if step % 10 == 0 || last_print.elapsed().as_secs_f64() > 0.5 {
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
                0 => "L",
                1 => "R",
                2 => "T",
                3 => "B",
                4 => "TL",
                5 => "TR",
                6 => "BL",
                7 => "BR",
                _ => "U",
            };
            
            let correct = if pred_idx == target_label { "✓" } else { "✗" };
            
            let bar_width = 30;
            let filled = ((progress / 100.0) * bar_width as f64) as usize;
            let bar: String = (0..bar_width).map(|i| if i < filled { '█' } else { '░' }).collect();
            
            let loss_scalar = loss_val.to_vec0::<f32>()?;
            
            print!("[{}] {:5.1}% | Step {:4}/{} | Loss: {:.4} | {:.1} steps/s | ETA: {:3.0}s | {} {} {}\r",
                bar, progress, step, training_steps, loss_scalar, steps_per_sec, eta_secs,
                pred_label, label_str, correct
            );
            std::io::stdout().flush().unwrap();
            last_print = std::time::Instant::now();
        }
    }
    
    // Final evaluation
    println!("\n\n--- Final Evaluation ---");
    
    let mut correct = 0;
    let mut total = 0;
    let test_samples = 200.min(lines.len() / 10);
    
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
        
        let block_output = block.forward(&grid_pixels)?;
        let logits = evaluator.forward(&block_output.unsqueeze(0)?)?;
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
    
    let accuracy = (correct as f64 / total as f64) * 100.0;
    let training_time = start_time.elapsed().as_secs_f64();
    let final_loss = {
        // Get final loss from last batch
        let line = lines[0];
        let grid_str = &line[0..9];
        let mut grid_pixels = vec![0u8; 9];
        for (i, ch) in grid_str.chars().enumerate() {
            grid_pixels[i] = ch.to_digit(10).unwrap_or(0) as u8;
        }
        
        let block_output = block.forward(&grid_pixels)?;
        let logits = evaluator.forward(&block_output.unsqueeze(0)?)?;
        let logits_2d = logits.squeeze(0)?;
        let target = Tensor::from_vec(vec![0u32], 1, &device)?;
        loss::cross_entropy(&logits_2d.unsqueeze(0)?, &target)?.to_vec0::<f32>()?
    };
    
    println!("Test accuracy: {}/{} ({:.2}%)", correct, total, accuracy);
    println!("Training time: {:.1}s ({:.2} steps/s)", training_time, training_steps as f64 / training_time);
    println!("Final loss: {:.4}", final_loss);
    
    // Extract and save genome
    println!("\n--- Saving Genome ---");
    
    let genome = block.extract_genome()?;
    let metrics = GenomeMetrics {
        accuracy,
        loss: final_loss as f64,
        training_steps,
        training_time,
        created_at: chrono::Utc::now().to_rfc3339(),
    };
    
    collection.add_genome(genome, metrics);
    
    // Save collection
    std::fs::create_dir_all("genomes")?;
    collection.save_to_file(&format!("genomes/{}.json", collection_id))
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    
    println!("Genome saved to: genomes/{}.json", collection_id);
    
    // Print genome info
    let stats = collection.get_statistics();
    println!("\n--- Collection Statistics ---");
    println!("Total genomes: {}", stats.total_genomes);
    println!("Best accuracy: {:.2}%", stats.best_accuracy);
    println!("Average training time: {:.1}s", stats.avg_training_time);
    
    println!("\n=== Modular 3x3 Block Training Complete ===");
    println!("✅ Trained single 3x3 vision block");
    println!("✅ Extracted and saved genome");
    println!("✅ Ready for assembly into larger grids");
    
    Ok(())
}
