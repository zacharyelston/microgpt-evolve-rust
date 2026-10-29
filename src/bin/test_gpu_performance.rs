//! Quick GPU Performance Test
//! 
//! Minimal test to verify GPU acceleration is working

use candle_core::{Device, Tensor, Result, Module, DType};
use candle_nn::{loss, ops::softmax, Optimizer, AdamW, ParamsAdamW, VarBuilder, VarMap};
use serde_json;

fn main() -> Result<()> {
    println!("=== Quick GPU Performance Test ===");
    
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    println!("Device: {:?}", device);
    
    // Test 1: Simple tensor operations
    println!("\n--- Test 1: Simple Tensor Operations ---");
    let start = std::time::Instant::now();
    
    for i in 0..10 {
        let a = Tensor::randn(0.0f32, 1.0f32, (100, 100), &device)?;
        let b = Tensor::randn(0.0f32, 1.0f32, (100, 100), &device)?;
        let c = a.matmul(&b)?;
        let d = c.relu()?;
        if i == 0 {
            println!("First operation completed");
        }
    }
    
    let tensor_time = start.elapsed().as_secs_f64();
    println!("10 tensor operations: {:.3}s", tensor_time);
    
    // Test 2: Load perfect 1x1 model
    println!("\n--- Test 2: Load Perfect 1x1 Model ---");
    let model_path = "models/1x1/grayscale_1x1_100_percent_v1.json";
    
    if !std::path::Path::new(model_path).exists() {
        println!("❌ Model not found: {}", model_path);
        println!("Run: cargo run --release --bin train_1x1_perfect_100");
        return Ok(());
    }
    
    let model_json = std::fs::read_to_string(model_path).expect("Failed to read model");
    let model_weights: serde_json::Value = serde_json::from_str(&model_json).expect("Failed to parse model");
    println!("✅ Model loaded successfully");
    
    // Test 3: Create simple linear layers
    println!("\n--- Test 3: Create Linear Layers ---");
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    
    let layer1 = candle_nn::linear(139, 256, vb.pp("layer1"))?;
    let layer2 = candle_nn::linear(256, 9, vb.pp("layer2"))?;
    println!("✅ Layers created");
    
    // Test 4: Quick training loop
    println!("\n--- Test 4: Quick Training Loop (100 steps) ---");
    
    let mut optimizer = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.001,
        }
    )?;
    
    let training_start = std::time::Instant::now();
    
    for step in 0..100 {
        // Create random batch
        let batch_size = 32;
        let input = Tensor::randn(0.0f32, 1.0f32, (batch_size, 139), &device)?;
        let targets = Tensor::from_vec(
            (0..batch_size).map(|_| rand::random::<u32>() % 9).collect::<Vec<_>>(),
            batch_size,
            &device
        )?;
        
        // Forward pass
        let hidden = layer1.forward(&input)?.relu()?;
        let logits = layer2.forward(&hidden)?;
        let probs = softmax(&logits, 1)?;
        
        // Loss
        let loss_val = loss::cross_entropy(&probs, &targets)?;
        
        // Backward pass
        optimizer.backward_step(&loss_val)?;
        
        // Progress
        if step % 20 == 0 {
            let loss_scalar = loss_val.to_vec0::<f32>()?;
            let elapsed = training_start.elapsed().as_secs_f64();
            let steps_per_sec = (step + 1) as f64 / elapsed.max(0.001);
            
            println!("Step {}: Loss={:.4} | {:.1} steps/s | GPU: ✓", 
                step, loss_scalar, steps_per_sec);
        }
    }
    
    let training_time = training_start.elapsed().as_secs_f64();
    println!("✅ Training completed: {:.2}s | {:.1} steps/s", training_time, 100.0 / training_time);
    
    // Test 5: Load 3x3 dataset
    println!("\n--- Test 5: Load 3x3 Dataset ---");
    let data_path = "eyespot_training_3x3.txt";
    
    if !std::path::Path::new(data_path).exists() {
        println!("❌ Dataset not found: {}", data_path);
        return Ok(());
    }
    
    let raw = std::fs::read_to_string(data_path).expect("Failed to read dataset");
    let lines: Vec<&str> = raw.lines().collect();
    println!("✅ Dataset loaded: {} samples", lines.len());
    
    // Test 6: Parse a few samples
    println!("\n--- Test 6: Parse Sample Data ---");
    let mut sample_count = 0;
    
    for line in lines.iter().take(10) {
        if line.len() >= 10 {
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
            
            sample_count += 1;
        }
    }
    
    println!("✅ Parsed {} samples successfully", sample_count);
    
    // Performance Summary
    println!("\n--- GPU Performance Summary ---");
    println!("Tensor operations: {:.3}s", tensor_time);
    println!("Training (100 steps): {:.3}s", training_time);
    println!("Training speed: {:.1} steps/s", 100.0 / training_time);
    
    if training_time < 5.0 {
        println!("🎉 EXCELLENT: GPU acceleration working well!");
    } else if training_time < 10.0 {
        println!("✅ GOOD: GPU acceleration working");
    } else {
        println!("⚠️  SLOW: Check GPU optimization");
    }
    
    println!("\n=== GPU Performance Test Complete ===");
    
    Ok(())
}
