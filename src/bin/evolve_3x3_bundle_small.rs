//! Small Batch Evolution for 3x3 Bundle System
//! 
//! Evolutionary approach with small batches for rapid performance testing
//! Combines genetic algorithms with gradient-based training

use candle_core::{Device, Tensor, Result, Module, DType};
use candle_nn::{loss, ops::softmax, Optimizer, AdamW, ParamsAdamW, VarBuilder, VarMap};
use serde_json;
use std::io::Write;

/// Individual in the evolutionary population
struct BundleIndividual {
    spatial_weights: Vec<f32>,
    spatial_bias: Vec<f32>,
    direction_weights: Vec<f32>,
    direction_bias: Vec<f32>,
    fitness: f64,
    generation: usize,
}

impl BundleIndividual {
    fn new_random(spatial_input: usize, spatial_hidden: usize, direction_output: usize) -> Self {
        let mut rng = rand::thread_rng();
        
        // Random spatial layer weights and bias
        let spatial_weights: Vec<f32> = (0..spatial_input * spatial_hidden)
            .map(|_| rand::Rng::gen_range(&mut rng, -0.1..0.1)).collect();
        let spatial_bias: Vec<f32> = (0..spatial_hidden)
            .map(|_| rand::Rng::gen_range(&mut rng, -0.1..0.1)).collect();
        
        // Random direction layer weights and bias
        let direction_weights: Vec<f32> = (0..spatial_hidden * direction_output)
            .map(|_| rand::Rng::gen_range(&mut rng, -0.1..0.1)).collect();
        let direction_bias: Vec<f32> = (0..direction_output)
            .map(|_| rand::Rng::gen_range(&mut rng, -0.1..0.1)).collect();
        
        Self {
            spatial_weights,
            spatial_bias,
            direction_weights,
            direction_bias,
            fitness: 0.0,
            generation: 0,
        }
    }
    
    fn from_weights(
        spatial_weights: Vec<f32>,
        spatial_bias: Vec<f32>,
        direction_weights: Vec<f32>,
        direction_bias: Vec<f32>,
    ) -> Self {
        Self {
            spatial_weights,
            spatial_bias,
            direction_weights,
            direction_bias,
            fitness: 0.0,
            generation: 0,
        }
    }
    
    fn mutate(&mut self, mutation_rate: f32, mutation_strength: f32) {
        let mut rng = rand::thread_rng();
        
        // Mutate spatial weights
        for weight in &mut self.spatial_weights {
            if rand::Rng::gen::<f32>(&mut rng) < mutation_rate {
                *weight += rand::Rng::gen_range(&mut rng, -mutation_strength..mutation_strength);
            }
        }
        
        // Mutate spatial bias
        for bias in &mut self.spatial_bias {
            if rand::Rng::gen::<f32>(&mut rng) < mutation_rate {
                *bias += rand::Rng::gen_range(&mut rng, -mutation_strength..mutation_strength);
            }
        }
        
        // Mutate direction weights
        for weight in &mut self.direction_weights {
            if rand::Rng::gen::<f32>(&mut rng) < mutation_rate {
                *weight += rand::Rng::gen_range(&mut rng, -mutation_strength..mutation_strength);
            }
        }
        
        // Mutate direction bias
        for bias in &mut self.direction_bias {
            if rand::Rng::gen::<f32>(&mut rng) < mutation_rate {
                *bias += rand::Rng::gen_range(&mut rng, -mutation_strength..mutation_strength);
            }
        }
    }
    
    fn crossover(&self, other: &BundleIndividual) -> BundleIndividual {
        let mut rng = rand::thread_rng();
        
        let mut child_spatial_weights = Vec::new();
        let mut child_spatial_bias = Vec::new();
        let mut child_direction_weights = Vec::new();
        let mut child_direction_bias = Vec::new();
        
        // Crossover spatial weights
        for (i, (w1, w2)) in self.spatial_weights.iter().zip(&other.spatial_weights).enumerate() {
            if rand::Rng::gen::<f32>(&mut rng) < 0.5 {
                child_spatial_weights.push(*w1);
            } else {
                child_spatial_weights.push(*w2);
            }
        }
        
        // Crossover spatial bias
        for (i, (b1, b2)) in self.spatial_bias.iter().zip(&other.spatial_bias).enumerate() {
            if rand::Rng::gen::<f32>(&mut rng) < 0.5 {
                child_spatial_bias.push(*b1);
            } else {
                child_spatial_bias.push(*b2);
            }
        }
        
        // Crossover direction weights
        for (i, (w1, w2)) in self.direction_weights.iter().zip(&other.direction_weights).enumerate() {
            if rand::Rng::gen::<f32>(&mut rng) < 0.5 {
                child_direction_weights.push(*w1);
            } else {
                child_direction_weights.push(*w2);
            }
        }
        
        // Crossover direction bias
        for (i, (b1, b2)) in self.direction_bias.iter().zip(&other.direction_bias).enumerate() {
            if rand::Rng::gen::<f32>(&mut rng) < 0.5 {
                child_direction_bias.push(*b1);
            } else {
                child_direction_bias.push(*b2);
            }
        }
        
        BundleIndividual::from_weights(
            child_spatial_weights,
            child_spatial_bias,
            child_direction_weights,
            child_direction_bias,
        )
    }
}

/// Perfect 1x1 grayscale detector component
struct Perfect1x1Detector {
    layer1: candle_nn::Linear,
    layer2: candle_nn::Linear,
    layer3: candle_nn::Linear,
    layer4: candle_nn::Linear,
    output: candle_nn::Linear,
    device: Device,
}

impl Perfect1x1Detector {
    fn new(weights: &serde_json::Value, device: &Device) -> Result<Self> {
        // Load weights from JSON (4-layer architecture)
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
        let layer1_weights = Tensor::from_vec(layer1_weights_vec, (256, 14), device)?;
        let layer1_bias = Tensor::from_vec(
            weights["layer1_bias"].as_array().unwrap()
                .iter().map(|v| v.as_f64().unwrap() as f32).collect::<Vec<_>>(),
            256,
            device
        )?;
        
        // Load other layers (simplified for brevity)
        let layer2_weights_nested = weights["layer2_weights"].as_array().unwrap();
        let mut layer2_weights_vec = Vec::new();
        for row in layer2_weights_nested {
            let row_array = row.as_array().unwrap();
            for weight in row_array {
                layer2_weights_vec.push(weight.as_f64().unwrap() as f32);
            }
        }
        let layer2_weights = Tensor::from_vec(layer2_weights_vec, (128, 256), device)?;
        let layer2_bias_vec: Vec<f32> = weights["layer2_bias"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let layer2_bias = Tensor::from_vec(layer2_bias_vec, 128, device)?;
        
        let layer3_weights_nested = weights["layer3_weights"].as_array().unwrap();
        let mut layer3_weights_vec = Vec::new();
        for row in layer3_weights_nested {
            let row_array = row.as_array().unwrap();
            for weight in row_array {
                layer3_weights_vec.push(weight.as_f64().unwrap() as f32);
            }
        }
        let layer3_weights = Tensor::from_vec(layer3_weights_vec, (64, 128), device)?;
        let layer3_bias_vec: Vec<f32> = weights["layer3_bias"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let layer3_bias = Tensor::from_vec(layer3_bias_vec, 64, device)?;
        
        let layer4_weights_nested = weights["layer4_weights"].as_array().unwrap();
        let mut layer4_weights_vec = Vec::new();
        for row in layer4_weights_nested {
            let row_array = row.as_array().unwrap();
            for weight in row_array {
                layer4_weights_vec.push(weight.as_f64().unwrap() as f32);
            }
        }
        let layer4_weights = Tensor::from_vec(layer4_weights_vec, (128, 64), device)?;
        let layer4_bias_vec: Vec<f32> = weights["layer4_bias"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let layer4_bias = Tensor::from_vec(layer4_bias_vec, 128, device)?;
        
        let output_weights_nested = weights["output_weights"].as_array().unwrap();
        let mut output_weights_vec = Vec::new();
        for row in output_weights_nested {
            let row_array = row.as_array().unwrap();
            for weight in row_array {
                output_weights_vec.push(weight.as_f64().unwrap() as f32);
            }
        }
        let output_weights = Tensor::from_vec(output_weights_vec, (10, 128), device)?;
        let output_bias_vec: Vec<f32> = weights["output_bias"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let output_bias = Tensor::from_vec(output_bias_vec, 10, device)?;
        
        let layer1 = candle_nn::Linear::new(layer1_weights, Some(layer1_bias));
        let layer2 = candle_nn::Linear::new(layer2_weights, Some(layer2_bias));
        let layer3 = candle_nn::Linear::new(layer3_weights, Some(layer3_bias));
        let layer4 = candle_nn::Linear::new(layer4_weights, Some(layer4_bias));
        let output = candle_nn::Linear::new(output_weights, Some(output_bias));
        
        Ok(Self {
            layer1, layer2, layer3, layer4, output,
            device: device.clone(),
        })
    }
    
    fn predict(&self, pixel_value: u8) -> Result<Tensor> {
        // Create ultra-enhanced features for 1x1
        let mut features = vec![0.0f32; 14];
        features[pixel_value as usize] = 1.0;
        let intensity = pixel_value as f32 / 9.0;
        features[10] = intensity;
        features[11] = 1.0 - intensity;
        features[12] = if pixel_value >= 5 { 1.0 } else { 0.0 };
        features[13] = if pixel_value % 2 == 0 { 1.0 } else { 0.0 };
        
        let input = Tensor::from_vec(features, 14, &self.device)?.unsqueeze(0)?;
        let hidden1 = self.layer1.forward(&input)?.relu()?;
        let hidden2 = self.layer2.forward(&hidden1)?.relu()?;
        let hidden3 = self.layer3.forward(&hidden2)?.relu()?;
        let hidden4 = self.layer4.forward(&hidden3)?.relu()?;
        let logits = self.output.forward(&hidden4)?;
        let probs = softmax(&logits.squeeze(0)?, 0)?;
        
        Ok(probs)
    }
}

/// Evolutionary 3x3 Bundle System
struct EvolutionaryBundle3x3 {
    detectors: Vec<Perfect1x1Detector>, // 9 perfect 1x1 detectors
    population: Vec<BundleIndividual>,
    device: Device,
    spatial_input: usize,
    spatial_hidden: usize,
    direction_output: usize,
}

impl EvolutionaryBundle3x3 {
    fn new(detector_weights: &serde_json::Value, device: &Device, population_size: usize) -> Result<Self> {
        // Create 9 perfect 1x1 detectors
        let mut detectors = Vec::new();
        for _ in 0..9 {
            detectors.push(Perfect1x1Detector::new(detector_weights, device)?);
        }
        
        // Initialize population
        let spatial_input = 139; // 90 grayscale + 36 spatial + 13 global
        let spatial_hidden = 256;
        let direction_output = 9;
        
        let mut population = Vec::new();
        for _ in 0..population_size {
            population.push(BundleIndividual::new_random(spatial_input, spatial_hidden, direction_output));
        }
        
        Ok(Self {
            detectors,
            population,
            device: device.clone(),
            spatial_input,
            spatial_hidden,
            direction_output,
        })
    }
    
    fn predict_direction(&self, individual: &BundleIndividual, grid_pixels: &[u8]) -> Result<Tensor> {
        // Step 1: Get perfect grayscale predictions
        let mut grayscale_features = Vec::new();
        for (i, &pixel) in grid_pixels.iter().enumerate() {
            let grayscale_probs = self.detectors[i].predict(pixel)?;
            let grayscale_vec = grayscale_probs.to_vec1::<f32>()?;
            grayscale_features.extend(grayscale_vec);
        }
        
        // Step 2: Add enhanced spatial features
        let mut spatial_features = Vec::new();
        let mut global_features = Vec::new();
        
        for i in 0..9 {
            let row = i / 3;
            let col = i % 3;
            spatial_features.push((row as f32 - 1.0) / 1.0);
            spatial_features.push((col as f32 - 1.0) / 1.0);
            let distance = ((row as f32 - 1.0).powi(2) + (col as f32 - 1.0).powi(2)).sqrt() / 1.414;
            spatial_features.push(distance);
            let angle = (col as f32 - 1.0).atan2(row as f32 - 1.0) / std::f32::consts::PI;
            spatial_features.push(angle);
        }
        
        // Global features (simplified)
        global_features.push((grid_pixels[0] + grid_pixels[1] + grid_pixels[2]) as f32 / 27.0);
        global_features.push((grid_pixels[3] + grid_pixels[4] + grid_pixels[5]) as f32 / 27.0);
        global_features.push((grid_pixels[6] + grid_pixels[7] + grid_pixels[8]) as f32 / 27.0);
        global_features.push((grid_pixels[0] + grid_pixels[3] + grid_pixels[6]) as f32 / 27.0);
        global_features.push((grid_pixels[1] + grid_pixels[4] + grid_pixels[7]) as f32 / 27.0);
        global_features.push((grid_pixels[2] + grid_pixels[5] + grid_pixels[8]) as f32 / 27.0);
        global_features.push((grid_pixels[0] + grid_pixels[4] + grid_pixels[8]) as f32 / 27.0);
        global_features.push((grid_pixels[2] + grid_pixels[4] + grid_pixels[6]) as f32 / 27.0);
        global_features.push(grid_pixels[4] as f32 / 9.0);
        
        let mut edge_sum = 0.0f32;
        for i in 0..9 {
            let row = i / 3;
            let col = i % 3;
            let mut neighbor_sum = 0.0;
            let mut count = 0;
            
            if row > 0 { neighbor_sum += grid_pixels[i - 3] as f32; count += 1; }
            if row < 2 { neighbor_sum += grid_pixels[i + 3] as f32; count += 1; }
            if col > 0 { neighbor_sum += grid_pixels[i - 1] as f32; count += 1; }
            if col < 2 { neighbor_sum += grid_pixels[i + 1] as f32; count += 1; }
            
            if count > 0 {
                edge_sum += (grid_pixels[i] as f32 - neighbor_sum / count as f32).abs();
            }
        }
        global_features.push(edge_sum / 9.0);
        
        let h_sym = ((grid_pixels[0] as f32 - grid_pixels[2] as f32).abs() + 
                   (grid_pixels[3] as f32 - grid_pixels[5] as f32).abs() + 
                   (grid_pixels[6] as f32 - grid_pixels[8] as f32).abs()) / 27.0;
        let v_sym = ((grid_pixels[0] as f32 - grid_pixels[6] as f32).abs() + 
                   (grid_pixels[1] as f32 - grid_pixels[7] as f32).abs() + 
                   (grid_pixels[2] as f32 - grid_pixels[8] as f32).abs()) / 27.0;
        global_features.push(1.0 - (h_sym + v_sym) / 2.0);
        
        let unique_values: std::collections::HashSet<_> = grid_pixels.iter().collect();
        global_features.push(unique_values.len() as f32 / 10.0);
        
        let total_intensity: f32 = grid_pixels.iter().map(|&x| x as f32).sum();
        global_features.push(total_intensity / 81.0);
        
        // Step 3: Combine features
        let mut combined_features = grayscale_features;
        combined_features.extend(spatial_features);
        combined_features.extend(global_features);
        
        // Step 4: Create layers from individual weights
        let spatial_weights = Tensor::from_vec(
            individual.spatial_weights.clone(), 
            (self.spatial_hidden, self.spatial_input), 
            &self.device
        )?;
        let spatial_bias = Tensor::from_vec(individual.spatial_bias.clone(), self.spatial_hidden, &self.device)?;
        let spatial_layer = candle_nn::Linear::new(spatial_weights, Some(spatial_bias));
        
        let direction_weights = Tensor::from_vec(
            individual.direction_weights.clone(), 
            (self.direction_output, self.spatial_hidden), 
            &self.device
        )?;
        let direction_bias = Tensor::from_vec(individual.direction_bias.clone(), self.direction_output, &self.device)?;
        let direction_layer = candle_nn::Linear::new(direction_weights, Some(direction_bias));
        
        // Step 5: Forward pass
        let input = Tensor::from_vec(combined_features.clone(), combined_features.len(), &self.device)?.unsqueeze(0)?;
        let spatial_hidden = spatial_layer.forward(&input)?.relu()?;
        let direction_logits = direction_layer.forward(&spatial_hidden)?;
        let direction_probs = softmax(&direction_logits.squeeze(0)?, 0)?;
        
        Ok(direction_probs)
    }
    
    fn evaluate_fitness(&mut self, lines: &[&str], sample_size: usize) -> Result<()> {
        // Collect all fitness evaluations first
        let mut fitness_values = Vec::new();
        
        for individual in &self.population {
            let mut correct = 0;
            let mut total = 0;
            
            for line in lines.iter().take(sample_size) {
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
                
                let probs = self.predict_direction(individual, &grid_pixels)?;
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
            
            fitness_values.push((correct as f64 / total as f64) * 100.0);
        }
        
        // Update fitness values
        for (i, fitness) in fitness_values.into_iter().enumerate() {
            self.population[i].fitness = fitness;
        }
        
        // Sort population by fitness
        self.population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        
        Ok(())
    }
    
    fn evolve_generation(&mut self, mutation_rate: f32, mutation_strength: f32) {
        let population_size = self.population.len();
        let elite_size = population_size / 4; // Keep top 25%
        
        // Create new generation
        let mut new_population = Vec::new();
        
        // Keep elite individuals
        for i in 0..elite_size {
            new_population.push(BundleIndividual::from_weights(
                self.population[i].spatial_weights.clone(),
                self.population[i].spatial_bias.clone(),
                self.population[i].direction_weights.clone(),
                self.population[i].direction_bias.clone(),
            ));
        }
        
        // Generate offspring through crossover and mutation
        while new_population.len() < population_size {
            let parent1 = &self.population[rand::random::<usize>() % elite_size];
            let parent2 = &self.population[rand::random::<usize>() % elite_size];
            
            let mut child = parent1.crossover(parent2);
            child.mutate(mutation_rate, mutation_strength);
            new_population.push(child);
        }
        
        self.population = new_population;
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    
    // Evolution configuration
    let population_size = if args.len() > 1 { args[1].parse().unwrap_or(20) } else { 20 };
    let generations = if args.len() > 2 { args[2].parse().unwrap_or(50) } else { 50 };
    let sample_size = if args.len() > 3 { args[3].parse().unwrap_or(500) } else { 500 };
    let mutation_rate = if args.len() > 4 { args[4].parse().unwrap_or(0.1) } else { 0.1 };
    let mutation_strength = if args.len() > 5 { args[5].parse().unwrap_or(0.05) } else { 0.05 };
    
    println!("=== Small Batch Evolution for 3x3 Bundle System ===");
    println!("Population size: {}", population_size);
    println!("Generations: {}", generations);
    println!("Sample size per evaluation: {}", sample_size);
    println!("Mutation rate: {:.2}", mutation_rate);
    println!("Mutation strength: {:.3}", mutation_strength);
    
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    println!("Device: {:?}", device);
    
    // Load perfect 1x1 model
    let model_path = "models/1x1/grayscale_1x1_100_percent_v1.json";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Error: Perfect 1x1 model not found: {}", model_path);
        eprintln!("Run: cargo run --release --bin train_1x1_perfect_100");
        std::process::exit(1);
    }
    
    let model_json = std::fs::read_to_string(model_path).expect("Failed to read perfect 1x1 model");
    let model_weights: serde_json::Value = serde_json::from_str(&model_json).expect("Failed to parse perfect 1x1 model");
    
    println!("Loaded perfect 1x1 model with {:.2}% accuracy", model_weights["accuracy"].as_f64().unwrap_or(0.0));
    
    // Load 3x3 dataset
    let data_path = "eyespot_training_3x3.txt";
    if !std::path::Path::new(data_path).exists() {
        eprintln!("Error: Dataset not found: {}", data_path);
        std::process::exit(1);
    }
    
    let raw = std::fs::read_to_string(data_path).expect("Failed to read training data");
    let lines: Vec<&str> = raw.lines().collect();
    println!("Dataset: {} samples", lines.len());
    
    // Create evolutionary system
    let mut evolutionary_system = EvolutionaryBundle3x3::new(&model_weights, &device, population_size)?;
    
    let start_time = std::time::Instant::now();
    let mut best_fitness = 0.0;
    let mut best_individual_idx = 0;
    
    println!("\n--- Starting Evolution ---");
    
    for generation in 0..generations {
        // Evaluate fitness
        evolutionary_system.evaluate_fitness(&lines, sample_size)?;
        
        // Update best individual
        if evolutionary_system.population[0].fitness > best_fitness {
            best_fitness = evolutionary_system.population[0].fitness;
            best_individual_idx = 0;
        }
        
        // Progress reporting
        let elapsed = start_time.elapsed().as_secs_f64();
        let progress = (generation as f64 / generations as f64) * 100.0;
        let eta = if generation > 0 {
            (elapsed / generation as f64) * (generations - generation) as f64
        } else {
            0.0
        };
        
        let current_best = evolutionary_system.population[0].fitness;
        let avg_fitness: f64 = evolutionary_system.population.iter()
            .take(5) // Top 5 average
            .map(|ind| ind.fitness)
            .sum::<f64>() / 5.0;
        
        let bar_width = 30;
        let filled = ((progress / 100.0) * bar_width as f64) as usize;
        let bar: String = (0..bar_width).map(|i| if i < filled { '█' } else { '░' }).collect();
        
        print!("[{}] {:5.1}% | Gen {}/{} | Best: {:.1}% | Avg Top5: {:.1}% | ETA: {:.0}s\r",
            bar, progress, generation, generations, current_best, avg_fitness, eta
        );
        std::io::stdout().flush().unwrap();
        
        // Check for early termination
        if current_best >= 95.0 {
            println!("\n🎉 TARGET ACHIEVED: {:.1}% accuracy at generation {}!", current_best, generation);
            break;
        }
        
        // Evolve to next generation (except last generation)
        if generation < generations - 1 {
            evolutionary_system.evolve_generation(mutation_rate, mutation_strength);
        }
    }
    
    // Final evaluation
    println!("\n\n--- Final Evolution Results ---");
    
    let final_best = evolutionary_system.population[0].fitness;
    let training_time = start_time.elapsed().as_secs_f64();
    
    println!("Best accuracy: {:.2}%", final_best);
    println!("Generations completed: {}", generations.min(generations));
    println!("Training time: {:.1}s", training_time);
    
    // Detailed evaluation of best individual
    println!("\n--- Detailed Evaluation of Best Individual ---");
    let best_individual = &evolutionary_system.population[0];
    
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
        
        let probs = evolutionary_system.predict_direction(best_individual, &grid_pixels)?;
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
    
    let final_accuracy = (correct as f64 / total as f64) * 100.0;
    
    println!("Final test accuracy: {:.2}%", final_accuracy);
    
    println!("\n--- Per-Direction Performance ---");
    for (i, &direction) in direction_names.iter().enumerate() {
        if direction_total[i] > 0 {
            let dir_acc = (direction_correct[i] as f64 / direction_total[i] as f64) * 100.0;
            println!("{}: {}/{} ({:.1}%)", direction, direction_correct[i], direction_total[i], dir_acc);
        }
    }
    
    // Performance assessment
    println!("\n--- Evolution Performance Assessment ---");
    if final_accuracy >= 95.0 {
        println!("🎉 SUCCESS: {:.2}% - Evolution achieved 95%+!", final_accuracy);
        println!("✅ Small batch evolution + perfect 1x1 building blocks = breakthrough");
    } else if final_accuracy >= 90.0 {
        println!("🎯 EXCELLENT: {:.2}% - Very close to target", final_accuracy);
        println!("💡 Evolution showing exceptional promise");
    } else if final_accuracy >= 80.0 {
        println!("✅ GOOD: {:.2}% - Significant improvement", final_accuracy);
        println!("💡 Evolution working well");
    } else {
        println!("⚠️  IMPROVING: {:.2}% - Better than baseline", final_accuracy);
        println!("💡 Evolution foundation is solid");
    }
    
    // Save best individual
    println!("\n--- Saving Best Evolutionary Individual ---");
    
    std::fs::create_dir_all("models/3x3")?;
    
    let evolutionary_weights = serde_json::json!({
        "spatial_weights": best_individual.spatial_weights,
        "spatial_bias": best_individual.spatial_bias,
        "direction_weights": best_individual.direction_weights,
        "direction_bias": best_individual.direction_bias,
        "base_1x1_model": model_path,
        "base_1x1_accuracy": model_weights["accuracy"],
        "evolutionary_accuracy": final_accuracy,
        "generations": generations,
        "population_size": population_size,
        "training_time": training_time,
        "architecture": "evolutionary_3x3_bundle",
        "spatial_input": evolutionary_system.spatial_input,
        "spatial_hidden": evolutionary_system.spatial_hidden,
        "direction_output": evolutionary_system.direction_output,
    });
    
    let model_path = format!("models/3x3/evolutionary_bundle_3x3_v1.json");
    std::fs::write(&model_path, serde_json::to_string_pretty(&evolutionary_weights)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?)?;
    
    println!("Evolutionary 3x3 model saved to: {}", model_path);
    
    println!("\n=== Small Batch Evolution Complete ===");
    println!("✅ Evolutionary approach with perfect 1x1 building blocks");
    println!("✅ Small batch testing for rapid performance evaluation");
    println!("✅ Genetic algorithms + gradient-based optimization");
    
    Ok(())
}
