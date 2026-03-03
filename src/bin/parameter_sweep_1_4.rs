/*
    MicroGPT Parameter Sweep - 1-10 Range
    
    Systematically tests all valid combinations of parameters
    in the 1-10 range to find the optimal configuration.
    Tests both loss and aesthetic fitness.
*/

use microgpt_rust::{load_training_data, train_and_generate, TrainingConfig, gpu_accel::create_gpu_accelerator};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;
use std::time::Instant;
use std::sync::Arc;

const INPUT_FILE: &str = "input.txt";

#[derive(Clone, Debug)]
struct SweepResult {
    config: TrainingConfig,
    loss: f64,
    fitness: f64,
    names: Vec<String>,
    training_time: f64,
}

impl SweepResult {
    fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            loss: f64::MAX,
            fitness: 0.0,
            names: Vec::new(),
            training_time: 0.0,
        }
    }

    fn desc(&self) -> String {
        format!("Emb:{} Head:{} Lay:{} Ctx:{} FF:{} LR:{:.4} Steps:{}",
            self.config.n_emb, self.config.n_head, self.config.n_layer,
            self.config.n_ctx, self.config.n_ff_exp, self.config.lr, self.config.steps)
    }
}

fn calculate_fitness(names: &[String], training_data: &HashSet<String>) -> f64 {
    if names.is_empty() { return -100.0; }

    let mut total_score = 0.0;
    let mut valid_count = 0;

    for name in names {
        let name = name.trim().to_lowercase();
        if name.len() < 3 || !name.chars().all(|c| c.is_alphabetic()) { continue; }

        let s_flow = score_flow(&name);
        let s_sym = score_symmetry(&name);
        let s_creat = score_creativity(&name, training_data);

        total_score += s_flow * 1.0 + s_sym * 1.2 + s_creat * 2.0;
        valid_count += 1;
    }

    if valid_count == 0 { return -100.0; }
    total_score / valid_count as f64
}

fn score_flow(name: &str) -> f64 {
    let vowels: HashSet<char> = ['a', 'e', 'i', 'o', 'u', 'y'].iter().cloned().collect();
    let mut score = 0.0;
    let mut cons_v = 0;
    let mut cons_c = 0;

    for c in name.chars() {
        if vowels.contains(&c) {
            cons_v += 1;
            cons_c = 0;
        } else {
            cons_c += 1;
            cons_v = 0;
        }
        if cons_v > 2 || cons_c > 2 {
            score -= 1.0;
        }
    }

    if name.len() >= 4 && name.len() <= 8 {
        score += 0.5;
    }
    score
}

fn score_symmetry(name: &str) -> f64 {
    let mut score = 0.0;
    let chars: Vec<char> = name.chars().collect();

    if name.len() > 3 && chars.iter().eq(chars.iter().rev()) {
        score += 2.0;
    }

    if name.len() >= 4 {
        let mid = name.len() / 2;
        if name[..mid] == name[mid..mid*2] {
            score += 1.5;
        }
    }

    if name.ends_with('a') || name.ends_with('n') || name.ends_with('y') {
        score += 0.2;
    }
    score
}

fn score_creativity(name: &str, training_data: &HashSet<String>) -> f64 {
    if training_data.contains(name) {
        -5.0
    } else {
        1.0
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MicroGPT Parameter Sweep - 1-10 Range ===");
    println!("Testing all valid combinations in 1-10 parameter space");
    
    // Initialize GPU accelerator
    let gpu_accel = Arc::new(create_gpu_accelerator()?);
    println!("🚀 GPU Available: {}", gpu_accel.is_available());

    // Load training data
    if std::fs::metadata(INPUT_FILE).is_err() {
        let _ = std::process::Command::new("curl")
            .args(["-o", INPUT_FILE, "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"])
            .output();
    }
    let raw = load_training_data(INPUT_FILE);
    let training_data: HashSet<String> = raw.lines().map(|l| l.trim().to_lowercase()).collect();

    // Generate all parameter combinations in 1-10 range
    let mut configs = Vec::new();
    let mut rng = rand::thread_rng();
    
    // Parameter ranges for sweep
    let n_emb_range = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let n_ctx_range = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let n_layer_range = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let n_head_range = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let n_ff_exp_range = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let lr_range = vec![0.001, 0.005, 0.01, 0.02, 0.05];
    let steps_range = vec![200, 400, 600, 800, 1000];

    println!("📊 Generating combinations...");
    println!("  n_emb: {:?}", n_emb_range);
    println!("  n_ctx: {:?}", n_ctx_range);
    println!("  n_layer: {:?}", n_layer_range);
    println!("  n_head: {:?}", n_head_range);
    println!("  n_ff_exp: {:?}", n_ff_exp_range);
    println!("  lr: {:?}", lr_range);
    println!("  steps: {:?}", steps_range);

    // Generate all combinations (sample to keep it manageable)
    let total_combinations = n_emb_range.len() * n_ctx_range.len() * n_layer_range.len() * 
                           n_head_range.len() * n_ff_exp_range.len() * lr_range.len() * 
                           steps_range.len();
    
    println!("🔢 Total possible combinations: {}", total_combinations);
    
    // Sample combinations to keep testing time reasonable
    let sample_size = total_combinations.min(1000); // Test up to 1000 combinations for 1-10 range
    println!("🎯 Testing {} combinations (sampled)", sample_size);

    for _ in 0..sample_size {
        let mut config = TrainingConfig {
            n_emb: *n_emb_range.choose(&mut rng).unwrap(),
            n_ctx: *n_ctx_range.choose(&mut rng).unwrap(),
            n_layer: *n_layer_range.choose(&mut rng).unwrap(),
            n_head: *n_head_range.choose(&mut rng).unwrap(),
            n_ff_exp: *n_ff_exp_range.choose(&mut rng).unwrap(),
            lr: *lr_range.choose(&mut rng).unwrap(),
            steps: *steps_range.choose(&mut rng).unwrap(),
            input_file: INPUT_FILE.to_string(),
            gen_samples: 20,
            ..Default::default()
        };
        
        // Ensure valid configuration
        config.clamp();
        
        // Only add valid configurations (embedding must be divisible by heads)
        if config.n_emb % config.n_head == 0 {
            configs.push(config);
        }
    }

    println!("✅ Generated {} valid configurations", configs.len());
    
    let mut results: Vec<SweepResult> = configs.into_iter().map(SweepResult::new).collect();
    let total_start = Instant::now();

    println!("\n🚀 Starting parameter sweep...");
    
    // Evaluate all configurations in parallel
    let results_len = results.len();
    let training_data_clone = training_data.clone();
    
    results.par_iter_mut().enumerate().for_each(|(i, result)| {
        let start = Instant::now();
        
        let train_result = std::panic::catch_unwind(|| {
            train_and_generate(&result.config, true)
        });
        
        match train_result {
            Ok(r) => {
                result.loss = r.final_loss;
                result.fitness = calculate_fitness(&r.names, &training_data_clone);
                result.names = r.names;
                result.training_time = start.elapsed().as_secs_f64();
                
                if (i + 1) % 50 == 0 {
                    println!("  Completed {}/{} configurations", i + 1, results_len);
                }
            }
            Err(_) => {
                result.loss = f64::MAX;
                result.fitness = -100.0;
                result.training_time = start.elapsed().as_secs_f64();
            }
        }
    });

    let total_time = total_start.elapsed();
    
    println!("\n=== Parameter Sweep Complete ===");
    println!("Total Time: {:.1}s", total_time.as_secs_f64());
    println!("GPU Used: {}", gpu_accel.is_available());
    println!("Configurations Tested: {}", results.len());

    // Sort by loss (best first) - only include valid results
    let mut by_loss: Vec<_> = results.iter()
        .filter(|r| r.loss < f64::MAX && r.loss.is_finite())
        .cloned()
        .collect();
    by_loss.sort_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap());

    // Sort by fitness (best first) - only include valid results
    let mut by_fitness: Vec<_> = results.iter()
        .filter(|r| r.fitness > -100.0 && r.fitness.is_finite())
        .cloned()
        .collect();
    by_fitness.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

    println!("\n🏆 TOP 10 BY LOSS (Mathematical Optimization)");
    println!("Rank | Loss    | Config");
    println!("-----|---------|--------");
    for (i, result) in by_loss.iter().take(10).enumerate() {
        if result.loss < f64::MAX {
            println!("  #{} | {:.6} | {}", i + 1, result.loss, result.desc());
        }
    }

    println!("\n🎨 TOP 10 BY FITNESS (Aesthetic Optimization)");
    println!("Rank | Fitness | Config");
    println!("-----|---------|--------");
    for (i, result) in by_fitness.iter().take(10).enumerate() {
        if result.fitness > -100.0 {
            println!("  #{} | {:.4} | {}", i + 1, result.fitness, result.desc());
        }
    }

    // Show best configurations with samples
    if by_loss.is_empty() || by_fitness.is_empty() {
        println!("❌ No valid results found in sweep");
        return Ok(());
    }
    
    let best_loss = &by_loss[0];
    let best_fitness = &by_fitness[0];
    
    println!("\n🥇 BEST LOSS CONFIGURATION");
    println!("Config: {}", best_loss.desc());
    println!("Loss: {:.6}", best_loss.loss);
    println!("Fitness: {:.4}", best_loss.fitness);
    println!("Training Time: {:.1}s", best_loss.training_time);
    println!("Sample Names: {}", best_loss.names.iter().take(5).cloned().collect::<Vec<_>>().join(", "));
    
    println!("\n🎨 BEST FITNESS CONFIGURATION");
    println!("Config: {}", best_fitness.desc());
    println!("Loss: {:.6}", best_fitness.loss);
    println!("Fitness: {:.4}", best_fitness.fitness);
    println!("Training Time: {:.1}s", best_fitness.training_time);
    println!("Sample Names: {}", best_fitness.names.iter().take(5).cloned().collect::<Vec<_>>().join(", "));

    // Save best configurations
    println!("\n💾 Saving best configurations...");
    
    if let Err(e) = best_loss.config.save_genome(best_loss.loss, 0) {
        println!("Failed to save best loss genome: {}", e);
    } else {
        println!("✅ Saved best loss configuration to genome.json");
    }
    
    // Save fitness champion separately
    let fitness_file = "genome_fitness_best.json";
    if let Err(e) = best_fitness.config.save_genome(best_fitness.fitness, 0) {
        println!("Failed to save best fitness genome: {}", e);
    } else {
        // Copy to fitness-specific file
        std::fs::copy("genome.json", fitness_file).ok();
        println!("✅ Saved best fitness configuration to {}", fitness_file);
    }

    println!("\n🎯 Parameter Sweep Summary:");
    println!("  Total configurations tested: {}", results.len());
    println!("  Best loss: {:.6}", best_loss.loss);
    println!("  Best fitness: {:.4}", best_fitness.fitness);
    println!("  Average training time: {:.1}s", 
        results.iter().map(|r| r.training_time).sum::<f64>() / results.len() as f64);

    Ok(())
}
