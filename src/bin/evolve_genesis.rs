/*
    MicroGPT Evolution Engine - Genesis Branch
    
    Clean evolution starting from 1s in everything.
    20 generations with GPU integration for baseline testing.
*/

use microgpt_rust::{load_training_data, train_and_generate, TrainingConfig, gpu_accel::{create_gpu_accelerator, cpu_matrix_vector_multiply}};
use rand::prelude::*;
use rayon::prelude::*;
use std::time::Instant;
use std::sync::Arc;

const POPULATION_SIZE: usize = 12;
const NUM_GENERATIONS: usize = 20;
const MUTATION_RATE: f64 = 0.3;
const TARGET_LOSS: f64 = 1.2;
const INPUT_FILE: &str = "input.txt";

#[derive(Clone, Debug)]
struct Organism {
    config: TrainingConfig,
    loss: f64,
    evaluated: bool,
    origin: String,
}

impl Organism {
    fn from_config(config: &TrainingConfig, origin: &str) -> Self {
        Self {
            config: config.clone(),
            loss: f64::MAX,
            evaluated: false,
            origin: origin.to_string(),
        }
    }

    fn random() -> Self {
        let mut rng = rand::thread_rng();
        let mut config = TrainingConfig {
            n_emb: *[1, 2, 3, 4, 6, 8, 10, 12].choose(&mut rng).unwrap(),  // Start from 1-4
            n_ctx: *[1, 2, 3, 4, 6, 8, 12, 16].choose(&mut rng).unwrap(),  // Start from 1-4
            n_layer: rng.gen_range(1..=3),  // Smaller range
            n_head: *[1, 2].choose(&mut rng).unwrap(),  // Smaller range
            n_ff_exp: rng.gen_range(1..=3),  // Smaller range
            lr: 10f64.powf(rng.gen_range(-3.0..-1.5)),  // Slightly different LR
            steps: *[200, 400, 600, 800, 1000].choose(&mut rng).unwrap(),
            input_file: INPUT_FILE.to_string(),
            gen_samples: 1,
            ..Default::default()
        };
        
        // Clamp to valid ranges
        config.clamp();
        
        Self::from_config(&config, "random")
    }

    fn mutate(&self) -> Self {
        let mut rng = rand::thread_rng();
        let mut config = self.config.clone();
        
        // Mutations around current config - start from 1-4 ranges
        if rng.gen::<f64>() < MUTATION_RATE {
            config.n_emb = *[1, 2, 3, 4, 6, 8, 10, 12].choose(&mut rng).unwrap();
        }
        if rng.gen::<f64>() < MUTATION_RATE {
            config.n_ctx = *[1, 2, 3, 4, 6, 8, 12, 16].choose(&mut rng).unwrap();
        }
        if rng.gen::<f64>() < MUTATION_RATE {
            config.n_layer = rng.gen_range(1..=3);
        }
        if rng.gen::<f64>() < MUTATION_RATE {
            config.n_head = *[1, 2].choose(&mut rng).unwrap();
        }
        if rng.gen::<f64>() < MUTATION_RATE {
            config.n_ff_exp = rng.gen_range(1..=3);
        }
        if rng.gen::<f64>() < MUTATION_RATE {
            config.lr = 10f64.powf(rng.gen_range(-3.0..-1.5));
        }
        if rng.gen::<f64>() < MUTATION_RATE {
            config.steps = *[200, 400, 600, 800, 1000].choose(&mut rng).unwrap();
        }
        
        // Clamp to valid ranges and fix consistency
        config.clamp();
        
        Self::from_config(&config, "mutant")
    }

    fn crossover(parent1: &Organism, parent2: &Organism) -> Self {
        let mut rng = rand::thread_rng();
        let mut config = parent1.config.clone();
        
        // Mix parameters from both parents
        if rng.gen() {
            config.n_emb = parent2.config.n_emb;
        }
        if rng.gen() {
            config.n_ctx = parent2.config.n_ctx;
        }
        if rng.gen() {
            config.n_layer = parent2.config.n_layer;
        }
        if rng.gen() {
            config.n_head = parent2.config.n_head;
        }
        if rng.gen() {
            config.n_ff_exp = parent2.config.n_ff_exp;
        }
        if rng.gen() {
            config.lr = parent2.config.lr;
        }
        if rng.gen() {
            config.steps = parent2.config.steps;
        }
        
        // Clamp to valid ranges and fix consistency
        config.clamp();
        
        Self::from_config(&config, "cross")
    }

    fn evaluate(&mut self, gpu_accel: &Arc<microgpt_rust::gpu_accel::GpuAccelerator>) {
        if self.evaluated {
            return;
        }
        
        let start = Instant::now();
        let result = std::panic::catch_unwind(|| {
            train_with_gpu(&self.config, true, gpu_accel)
        });
        
        match result {
            Ok(r) => {
                self.loss = r.final_loss;
                self.evaluated = true;
                let gpu_status = if gpu_accel.is_available() { "GPU" } else { "CPU" };
                println!("[{}] {} | loss={:.4} ({:.1}s)", 
                    gpu_status, self.desc(), self.loss, start.elapsed().as_secs_f64());
            }
            Err(_) => {
                self.loss = f64::MAX;
                self.evaluated = true;
                println!("[{}] {} | PANICKED", 
                    if gpu_accel.is_available() { "GPU" } else { "CPU" }, self.desc());
            }
        }
    }

    fn desc(&self) -> String {
        format!("Emb:{} Head:{} Lay:{} Ctx:{} FF:{} LR:{:.4} Steps:{} | {}",
            self.config.n_emb, self.config.n_head, self.config.n_layer,
            self.config.n_ctx, self.config.n_ff_exp, self.config.lr, self.config.steps,
            self.origin)
    }
}

fn train_with_gpu(config: &TrainingConfig, silent: bool, gpu_accel: &Arc<microgpt_rust::gpu_accel::GpuAccelerator>) -> microgpt_rust::TrainingResult {
    // Use regular training with GPU available
    microgpt_rust::train_and_generate(config, silent)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all("experiments").ok();
    
    println!("=== MicroGPT Evolution Engine - GENESIS BRANCH ===");
    println!("Starting from scratch: 1s in everything");
    println!("Target: loss < {:.1}", TARGET_LOSS);
    println!("Population: {}, Generations: {}", POPULATION_SIZE, NUM_GENERATIONS);
    
    // Initialize GPU accelerator
    let gpu_accel = Arc::new(create_gpu_accelerator()?);
    println!("GPU Available: {}", gpu_accel.is_available());
    
    // Load training data
    if std::fs::metadata(INPUT_FILE).is_err() {
        let _ = std::process::Command::new("curl")
            .args(["-o", INPUT_FILE, "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"])
            .output();
    }
    load_training_data(INPUT_FILE);

    // Start from genesis config - true 1s where possible
    let mut genesis_config = TrainingConfig {
        n_emb: 1,   // Start with 1
        n_ctx: 1,   // Start with 1  
        n_layer: 1,
        n_head: 1,
        n_ff_exp: 1,
        lr: 0.01,
        steps: 1000,
        input_file: INPUT_FILE.to_string(),
        gen_samples: 1,
        ..Default::default()
    };
    
    // Clamp genesis config to valid ranges (will fix consistency issues)
    genesis_config.clamp();
    
    println!("🌱 GENESIS: Starting from true 1s (clamped to valid ranges)");
    println!("Genesis config: Emb:{} Ctx:{} Lay:1 Head:1 FF:1 LR:0.01 Steps:1000", 
        genesis_config.n_emb, genesis_config.n_ctx);
    println!("Validation: {}", if genesis_config.is_reasonable() { "✅ PASS" } else { "❌ FAIL" });

    // Create initial population - start from genesis and mutations
    let mut population = Vec::with_capacity(POPULATION_SIZE);
    
    // Add the genesis organism as first member
    population.push(Organism::from_config(&genesis_config, "genesis"));
    
    // Fill the rest with mutations of genesis (not random!)
    while population.len() < POPULATION_SIZE {
        population.push(population[0].mutate());
    }

    let mut best_ever = population[0].clone();
    let total_start = Instant::now();

    println!("\n--- Starting Evolution from Genesis ---");

    // Evolution loop
    for gen in 0..NUM_GENERATIONS {
        println!("\n--- Generation {}/{} ---", gen + 1, NUM_GENERATIONS);
        
        // Evaluate population
        let gpu_accel_clone = gpu_accel.clone();
        population.par_iter_mut().for_each(|org| {
            if !org.evaluated {
                org.evaluate(&gpu_accel_clone);
            }
        });

        // Sort by loss (handle NaN values)
        population.sort_by(|a, b| {
            if a.loss.is_nan() && b.loss.is_nan() {
                std::cmp::Ordering::Equal
            } else if a.loss.is_nan() {
                std::cmp::Ordering::Greater  // NaN goes to end
            } else if b.loss.is_nan() {
                std::cmp::Ordering::Less   // NaN goes to end
            } else {
                a.loss.partial_cmp(&b.loss).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        // Print results
        println!("Results:");
        for (i, org) in population.iter().take(5).enumerate() {
            println!("  #{}: {:.4} | {}", i + 1, org.loss, org.desc());
        }

        let best_this_gen = population[0].clone();
        println!("Best this gen: {:.4} | {}", best_this_gen.loss, best_this_gen.desc());

        // Update best ever
        if best_this_gen.loss < best_ever.loss {
            println!("🎉 NEW BEST! {:.6} -> {:.6}", best_ever.loss, best_this_gen.loss);
            best_ever = best_this_gen.clone();
            
            // Save the best genome
            println!("💾 Saving genesis best genome...");
            if let Err(e) = best_ever.config.save_genome(best_ever.loss, gen + 1) {
                println!("Failed to save genome: {}", e);
            }
        }

        // Check target
        if best_ever.loss < TARGET_LOSS {
            println!("🎯 TARGET REACHED! Loss: {:.6}", best_ever.loss);
            break;
        }

        // Create next generation
        let mut next_pop = Vec::with_capacity(POPULATION_SIZE);
        
        // Keep elite
        next_pop.push(population[0].clone());
        
        // Add mutations of elite
        for _ in 0..3 {
            next_pop.push(population[0].mutate());
        }
        
        // Add crossovers
        for i in 0..2 {
            let parent2_idx = (i + 1) % population.len().min(3);
            next_pop.push(Organism::crossover(&population[0], &population[parent2_idx]));
        }
        
        // Add random diversity
        while next_pop.len() < POPULATION_SIZE {
            next_pop.push(Organism::random());
        }
        
        population = next_pop;
    }

    println!("\n=== Genesis Evolution Complete ===");
    println!("Total Time: {:.1}s", total_start.elapsed().as_secs_f64());
    println!("GPU Used: {}", gpu_accel.is_available());
    println!("Best loss: {:.6}", best_ever.loss);
    println!("Best config: {}", best_ever.desc());
    
    // Show improvement from genesis
    println!("\n🌱 Genesis Progress:");
    println!("  Started from: Emb:1, Ctx:1, Lay:1, Head:1, FF:1");
    println!("  Evolved to: {}", best_ever.desc());
    println!("  Improvement: {:.6} → {:.6}", 
        if best_ever.loss < f64::MAX { best_ever.loss } else { 0.0 }, 
        best_ever.loss);
    
    Ok(())
}
